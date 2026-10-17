#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Measure chalk strokes on a blackboard with the D455, in millimetres.

White strokes are found by a white top-hat against the dark board. Length and
width come from the minimum rotated rectangle. Pixel size is converted with
the median depth inside each stroke and the colour-camera focal length:

    millimetres = pixels * depth_m * 1000 / fx

That assumes the board faces the camera. Keys: q / ESC quit, SPACE save.
"""

import argparse
import os
import time

import cv2
import numpy as np
import pyrealsense2 as rs
from PIL import Image, ImageDraw, ImageFont

from rs_common import DepthFilter, intrinsics_dict, list_devices, save_bgr, start_pipeline

HERE = os.path.dirname(os.path.abspath(__file__))


def _axis(part):
    ys, xs = np.nonzero(part)
    points = np.stack([xs, ys], axis=1).astype(np.float64)
    center = points.mean(axis=0)
    if len(points) < 3:
        direction = np.array([1.0, 0.0])
    else:
        _, vectors = np.linalg.eigh(np.cov((points - center).T))
        direction = vectors[:, -1]
    normal = np.array([-direction[1], direction[0]])
    proj = (points - center) @ direction
    perp = (points - center) @ normal
    return center, direction, normal, proj, perp


def _pixel_size(part):
    """End-to-end length, and the median thickness across the stroke."""
    center, direction, normal, proj, perp = _axis(part)
    length_px = float(proj.max() - proj.min() + 1.0)
    widths = []
    for station in np.linspace(proj.min(), proj.max(), 21):
        band = np.abs(proj - station) <= 1.5
        if int(band.sum()) < 3:
            continue
        widths.append(float(np.percentile(np.abs(perp[band]), 90) * 2.0))
    width_px = float(np.median(widths)) if widths else 1.0
    rect = cv2.minAreaRect(np.stack(np.nonzero(part)[::-1], axis=1).astype(np.float32))
    return max(length_px, 1.0), max(width_px, 1.0), rect


def _extend_faint(part, faint):
    """Add the dim continuation of a stroke, staying inside its own corridor."""
    count, labels = cv2.connectedComponents(faint.astype(np.uint8), connectivity=8)
    labels_on_core = np.unique(labels[part])
    grown = np.zeros(part.shape, dtype=bool)
    for label_id in labels_on_core:
        if label_id == 0:
            continue
        grown |= labels == label_id
    center, direction, normal, proj, perp = _axis(part)
    half = max(8.0, float(np.percentile(np.abs(perp), 95)) * 1.6)
    ys, xs = np.nonzero(grown)
    offset = (xs - center[0]) * normal[0] + (ys - center[1]) * normal[1]
    along = (xs - center[0]) * direction[0] + (ys - center[1]) * direction[1]
    keep = (np.abs(offset) <= half) & (along >= proj.min() - 40) & (along <= proj.max() + 40)
    grown[ys[~keep], xs[~keep]] = False
    return grown | part


def find_strokes(gray, bright=175, min_area=500, min_length_px=100, min_aspect=8.0):
    """White chalk on a green board. Tuned on Desktop/收件箱/划痕.

    A bright core locates the stroke. A lower threshold then pulls in the faint
    ends, but only along that stroke, so board scratches stay out.
    """
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (51, 51))
    tophat = cv2.morphologyEx(blurred, cv2.MORPH_TOPHAT, kernel)
    core = np.where((tophat >= 28) & (blurred >= bright), 255, 0).astype(np.uint8)
    core = cv2.morphologyEx(core, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    faint = (tophat >= 12) & (blurred >= bright - 30)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(core, connectivity=8)
    strokes = []
    for label_id in range(1, count):
        if int(stats[label_id, cv2.CC_STAT_AREA]) < min_area:
            continue
        part = _extend_faint(labels == label_id, faint)
        length_px, width_px, rect = _pixel_size(part)
        if length_px < min_length_px or length_px / max(width_px, 1.0) < min_aspect:
            continue
        strokes.append({"part": part, "rect": rect, "length_px": length_px, "width_px": width_px})
    return strokes


def _valid_depth(values):
    """D455 reports about 65.5 m when stereo depth failed. Indoor board shots stay within 4 m."""
    values = values[(values >= 0.35) & (values <= 4.0)]
    if values.size < 30:
        return 0.0
    return float(np.median(values))


def attach_millimetres(strokes, depth_m, fx):
    kernel = np.ones((15, 15), np.uint8)
    for stroke in strokes:
        depth = _valid_depth(depth_m[stroke["part"]])
        if depth == 0.0:
            near = cv2.dilate(stroke["part"].astype(np.uint8), kernel).astype(bool) & ~stroke["part"]
            depth = _valid_depth(depth_m[near])
        scale = depth * 1000.0 / fx if depth > 0 else 0.0
        stroke["depth_m"] = depth
        stroke["length_mm"] = stroke["length_px"] * scale
        stroke["width_mm"] = stroke["width_px"] * scale


def draw_overlay(color, strokes):
    for stroke in strokes:
        box = np.round(cv2.boxPoints(stroke["rect"])).astype(np.int32)
        cv2.polylines(color, [box], True, (0, 255, 0), 2)
    image = Image.fromarray(cv2.cvtColor(color, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(image)
    font_path = r"C:\Windows\Fonts\msyh.ttc"
    font = ImageFont.truetype(font_path, 28) if os.path.isfile(font_path) else ImageFont.load_default()
    for index, stroke in enumerate(strokes):
        text = "划痕%d  长 %.0f px  宽 %.0f px" % (
            index + 1, stroke["length_px"], stroke["width_px"])
        draw.text((16, 16 + index * 36), text, fill=(0, 255, 0), font=font)
    if not strokes:
        draw.text((16, 16), "没有检出粉笔印", fill=(255, 80, 80), font=font)
    return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)


def grab_measured(pipeline, align, depth_filter, depth_scale, fx, bright):
    frames = align.process(pipeline.wait_for_frames())
    color_frame = frames.get_color_frame()
    depth_frame = frames.get_depth_frame()
    if not color_frame or not depth_frame:
        return None
    depth_frame = depth_filter(depth_frame)
    color = np.asanyarray(color_frame.get_data())
    depth_m = np.asanyarray(depth_frame.get_data()).astype(np.float32) * depth_scale
    gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
    strokes = find_strokes(gray, bright=bright)
    attach_millimetres(strokes, depth_m, fx)
    return color, depth_m, strokes


def main():
    parser = argparse.ArgumentParser(description="D455 测量黑板粉笔印的长宽（毫米）")
    parser.add_argument("--bright", type=int, default=150, help="粉笔印最低亮度，0-255")
    parser.add_argument("--once", action="store_true", help="只测一帧并保存，不打开窗口")
    args = parser.parse_args()
    devices = list_devices()
    if not devices:
        print("没有检测到 D455。")
        return
    pipeline, profile, depth_scale = start_pipeline(devices[0]["serial"])
    align = rs.align(rs.stream.color)
    depth_filter = DepthFilter()
    fx = intrinsics_dict(profile.get_stream(rs.stream.color))["fx"]
    try:
        for _ in range(20):
            pipeline.wait_for_frames()
        if args.once:
            grabbed = grab_measured(pipeline, align, depth_filter, depth_scale, fx, args.bright)
            if grabbed is None:
                print("没有读到帧。")
                return
            color, depth_m, strokes = grabbed
            out = os.path.join(HERE, "measure_once.png")
            save_bgr(out, draw_overlay(color, strokes))
            print("检出 %d 条，画面 %s" % (len(strokes), out))
            for index, stroke in enumerate(strokes, start=1):
                print("  %d  长 %.1f mm  宽 %.1f mm  距离 %.2f m" % (
                    index, stroke["length_mm"], stroke["width_mm"], stroke["depth_m"]))
            return
        window = "D455 实时画面"
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window, 1280, 480)
        cv2.setWindowProperty(window, cv2.WND_PROP_TOPMOST, 1)
        print("实时窗口已打开。q 退出，空格保存当前画面。")
        while True:
            grabbed = grab_measured(pipeline, align, depth_filter, depth_scale, fx, args.bright)
            if grabbed is None:
                continue
            color, depth_m, strokes = grabbed
            view = draw_overlay(color, strokes)
            depth_u8 = cv2.convertScaleAbs(np.clip(depth_m, 0, 4.0), alpha=255.0 / 4.0)
            depth_view = cv2.applyColorMap(depth_u8, cv2.COLORMAP_JET)
            depth_view[depth_m <= 0] = 0
            panel = np.hstack([cv2.resize(view, (640, 360)), cv2.resize(depth_view, (640, 360))])
            cv2.imshow(window, panel)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord(" "):
                path = os.path.join(HERE, "measure_%s.png" % time.strftime("%H%M%S"))
                save_bgr(path, view)
                print("已保存 %s ，%d 条" % (path, len(strokes)))
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
