#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Live D455 preview; press SPACE to save an aligned colour + depth shot.

Each shot writes, under captures/<session>/:
  <id>_color.png   colour image (BGR, 1280x720)
  <id>_depth.png   depth aligned to the colour image, uint16, unit = depth_scale metres
  <id>_meta.json   intrinsics, depth scale, centre distance and mm per pixel
Keys: SPACE save, q / ESC quit.
"""

import argparse
import json
import os
import time
from datetime import datetime

import cv2
import numpy as np
import pyrealsense2 as rs

from rs_common import DepthFilter, center_depth_m, intrinsics_dict, list_devices, mm_per_pixel, save_bgr, start_pipeline

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    parser = argparse.ArgumentParser(description="D455 采集：空格保存，q 退出")
    parser.add_argument("--out", default=os.path.join(HERE, "captures"))
    parser.add_argument("--session", default=datetime.now().strftime("%Y%m%d_%H%M%S"))
    parser.add_argument("--no-filter", action="store_true", help="保存未滤波的原始深度")
    args = parser.parse_args()

    devices = list_devices()
    if not devices:
        print("没有检测到 D455，请先运行 check_camera.py 排查。")
        return
    session_dir = os.path.join(args.out, args.session)
    os.makedirs(session_dir, exist_ok=True)
    pipeline, profile, depth_scale = start_pipeline(devices[0]["serial"])
    align = rs.align(rs.stream.color)
    depth_filter = DepthFilter()
    color_intr = intrinsics_dict(profile.get_stream(rs.stream.color))
    colorizer = rs.colorizer()
    saved = 0
    print("保存到 %s ；空格保存，q 退出" % session_dir)
    try:
        while True:
            frames = align.process(pipeline.wait_for_frames())
            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()
            if not color_frame or not depth_frame:
                continue
            if not args.no_filter:
                depth_frame = depth_filter(depth_frame)
            color = np.asanyarray(color_frame.get_data())
            depth_raw = np.asanyarray(depth_frame.get_data())
            depth_m = depth_raw.astype(np.float32) * depth_scale
            distance = center_depth_m(depth_m)
            scale = mm_per_pixel(distance, color_intr["fx"])

            preview = color.copy()
            h, w = preview.shape[:2]
            cv2.drawMarker(preview, (w // 2, h // 2), (0, 255, 0), cv2.MARKER_CROSS, 30, 2)
            cv2.putText(preview, "center %.3f m  %.3f mm/px  saved %d" % (distance, scale, saved),
                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
            depth_view = np.asanyarray(colorizer.colorize(depth_frame).get_data())
            both = np.hstack([preview, depth_view])
            cv2.imshow("D455  colour | depth   (SPACE save, q quit)", cv2.resize(both, (both.shape[1] // 2, both.shape[0] // 2)))

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord(" "):
                saved += 1
                stem = os.path.join(session_dir, "%04d" % saved)
                save_bgr(stem + "_color.png", color)
                save_bgr(stem + "_depth.png", depth_raw)
                meta = {
                    "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "device": devices[0],
                    "depth_scale_m": depth_scale,
                    "depth_aligned_to": "color",
                    "depth_filtered": not args.no_filter,
                    "color_intrinsics": color_intr,
                    "center_depth_m": distance,
                    "center_mm_per_px": scale,
                }
                with open(stem + "_meta.json", "w", encoding="utf-8") as handle:
                    json.dump(meta, handle, ensure_ascii=False, indent=2)
                print("已保存 %s  中心 %.3f m  %.3f mm/px" % (os.path.basename(stem), distance, scale))
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
