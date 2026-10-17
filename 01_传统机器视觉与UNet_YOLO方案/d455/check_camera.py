#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Check that the D455 is connected and streaming, and print what measurement needs."""

import json
import os
import sys

import cv2
import numpy as np
import pyrealsense2 as rs

from rs_common import DepthFilter, center_depth_m, intrinsics_dict, list_devices, mm_per_pixel, save_bgr, start_pipeline

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    print("pyrealsense2", rs.__version__ if hasattr(rs, "__version__") else "")
    devices = list_devices()
    if not devices:
        print("没有检测到 RealSense 相机：请确认 D455 已接到电脑的 USB 3 口（蓝色口），线缆为 USB 3 数据线。")
        return 1
    for device in devices:
        print("相机 %s  序列号 %s  固件 %s  USB %s" % (
            device["name"], device["serial"], device["firmware"], device["usb"]))
        if device["usb"] and not device["usb"].startswith("3"):
            print("  警告：当前是 USB %s，1280x720 双流可能起不来，请换 USB 3 口或线。" % device["usb"])

    pipeline, profile, depth_scale = start_pipeline(devices[0]["serial"])
    align = rs.align(rs.stream.color)
    depth_filter = DepthFilter()
    try:
        for _ in range(30):
            pipeline.wait_for_frames()
        frames = align.process(pipeline.wait_for_frames())
        color = np.asanyarray(frames.get_color_frame().get_data())
        depth = depth_filter(frames.get_depth_frame())
        depth_m = np.asanyarray(depth.get_data()).astype(np.float32) * depth_scale
        color_intr = intrinsics_dict(profile.get_stream(rs.stream.color))
    finally:
        pipeline.stop()

    distance = center_depth_m(depth_m)
    valid_ratio = float((depth_m > 0).mean())
    print("彩色内参 %s" % json.dumps(color_intr, ensure_ascii=False))
    print("深度比例 %.6f 米/单位，有效深度像素 %.0f%%" % (depth_scale, 100 * valid_ratio))
    if distance > 0:
        print("画面中心距离 %.3f m，此处 1 像素约 %.3f mm" % (distance, mm_per_pixel(distance, color_intr["fx"])))
    else:
        print("画面中心没有有效深度（太近、反光或遮挡）。D455 最近约 0.4~0.6 m。")
    out = os.path.join(HERE, "check_frame.png")
    save_bgr(out, color)
    print("测试画面已保存 %s" % out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
