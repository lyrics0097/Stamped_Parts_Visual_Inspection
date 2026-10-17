# -*- coding: utf-8 -*-
"""Shared RealSense D455 helpers: stream setup, depth filtering, intrinsics."""

import os

import cv2
import numpy as np
import pyrealsense2 as rs

COLOR_SIZE = (1280, 720)
DEPTH_SIZE = (1280, 720)
FPS = 30


def list_devices():
    devices = []
    for device in rs.context().query_devices():
        def info(key):
            return device.get_info(key) if device.supports(key) else ""
        devices.append({
            "name": info(rs.camera_info.name),
            "serial": info(rs.camera_info.serial_number),
            "firmware": info(rs.camera_info.firmware_version),
            "usb": info(rs.camera_info.usb_type_descriptor),
            "product_line": info(rs.camera_info.product_line),
        })
    return devices


def start_pipeline(serial=None):
    pipeline = rs.pipeline()
    config = rs.config()
    if serial:
        config.enable_device(serial)
    config.enable_stream(rs.stream.color, COLOR_SIZE[0], COLOR_SIZE[1], rs.format.bgr8, FPS)
    config.enable_stream(rs.stream.depth, DEPTH_SIZE[0], DEPTH_SIZE[1], rs.format.z16, FPS)
    profile = pipeline.start(config)
    depth_sensor = profile.get_device().first_depth_sensor()
    return pipeline, profile, depth_sensor.get_depth_scale()


class DepthFilter:
    """Spatial + temporal smoothing and hole filling, run in the disparity domain."""

    def __init__(self):
        self.to_disparity = rs.disparity_transform(True)
        self.to_depth = rs.disparity_transform(False)
        self.spatial = rs.spatial_filter()
        self.temporal = rs.temporal_filter()
        self.holes = rs.hole_filling_filter()

    def __call__(self, depth_frame):
        frame = self.to_disparity.process(depth_frame)
        frame = self.spatial.process(frame)
        frame = self.temporal.process(frame)
        frame = self.to_depth.process(frame)
        return self.holes.process(frame)


def intrinsics_dict(stream_profile):
    intr = stream_profile.as_video_stream_profile().get_intrinsics()
    return {
        "width": intr.width, "height": intr.height,
        "fx": intr.fx, "fy": intr.fy, "ppx": intr.ppx, "ppy": intr.ppy,
        "model": str(intr.model), "coeffs": list(intr.coeffs),
    }


def center_depth_m(depth_m, size=21):
    """Median of valid depth in a small window at the image center, in metres."""
    height, width = depth_m.shape
    half = size // 2
    window = depth_m[height // 2 - half: height // 2 + half + 1, width // 2 - half: width // 2 + half + 1]
    valid = window[window > 0]
    return float(np.median(valid)) if valid.size else 0.0


def save_bgr(path, image):
    """cv2.imwrite cannot open paths that contain non-ASCII characters."""
    ok, encoded = cv2.imencode(os.path.splitext(path)[1] or ".png", image)
    if not ok:
        raise OSError("编码图片失败: %s" % path)
    encoded.tofile(path)


def mm_per_pixel(depth_m, fx):
    """Size on a fronto-parallel plane covered by one pixel at the given depth."""
    return depth_m * 1000.0 / fx if depth_m > 0 else 0.0
