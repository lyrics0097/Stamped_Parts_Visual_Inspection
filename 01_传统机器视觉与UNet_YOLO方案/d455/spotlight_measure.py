#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Preview the D455, and measure only when the Logitech Spotlight button is pressed.

The live view is not detected. A press collects several frames, takes the
median colour and depth, then runs chalk measurement once. That is steadier
than judging every frame. Space and Page Down also trigger a shot.
"""

import argparse
import ctypes
import os
import threading
import time
from ctypes import wintypes
from datetime import datetime

import cv2
import numpy as np
import pyrealsense2 as rs
from PIL import Image, ImageDraw, ImageFont

from measure_chalk import attach_millimetres, draw_overlay, find_strokes
from rs_common import DepthFilter, intrinsics_dict, list_devices, save_bgr, start_pipeline

HERE = os.path.dirname(os.path.abspath(__file__))
WM_INPUT = 0x00FF
RID_INPUT = 0x10000003
RIM_TYPEKEYBOARD = 1
RIDEV_INPUTSINK = 0x00000100
RIDI_DEVICENAME = 0x20000007
RI_KEY_BREAK = 0x01
# Spotlight dongle and the usual presenter keys (Page Down / Page Up).
SPOTLIGHT_PIDS = ("PID_C53E", "PID_C53C", "PID_C52D", "PID_C539", "PID_B503")
FALLBACK_KEYS = {0x22, 0x21, 0x20}  # Page Down, Page Up, Space

user32 = ctypes.windll.user32
kernel32 = ctypes.windll.kernel32
user32.DefWindowProcW.argtypes = [wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM]
user32.DefWindowProcW.restype = ctypes.c_ssize_t
user32.CreateWindowExW.restype = wintypes.HWND
user32.CreateWindowExW.argtypes = [
    wintypes.DWORD, wintypes.LPCWSTR, wintypes.LPCWSTR, wintypes.DWORD,
    ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
    wintypes.HWND, wintypes.HMENU, wintypes.HINSTANCE, wintypes.LPVOID,
]
user32.RegisterClassW.restype = wintypes.ATOM
kernel32.GetModuleHandleW.restype = wintypes.HMODULE
kernel32.GetModuleHandleW.argtypes = [wintypes.LPCWSTR]
user32.GetRawInputData.argtypes = [
    wintypes.HANDLE, wintypes.UINT, wintypes.LPVOID, ctypes.POINTER(wintypes.UINT), wintypes.UINT,
]
user32.GetRawInputDeviceInfoW.argtypes = [
    wintypes.HANDLE, wintypes.UINT, wintypes.LPVOID, ctypes.POINTER(wintypes.UINT),
]

WNDPROC = ctypes.WINFUNCTYPE(ctypes.c_ssize_t, wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM)


class RAWINPUTDEVICE(ctypes.Structure):
    _fields_ = [
        ("usUsagePage", wintypes.USHORT),
        ("usUsage", wintypes.USHORT),
        ("dwFlags", wintypes.DWORD),
        ("hwndTarget", wintypes.HWND),
    ]


user32.RegisterRawInputDevices.argtypes = [ctypes.POINTER(RAWINPUTDEVICE), wintypes.UINT, wintypes.UINT]
user32.RegisterRawInputDevices.restype = wintypes.BOOL


class RAWINPUTHEADER(ctypes.Structure):
    _fields_ = [
        ("dwType", wintypes.DWORD),
        ("dwSize", wintypes.DWORD),
        ("hDevice", wintypes.HANDLE),
        ("wParam", wintypes.WPARAM),
    ]


class RAWKEYBOARD(ctypes.Structure):
    _fields_ = [
        ("MakeCode", wintypes.USHORT),
        ("Flags", wintypes.USHORT),
        ("Reserved", wintypes.USHORT),
        ("VKey", wintypes.USHORT),
        ("Message", wintypes.UINT),
        ("ExtraInformation", wintypes.ULONG),
    ]


class RAWINPUT(ctypes.Structure):
    class _DATA(ctypes.Union):
        _fields_ = [("keyboard", RAWKEYBOARD)]

    _anonymous_ = ("data",)
    _fields_ = [("header", RAWINPUTHEADER), ("data", _DATA)]


class WNDCLASSW(ctypes.Structure):
    _fields_ = [
        ("style", wintypes.UINT),
        ("lpfnWndProc", WNDPROC),
        ("cbClsExtra", ctypes.c_int),
        ("cbWndExtra", ctypes.c_int),
        ("hInstance", wintypes.HINSTANCE),
        ("hIcon", wintypes.HICON),
        ("hCursor", wintypes.HANDLE),
        ("hbrBackground", wintypes.HBRUSH),
        ("lpszMenuName", wintypes.LPCWSTR),
        ("lpszClassName", wintypes.LPCWSTR),
    ]


trigger = threading.Event()
_wndproc_ref = None


def _device_name(handle):
    size = wintypes.UINT(0)
    user32.GetRawInputDeviceInfoW(handle, RIDI_DEVICENAME, None, ctypes.byref(size))
    if not size.value:
        return ""
    buffer = ctypes.create_unicode_buffer(size.value)
    user32.GetRawInputDeviceInfoW(handle, RIDI_DEVICENAME, buffer, ctypes.byref(size))
    return buffer.value


def _is_spotlight(name):
    upper = name.upper()
    if "SPOTLIGHT" in upper or "PRESENTER" in upper:
        return True
    return "VID_046D" in upper and any(pid in upper for pid in SPOTLIGHT_PIDS)


def _on_input(lparam):
    size = wintypes.UINT(0)
    user32.GetRawInputData(lparam, RID_INPUT, None, ctypes.byref(size), ctypes.sizeof(RAWINPUTHEADER))
    if not size.value:
        return
    raw_buffer = ctypes.create_string_buffer(size.value)
    user32.GetRawInputData(lparam, RID_INPUT, raw_buffer, ctypes.byref(size), ctypes.sizeof(RAWINPUTHEADER))
    raw = ctypes.cast(raw_buffer, ctypes.POINTER(RAWINPUT)).contents
    if raw.header.dwType != RIM_TYPEKEYBOARD or raw.keyboard.Flags & RI_KEY_BREAK:
        return
    name = _device_name(raw.header.hDevice)
    vkey = raw.keyboard.VKey
    if _is_spotlight(name) or vkey in FALLBACK_KEYS:
        print("按键触发  vk=%s  设备=%s" % (vkey, name), flush=True)
        trigger.set()


def _wndproc(hwnd, message, wparam, lparam):
    if message == WM_INPUT:
        _on_input(lparam)
        return 0
    return user32.DefWindowProcW(hwnd, message, wparam, lparam)


def _listen():
    global _wndproc_ref
    _wndproc_ref = WNDPROC(_wndproc)
    window_class = WNDCLASSW()
    window_class.lpfnWndProc = _wndproc_ref
    window_class.hInstance = kernel32.GetModuleHandleW(None)
    window_class.lpszClassName = "D455SpotlightHook"
    user32.RegisterClassW(ctypes.byref(window_class))
    hwnd = user32.CreateWindowExW(0, window_class.lpszClassName, "D455SpotlightHook", 0, 0, 0, 0, 0, 0, None, window_class.hInstance, None)
    device = RAWINPUTDEVICE(0x01, 0x06, RIDEV_INPUTSINK, hwnd)
    if not user32.RegisterRawInputDevices(ctypes.byref(device), 1, ctypes.sizeof(device)):
        print("Spotlight 按键监听注册失败，仍可用空格或 Page Down。", flush=True)
    message = wintypes.MSG()
    while user32.GetMessageW(ctypes.byref(message), None, 0, 0) > 0:
        user32.TranslateMessage(ctypes.byref(message))
        user32.DispatchMessageW(ctypes.byref(message))


def _banner(image, text):
    canvas = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(canvas)
    font_path = r"C:\Windows\Fonts\msyh.ttc"
    font = ImageFont.truetype(font_path, 32) if os.path.isfile(font_path) else ImageFont.load_default()
    draw.rectangle((0, 0, canvas.size[0], 52), fill=(0, 0, 0))
    draw.text((16, 8), text, fill=(255, 255, 255), font=font)
    return cv2.cvtColor(np.array(canvas), cv2.COLOR_RGB2BGR)


def _read_pair(pipeline, align, depth_filter, depth_scale):
    try:
        frames = align.process(pipeline.wait_for_frames(1000))
    except RuntimeError:
        return None
    color_frame = frames.get_color_frame()
    depth_frame = frames.get_depth_frame()
    if not color_frame or not depth_frame:
        return None
    depth_frame = depth_filter(depth_frame)
    color = np.asanyarray(color_frame.get_data())
    depth_m = np.asanyarray(depth_frame.get_data()).astype(np.float32) * depth_scale
    return color, depth_m


def collect_median(pipeline, align, depth_filter, depth_scale, count):
    colors, depths = [], []
    while len(colors) < count:
        pair = _read_pair(pipeline, align, depth_filter, depth_scale)
        if pair is None:
            continue
        colors.append(pair[0])
        depths.append(pair[1])
    color = np.median(np.stack(colors), axis=0).astype(np.uint8)
    depth_stack = np.stack(depths).astype(np.float32)
    depth_stack[depth_stack <= 0] = np.nan
    depth_m = np.nan_to_num(np.nanmedian(depth_stack, axis=0)).astype(np.float32)
    return color, depth_m


def main():
    parser = argparse.ArgumentParser(description="按 Spotlight 采集多帧后再测量粉笔印长宽")
    parser.add_argument("--frames", type=int, default=8, help="每次按键采集并取中值的帧数")
    parser.add_argument("--bright", type=int, default=175)
    args = parser.parse_args()
    devices = list_devices()
    if not devices:
        print("没有检测到 D455。")
        return
    session = os.path.join(HERE, "captures", datetime.now().strftime("spotlight_%Y%m%d_%H%M%S"))
    os.makedirs(session, exist_ok=True)
    threading.Thread(target=_listen, daemon=True).start()
    pipeline, profile, depth_scale = start_pipeline(devices[0]["serial"])
    align = rs.align(rs.stream.color)
    depth_filter = DepthFilter()
    fx = intrinsics_dict(profile.get_stream(rs.stream.color))["fx"]
    window = "D455 按键测量"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window, 960, 540)
    cv2.setWindowProperty(window, cv2.WND_PROP_TOPMOST, 1)
    saved = 0
    last_lines = []
    quiet_until = 0.0
    print("实时预览已打开。对准后按 Spotlight、Page Down 或空格再测量。q 退出。", flush=True)
    print("结果保存在 %s" % session, flush=True)
    try:
        while True:
            now = time.perf_counter()
            if trigger.is_set() and now >= quiet_until:
                trigger.clear()
                quiet_until = now + 1.5
                color, depth_m = collect_median(pipeline, align, depth_filter, depth_scale, args.frames)
                gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
                strokes = find_strokes(gray, bright=args.bright)
                attach_millimetres(strokes, depth_m, fx)
                saved += 1
                annotated = draw_overlay(color, strokes)
                path = os.path.join(session, "%04d.png" % saved)
                save_bgr(path, annotated)
                last_lines = ["第%d次  检出 %d 条" % (saved, len(strokes))]
                for index, stroke in enumerate(strokes, start=1):
                    last_lines.append("%d  长 %.0f px  宽 %.0f px" % (
                        index, stroke["length_px"], stroke["width_px"]))
                print("第 %d 次  检出 %d 条  %s" % (saved, len(strokes), path), flush=True)
                for index, stroke in enumerate(strokes, start=1):
                    print("  %d  长 %.0f px  宽 %.0f px" % (
                        index, stroke["length_px"], stroke["width_px"]), flush=True)
            elif trigger.is_set():
                trigger.clear()
            pair = _read_pair(pipeline, align, depth_filter, depth_scale)
            if pair is None:
                continue
            status = "实时预览，对准后按键测量"
            if last_lines:
                status = "实时预览 | " + "  ".join(last_lines[:3])
            shown = _banner(pair[0], status)
            cv2.imshow(window, cv2.resize(shown, (960, 540)))
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord(" "):
                trigger.set()
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
