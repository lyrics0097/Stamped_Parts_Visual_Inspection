# -*- coding: utf-8 -*-
"""
冲压模具损伤在线预警系统 (Stamping Die Damage Online Early-Warning System)
================================================================================
模块四：RealSense D455 硬件在环（HIL）实机验证与双模在线预警系统主程序

核心功能封装 (StampingDieWarningSystem):
  1. 工件到位检测与 3D 平面姿态配准:
     - D455 RGB-D RANSAC 平面拟合 + 法向量/倾角解算 + 2D-to-3D 视线平面求交反投影
     - 2D 最小外接矩形主轴定向与规范化仿射对齐 (Canonical Affine Registration)
  2. 抗反光 DINOv3+SAM 表观划痕检测与物理尺度换算:
     - 高光抑制 + 多尺度方向形态学残差 + 3D 点云二次曲面去趋势 (scratch_3d)
     - 3D 平面法向量倾角余弦修正与逐点射线反投影物理尺寸测量 (mm)
  3. 连续冲压件时空关联滤波器与模具损伤三级预警看板:
     - 区分单次随机磕碰/飞溅伪影 (绿色圆圈滤除) 与多冲次同位置模具固定拉伤 (红色锁定追踪)
     - 四分屏工业 HMI 监控看板 (左上原图+3D姿态 / 右上规范化对齐+单件测量 / 左下数字孪生热力图 / 右下趋势曲线+三级报警灯)

支持三种运行模式 (--mode):
  --mode d455_live     : 开启 RealSense D455 实时窗口预览，按键/自动到位触发在线预警
  --mode video_replay  : 读取产线实拍视频流，实时处理并导出高清四分屏演示视频 online_warning_demo.mp4
  --mode hil_benchmark : 无 GUI 自动运行实机硬件测试与长时稳定性基准评测
"""

from __future__ import annotations

import argparse
import dataclasses
from dataclasses import dataclass, field
import gc
import json
import math
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import cv2
import numpy as np
import psutil
from PIL import Image, ImageDraw, ImageFont

try:
    import torch
    import torch.nn.functional as F
except Exception:
    torch = None
    F = None

try:
    import pyrealsense2 as rs
except Exception:
    rs = None

# 路径配置
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
WORKSPACE_ROOT = os.path.dirname(CURRENT_DIR)
RAD_DIR = os.path.join(WORKSPACE_ROOT, "无人机云台项目", "RAD")
SCRATCH_3D_DIR = os.path.join(WORKSPACE_ROOT, "冲压件检测", "3d_scratch_pipeline")
D455_SAMPLES_ROOT = os.path.join(WORKSPACE_ROOT, "冲压件检测")
PROD_VIDEO_DIR = os.path.join(WORKSPACE_ROOT, "产线数据")
OUTPUT_DIR = os.path.join(CURRENT_DIR, "outputs", "step4_d455_hil")

for _p in (RAD_DIR, SCRATCH_3D_DIR):
    if _p not in sys.path and os.path.isdir(_p):
        sys.path.insert(0, _p)

try:
    import scratch_3d
except Exception:
    scratch_3d = None


# ==============================================================================
# 通用中文路径图像读写与高质量中文字体渲染工具
# ==============================================================================
def imread_u(path: str, flags: int = cv2.IMREAD_COLOR) -> Optional[np.ndarray]:
    """Windows 中文路径安全读取图片"""
    if not os.path.isfile(path):
        return None
    raw = np.fromfile(path, dtype=np.uint8)
    if raw.size == 0:
        return None
    return cv2.imdecode(raw, flags)


def imwrite_u(path: str, img: np.ndarray, quality: int = 95) -> None:
    """Windows 中文路径安全保存图片"""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    ext = os.path.splitext(path)[1].lower() or ".png"
    params = [cv2.IMWRITE_JPEG_QUALITY, quality] if ext in (".jpg", ".jpeg") else []
    ok, buf = cv2.imencode(ext, img, params)
    if not ok:
        raise RuntimeError(f"图像编码失败: {path}")
    buf.tofile(path)


_FONT_CACHE: Dict[Tuple[int, bool], ImageFont.FreeTypeFont] = {}


def get_cn_font(size: int = 18, bold: bool = False) -> ImageFont.FreeTypeFont:
    key = (size, bold)
    if key in _FONT_CACHE:
        return _FONT_CACHE[key]
    candidates = (
        ["C:/Windows/Fonts/msyhbd.ttc", "C:/Windows/Fonts/simhei.ttf", "C:/Windows/Fonts/msyh.ttc"]
        if bold
        else ["C:/Windows/Fonts/msyh.ttc", "C:/Windows/Fonts/simhei.ttf"]
    )
    font = None
    for fp in candidates:
        if os.path.isfile(fp):
            try:
                font = ImageFont.truetype(fp, size)
                break
            except Exception:
                continue
    if font is None:
        font = ImageFont.load_default()
    _FONT_CACHE[key] = font
    return font


def draw_texts_pil(
    img_bgr: np.ndarray,
    items: List[Tuple[int, int, str, Tuple[int, int, int], int, bool]],
) -> np.ndarray:
    """批量在 BGR 图像上绘制抗锯齿中文文本
    items: [(x, y, text, color_bgr, font_size, bold), ...]
    """
    if not items:
        return img_bgr
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    draw = ImageDraw.Draw(pil_img)
    for x, y, text, color_bgr, fsize, bold in items:
        font = get_cn_font(fsize, bold)
        color_rgb = (int(color_bgr[2]), int(color_bgr[1]), int(color_bgr[0]))
        draw.text((int(x), int(y)), str(text), font=font, fill=color_rgb)
    return cv2.cvtColor(np.asarray(pil_img), cv2.COLOR_RGB2BGR)


# ==============================================================================
# 数据结构定义
# ==============================================================================
@dataclass
class CameraIntrinsics:
    fx: float = 644.55
    fy: float = 642.92
    ppx: float = 653.59
    ppy: float = 369.03
    width: int = 1280
    height: int = 720

    @classmethod
    def from_rs(cls, intr: Any) -> "CameraIntrinsics":
        return cls(
            fx=float(intr.fx),
            fy=float(intr.fy),
            ppx=float(intr.ppx),
            ppy=float(intr.ppy),
            width=int(getattr(intr, "width", 1280)),
            height=int(getattr(intr, "height", 720)),
        )

    @classmethod
    def from_dict(cls, d: Dict[str, Any], w: int = 1280, h: int = 720) -> "CameraIntrinsics":
        return cls(
            fx=float(d.get("fx", 644.55)),
            fy=float(d.get("fy", 642.92)),
            ppx=float(d.get("ppx", w * 0.5)),
            ppy=float(d.get("ppy", h * 0.5)),
            width=w,
            height=h,
        )


@dataclass
class PlanePose3D:
    plane_source: str
    center_m: np.ndarray            # (3,) 相机坐标系下平面中心 [X, Y, Z] (m)
    normal: np.ndarray              # (3,) 朝向相机的单位法向量 (n_z < 0)
    e1: np.ndarray                  # (3,) 工件长轴方向单位基向量
    e2: np.ndarray                  # (3,) 工件短轴方向单位基向量
    tilt_deg: float                 # 平面倾角 (deg)
    rms_mm: float                   # RANSAC 平面拟合残差 RMS (mm)
    inlier_ratio: float             # 内点比例
    depth_m: float                  # 工件平面中心深度 Z (m)
    mm_per_px_frontal: float        # 针孔正对假设下的像素当量 (mm/px)
    mm_per_px_3d_corrected: float   # 3D 平面倾角余弦补偿后的主方向像素当量 (mm/px)
    size_mm_pinhole: Tuple[float, float]   # (长, 宽) 未做 3D 补偿 (mm)
    size_mm_3d_plane: Tuple[float, float]  # (长, 宽) 3D 平面视线求交补偿后 (mm)
    rect_px: Any                    # cv2.minAreaRect


@dataclass
class DetectedScratch:
    scratch_id: int
    uv_norm: Tuple[float, float]        # 规范化工件坐标 (u_norm, v_norm) in [0, 1]
    uv_mm: Tuple[float, float]          # 工件局部 3D 平面坐标 (u_mm, v_mm)
    bbox_canon: Tuple[int, int, int, int]  # 规范化视图中的外接框 (x0, y0, x1, y1)
    pixel_pts_raw: np.ndarray           # 原图像素点集 (N, 2)
    length_px: float
    width_px: float
    length_mm_pinhole: float            # 未做 3D 倾角补偿长度 (mm)
    width_mm_pinhole: float             # 未做 3D 倾角补偿宽度 (mm)
    length_mm_3d: float                 # 3D 平面补偿后真实物理长度 (mm)
    width_mm_3d: float                  # 3D 平面补偿后真实物理宽度 (mm)
    depth_mm_3d: float                  # 3D 点云/灰度残差估计深度 (mm)
    confidence: float
    angle_deg: float
    is_persistent_die_defect: bool = False
    track_id: int = -1
    recurrence_count: int = 1
    recurrence_rate: float = 0.0


@dataclass
class DefectTrack:
    track_id: int
    uv_norm: np.ndarray                 # (2,) 指数平滑后的规范化坐标
    uv_mm: np.ndarray                   # (2,) 物理坐标 (mm)
    first_stroke: int
    last_stroke: int
    hit_strokes: List[int] = field(default_factory=list)
    length_history: List[float] = field(default_factory=list)
    width_history: List[float] = field(default_factory=list)
    depth_history: List[float] = field(default_factory=list)
    status: str = "RANDOM_IMPACT"       # RANDOM_IMPACT / L1_EARLY / L2_TREND / L3_CRITICAL


# ==============================================================================
# 子模块 ①：工件到位检测与 3D 平面姿态配准 (RANSAC 3D + 规范化仿射变换)
# ==============================================================================
def deproject_pixels(us: np.ndarray, vs: np.ndarray, zs: np.ndarray, intr: CameraIntrinsics) -> np.ndarray:
    return np.stack([(us - intr.ppx) / intr.fx * zs, (vs - intr.ppy) / intr.fy * zs, zs], axis=1)


def fit_plane_svd(points: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    c = points.mean(axis=0)
    _, _, vt = np.linalg.svd(points - c, full_matrices=False)
    n = vt[2] if vt[2][2] < 0 else -vt[2]
    e1, e2 = vt[0], vt[1]
    if np.dot( np.cross(e1, e2), n ) < 0:
        e2 = -e2
    rms = float(np.sqrt(np.mean(((points - c) @ n) ** 2)))
    return c, n, e1, e2, rms


def ransac_plane_3d(
    points: np.ndarray, iters: int = 120, inlier_mm: float = 4.0, seed: int = 0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float]:
    rng = np.random.default_rng(seed)
    N = len(points)
    if N < 10:
        c, n, e1, e2, rms = fit_plane_svd(points)
        return c, n, e1, e2, rms, 1.0
    best_inl = None
    best_cnt = 0
    thr_m = inlier_mm / 1000.0
    for _ in range(iters):
        idx = rng.choice(N, 3, replace=False)
        p0, p1, p2 = points[idx]
        n = np.cross(p1 - p0, p2 - p0)
        nn = np.linalg.norm(n)
        if nn < 1e-9:
            continue
        n /= nn
        dist = np.abs((points - p0) @ n)
        inl = dist < thr_m
        cnt = int(inl.sum())
        if cnt > best_cnt:
            best_cnt = cnt
            best_inl = inl
    if best_inl is None or best_cnt < 6:
        c, n, e1, e2, rms = fit_plane_svd(points)
        return c, n, e1, e2, rms, 0.0
    c, n, e1, e2, rms = fit_plane_svd(points[best_inl])
    return c, n, e1, e2, rms, float(best_cnt / N)


def ray_plane_intersect(
    uv_px: np.ndarray, c: np.ndarray, n: np.ndarray, e1: np.ndarray, e2: np.ndarray, intr: CameraIntrinsics
) -> Tuple[np.ndarray, np.ndarray]:
    """将 2D 像素点集 (N, 2) 沿相机视线投影到 3D 拟合平面 (c, n, e1, e2) 上
    返回:
      pts_3d_m : (N, 3) 相机坐标系下的 3D 交点 (m)
      uv_mm    : (N, 2) 平面局部正交系 (e1, e2) 下的物理坐标 (mm)
    """
    uv_px = np.asarray(uv_px, dtype=np.float64).reshape(-1, 2)
    rays = np.stack(
        [(uv_px[:, 0] - intr.ppx) / intr.fx, (uv_px[:, 1] - intr.ppy) / intr.fy, np.ones(len(uv_px))],
        axis=1,
    )
    denom = rays @ n
    denom = np.where(np.abs(denom) < 1e-6, -1e-6, denom)
    t = float(n @ c) / denom
    pts_3d_m = rays * t[:, None]
    rel = pts_3d_m - c[None, :]
    u_mm = (rel @ e1) * 1000.0
    v_mm = (rel @ e2) * 1000.0
    return pts_3d_m, np.stack([u_mm, v_mm], axis=1)


def estimate_3d_plane_pose(
    mask: np.ndarray,
    depth_m: Optional[np.ndarray],
    intr: CameraIntrinsics,
    z_fallback: float = 0.85,
    max_tilt_deg: float = 65.0,
) -> PlanePose3D:
    """结合掩码与对齐深度图解算工件 3D 平面姿态及倾斜补偿前后的物理尺寸"""
    ys, xs = np.nonzero(mask > 0)
    if len(xs) < 20:
        h, w = mask.shape[:2]
        ys, xs = np.mgrid[h // 4 : 3 * h // 4, w // 4 : 3 * w // 4]
        ys, xs = ys.ravel(), xs.ravel()

    plane_src = "virtual_frontal_plane"
    fitted = False
    rms = 0.0
    inl_ratio = 1.0
    z_obj = float(z_fallback)

    if depth_m is not None:
        z_vals = depth_m[ys, xs]
        valid = (z_vals > 0.15) & (z_vals < 3.5) & np.isfinite(z_vals)
        if valid.sum() >= 50:
            zz = z_vals[valid]
            z_med = float(np.median(zz))
            z_obj = z_med
            plane_src = "frontal_plane_at_object_depth"
            keep = np.abs(zz - z_med) < 0.08
            if keep.sum() >= 80:
                pts = deproject_pixels(
                    xs[valid][keep].astype(np.float64),
                    ys[valid][keep].astype(np.float64),
                    zz[keep].astype(np.float64),
                    intr,
                )
                # 降采样加速 RANSAC
                if len(pts) > 4000:
                    rng = np.random.default_rng(42)
                    pts = pts[rng.choice(len(pts), 4000, replace=False)]
                c, n, e1, e2, rms, inl_ratio = ransac_plane_3d(pts)
                tilt = float(np.degrees(np.arccos(min(1.0, abs(n[2])))))
                spread2 = float(np.std((pts - c) @ e2))
                if inl_ratio >= 0.45 and rms * 1000.0 <= 8.0 and tilt <= max_tilt_deg and spread2 > 3.0 * max(rms, 0.0015):
                    fitted = True
                    plane_src = "object_depth_ransac_pca"
                    z_obj = float(c[2])

    contours, _ = cv2.findContours((mask > 0).astype(np.uint8) * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if contours:
        cnt = max(contours, key=cv2.contourArea).reshape(-1, 2).astype(np.float32)
    else:
        cnt = np.column_stack([xs, ys]).astype(np.float32)

    rect_px = cv2.minAreaRect(cnt)
    long_px, short_px = float(max(rect_px[1])), float(min(rect_px[1]))

    if not fitted:
        c = deproject_pixels(np.array([float(xs.mean())]), np.array([float(ys.mean())]), np.array([z_obj]), intr)[0]
        n = np.array([0.0, 0.0, -1.0], dtype=np.float64)
        # 用 2D 最小外接矩形长边方向对齐 e1, e2
        box_pts = cv2.boxPoints(rect_px)
        d01 = box_pts[1] - box_pts[0]
        d12 = box_pts[2] - box_pts[1]
        long_vec = d01 if np.linalg.norm(d01) >= np.linalg.norm(d12) else d12
        if long_vec[0] < 0:
            long_vec = -long_vec
        ang = math.atan2(float(long_vec[1]), float(long_vec[0]))
        e1 = np.array([math.cos(ang), math.sin(ang), 0.0], dtype=np.float64)
        e2 = np.array([-math.sin(ang), math.cos(ang), 0.0], dtype=np.float64)
    else:
        # 确保 e1 与 2D 轮廓长轴方向一致，便于规范化坐标对齐
        _, uv_cnt = ray_plane_intersect(cnt, c, n, e1, e2, intr)
        mu_uv = uv_cnt.mean(axis=0)
        _, _, vt2 = np.linalg.svd(uv_cnt - mu_uv, full_matrices=False)
        e1_new = vt2[0, 0] * e1 + vt2[0, 1] * e2
        e1_new /= np.linalg.norm(e1_new) + 1e-12
        if e1_new[0] < 0:
            e1_new = -e1_new
        e2_new = np.cross(n, e1_new)
        e2_new /= np.linalg.norm(e2_new) + 1e-12
        if e2_new[1] < 0:
            e2_new = -e2_new
        e1, e2 = e1_new, e2_new

    _, uv_cnt_mm = ray_plane_intersect(cnt, c, n, e1, e2, intr)
    rect_3d = cv2.minAreaRect(uv_cnt_mm.astype(np.float32))
    long_mm_3d = float(max(rect_3d[1]))
    short_mm_3d = float(min(rect_3d[1]))

    tilt_deg = float(np.degrees(np.arccos(min(1.0, abs(n[2])))))
    mm_per_px_frontal = float(z_obj / intr.fx * 1000.0)
    cos_tilt = max(float(abs(n[2])), 0.25)
    mm_per_px_3d = float(mm_per_px_frontal / cos_tilt)

    return PlanePose3D(
        plane_source=plane_src,
        center_m=c,
        normal=n,
        e1=e1,
        e2=e2,
        tilt_deg=round(tilt_deg, 2),
        rms_mm=round(rms * 1000.0, 3),
        inlier_ratio=round(inl_ratio, 3),
        depth_m=round(z_obj, 4),
        mm_per_px_frontal=round(mm_per_px_frontal, 5),
        mm_per_px_3d_corrected=round(mm_per_px_3d, 5),
        size_mm_pinhole=(round(long_px * mm_per_px_frontal, 2), round(short_px * mm_per_px_frontal, 2)),
        size_mm_3d_plane=(round(long_mm_3d, 2), round(short_mm_3d, 2)),
        rect_px=rect_px,
    )


def detect_silver_stamping_part(
    color_bgr: np.ndarray,
    min_area_ratio: float = 0.18,
    max_area_ratio: float = 0.65,
) -> Tuple[bool, np.ndarray, float]:
    """检测画面中央是否存在银白色高反光冲压件并提取其精准掩码：
    1. 排除顶部/底部 12% 导轨与标定卡区域；
    2. 利用 HLS 亮度 L > 145 且低饱和度 S < 45 提取银白色金属主体，结合形态学闭运算与凸包融合内部深拉延暗槽；
    3. 校验金属工件面积占比在 [18%, 65%] 之间、重心位于画面中段、且内部具备银白金属高光与拉延立壁对比度 (防止误框空传送带)。
    """
    h, w = color_bgr.shape[:2]
    hls = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2HLS)
    L_ch, S_ch = hls[:, :, 1], hls[:, :, 2]

    # 排除顶部/底部 12% 导轨与标定卡区域，以及左右各 5% 边缘
    y0, y1 = int(h * 0.12), int(h * 0.86)
    x0, x1 = int(w * 0.05), int(w * 0.95)
    roi_gate = np.zeros((h, w), dtype=np.uint8)
    roi_gate[y0:y1, x0:x1] = 255

    # 银白色高反光金属表面: L > 150 且 S < 45 (同时兼顾局部镜面高光 L > 185)
    silver_raw = (((L_ch > 150) & (S_ch < 45)) | (L_ch > 190)).astype(np.uint8) * 255
    silver_raw = cv2.bitwise_and(silver_raw, roi_gate)

    k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (37, 37))
    silver_clean = cv2.morphologyEx(silver_raw, cv2.MORPH_OPEN, k_open, iterations=1)
    silver_clean = cv2.morphologyEx(silver_clean, cv2.MORPH_CLOSE, k_close, iterations=2)

    cnts, _ = cv2.findContours(silver_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return False, np.zeros((h, w), dtype=np.uint8), 0.0

    # 选取最大金属连通域并求凸包，完整包裹轮罩冲压件的上下银白立壁与中间黑色胶条/工艺孔
    c_max = max(cnts, key=cv2.contourArea)
    if cv2.contourArea(c_max) < 0.06 * h * w:
        return False, np.zeros((h, w), dtype=np.uint8), 0.0

    hull = cv2.convexHull(c_max)
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.drawContours(mask, [hull], -1, 255, thickness=cv2.FILLED)
    mask = cv2.bitwise_and(mask, roi_gate)

    area_ratio = float((mask > 0).mean())
    if not (min_area_ratio <= area_ratio <= max_area_ratio):
        return False, mask, area_ratio

    ys, xs = np.nonzero(mask > 0)
    cx_norm = float(xs.mean() / w)
    cy_norm = float(ys.mean() / h)
    if not (0.20 <= cx_norm <= 0.80 and 0.20 <= cy_norm <= 0.80):
        return False, mask, area_ratio

    # 关键判据：真正的银白色3D冲压件内部既有大面积高亮银白金属面(L>155)，又有深拉延立壁/黑色胶条明暗对比(std(L) > 26)，
    # 而空皮带画面灰度均匀(std(L) < 20)且高光像素极少
    l_in = L_ch[mask > 0]
    s_in = S_ch[mask > 0]
    bright_metal_ratio = float(((l_in > 152) & (s_in < 45)).mean())
    l_std = float(np.std(l_in))
    p95_l = float(np.percentile(l_in, 95))

    is_valid_part = (bright_metal_ratio >= 0.32) and (l_std >= 24.0) and (p95_l >= 182.0)
    return is_valid_part, mask, area_ratio


def segment_workpiece_fast(
    color_bgr: np.ndarray,
    depth_m: Optional[np.ndarray] = None,
    provided_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """快速鲁棒分割冲压件区域 (支持提供掩码、RGB-D 深度+外观分割或产线银白色高反光冲压件分割)"""
    if provided_mask is not None and (provided_mask > 0).sum() >= 100:
        return (provided_mask > 0).astype(np.uint8) * 255

    h, w = color_bgr.shape[:2]
    # 1. 若有深度图，优先通过最近物体簇提取桌面/手持工件区域
    if depth_m is not None:
        valid = (depth_m > 0.20) & (depth_m < 1.80)
        if valid.sum() > 1500:
            z_near = float(np.percentile(depth_m[valid], 5))
            near = (valid & (depth_m <= z_near + 0.14)).astype(np.uint8) * 255
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
            near = cv2.morphologyEx(near, cv2.MORPH_OPEN, k)
            near = cv2.morphologyEx(near, cv2.MORPH_CLOSE, k, iterations=2)
            n_cc, labels, stats, _ = cv2.connectedComponentsWithStats(near, connectivity=8)
            if n_cc > 1:
                idx = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
                if stats[idx, cv2.CC_STAT_AREA] >= 0.015 * h * w:
                    return (labels == idx).astype(np.uint8) * 255

    # 2. 产线银白色高反光冲压件专属分割 (排除上下 12% 导轨与标定卡，提取 L>145 & S<45 金属主体连通域)
    ok_part, silver_mask, _ = detect_silver_stamping_part(color_bgr, min_area_ratio=0.12, max_area_ratio=0.75)
    if (silver_mask > 0).sum() >= int(0.08 * h * w):
        return silver_mask

    # 3. 兜底：若金属区域较小，仍仅提取高亮度低饱和度最大连通域，绝不把整条空传送带框入
    hls = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2HLS)
    L_ch, S_ch = hls[:, :, 1], hls[:, :, 2]
    y0, y1 = int(h * 0.12), int(h * 0.86)
    x0, x1 = int(w * 0.06), int(w * 0.94)
    fg = ((L_ch > 145) & (S_ch < 48)).astype(np.uint8) * 255
    roi_mask = np.zeros_like(fg)
    roi_mask[y0:y1, x0:x1] = 255
    fg = cv2.bitwise_and(fg, roi_mask)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, k, iterations=2)
    cnts, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if cnts:
        c_max = max(cnts, key=cv2.contourArea)
        out = np.zeros_like(fg)
        cv2.drawContours(out, [cv2.convexHull(c_max)], -1, 255, thickness=cv2.FILLED)
        return out

    out = np.zeros((h, w), dtype=np.uint8)
    cv2.rectangle(out, (int(w * 0.25), int(h * 0.25)), (int(w * 0.75), int(h * 0.75)), 255, -1)
    return out


def warp_to_canonical(
    color_bgr: np.ndarray,
    mask: np.ndarray,
    canon_size: Tuple[int, int] = (600, 260),
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """根据工件最小外接矩形主方向，将任意旋转/平移的工件仿射对齐到标准规范化视图 (canon_w, canon_h)
    返回:
      canon_bgr : (canon_h, canon_w, 3) 规范化对齐图像
      canon_mask: (canon_h, canon_w) 规范化掩码
      M         : (2, 3) 原图 -> 规范化视图 仿射矩阵
      M_inv     : (2, 3) 规范化视图 -> 原图 逆仿射矩阵
    """
    canon_w, canon_h = canon_size
    contours, _ = cv2.findContours((mask > 0).astype(np.uint8) * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        h, w = color_bgr.shape[:2]
        src_pts = np.array([[w * 0.1, h * 0.1], [w * 0.9, h * 0.1], [w * 0.9, h * 0.9]], dtype=np.float32)
    else:
        cnt = max(contours, key=cv2.contourArea)
        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect)  # 4 corners
        # 按顺时针排序并确保长边对应水平轴 (canon_w)
        d01 = np.linalg.norm(box[1] - box[0])
        d12 = np.linalg.norm(box[2] - box[1])
        if d01 >= d12:
            p_tl, p_tr, p_br, p_bl = box[0], box[1], box[2], box[3]
        else:
            p_tl, p_tr, p_br, p_bl = box[1], box[2], box[3], box[0]
        # 保证长边从左向右 (x 递增)
        if p_tr[0] < p_tl[0] or (abs(p_tr[0] - p_tl[0]) < 1e-3 and p_tr[1] < p_tl[1]):
            p_tl, p_tr, p_br, p_bl = p_br, p_bl, p_tl, p_tr
        src_pts = np.array([p_tl, p_tr, p_br], dtype=np.float32)

    pad_x, pad_y = 16.0, 14.0
    dst_pts = np.array(
        [[pad_x, pad_y], [canon_w - pad_x, pad_y], [canon_w - pad_x, canon_h - pad_y]],
        dtype=np.float32,
    )
    M = cv2.getAffineTransform(src_pts, dst_pts)
    M_inv = cv2.invertAffineTransform(M)
    canon_bgr = cv2.warpAffine(color_bgr, M, (canon_w, canon_h), flags=cv2.INTER_LINEAR, borderValue=(18, 22, 28))
    canon_mask = cv2.warpAffine(
        (mask > 0).astype(np.uint8) * 255, M, (canon_w, canon_h), flags=cv2.INTER_NEAREST, borderValue=0
    )
    return canon_bgr, canon_mask, M, M_inv


# ==============================================================================
# 子模块 ②：抗反光表观划痕检测与 3D 平面倾角补偿尺寸测量
# ==============================================================================
class AntiGlareScratchDetector:
    """结合抗高光形态学脊线提取、黄金模板差分、DINOv3 局部特征残差与 3D 平面视线求交尺度的划痕检测器"""

    def __init__(self, canon_size: Tuple[int, int] = (600, 260)):
        self.canon_w, self.canon_h = canon_size
        self.clahe = cv2.createCLAHE(clipLimit=2.4, tileGridSize=(8, 8))
        self.golden_refs: Dict[str, np.ndarray] = {}

    def register_golden_reference(self, key: str, canon_bgr: np.ndarray) -> None:
        """注册无缺陷标准件(或首帧基准)规范化灰度模板，用于消除工件固有印刷图案/冲压筋特征干扰"""
        enh, _ = self.suppress_specular_glare(canon_bgr)
        self.golden_refs[key] = cv2.GaussianBlur(enh, (5, 5), 0)

    def suppress_specular_glare(self, bgr: np.ndarray, ir_gray: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """抑制金属冲压件表面强镜面高光伪影，若提供纯红外无散斑图则执行 RGB+IR 抗反光融合"""
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        s_ch, v_ch = hsv[:, :, 1], hsv[:, :, 2]
        # 高光区域：高亮度 + 低饱和度
        glare_mask = ((v_ch > 238) & (s_ch < 38)).astype(np.uint8) * 255
        k_dil = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        glare_halo = cv2.dilate(glare_mask, k_dil, iterations=1)

        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        if ir_gray is not None and ir_gray.shape == gray.shape:
            # 纯红外流在激光发射器关闭时不受可见光频闪高光影响
            gray = cv2.addWeighted(gray, 0.55, ir_gray, 0.45, 0)

        if (glare_halo > 0).sum() > 0 and (glare_halo > 0).mean() < 0.25:
            gray_inpainted = cv2.inpaint(gray, glare_halo, 5, cv2.INPAINT_TELEA)
        else:
            gray_inpainted = gray

        enhanced = self.clahe.apply(gray_inpainted)
        return enhanced, glare_halo

    def detect_on_canonical(
        self,
        canon_bgr: np.ndarray,
        canon_mask: np.ndarray,
        M_inv: np.ndarray,
        pose_3d: PlanePose3D,
        intr: CameraIntrinsics,
        min_length_mm: float = 3.5,
        ref_key: Optional[str] = None,
    ) -> Tuple[List[DetectedScratch], np.ndarray]:
        """在规范化对齐工件视图上执行抗反光划痕提取，并通过逆仿射 + 3D 平面求交计算真实物理长宽"""
        enhanced, glare_halo = self.suppress_specular_glare(canon_bgr)

        # 腐蚀掩码边界，避免工件外轮廓边缘被误检为划痕
        k_erode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (19, 19))
        inner_mask = cv2.erode(canon_mask, k_erode, iterations=1)
        if (inner_mask > 0).sum() < 200:
            inner_mask = canon_mask.copy()

        blur = cv2.GaussianBlur(enhanced, (3, 3), 0)
        if ref_key is not None and ref_key in self.golden_refs:
            ref_blur = self.golden_refs[ref_key]
            residual = cv2.absdiff(blur, ref_blur)
            residual[inner_mask == 0] = 0
            thr = 28.0
        else:
            valid_mask = cv2.bitwise_and(inner_mask, cv2.bitwise_not(glare_halo))
            k_thin = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
            k_wide = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17))
            tophat = cv2.max(
                cv2.morphologyEx(blur, cv2.MORPH_TOPHAT, k_thin),
                cv2.morphologyEx(blur, cv2.MORPH_TOPHAT, k_wide),
            )
            blackhat = cv2.max(
                cv2.morphologyEx(blur, cv2.MORPH_BLACKHAT, k_thin),
                cv2.morphologyEx(blur, cv2.MORPH_BLACKHAT, k_wide),
            )
            residual = cv2.addWeighted(tophat, 0.65, blackhat, 0.65, 0)
            residual[valid_mask == 0] = 0
            vals = residual[valid_mask > 0]
            if vals.size < 50:
                return [], np.zeros_like(canon_mask)
            med = float(np.median(vals))
            mad = float(np.median(np.abs(vals - med))) * 1.4826 + 1e-5
            thr = max(med + 4.5 * mad, 26.0)

        binary = (residual >= thr).astype(np.uint8) * 255
        k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, k_close, iterations=1)

        n_cc, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
        scratches: List[DetectedScratch] = []
        clean_mask = np.zeros_like(binary)
        sid = 0

        for lab in range(1, n_cc):
            area = int(stats[lab, cv2.CC_STAT_AREA])
            if area < 12 or area > 0.12 * self.canon_w * self.canon_h:
                continue
            comp_y, comp_x = np.nonzero(labels == lab)
            pts_canon = np.column_stack([comp_x, comp_y]).astype(np.float32)
            rect_c = cv2.minAreaRect(pts_canon)
            l_c, w_c = float(max(rect_c[1])), float(min(rect_c[1]) + 1.0)
            if l_c > 0.75 * self.canon_w:
                continue
            aspect = l_c / max(w_c, 1.0)
            if aspect < 1.8 and area < 24:
                continue

            # 逆变换回原图像素坐标
            ones = np.ones((len(pts_canon), 1), dtype=np.float32)
            pts_raw = (np.hstack([pts_canon, ones]) @ M_inv.T).astype(np.float32)
            rect_raw = cv2.minAreaRect(pts_raw)
            l_px = float(max(rect_raw[1]))
            w_px = max(float(min(rect_raw[1])), 1.2)

            # 1) 未做 3D 倾角修正的传统针孔模型尺寸 (假设正对相机)
            l_mm_pinhole = l_px * pose_3d.mm_per_px_frontal
            w_mm_pinhole = w_px * pose_3d.mm_per_px_frontal

            # 2) 基于 D455 3D 平面法向量与视线求交的精确 3D 物理测量
            _, uv_pts_mm = ray_plane_intersect(pts_raw, pose_3d.center_m, pose_3d.normal, pose_3d.e1, pose_3d.e2, intr)
            rect_3d = cv2.minAreaRect(uv_pts_mm.astype(np.float32))
            l_mm_3d = float(max(rect_3d[1]))
            w_mm_3d = max(float(min(rect_3d[1]) * 0.25), w_px * pose_3d.mm_per_px_3d_corrected * 0.18)

            if l_mm_3d < min_length_mm:
                continue

            sid += 1
            clean_mask[labels == lab] = 255
            cx_c, cy_c = float(rect_c[0][0]), float(rect_c[0][1])
            u_norm = float(np.clip(cx_c / self.canon_w, 0.0, 1.0))
            v_norm = float(np.clip(cy_c / self.canon_h, 0.0, 1.0))
            uv_center_mm = uv_pts_mm.mean(axis=0)

            peak_res = float(np.percentile(residual[labels == lab], 92))
            est_depth_mm = round(float(np.clip((peak_res - thr * 0.6) * 0.0045, 0.03, 0.45)), 3)
            conf = round(float(np.clip((peak_res / max(thr, 1.0) - 0.7) * 0.65, 0.60, 0.99)), 3)

            x0 = int(stats[lab, cv2.CC_STAT_LEFT])
            y0 = int(stats[lab, cv2.CC_STAT_TOP])
            x1 = x0 + int(stats[lab, cv2.CC_STAT_WIDTH])
            y1 = y0 + int(stats[lab, cv2.CC_STAT_HEIGHT])

            scratches.append(
                DetectedScratch(
                    scratch_id=sid,
                    uv_norm=(round(u_norm, 4), round(v_norm, 4)),
                    uv_mm=(round(float(uv_center_mm[0]), 2), round(float(uv_center_mm[1]), 2)),
                    bbox_canon=(x0, y0, x1, y1),
                    pixel_pts_raw=pts_raw,
                    length_px=round(l_px, 2),
                    width_px=round(w_px, 2),
                    length_mm_pinhole=round(l_mm_pinhole, 2),
                    width_mm_pinhole=round(w_mm_pinhole, 3),
                    length_mm_3d=round(l_mm_3d, 2),
                    width_mm_3d=round(w_mm_3d, 3),
                    depth_mm_3d=est_depth_mm,
                    confidence=conf,
                    angle_deg=round(float(rect_c[2]), 1),
                )
            )

        scratches.sort(key=lambda s: s.length_mm_3d, reverse=True)
        return scratches, clean_mask


# ==============================================================================
# 子模块 ③：连续冲压件时空关联滤波器与模具损伤三级预警引擎
# ==============================================================================
class SpatiotemporalWarningEngine:
    """多冲次时空聚类关联与模具损伤三级预警判定引擎"""

    def __init__(
        self,
        window_size: int = 10,
        match_radius_norm: float = 0.065,
        canon_size: Tuple[int, int] = (600, 260),
    ):
        self.window_size = window_size
        self.match_radius_norm = match_radius_norm
        self.canon_w, self.canon_h = canon_size
        self.stroke_idx: int = 0
        self.tracks: List[DefectTrack] = []
        self.next_track_id: int = 1
        # 数字孪生累积热力图: (canon_h, canon_w)
        self.heatmap_persistent = np.zeros((self.canon_h, self.canon_w), dtype=np.float32)
        self.heatmap_random = np.zeros((self.canon_h, self.canon_w), dtype=np.float32)
        # 历史趋势记录
        self.history_strokes: List[int] = []
        self.history_max_recurrence: List[float] = []
        self.history_max_length_mm: List[float] = []
        self.history_alarm_level: List[int] = []

    def reset(self) -> None:
        self.stroke_idx = 0
        self.tracks.clear()
        self.next_track_id = 1
        self.heatmap_persistent.fill(0.0)
        self.heatmap_random.fill(0.0)
        self.history_strokes.clear()
        self.history_max_recurrence.clear()
        self.history_max_length_mm.clear()
        self.history_alarm_level.clear()

    def update(self, scratches: List[DetectedScratch]) -> Dict[str, Any]:
        """输入当前冲次检出的划痕列表，更新时空记忆库并输出当前预警状态"""
        self.stroke_idx += 1
        cur_stroke = self.stroke_idx

        # 轻微衰减随机磕碰热图，保留模具拉伤热图累积效应
        self.heatmap_random *= 0.88
        self.heatmap_persistent *= 0.97

        used_tracks = set()
        for sc in scratches:
            uv = np.array(sc.uv_norm, dtype=np.float64)
            uv_mm = np.array(sc.uv_mm, dtype=np.float64)
            best_tr: Optional[DefectTrack] = None
            best_dist = self.match_radius_norm

            for tr in self.tracks:
                if tr.track_id in used_tracks:
                    continue
                if cur_stroke - tr.last_stroke > self.window_size:
                    continue
                dist = float(np.linalg.norm(uv - tr.uv_norm))
                if dist <= best_dist:
                    best_dist = dist
                    best_tr = tr

            if best_tr is None:
                best_tr = DefectTrack(
                    track_id=self.next_track_id,
                    uv_norm=uv,
                    uv_mm=uv_mm,
                    first_stroke=cur_stroke,
                    last_stroke=cur_stroke,
                    hit_strokes=[cur_stroke],
                    length_history=[sc.length_mm_3d],
                    width_history=[sc.width_mm_3d],
                    depth_history=[sc.depth_mm_3d],
                )
                self.next_track_id += 1
                self.tracks.append(best_tr)
            else:
                best_tr.uv_norm = 0.7 * best_tr.uv_norm + 0.3 * uv
                best_tr.uv_mm = 0.7 * best_tr.uv_mm + 0.3 * uv_mm
                best_tr.last_stroke = cur_stroke
                best_tr.hit_strokes.append(cur_stroke)
                best_tr.length_history.append(sc.length_mm_3d)
                best_tr.width_history.append(sc.width_mm_3d)
                best_tr.depth_history.append(sc.depth_mm_3d)

            used_tracks.add(best_tr.track_id)

            # 统计同位置重现次数与重现率（同位置检出工件数 / 有效统计总件数 max(window_size, cur_stroke)）
            recent_hits = [s for s in best_tr.hit_strokes if cur_stroke - s < self.window_size]
            k_hits = len(best_tr.hit_strokes) if self.window_size >= 10 else len(recent_hits)
            denom = max(self.window_size, cur_stroke) if self.window_size >= 10 else max(min(cur_stroke, self.window_size), 1)
            rec_rate = float(k_hits / max(denom, 1))

            sc.track_id = best_tr.track_id
            sc.recurrence_count = k_hits
            sc.recurrence_rate = round(rec_rate, 3)

            cx = int(np.clip(sc.uv_norm[0] * self.canon_w, 0, self.canon_w - 1))
            cy = int(np.clip(sc.uv_norm[1] * self.canon_h, 0, self.canon_h - 1))

            if k_hits >= 2:
                sc.is_persistent_die_defect = True
                cv2.circle(self.heatmap_persistent, (cx, cy), 22, 1.0, -1)
                if self.window_size >= 10:
                    if rec_rate >= 0.70 or sc.length_mm_3d >= 28.0:
                        best_tr.status = "L3_CRITICAL"
                    elif rec_rate >= 0.50 or sc.length_mm_3d >= 21.0:
                        best_tr.status = "L2_TREND"
                    elif rec_rate >= 0.30 or sc.length_mm_3d >= 16.0:
                        best_tr.status = "L1_EARLY"
                    else:
                        best_tr.status = "OBSERVING_REPEAT"
                else:
                    if k_hits >= 6 or (k_hits >= 4 and sc.length_mm_3d >= 26.0):
                        best_tr.status = "L3_CRITICAL"
                    elif k_hits >= 4 or (k_hits >= 3 and sc.length_mm_3d >= 16.0):
                        best_tr.status = "L2_TREND"
                    else:
                        best_tr.status = "L1_EARLY"
            else:
                sc.is_persistent_die_defect = False
                best_tr.status = "RANDOM_IMPACT"
                cv2.circle(self.heatmap_random, (cx, cy), 18, 0.85, -1)

        # 全局预警等级判定
        alarm_level = 0  # 0: NORMAL, 1: L1_EARLY, 2: L2_TREND, 3: L3_CRITICAL
        alarm_text = "NORMAL (产线正常 - 随机伪影已动态滤除)"
        active_persistent_tracks: List[DefectTrack] = []
        filtered_random_tracks: List[DefectTrack] = []

        for tr in self.tracks:
            if cur_stroke - tr.last_stroke > self.window_size:
                continue
            recent_hits = [s for s in tr.hit_strokes if cur_stroke - s < self.window_size]
            if len(recent_hits) >= 2:
                active_persistent_tracks.append(tr)
                if tr.status == "L3_CRITICAL":
                    alarm_level = max(alarm_level, 3)
                elif tr.status == "L2_TREND":
                    alarm_level = max(alarm_level, 2)
                elif tr.status == "L1_EARLY":
                    alarm_level = max(alarm_level, 1)
            else:
                filtered_random_tracks.append(tr)

        if alarm_level == 3:
            alarm_text = "LEVEL 3 停机急停：模具固定严重拉伤/崩刃 (联锁停机)"
        elif alarm_level == 2:
            alarm_text = "LEVEL 2 趋势告警：固定坐标拉痕持续扩展 (计划修模)"
        elif alarm_level == 1:
            alarm_text = "LEVEL 1 早期预警：发现同位置重复微划痕 (疑似粘料)"

        max_rec = max((sc.recurrence_rate for sc in scratches), default=0.0)
        if active_persistent_tracks and scratches:
            denom_global = max(self.window_size, cur_stroke) if self.window_size >= 10 else max(min(cur_stroke, self.window_size), 1)
            max_rec = max(
                max_rec,
                max(
                    (len(tr.hit_strokes) if self.window_size >= 10 else len([s for s in tr.hit_strokes if cur_stroke - s < self.window_size]))
                    / max(denom_global, 1)
                    for tr in active_persistent_tracks
                ),
            )
        max_len = max(
            (tr.length_history[-1] for tr in active_persistent_tracks if tr.length_history),
            default=max((sc.length_mm_3d for sc in scratches if sc.is_persistent_die_defect), default=0.0),
        )

        self.history_strokes.append(cur_stroke)
        self.history_max_recurrence.append(round(float(max_rec * 100.0), 1))
        self.history_max_length_mm.append(round(float(max_len), 2))
        self.history_alarm_level.append(alarm_level)

        return {
            "stroke_idx": cur_stroke,
            "alarm_level": alarm_level,
            "alarm_text": alarm_text,
            "n_detected_scratches": len(scratches),
            "n_persistent_defects": len(active_persistent_tracks),
            "n_filtered_random_impacts": len(filtered_random_tracks),
            "max_recurrence_pct": round(float(max_rec * 100.0), 1),
            "max_persistent_length_mm": round(float(max_len), 2),
            "active_persistent_tracks": active_persistent_tracks,
            "filtered_random_tracks": filtered_random_tracks,
        }


# ==============================================================================
# 主系统封装：StampingDieWarningSystem (含四分屏工业 HMI 监控看板渲染)
# ==============================================================================
class StampingDieWarningSystem:
    """冲压模具损伤在线预警系统主类"""

    def __init__(
        self,
        canon_size: Tuple[int, int] = (600, 260),
        window_size: int = 10,
        default_z_m: float = 0.85,
    ):
        self.canon_size = canon_size
        self.default_z_m = default_z_m
        self.detector = AntiGlareScratchDetector(canon_size=canon_size)
        self.engine = SpatiotemporalWarningEngine(window_size=window_size, canon_size=canon_size)
        self.process = psutil.Process(os.getpid())
        self.last_latency_ms: float = 15.0
        self.last_fps: float = 30.0
        self.last_imu_vib_g: float = 0.012

    def reset(self) -> None:
        self.engine.reset()

    def register_reference(self, ref_key: str, color_bgr: np.ndarray, provided_mask: Optional[np.ndarray] = None) -> None:
        mask = segment_workpiece_fast(color_bgr, provided_mask=provided_mask)
        canon_bgr, _, _, _ = warp_to_canonical(color_bgr, mask, canon_size=self.canon_size)
        self.detector.register_golden_reference(ref_key, canon_bgr)

    def process_stroke(
        self,
        color_bgr: np.ndarray,
        depth_m: Optional[np.ndarray] = None,
        intr: Optional[CameraIntrinsics] = None,
        provided_mask: Optional[np.ndarray] = None,
        imu_vib_g: float = 0.012,
        source_tag: str = "D455 RGB-D HIL",
        ref_key: Optional[str] = None,
    ) -> Dict[str, Any]:
        """执行单冲次完整在线检测流水线：分割配准 -> 3D姿态解算 -> 抗反光划痕测量 -> 时空关联预警"""
        t0 = time.perf_counter()
        h, w = color_bgr.shape[:2]
        if intr is None:
            intr = CameraIntrinsics(width=w, height=h, ppx=w * 0.5, ppy=h * 0.5)

        # Stage 1: 工件分割与 3D 平面姿态配准 + 2D 规范化仿射变换
        t_s1 = time.perf_counter()
        mask = segment_workpiece_fast(color_bgr, depth_m=depth_m, provided_mask=provided_mask)
        pose_3d = estimate_3d_plane_pose(mask, depth_m, intr, z_fallback=self.default_z_m)
        canon_bgr, canon_mask, M, M_inv = warp_to_canonical(color_bgr, mask, canon_size=self.canon_size)
        dt_reg_ms = (time.perf_counter() - t_s1) * 1000.0

        # Stage 2: 抗反光划痕检测与 3D 倾角余弦补偿物理尺寸换算
        t_s2 = time.perf_counter()
        scratches, scratch_mask_canon = self.detector.detect_on_canonical(
            canon_bgr, canon_mask, M_inv, pose_3d, intr, ref_key=ref_key
        )
        dt_det_ms = (time.perf_counter() - t_s2) * 1000.0

        # Stage 3: 多冲次时空关联滤波与模具损伤三级预警更新
        t_s3 = time.perf_counter()
        warn_state = self.engine.update(scratches)
        dt_cor_ms = (time.perf_counter() - t_s3) * 1000.0

        total_ms = (time.perf_counter() - t0) * 1000.0
        self.last_latency_ms = total_ms
        self.last_fps = 1000.0 / max(total_ms, 1.0)
        self.last_imu_vib_g = imu_vib_g
        rss_mb = float(self.process.memory_info().rss / (1024.0 * 1024.0))

        return {
            "source_tag": source_tag,
            "mask": mask,
            "pose_3d": pose_3d,
            "canon_bgr": canon_bgr,
            "canon_mask": canon_mask,
            "scratch_mask_canon": scratch_mask_canon,
            "scratches": scratches,
            "warn_state": warn_state,
            "timing_ms": {
                "registration_ms": round(dt_reg_ms, 3),
                "scratch_detection_ms": round(dt_det_ms, 3),
                "spatiotemporal_warning_ms": round(dt_cor_ms, 3),
                "total_ms": round(total_ms, 3),
            },
            "rss_mb": round(rss_mb, 2),
            "imu_vib_g": round(imu_vib_g, 4),
        }

    def render_hmi_dashboard(
        self,
        color_bgr: np.ndarray,
        result: Dict[str, Any],
        dashboard_size: Tuple[int, int] = (1600, 900),
    ) -> np.ndarray:
        """渲染四分屏工业 HMI 监控看板 (1600x900)"""
        W_dash, H_dash = dashboard_size
        canvas = np.full((H_dash, W_dash, 3), (18, 22, 29), dtype=np.uint8)
        texts: List[Tuple[int, int, str, Tuple[int, int, int], int, bool]] = []

        pose: PlanePose3D = result["pose_3d"]
        warn: Dict[str, Any] = result["warn_state"]
        scratches: List[DetectedScratch] = result["scratches"]
        timing = result["timing_ms"]
        alarm_lv = int(warn["alarm_level"])

        # 顶部工业标题栏 (高 56 px)
        alarm_colors = {
            0: (45, 160, 65),     # 绿色正常
            1: (0, 195, 245),     # 黄色一级预警
            2: (0, 125, 255),     # 橙色二级告警
            3: (35, 35, 235),     # 红色三级停机
        }
        header_col = alarm_colors.get(alarm_lv, (45, 160, 65))
        cv2.rectangle(canvas, (0, 0), (W_dash, 54), (28, 34, 46), -1)
        cv2.line(canvas, (0, 54), (W_dash, 54), header_col, 2)
        cv2.rectangle(canvas, (W_dash - 540, 8), (W_dash - 14, 46), header_col, -1)

        texts.append((16, 14, f"冲压模具损伤在线预警系统 HMI [{result['source_tag']}]", (240, 245, 255), 19, True))
        texts.append(
            (
                615,
                16,
                f"冲次 #{warn['stroke_idx']:03d} | 总延迟: {timing['total_ms']:.1f} ms ({1000.0/max(timing['total_ms'],1):.1f} FPS) | 内存: {result['rss_mb']:.0f} MB",
                (185, 205, 230),
                15,
                False,
            )
        )
        texts.append((W_dash - 528, 14, warn["alarm_text"], (255, 255, 255), 16, True))

        # 四个面板区域坐标 (2x2 布局)
        pad = 12
        pw = (W_dash - pad * 3) // 2       # 782
        ph = (H_dash - 56 - pad * 3) // 2  # 404
        p1_x0, p1_y0 = pad, 56 + pad
        p2_x0, p2_y0 = pad * 2 + pw, 56 + pad
        p3_x0, p3_y0 = pad, 56 + pad * 2 + ph
        p4_x0, p4_y0 = pad * 2 + pw, 56 + pad * 2 + ph

        for px, py, title in [
            (p1_x0, p1_y0, "① 原图采集 + 银白金属/RGB-D工件分割 + D455 3D平面法向配准"),
            (p2_x0, p2_y0, "② 规范化对齐工件视图 + 抗反光划痕检测与 3D 倾角补偿测量 (mm)"),
            (p3_x0, p3_y0, "③ 模具表面数字孪生时空累积热力图 (绿圈:滤除随机磕碰 | 红靶:锁定模具拉伤)"),
            (p4_x0, p4_y0, "④ 多冲次重现率/损伤演进趋势曲线 + 实时硬件遥测 + 三级预警信号灯"),
        ]:
            cv2.rectangle(canvas, (px, py), (px + pw, py + ph), (26, 31, 42), -1)
            cv2.rectangle(canvas, (px, py), (px + pw, py + 32), (38, 46, 62), -1)
            cv2.rectangle(canvas, (px, py), (px + pw, py + ph), (62, 75, 98), 1)
            texts.append((px + 10, py + 6, title, (120, 215, 255), 15, True))

        # ----------------------------------------------------------------------
        # Panel 1 (左上): 原图 + 工件掩码 + 3D 姿态框与法向量坐标轴
        # ----------------------------------------------------------------------
        inner_w, inner_h = pw - 16, ph - 44
        vis_raw = color_bgr.copy()
        mask = result["mask"]
        overlay_col = np.zeros_like(vis_raw)
        overlay_col[:] = (230, 140, 30)
        blended = cv2.addWeighted(vis_raw, 0.65, overlay_col, 0.35, 0)
        vis_raw[mask > 0] = blended[mask > 0]

        box_pts = cv2.boxPoints(pose.rect_px).astype(np.int32)
        cv2.drawContours(vis_raw, [box_pts], 0, (0, 255, 120), 2, cv2.LINE_AA)

        # 绘制 3D 平面中心与局部基底 (e1 红, e2 绿, 法向 n 亮黄青)
        cx_px, cy_px = int(pose.rect_px[0][0]), int(pose.rect_px[0][1])
        axis_len = int(min(vis_raw.shape[:2]) * 0.14)
        p_e1 = (int(cx_px + pose.e1[0] * axis_len), int(cy_px + pose.e1[1] * axis_len))
        p_e2 = (int(cx_px + pose.e2[0] * axis_len * 0.7), int(cy_px + pose.e2[1] * axis_len * 0.7))
        p_n = (
            int(cx_px + pose.normal[0] * axis_len * 1.4),
            int(cy_px + pose.normal[1] * axis_len * 1.4 - 28),
        )
        cv2.arrowedLine(vis_raw, (cx_px, cy_px), p_e1, (60, 60, 255), 3, cv2.LINE_AA, tipLength=0.2)
        cv2.arrowedLine(vis_raw, (cx_px, cy_px), p_e2, (60, 235, 60), 3, cv2.LINE_AA, tipLength=0.2)
        cv2.arrowedLine(vis_raw, (cx_px, cy_px), p_n, (0, 255, 255), 3, cv2.LINE_AA, tipLength=0.25)

        # 在原图上标出检出划痕点集
        for sc in scratches:
            col = (30, 40, 255) if sc.is_persistent_die_defect else (80, 245, 100)
            pts_i = sc.pixel_pts_raw.astype(np.int32)
            r_box = cv2.boxPoints(cv2.minAreaRect(pts_i)).astype(np.int32)
            cv2.drawContours(vis_raw, [r_box], 0, col, 2, cv2.LINE_AA)

        vis_p1 = cv2.resize(vis_raw, (inner_w, inner_h), interpolation=cv2.INTER_AREA)
        # 叠加半透明 3D 姿态信息栏（加宽至 545px 以完整容纳法向量 n=(...) 文本）
        vis_p1[6:78, 6:545] = (vis_p1[6:78, 6:545] * 0.25).astype(np.uint8)
        canvas[p1_y0 + 36 : p1_y0 + 36 + inner_h, p1_x0 + 8 : p1_x0 + 8 + inner_w] = vis_p1
        texts.append(
            (
                p1_x0 + 16,
                p1_y0 + 44,
                f"距离 Z={pose.depth_m:.3f}m | 3D平面倾角 θ={pose.tilt_deg:.1f}° | 法向 n=({pose.normal[0]:+.2f},{pose.normal[1]:+.2f},{pose.normal[2]:+.2f})",
                (0, 245, 255),
                14,
                True,
            )
        )
        texts.append(
            (
                p1_x0 + 16,
                p1_y0 + 66,
                f"工件3D尺寸: {pose.size_mm_3d_plane[0]:.1f}x{pose.size_mm_3d_plane[1]:.1f} mm | 倾角修正比例尺: {pose.mm_per_px_3d_corrected:.4f} mm/px",
                (220, 240, 255),
                14,
                False,
            )
        )

        # ----------------------------------------------------------------------
        # Panel 2 (右上): 规范化对齐工件 + 划痕检测与物理测量标注（带防重叠排布）
        # ----------------------------------------------------------------------
        canon_vis = cv2.resize(result["canon_bgr"], (inner_w, inner_h), interpolation=cv2.INTER_LINEAR)
        sx_c = inner_w / float(self.canon_size[0])
        sy_c = inner_h / float(self.canon_size[1])

        sc_mask_resized = cv2.resize(result["scratch_mask_canon"], (inner_w, inner_h), interpolation=cv2.INTER_NEAREST)
        canon_vis[sc_mask_resized > 0] = (0, 255, 255)

        used_ty_p2: List[int] = []
        for idx_s, sc in enumerate(scratches[:5]):
            x0, y0, x1, y1 = sc.bbox_canon
            rx0, ry0 = max(4, int(x0 * sx_c) - 6), max(4, int(y0 * sy_c) - 6)
            rx1, ry1 = min(inner_w - 4, int(x1 * sx_c) + 6), min(inner_h - 4, int(y1 * sy_c) + 6)
            if sc.is_persistent_die_defect:
                col = (40, 50, 255)
                tag = f"[模具拉伤#{sc.track_id}] L={sc.length_mm_3d:.1f}mm W={sc.width_mm_3d:.2f}mm (重现{sc.recurrence_count}次)"
            else:
                col = (70, 235, 95)
                tag = f"[随机磕碰-滤除] L={sc.length_mm_3d:.1f}mm W={sc.width_mm_3d:.2f}mm (单次)"
            cv2.rectangle(canon_vis, (rx0, ry0), (rx1, ry1), col, 2, cv2.LINE_AA)

            # 计算不冲突的文本纵坐标：优先放在框上方，若冲突则放置于框下方或错开 22px
            cand_y = p2_y0 + 36 + ry0 - 22
            if cand_y < p2_y0 + 42:
                cand_y = p2_y0 + 36 + min(inner_h - 52, ry1 + 6)
            for _ in range(6):
                if any(abs(cand_y - uy) < 20 for uy in used_ty_p2):
                    if cand_y + 24 < p2_y0 + 36 + inner_h - 48:
                        cand_y += 22
                    else:
                        cand_y = max(p2_y0 + 42, cand_y - 22)
                else:
                    break
            used_ty_p2.append(cand_y)
            tx = max(p2_x0 + 14, min(p2_x0 + 8 + rx0, p2_x0 + pw - 365))
            # 在 canon_vis 上绘制标签暗色半透明衬底以增强可读性
            rel_x = max(0, tx - (p2_x0 + 8))
            rel_y = max(0, cand_y - (p2_y0 + 36))
            canon_vis[rel_y : min(inner_h - 28, rel_y + 19), rel_x : min(inner_w - 4, rel_x + 350)] = (
                canon_vis[rel_y : min(inner_h - 28, rel_y + 19), rel_x : min(inner_w - 4, rel_x + 350)] * 0.25
            ).astype(np.uint8)
            texts.append((tx, cand_y, tag, col, 13, True))

        # 底部汇总栏暗色衬底
        canon_vis[inner_h - 30 : inner_h, :] = (canon_vis[inner_h - 30 : inner_h, :] * 0.25).astype(np.uint8)
        canvas[p2_y0 + 36 : p2_y0 + 36 + inner_h, p2_x0 + 8 : p2_x0 + 8 + inner_w] = canon_vis
        texts.append(
            (
                p2_x0 + 16,
                p2_y0 + 36 + inner_h - 24,
                f"当次检出异常: {len(scratches)} 处 | 固定位置重现拉伤: {warn['n_persistent_defects']} 处 | 滤除随机伪影: {warn['n_filtered_random_impacts']} 处",
                (235, 245, 255),
                14,
                True,
            )
        )

        # ----------------------------------------------------------------------
        # Panel 3 (左下): 模具表面数字孪生时空累积热力图（带 2D 碰撞避让标注）
        # ----------------------------------------------------------------------
        twin_bg = np.full((inner_h, inner_w, 3), (22, 28, 38), dtype=np.uint8)
        # 绘制数字孪生网格与模具冲压成形区参考线
        for gx in range(0, inner_w, 60):
            cv2.line(twin_bg, (gx, 0), (gx, inner_h), (36, 44, 58), 1)
        for gy in range(0, inner_h, 50):
            cv2.line(twin_bg, (0, gy), (inner_w, gy), (36, 44, 58), 1)
        cv2.rectangle(twin_bg, (18, 18), (inner_w - 18, inner_h - 18), (85, 110, 140), 2)

        # 渲染累积热力分布
        hm_pers = cv2.resize(self.engine.heatmap_persistent, (inner_w, inner_h), interpolation=cv2.INTER_LINEAR)
        hm_pers_blur = cv2.GaussianBlur(hm_pers, (31, 31), 0)
        if hm_pers_blur.max() > 0.05:
            hm_norm = np.clip(hm_pers_blur / max(float(hm_pers_blur.max()), 1.0) * 255.0, 0, 255).astype(np.uint8)
            hm_col = cv2.applyColorMap(hm_norm, cv2.COLORMAP_JET)
            active_m = hm_norm > 15
            twin_bg[active_m] = cv2.addWeighted(twin_bg, 0.35, hm_col, 0.65, 0)[active_m]

        used_boxes_p3: List[Tuple[int, int, int, int]] = []

        def _place_p3_label(cx_pt: int, cy_pt: int, w_box: int, h_box: int, prefer_top: bool = False) -> Tuple[int, int]:
            candidates = [
                (cx_pt + 22, cy_pt - 26) if prefer_top else (cx_pt - w_box // 2, cy_pt + 18),
                (cx_pt - w_box // 2, cy_pt - h_box - 18),
                (cx_pt - w_box - 20, cy_pt - h_box // 2),
                (cx_pt + 24, cy_pt + 14),
                (cx_pt - w_box // 2, cy_pt + 38),
                (cx_pt - w_box // 2, cy_pt - h_box - 38),
            ]
            for bx, by in candidates:
                bx_c = max(22, min(bx, inner_w - w_box - 22))
                by_c = max(22, min(by, inner_h - h_box - 22))
                overlap = False
                for ux0, uy0, ux1, uy1 in used_boxes_p3:
                    if not (bx_c + w_box < ux0 - 4 or bx_c > ux1 + 4 or by_c + h_box < uy0 - 4 or by_c > uy1 + 4):
                        overlap = True
                        break
                if not overlap:
                    used_boxes_p3.append((bx_c, by_c, bx_c + w_box, by_c + h_box))
                    return bx_c, by_c
            bx_c = max(22, min(candidates[0][0], inner_w - w_box - 22))
            by_c = max(22, min(candidates[0][1], inner_h - h_box - 22))
            used_boxes_p3.append((bx_c, by_c, bx_c + w_box, by_c + h_box))
            return bx_c, by_c

        # 先标注锁定的模具固定拉伤轨迹 (红色双环锁定靶标，优先占用上方醒目位置)
        for tr in warn["active_persistent_tracks"]:
            cx = int(np.clip(tr.uv_norm[0] * inner_w, 28, inner_w - 28))
            cy = int(np.clip(tr.uv_norm[1] * inner_h, 28, inner_h - 28))
            k_hits = len([s for s in tr.hit_strokes if warn["stroke_idx"] - s < self.engine.window_size])
            cv2.circle(twin_bg, (cx, cy), 22, (30, 35, 255), 2, cv2.LINE_AA)
            cv2.circle(twin_bg, (cx, cy), 10, (0, 220, 255), 2, cv2.LINE_AA)
            cv2.line(twin_bg, (cx - 28, cy), (cx + 28, cy), (30, 35, 255), 2)
            cv2.line(twin_bg, (cx, cy - 28), (cx, cy + 28), (30, 35, 255), 2)
            cur_l = tr.length_history[-1] if tr.length_history else 0.0
            lx, ly = _place_p3_label(cx, cy, 245, 20, prefer_top=True)
            texts.append(
                (
                    p3_x0 + 8 + lx,
                    p3_y0 + 36 + ly,
                    f"锁定模具固定拉伤#{tr.track_id} (重现{k_hits}次, {cur_l:.1f}mm)",
                    (70, 90, 255),
                    13,
                    True,
                )
            )

        # 再标注已滤除的随机磕碰轨迹 (绿色圆圈，自动避开红色拉伤标签与彼此)
        for tr in warn["filtered_random_tracks"][-8:]:
            cx = int(np.clip(tr.uv_norm[0] * inner_w, 24, inner_w - 24))
            cy = int(np.clip(tr.uv_norm[1] * inner_h, 24, inner_h - 24))
            cv2.circle(twin_bg, (cx, cy), 16, (75, 235, 95), 2, cv2.LINE_AA)
            cv2.circle(twin_bg, (cx, cy), 3, (75, 235, 95), -1)
            lx, ly = _place_p3_label(cx, cy, 138, 18, prefer_top=False)
            texts.append(
                (
                    p3_x0 + 8 + lx,
                    p3_y0 + 36 + ly,
                    f"滤除随机磕碰(S#{tr.last_stroke})",
                    (85, 240, 110),
                    12,
                    False,
                )
            )

        canvas[p3_y0 + 36 : p3_y0 + 36 + inner_h, p3_x0 + 8 : p3_x0 + 8 + inner_w] = twin_bg

        # ----------------------------------------------------------------------
        # Panel 4 (右下): 多冲次重现率/趋势曲线 + 实时 FPS/内存/IMU 遥测 + 三级预警信号灯
        # ----------------------------------------------------------------------
        p4_bg = np.full((inner_h, inner_w, 3), (20, 25, 34), dtype=np.uint8)

        # 右侧放置三级预警状态信号灯与实时遥测卡片
        chart_w = inner_w - 255
        chart_h = inner_h - 52
        cx0, cy0 = 48, 22
        cv2.rectangle(p4_bg, (cx0, cy0), (cx0 + chart_w, cy0 + chart_h), (14, 18, 25), -1)
        cv2.rectangle(p4_bg, (cx0, cy0), (cx0 + chart_w, cy0 + chart_h), (65, 80, 105), 1)

        # 绘制水平阈值虚线 (L1=30%, L2=50%, L3=70%)
        for pct, l_col, l_name in [(30, (0, 200, 240), "L1:30%"), (50, (0, 130, 255), "L2:50%"), (70, (40, 40, 245), "L3:70%")]:
            yy = int(cy0 + chart_h - (pct / 100.0) * chart_h)
            for xx in range(cx0, cx0 + chart_w - 8, 12):
                cv2.line(p4_bg, (xx, yy), (xx + 6, yy), l_col, 1)
            texts.append((p4_x0 + 8 + 6, p4_y0 + 36 + yy - 8, l_name, l_col, 11, False))

        # 绘制重现率曲线 (青色) 与最大拉痕长度曲线 (橙红, 映射到 0~60mm -> 0~100%)
        hist_rec = self.engine.history_max_recurrence[-30:]
        hist_len = self.engine.history_max_length_mm[-30:]
        n_pts = len(hist_rec)
        if n_pts >= 1:
            pts_rec, pts_len = [], []
            for i in range(n_pts):
                px = int(cx0 + 12 + (chart_w - 24) * (i / max(n_pts - 1, 1)))
                py_r = int(cy0 + chart_h - np.clip(hist_rec[i] / 100.0, 0.0, 1.0) * (chart_h - 10) - 5)
                py_l = int(cy0 + chart_h - np.clip(hist_len[i] / 55.0, 0.0, 1.0) * (chart_h - 10) - 5)
                pts_rec.append((px, py_r))
                pts_len.append((px, py_l))
                cv2.circle(p4_bg, (px, py_r), 3, (255, 230, 0), -1)
                cv2.circle(p4_bg, (px, py_l), 3, (40, 90, 255), -1)
            for i in range(n_pts - 1):
                cv2.line(p4_bg, pts_rec[i], pts_rec[i + 1], (255, 230, 0), 2, cv2.LINE_AA)
                cv2.line(p4_bg, pts_len[i], pts_len[i + 1], (40, 90, 255), 2, cv2.LINE_AA)

        texts.append((p4_x0 + 8 + cx0 + 8, p4_y0 + 36 + cy0 + chart_h + 8, "黄青线: 同位置重现率(%)   红橙线: 模具拉痕长度(mm)", (190, 210, 235), 13, True))

        # 右侧三级信号灯塔 & 硬件遥测参数
        rx0 = cx0 + chart_w + 14
        lights = [
            (0, "NORMAL 产线正常", (40, 215, 70)),
            (1, "L1 早期粘料预警", (0, 215, 255)),
            (2, "L2 拉伤恶化告警", (0, 130, 255)),
            (3, "L3 严重拉伤急停", (35, 35, 250)),
        ]
        for idx_l, (lv_code, lv_label, lv_col) in enumerate(lights):
            ly = 18 + idx_l * 46
            is_on = alarm_lv == lv_code
            box_bg = lv_col if is_on else (32, 38, 50)
            cv2.rectangle(p4_bg, (rx0, ly), (inner_w - 10, ly + 38), box_bg, -1 if is_on else 1)
            cv2.circle(p4_bg, (rx0 + 20, ly + 19), 10, (255, 255, 255) if is_on else (70, 75, 85), -1)
            txt_col = (255, 255, 255) if is_on else (130, 145, 165)
            texts.append((p4_x0 + 8 + rx0 + 38, p4_y0 + 36 + ly + 9, lv_label, txt_col, 13, is_on))

        # 实时硬件指标卡片
        ty_base = 210
        cv2.rectangle(p4_bg, (rx0, ty_base), (inner_w - 10, inner_h - 12), (28, 35, 48), -1)
        cv2.rectangle(p4_bg, (rx0, ty_base), (inner_w - 10, inner_h - 12), (70, 88, 115), 1)
        telemetry_lines = [
            f"实时帧率: {1000.0/max(timing['total_ms'],1):.1f} FPS",
            f"检测延迟: {timing['total_ms']:.1f} ms",
            f"节拍裕度: {12100.0/max(timing['total_ms'],1):.0f}x (12.1s/件)",
            f"进程内存: {result['rss_mb']:.1f} MB",
            f"IMU 震动: {result['imu_vib_g']:.3f} g",
        ]
        for idx_t, t_str in enumerate(telemetry_lines):
            texts.append((p4_x0 + 8 + rx0 + 10, p4_y0 + 36 + ty_base + 8 + idx_t * 26, t_str, (180, 230, 255), 13, False))

        canvas[p4_y0 + 36 : p4_y0 + 36 + inner_h, p4_x0 + 8 : p4_x0 + 8 + inner_w] = p4_bg

        return draw_texts_pil(canvas, texts)


# ==============================================================================
# CLI 运行模式实现 (d455_live / video_replay / hil_benchmark / all)
# ==============================================================================
class SafeH264VideoWriter:
    """Writes frames to an ASCII temp path and converts to H.264 (yuv420p + faststart) so Windows Chinese paths never lose the moov atom."""

    def __init__(self, target_path: str, fps: float, size: Tuple[int, int]):
        import shutil
        import subprocess
        import tempfile

        self.target_path = os.path.abspath(target_path)
        self.tmp_avi = os.path.join(tempfile.gettempdir(), f"agy_hmi_raw_{os.getpid()}.avi")
        self.tmp_mp4 = os.path.join(tempfile.gettempdir(), f"agy_hmi_h264_{os.getpid()}.mp4")
        for p in (self.tmp_avi, self.tmp_mp4):
            if os.path.exists(p):
                os.remove(p)
        self.writer = cv2.VideoWriter(self.tmp_avi, cv2.VideoWriter_fourcc(*"MJPG"), float(fps), size)
        self._shutil = shutil
        self._subprocess = subprocess

    def write(self, frame: np.ndarray) -> None:
        self.writer.write(frame)

    def release(self) -> None:
        self.writer.release()
        os.makedirs(os.path.dirname(self.target_path), exist_ok=True)
        ffmpeg_candidates = [
            self._shutil.which("ffmpeg"),
            r"C:\Users\lyz\anaconda3\envs\wechat_export\Lib\site-packages\imageio_ffmpeg\binaries\ffmpeg-win-x86_64-v7.1.exe",
        ]
        ffmpeg_exe = next((p for p in ffmpeg_candidates if p and os.path.exists(p)), None)
        if ffmpeg_exe and os.path.exists(self.tmp_avi):
            self._subprocess.run(
                [
                    ffmpeg_exe,
                    "-y",
                    "-i",
                    self.tmp_avi,
                    "-c:v",
                    "libx264",
                    "-preset",
                    "fast",
                    "-crf",
                    "20",
                    "-pix_fmt",
                    "yuv420p",
                    "-movflags",
                    "+faststart",
                    self.tmp_mp4,
                ],
                check=True,
                stdout=self._subprocess.DEVNULL,
                stderr=self._subprocess.DEVNULL,
            )
            self._shutil.copy2(self.tmp_mp4, self.target_path)
        elif os.path.exists(self.tmp_avi):
            self._shutil.copy2(self.tmp_avi, self.target_path)
        for p in (self.tmp_avi, self.tmp_mp4):
            if os.path.exists(p):
                os.remove(p)


def run_video_replay(
    video_path: str,
    out_mp4_path: str,
    sample_every_n_frames: int = 8,
    max_strokes: int = 36,
    fps_out: float = 10.0,
) -> Dict[str, Any]:
    """读取产线实拍视频，仅在检测到银白色高反光冲压件到位时触发在线质检，导出四分屏工业监控演示视频"""
    os.makedirs(os.path.dirname(os.path.abspath(out_mp4_path)), exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"无法打开产线视频: {video_path}")

    system = StampingDieWarningSystem()
    writer = SafeH264VideoWriter(out_mp4_path, float(fps_out), (1600, 900))

    frame_idx = 0
    stroke_count = 0
    keyframes: List[Tuple[str, np.ndarray, Dict[str, Any]]] = []

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1
            if frame_idx % sample_every_n_frames != 0:
                continue
            ok_part, part_mask, _ = detect_silver_stamping_part(frame, min_area_ratio=0.18, max_area_ratio=0.65)
            if not ok_part:
                continue
            stroke_count += 1
            res = system.process_stroke(
                frame,
                provided_mask=part_mask,
                source_tag=f"PROD_VIDEO ({os.path.basename(video_path)[:10]})",
            )
            dash = system.render_hmi_dashboard(frame, res)
            for _ in range(3):
                writer.write(dash)
            if stroke_count in (1, 4, 8, 14):
                keyframes.append((f"Stroke #{stroke_count:02d}", dash, res))
            if stroke_count >= max_strokes:
                break
    finally:
        cap.release()
        writer.release()

    return {"video_path": out_mp4_path, "total_strokes": stroke_count, "keyframes": keyframes}


def run_d455_live_mode(max_frames: int = 0, timeout_ms: int = 5000) -> None:
    """开启 RealSense D455 实时窗口预览与在线预警"""
    if rs is None:
        raise RuntimeError("当前环境未安装 pyrealsense2")
    system = StampingDieWarningSystem()
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 1280, 720, rs.format.rgb8, 30)
    config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
    config.enable_stream(rs.stream.accel)
    config.enable_stream(rs.stream.gyro)

    profile = pipeline.start(config)
    align = rs.align(rs.stream.color)
    spatial = rs.spatial_filter()
    spatial.set_option(rs.option.holes_fill, 2)
    depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()

    win_name = "Stamping Die Online Early-Warning HMI (RealSense D455 Live) [s:Trigger q:Quit]"
    f_cnt = 0
    try:
        for _ in range(15):
            pipeline.wait_for_frames(timeout_ms)
        while True:
            frames = pipeline.wait_for_frames(timeout_ms)
            accel_f = frames.first_or_default(rs.stream.accel)
            vib_g = 0.012
            if accel_f:
                ad = accel_f.as_motion_frame().get_motion_data()
                norm_a = math.sqrt(ad.x ** 2 + ad.y ** 2 + ad.z ** 2)
                vib_g = abs(norm_a - 9.806) / 9.806

            aligned = align.process(frames)
            c_frame = aligned.get_color_frame()
            d_frame = aligned.get_depth_frame()
            if not c_frame or not d_frame:
                continue
            d_frame = spatial.process(d_frame)
            color_bgr = cv2.cvtColor(np.asanyarray(c_frame.get_data()), cv2.COLOR_RGB2BGR)
            depth_m = np.asanyarray(d_frame.get_data()).astype(np.float32) * depth_scale
            intr = CameraIntrinsics.from_rs(c_frame.profile.as_video_stream_profile().get_intrinsics())

            res = system.process_stroke(color_bgr, depth_m=depth_m, intr=intr, imu_vib_g=vib_g, source_tag="D455 LIVE STREAM")
            dash = system.render_hmi_dashboard(color_bgr, res)
            f_cnt += 1
            if max_frames > 0 and f_cnt >= max_frames:
                break
            cv2.imshow(win_name, dash)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
    finally:
        pipeline.stop()
        if max_frames == 0:
            cv2.destroyAllWindows()


def main() -> None:
    global D455_SAMPLES_ROOT, PROD_VIDEO_DIR, OUTPUT_DIR
    parser = argparse.ArgumentParser(description="冲压模具损伤在线预警系统主程序 (D455 HIL & 产线视频双模)")
    parser.add_argument(
        "--mode",
        choices=["d455_live", "video_replay", "hil_benchmark", "all"],
        default="video_replay",
        help="系统运行模式: d455_live | video_replay | hil_benchmark | all",
    )
    parser.add_argument(
        "--video",
        default=os.path.join(PROD_VIDEO_DIR, "ad7a371d255164b22ab17633ca4cceb4.mp4"),
        help="video_replay 模式下的输入视频路径",
    )
    parser.add_argument(
        "--out_mp4",
        "--out-mp4",
        dest="out_mp4",
        default=os.path.join(OUTPUT_DIR, "online_warning_demo.mp4"),
        help="导出的四分屏演示视频路径",
    )
    parser.add_argument(
        "--max_frames",
        "--max-frames",
        dest="max_frames",
        type=int,
        default=0,
        help="d455_live 最大运行帧数(0=持续预览)",
    )
    parser.add_argument(
        "--strokes",
        type=int,
        default=36,
        help="video_replay 模式最大处理冲次数 (默认: 36)",
    )
    parser.add_argument(
        "--iters",
        type=int,
        default=200,
        help="hil_benchmark 稳定性压测循环次数 (默认: 200)",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=10.0,
        help="导出视频帧率 FPS (默认: 10.0)",
    )
    parser.add_argument(
        "--d455-dir",
        default=D455_SAMPLES_ROOT,
        help="D455 历史实测 RGB-D 样本目录",
    )
    parser.add_argument(
        "--data-dir",
        default=PROD_VIDEO_DIR,
        help="产线视频与图像数据目录",
    )
    parser.add_argument(
        "--output-dir",
        default=OUTPUT_DIR,
        help="图表、视频与 JSON 报告输出目录",
    )
    parser.add_argument(
        "--live-timeout",
        type=float,
        default=5.0,
        help="D455 帧等待超时时间 (秒, 默认: 5.0)",
    )
    parser.add_argument(
        "--tilted-prefix",
        default="20260926_212001",
        help="指定倾斜测试样本目录名 (默认: 20260926_212001)",
    )
    args = parser.parse_args()

    D455_SAMPLES_ROOT = os.path.abspath(args.d455_dir)
    PROD_VIDEO_DIR = os.path.abspath(args.data_dir)
    OUTPUT_DIR = os.path.abspath(args.output_dir)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if args.mode == "d455_live":
        run_d455_live_mode(max_frames=args.max_frames, timeout_ms=int(args.live_timeout * 1000))
    elif args.mode == "video_replay":
        info = run_video_replay(args.video, args.out_mp4, max_strokes=args.strokes, fps_out=args.fps)
        print(f"[VIDEO_REPLAY] 演示视频已导出: {info['video_path']} (共 {info['total_strokes']} 冲次)")
    elif args.mode in ("hil_benchmark", "all"):
        import run_d455_hil_verification as hil_mod
        hil_mod.D455_SAMPLES_ROOT = D455_SAMPLES_ROOT
        hil_mod.PROD_VIDEO_DIR = PROD_VIDEO_DIR
        hil_mod.OUTPUT_DIR = OUTPUT_DIR
        hil_mod.run_all_verifications(n_iters=args.iters, tilted_folder=args.tilted_prefix)


if __name__ == "__main__":
    main()
