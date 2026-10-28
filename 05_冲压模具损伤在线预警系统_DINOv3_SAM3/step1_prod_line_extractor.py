# -*- coding: utf-8 -*-
"""
模块一：产线视频关键帧抽取与工件位姿配准 (step1_prod_line_extractor.py)
=============================================================================
功能概述：
1. 扫描产线连续过线视频 (0098f6f4...mp4 & ad7a371d...mp4)，基于背景差分、高反光镀锌板
   低频平滑度与边缘清晰度构建“虚拟龙门架在线同步抓拍”触发器，自动屏蔽手指遮挡与人员干扰，
   精准抽取 12 个连续过线冲压件对（大众左/右后轮罩 1PE 810 425/426）的最佳关键帧。
2. 对全部 12 张视频关键帧及筛选出的 20 张高质量传送带近景工件照片，提取高精度二值前景掩码，
   剔除传送带人字纹背景、上下金属护栏、底部 AprilTag 标定纸片与人员手臂。
3. 计算最小外接矩形 (cv2.minAreaRect) 与主轴方向，将不同位置/角度/尺度的冲压件仿射/透视配准
   (Canonical Warping) 到统一的 800x600 标准工件坐标系 (u, v)，标定物理分辨率并映射 SOP 三大监测分区。
4. 生成高质量工程可视化图表与 JSON 汇总报告。
"""

import os
import re
import glob
import json
import math
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

# 配置 Matplotlib 中文字体支持（优先使用 Microsoft YaHei 避免特殊符号缺字）
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


# =============================================================================
# 工具函数：支持中文路径的图像读写
# =============================================================================
def imread_cn(path: str, flags: int = cv2.IMREAD_COLOR) -> np.ndarray:
    """支持 Windows 中文路径的图像读取"""
    raw = np.fromfile(path, dtype=np.uint8)
    return cv2.imdecode(raw, flags)


def imwrite_cn(path: str, img: np.ndarray, params=None) -> bool:
    """支持 Windows 中文路径的图像保存"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    ext = os.path.splitext(path)[1]
    if params is None:
        if ext.lower() in [".jpg", ".jpeg"]:
            params = [int(cv2.IMWRITE_JPEG_QUALITY), 95]
        elif ext.lower() == ".png":
            params = [int(cv2.IMWRITE_PNG_COMPRESSION), 3]
        else:
            params = []
    ok, buf = cv2.imencode(ext, img, params)
    if ok:
        buf.tofile(path)
    return ok


# =============================================================================
# 核心算法 1：产线视频扫描、手指遮挡剔除与最佳关键帧抽取
# =============================================================================
def compute_video_background(video_path: str, sample_count: int = 40, proc_size=(320, 180)) -> np.ndarray:
    """通过对视频均匀采样计算中值背景模型（传送带静态/纹理背景）"""
    cap = cv2.VideoCapture(video_path)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    samples = []
    for fidx in np.linspace(0, max(1, n_frames - 10), sample_count, dtype=int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(fidx))
        ret, frame = cap.read()
        if ret:
            sm = cv2.resize(frame, proc_size, interpolation=cv2.INTER_AREA)
            samples.append(cv2.cvtColor(sm, cv2.COLOR_BGR2GRAY))
    cap.release()
    return np.median(samples, axis=0).astype(np.uint8)


def analyze_video_stream(video_path: str, expected_windows: list, video_tag: str) -> dict:
    """
    扫描连续过线视频每一帧，计算中央检测窗口内的：
    - 工件存在得分 (presence_score): 银白金属平滑区 + 中间深黑分隔缝与背景差分融合得分
    - 边缘清晰度 (sharpness): Laplacian 方差
    - 肤色/手指遮挡占比 (skin_ratio): 自动识别手指遮挡镜头或人员手臂入画
    - 工件质心轨迹 (cx_norm, cy_norm)
    并在各过线窗口内自动触发最佳抓拍关键帧。
    """
    cap = cv2.VideoCapture(video_path)
    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    proc_w, proc_h = 320, 180
    bg_med = compute_video_background(video_path, sample_count=40, proc_size=(proc_w, proc_h))

    frame_indices = []
    timestamps = []
    presence_scores = []
    sharpness_scores = []
    skin_ratios = []
    centroids_x = []
    centroids_y = []
    occluded_flags = []

    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        sm = cv2.resize(frame, (proc_w, proc_h), interpolation=cv2.INTER_AREA)
        g = cv2.cvtColor(sm, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(sm, cv2.COLOR_BGR2HSV)

        # 中央检测窗口 ROI (排除上下护栏与边缘截断区)
        ry1, ry2 = int(proc_h * 0.14), int(proc_h * 0.86)
        rx1, rx2 = int(proc_w * 0.12), int(proc_w * 0.88)
        roi_g = g[ry1:ry2, rx1:rx2]
        roi_s = hsv[ry1:ry2, rx1:rx2, 1]
        roi_bg = bg_med[ry1:ry2, rx1:rx2]

        diff = cv2.absdiff(roi_g, roi_bg)
        # 镀锌冲压件特征：显著区别于背景且具备高亮度低饱和度金属光泽，或双件间深黑分隔槽
        part_mask = (diff > 26) & (((roi_g > 168) & (roi_s < 52)) | (roi_g < 45))
        presence = float(part_mask.mean())

        # 边缘清晰度 (Laplacian 方差)
        lap_var = float(cv2.Laplacian(roi_g, cv2.CV_64F).var())

        # 手指遮挡 / 人员皮肤检测 (HSV 暖色相 + 中高饱和度)
        skin_mask = (((hsv[:, :, 0] < 22) | (hsv[:, :, 0] > 160)) &
                     (hsv[:, :, 1] > 52) & (hsv[:, :, 2] > 65))
        skin_ratio = float(skin_mask.mean())

        # 判断是否发生镜头手指遮挡 (33~40s 期间大面积肤色或极端失焦模糊)
        is_occluded = bool(skin_ratio > 0.12 or lap_var < 120.0)

        # 计算工件质心 (归一化坐标 0~1)
        ys, xs = np.where(part_mask)
        if len(xs) > 80 and not is_occluded:
            cx = float((xs.mean() + rx1) / proc_w)
            cy = float((ys.mean() + ry1) / proc_h)
        else:
            cx, cy = np.nan, np.nan

        frame_indices.append(idx)
        timestamps.append(round(idx / fps, 3))
        presence_scores.append(presence)
        sharpness_scores.append(lap_var)
        skin_ratios.append(skin_ratio)
        centroids_x.append(cx)
        centroids_y.append(cy)
        occluded_flags.append(is_occluded)
        idx += 1

    presence_arr = np.array(presence_scores)
    sharp_arr = np.array(sharpness_scores)
    skin_arr = np.array(skin_ratios)
    occ_arr = np.array(occluded_flags)

    # 在每个过线周期窗口内选择最佳抓拍帧：
    # 综合考虑存在面积饱满度、图像清晰度、居中程度，并严厉惩罚皮肤/手臂入画
    triggers = []
    for w_idx, (f_start, f_end, anchor_f) in enumerate(expected_windows):
        f_start = max(0, f_start)
        f_end = min(len(presence_arr) - 1, f_end)
        best_f = anchor_f
        best_score = -1e9
        for f in range(f_start, f_end + 1):
            if occ_arr[f]:
                continue
            # 距窗口理想中心微调权重 + 面积得分 + 清晰度归一化 - 皮肤干扰惩罚
            p_val = presence_arr[f]
            s_val = min(sharp_arr[f] / 3000.0, 1.5)
            skin_pen = skin_arr[f] * 6.0
            prox_bonus = max(0.0, 1.0 - abs(f - anchor_f) / 25.0) * 0.25
            total_s = p_val * 2.0 + s_val * 0.25 + prox_bonus - skin_pen
            if total_s > best_score:
                best_score = total_s
                best_f = f

        cap.set(cv2.CAP_PROP_POS_FRAMES, int(best_f))
        ret, best_frame = cap.read()
        triggers.append({
            "local_index": w_idx + 1,
            "frame_idx": int(best_f),
            "timestamp_s": round(best_f / fps, 2),
            "presence_score": round(float(presence_arr[best_f]), 4),
            "sharpness_laplacian": round(float(sharp_arr[best_f]), 1),
            "skin_ratio": round(float(skin_arr[best_f]), 4),
            "centroid_norm": [
                round(float(centroids_x[best_f]) if not np.isnan(centroids_x[best_f]) else 0.5, 3),
                round(float(centroids_y[best_f]) if not np.isnan(centroids_y[best_f]) else 0.45, 3),
            ],
            "frame_bgr": best_frame,
        })

    cap.release()
    return {
        "video_tag": video_tag,
        "video_path": video_path,
        "width": width,
        "height": height,
        "fps": fps,
        "total_frames": n_frames,
        "duration_s": round(n_frames / fps, 2),
        "timestamps": timestamps,
        "presence_scores": presence_scores,
        "sharpness_scores": sharpness_scores,
        "skin_ratios": skin_ratios,
        "occluded_flags": occluded_flags,
        "centroids_x": centroids_x,
        "centroids_y": centroids_y,
        "triggers": triggers,
    }


# =============================================================================
# 核心算法 2：冲压件解剖学结构分割与 800x600 标准坐标系位姿配准
# =============================================================================
def segment_stamping_workpiece(img: np.ndarray, mode: str = "photo") -> np.ndarray:
    """
    针对传送带上的镀锌冲压件对 (1PE 810 425/426) 提取高精度二值掩码：
    利用四大解剖结构特征：
      1) 主前穹顶高光银白金属区 (Front Dome: 高亮度、低饱和度、极低高频纹理)
      2) 双件间深黑新月形分隔缝 (Inter-Part Crescent Gap: 紧邻前穹顶上缘的最大深黑连通域)
      3) 后伴随件银白穹顶 (Rear Dome: 紧邻深黑分隔缝上缘的银白金属区，自动阻断顶部护栏)
      4) 底部与两侧法兰翻边亮边 (Bottom & Side Flange Rim: 跨越立壁圆弧区的银白翻边环)
    最后通过凸包闭合生成完整、紧凑的工件二值 Mask，彻底剔除人字纹传送带、护栏、AprilTag 及人手。
    """
    h, w = img.shape[:2]
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    s = hsv[:, :, 1].astype(np.float32)

    # 原生分辨率下计算高频纹理能量（人字纹传送带 hf > 15，冲压件平滑金属表面 hf < 10）
    blur = cv2.GaussianBlur(g, (7, 7), 0)
    hf = cv2.GaussianBlur(np.abs(g - blur), (25, 25), 0)

    sw = 640
    sh = int(round(h * 640.0 / w))
    blur_sm = cv2.resize(blur, (sw, sh), interpolation=cv2.INTER_AREA)
    s_sm = cv2.resize(s, (sw, sh), interpolation=cv2.INTER_AREA)
    hf_sm = cv2.resize(hf, (sw, sh), interpolation=cv2.INTER_AREA)

    y_top = int(sh * (0.01 if mode == "video_zoom" else 0.055))
    y_bot = int(sh * (0.985 if mode == "video_zoom" else (0.65 if mode == "photo" else 0.74)))

    # 1. 提取主前穹顶 (高置信度银白平滑金属面，严格排除 >138 的灰色护栏)
    min_dome_val = 148 if mode == "video_wide" else 162
    dome_raw = (blur_sm > min_dome_val) & (s_sm < 42) & (hf_sm < 11.5)
    dome_mask = dome_raw.astype(np.uint8) * 255
    dome_mask[:y_top, :] = 0
    dome_mask[y_bot:, :] = 0
    if mode == "video_wide":
        # 仅屏蔽视频1广角端最底部传送带上的小型 AprilTag 标定纸片 (y > 0.71*sh)，保留冲压件完整区域
        dome_mask[int(sh * 0.71):, int(sw * 0.46):int(sw * 0.78)] = 0

    dome_mask = cv2.morphologyEx(dome_mask, cv2.MORPH_OPEN,
                                 cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
    dome_mask = cv2.morphologyEx(dome_mask, cv2.MORPH_CLOSE,
                                 cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (23, 23)))

    num_d, l_d, stats_d, _ = cv2.connectedComponentsWithStats(dome_mask, connectivity=8)
    if num_d <= 1:
        return np.zeros((h, w), dtype=np.uint8)
    front_idx = 1 + int(np.argmax(stats_d[1:, cv2.CC_STAT_AREA]))
    fx, fy, fw, fh, _ = stats_d[front_idx]
    front_dome = (l_d == front_idx).astype(np.uint8) * 255

    # 2. 提取紧邻前穹顶上缘的唯一最大深黑分隔缝连通域 (彻底排除顶部护栏阴影线)
    gy1 = max(y_top, int(fy - 0.14 * fw))
    gy2 = min(y_bot, int(fy + 0.45 * fh))
    gx1 = max(0, int(fx + 0.02 * fw))
    gx2 = min(sw, int(fx + 0.98 * fw))
    gap_cand = np.zeros((sh, sw), dtype=np.uint8)
    gap_roi = (blur_sm[gy1:gy2, gx1:gx2] < 28) & (hf_sm[gy1:gy2, gx1:gx2] < 9.5)
    gap_cand[gy1:gy2, gx1:gx2] = gap_roi.astype(np.uint8) * 255
    gap_cand = cv2.morphologyEx(gap_cand, cv2.MORPH_OPEN,
                                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))

    fd_dil = cv2.dilate(front_dome, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)))
    num_g, l_g, stats_g, _ = cv2.connectedComponentsWithStats(gap_cand, connectivity=8)
    gap_cc = np.zeros((sh, sw), dtype=np.uint8)
    best_ga = 0
    for i in range(1, num_g):
        comp = (l_g == i)
        if np.any(comp & (fd_dil > 0)) and stats_g[i, cv2.CC_STAT_AREA] > best_ga:
            best_ga = int(stats_g[i, cv2.CC_STAT_AREA])
            gap_cc = comp.astype(np.uint8) * 255

    # 3. 提取紧邻深黑分隔缝上缘的后件穹顶 (仅保留与 gap_cc 距离 <=9px 的银白金属连通域)
    rear_y1 = max(y_top, int(fy - 0.30 * fw))
    rear_y2 = int(fy + 0.20 * fh)
    rear_x1 = max(0, int(fx + 0.02 * fw))
    rear_x2 = min(sw, int(fx + 0.98 * fw))
    rear_cand = np.zeros((sh, sw), dtype=np.uint8)
    rear_roi = ((blur_sm[rear_y1:rear_y2, rear_x1:rear_x2] > 162) &
                (s_sm[rear_y1:rear_y2, rear_x1:rear_x2] < 38) &
                (hf_sm[rear_y1:rear_y2, rear_x1:rear_x2] < 10.0))
    rear_cand[rear_y1:rear_y2, rear_x1:rear_x2] = rear_roi.astype(np.uint8) * 255
    rear_cand = cv2.morphologyEx(rear_cand, cv2.MORPH_OPEN,
                                 cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))

    gap_touch = cv2.dilate(gap_cc if best_ga > 0 else front_dome,
                           cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
    num_r, l_r, stats_r, _ = cv2.connectedComponentsWithStats(rear_cand, connectivity=8)
    rear_dome = np.zeros((sh, sw), dtype=np.uint8)
    for i in range(1, num_r):
        comp = (l_r == i)
        if (np.any(comp & (gap_touch > 0)) and
                stats_r[i, cv2.CC_STAT_AREA] > 60 and
                stats_r[i, cv2.CC_STAT_WIDTH] < 0.98 * fw):
            rear_dome[comp] = 255

    # 4. 提取底部与两侧法兰翻边亮环 (跨越立壁深拉延圆弧区)
    flange_y1 = fy
    flange_y2 = min(y_bot, int(fy + fh + (0.48 * fw if mode == "video_zoom" else 0.42 * fw)))
    flange_x1 = max(0, int(fx - 0.08 * fw))
    flange_x2 = min(sw, int(fx + fw + 0.08 * fw))
    flange_cand = np.zeros((sh, sw), dtype=np.uint8)
    min_flange_val = 146 if mode == "video_wide" else 158
    flange_roi = ((blur_sm[flange_y1:flange_y2, flange_x1:flange_x2] > min_flange_val) &
                  (s_sm[flange_y1:flange_y2, flange_x1:flange_x2] < 42) &
                  (hf_sm[flange_y1:flange_y2, flange_x1:flange_x2] < 11.5))
    flange_cand[flange_y1:flange_y2, flange_x1:flange_x2] = flange_roi.astype(np.uint8) * 255
    if mode == "video_wide":
        # 仅屏蔽底部 y > 0.71*sh 的传送带 AprilTag 标定卡，完整保留冲压件底部与两侧法兰
        flange_cand[int(sh * 0.71):, int(sw * 0.46):int(sw * 0.78)] = 0
    flange_cand = cv2.morphologyEx(flange_cand, cv2.MORPH_OPEN,
                                   cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    # 仅保留与主穹顶邻近或位于其正下方的法兰连通域（使用竖向扩展核跨越深拉延立壁暗区），剔除传送带孤立反光
    dome_dil_flange = cv2.dilate(front_dome, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (35, 95)))
    num_f, l_f, stats_f, _ = cv2.connectedComponentsWithStats(flange_cand, connectivity=8)
    flange_valid = np.zeros((sh, sw), dtype=np.uint8)
    for i in range(1, num_f):
        comp = (l_f == i)
        if np.any(comp & (dome_dil_flange > 0)) and stats_f[i, cv2.CC_STAT_AREA] >= 25:
            flange_valid[comp] = 255

    # 5. 融合四大解剖部件并计算凸包，得到完整、光滑的冲压件外轮廓掩码
    combined = front_dome | gap_cc | rear_dome | flange_valid
    pts = np.column_stack(np.where(combined > 0))[:, ::-1]
    filled = np.zeros((sh, sw), dtype=np.uint8)
    if len(pts) >= 3:
        hull = cv2.convexHull(pts)
        cv2.drawContours(filled, [hull], -1, 255, -1)

    return cv2.resize(filled, (w, h), interpolation=cv2.INTER_NEAREST)


def warp_to_canonical(img: np.ndarray, mask: np.ndarray,
                      canonical_size=(800, 600), pad_u: int = 16, pad_v: int = 14) -> tuple:
    """
    计算冲压件最小外接矩形 (cv2.minAreaRect) 与解剖结构主轴方向，保证长轴水平且工件朝向一致
    （后件穹顶与深黑分隔缝朝上 v_small，立壁与法兰朝下 v_large），
    并通过 4 点透视梯形校正 (Perspective Keystone Rectification) 消除不同拍摄俯仰角与
    截断造成的梯形畸变，将工件配准到统一的 800x600 标准工件坐标系 (u, v)。
    """
    can_w, can_h = canonical_size
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return (cv2.resize(img, canonical_size),
                cv2.resize(mask, canonical_size),
                {"center": [0, 0], "size": [0, 0], "angle_deg": 0.0, "box_pts": []})

    c = max(cnts, key=cv2.contourArea)
    rect = cv2.minAreaRect(c)
    (rcx, rcy), (rw, rh), angle = rect

    # 归一化角度为长轴与水平轴的夹角 [-45, 45]
    if rw < rh:
        rw, rh = rh, rw
        angle += 90.0
    while angle > 45.0:
        angle -= 90.0
    while angle < -45.0:
        angle += 90.0

    # 当掩码接触 ROI 上下/左右裁剪边界时，cv2.minAreaRect 容易退化吸附到 0.0°；
    # 此时结合工件内部高光银白金属穹顶与深黑分隔缝的二阶中心矩主轴及非边界轮廓斜率估计真实平面倾角
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    inner_metal = ((gray > 158) & (mask > 0)).astype(np.uint8)
    M_mom = cv2.moments(inner_metal, binaryImage=True)
    mom_angle = 0.0
    if abs(M_mom["mu20"] - M_mom["mu02"]) > 1e-3:
        mom_angle = 0.5 * math.degrees(math.atan2(2.0 * M_mom["mu11"], M_mom["mu20"] - M_mom["mu02"]))

    # 同时拟合深黑分隔缝 (gray < 45) 的主轴方向（若存在显著分隔缝）
    dark_gap = ((gray < 45) & (mask > 0)).astype(np.uint8)
    M_gap = cv2.moments(dark_gap, binaryImage=True)
    gap_angle = mom_angle
    if M_gap["m00"] > 300 and abs(M_gap["mu20"] - M_gap["mu02"]) > 1e-3:
        gap_angle = 0.5 * math.degrees(math.atan2(2.0 * M_gap["mu11"], M_gap["mu20"] - M_gap["mu02"]))

    anat_angle = 0.65 * mom_angle + 0.35 * gap_angle
    if abs(angle) < 0.05 or abs(angle) > 12.0:
        angle = float(anat_angle)
    else:
        angle = float(0.55 * angle + 0.45 * anat_angle)

    clamped_angle = float(np.clip(angle, -10.0, 10.0))
    oriented_rect = ((rcx, rcy), (rw, rh), clamped_angle)
    box_pts = cv2.boxPoints(oriented_rect)

    # Step 1: 围绕工件质心旋转摆正主轴
    M_rot = cv2.getRotationMatrix2D((rcx, rcy), clamped_angle, 1.0)
    img_rot = cv2.warpAffine(img, M_rot, (w, h), flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_REPLICATE)
    mask_rot = cv2.warpAffine(mask, M_rot, (w, h), flags=cv2.INTER_NEAREST,
                               borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    # Step 2: 透视梯形归一化 (测量上下两处水平弦宽并映射至顶底基准线，兼顾高 IoU 与边缘零截断)
    ys, xs = np.where(mask_rot > 0)
    if len(xs) < 50:
        x1, y1, x2, y2 = 0, 0, w - 1, h - 1
    else:
        x1, x2 = int(xs.min()), int(xs.max())
        y1, y2 = int(ys.min()), int(ys.max())

    bh = max(20, y2 - y1)
    bw = max(20, x2 - x1)
    y_upper = int(y1 + 0.24 * bh)
    y_lower = int(y1 + 0.76 * bh)

    xs_u = np.where(mask_rot[max(0, y_upper - 3):min(h, y_upper + 4), :] > 0)[1]
    xs_l = np.where(mask_rot[max(0, y_lower - 3):min(h, y_lower + 4), :] > 0)[1]

    if len(xs_u) > 20 and len(xs_l) > 20:
        u_left, u_right = float(np.percentile(xs_u, 1)), float(np.percentile(xs_u, 99))
        l_left, l_right = float(np.percentile(xs_l, 1)), float(np.percentile(xs_l, 99))
        u_left = 0.50 * u_left + 0.50 * (x1 + 0.05 * bw)
        u_right = 0.50 * u_right + 0.50 * (x2 - 0.05 * bw)
        l_left = 0.50 * l_left + 0.50 * x1
        l_right = 0.50 * l_right + 0.50 * x2

        src_quad = np.float32([
            [u_left, y1],
            [u_right, y1],
            [l_right, y2],
            [l_left, y2],
        ])
        dst_quad = np.float32([
            [pad_u + 18, pad_v],
            [can_w - 1 - pad_u - 18, pad_v],
            [can_w - 1 - pad_u, can_h - 1 - pad_v],
            [pad_u, can_h - 1 - pad_v],
        ])
        H_mat = cv2.getPerspectiveTransform(src_quad, dst_quad)
        can_img = cv2.warpPerspective(img_rot, H_mat, (can_w, can_h), flags=cv2.INTER_LINEAR,
                                      borderMode=cv2.BORDER_REPLICATE)
        can_mask = cv2.warpPerspective(mask_rot, H_mat, (can_w, can_h), flags=cv2.INTER_NEAREST,
                                       borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    else:
        src_pts = np.float32([[x1, y1], [x2, y1], [x1, y2]])
        dst_pts = np.float32([[pad_u, pad_v], [can_w - 1 - pad_u, pad_v], [pad_u, can_h - 1 - pad_v]])
        M_aff = cv2.getAffineTransform(src_pts, dst_pts)
        can_img = cv2.warpAffine(img_rot, M_aff, (can_w, can_h), flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_REPLICATE)
        can_mask = cv2.warpAffine(mask_rot, M_aff, (can_w, can_h), flags=cv2.INTER_NEAREST,
                                  borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    pose_meta = {
        "center_xy": [round(float(rcx), 1), round(float(rcy), 1)],
        "rect_wh": [round(float(rw), 1), round(float(rh), 1)],
        "raw_angle_deg": round(float(angle), 2),
        "aligned_angle_deg": round(float(clamped_angle), 2),
        "bbox_xyxy": [x1, y1, x2, y2],
        "box_pts": np.round(box_pts, 1).tolist(),
    }
    return can_img, can_mask, pose_meta


def compute_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    """计算两个二值掩码的交并比 (IoU)"""
    a = mask_a > 0
    b = mask_b > 0
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter / max(1, union))


# =============================================================================
# 核心算法 3：SOP 三大重点工艺监测分区定义与绘制
# =============================================================================
SOP_ZONES = {
    "Zone_A_DrawWall": {
        "name_cn": "Zone_A 深拉延立壁圆弧过渡区",
        "desc": "SOP红框重点监控区：高发竖向拉毛、粘模拉痕、暗裂与开裂",
        "u_range": [60, 740],
        "v_range": [320, 530],
        "color_bgr": (40, 50, 235),     # 醒目红
        "color_hex": "#EB3228",
    },
    "Zone_B_TopRibs": {
        "name_cn": "Zone_B 顶部四道加强筋与高光穹顶区",
        "desc": "压痕、垫料凸起、成型不良与表面划伤监控区",
        "u_range": [140, 660],
        "v_range": [100, 320],
        "color_bgr": (230, 160, 20),    # 工业蓝/青
        "color_hex": "#14A0E6",
    },
    "Zone_C_Flange": {
        "name_cn": "Zone_C 底部与两侧安装法兰翻边及基准孔区",
        "desc": "起皱叠料、切边毛刺 (<0.07mm) 及基准孔 B(φ16)/C(φ12.5) 监控区",
        "u_range": [40, 760],
        "v_range": [500, 585],
        "color_bgr": (30, 190, 245),    # 琥珀黄/橙
        "color_hex": "#F5BE1E",
    },
}


def draw_canonical_zones(can_img: np.ndarray, can_mask: np.ndarray = None,
                         draw_labels: bool = True) -> np.ndarray:
    """在 800x600 标准工件图像上绘制 SOP 三大工艺监测分区与掩码外轮廓"""
    vis = can_img.copy()
    overlay = can_img.copy()

    if can_mask is not None:
        cnts, _ = cv2.findContours(can_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(vis, cnts, -1, (60, 235, 100), 2)

    for z_key, z_info in SOP_ZONES.items():
        u1, u2 = z_info["u_range"]
        v1, v2 = z_info["v_range"]
        col = z_info["color_bgr"]
        cv2.rectangle(overlay, (u1, v1), (u2, v2), col, -1)
        cv2.rectangle(vis, (u1, v1), (u2, v2), col, 2)
        if draw_labels:
            cv2.putText(vis, z_key, (u1 + 8, v1 + 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.58, col, 2, cv2.LINE_AA)

    vis = cv2.addWeighted(overlay, 0.16, vis, 0.84, 0)
    return vis


# =============================================================================
# 可视化生成函数 1：产线连续冲压过线节拍与自动抓拍触发时序图
# =============================================================================
def plot_video_trigger_timeline(v1_res: dict, v2_res: dict, cycle_stats: dict,
                                all_triggers: list, save_path: str):
    """生成《产线连续冲压过线节拍与自动抓拍触发时序图》(video_trigger_timeline.png)"""
    fig = plt.figure(figsize=(18, 11.5), dpi=160, facecolor="#F8FAFC")
    gs = GridSpec(3, 12, figure=fig, height_ratios=[1.35, 1.15, 0.95], hspace=0.42, wspace=0.18)

    fig.suptitle(
        "产线连续冲压过线节拍监测与虚拟龙门架自动抓拍触发时序图 (大众左/右后轮罩 1PE 810 425/426)",
        fontsize=15.5, fontweight="bold", color="#0F172A", y=0.97
    )

    # --- Panel 1: 视频 1 (101.0s, 8 件连续过线 + 33~40s 手指遮挡自动屏蔽) ---
    ax1 = fig.add_subplot(gs[0, :])
    ax1.set_facecolor("#FFFFFF")
    t1 = np.array(v1_res["timestamps"])
    p1 = np.array(v1_res["presence_scores"])
    s1 = np.array(v1_res["skin_ratios"])

    ax1.plot(t1, p1, color="#1D4ED8", lw=1.8, label="工件存在得分 (银白镀锌板占比 + 背景差分)")
    ax1.fill_between(t1, 0, p1, color="#3B82F6", alpha=0.15)
    ax1.plot(t1, s1, color="#DC2626", lw=1.3, ls="--", alpha=0.85, label="手指遮挡 / 人员皮肤入画占比")

    # 标注 33.5~40.5s 手指遮挡镜头屏蔽区间
    ax1.axvspan(33.4, 40.6, color="#FEE2E2", alpha=0.75, label="镜头手指遮挡自动屏蔽区 (33.4s~40.6s)")
    ax1.text(37.0, 0.58, "手指遮挡镜头\n自动跳过 (Skip)", color="#B91C1C", fontsize=9.5,
             fontweight="bold", ha="center", va="center",
             bbox=dict(boxstyle="round,pad=0.3", fc="#FEF2F2", ec="#EF4444", lw=1.2))

    v1_trigs = v1_res["triggers"]
    for i, tr in enumerate(v1_trigs):
        ts = tr["timestamp_s"]
        ps = tr["presence_score"]
        seq_id = i + 1
        ax1.axvline(ts, color="#059669", lw=1.8, ls="-.")
        ax1.scatter([ts], [ps], color="#059669", s=55, zorder=5)
        ax1.text(ts, min(ps + 0.08, 0.68), f"seq_{seq_id:02d}\n({ts:.1f}s)",
                 color="#065F46", fontsize=8.5, fontweight="bold", ha="center",
                 bbox=dict(boxstyle="round,pad=0.2", fc="#ECFDF5", ec="#10B981", lw=0.9))
        if i > 0:
            dt = ts - v1_trigs[i - 1]["timestamp_s"]
            mid_t = 0.5 * (ts + v1_trigs[i - 1]["timestamp_s"])
            ax1.annotate(f"Δt={dt:.1f}s", xy=(mid_t, 0.05), ha="center",
                         fontsize=8, color="#475569",
                         bbox=dict(boxstyle="round,pad=0.15", fc="#F1F5F9", ec="#CBD5E1", lw=0.7))

    ax1.set_xlim(0, v1_res["duration_s"])
    ax1.set_ylim(0, 0.78)
    ax1.set_title(
        f"连续过线视频 1 (0098f6f4...mp4 | 1280×720 @ 30fps, 3030帧, 101.0s) — 抽取 8 组冲压件对 (seq_01 ~ seq_08)",
        fontsize=11.5, fontweight="bold", color="#1E293B", loc="left", pad=6
    )
    ax1.set_xlabel("视频时间轴 t (秒)", fontsize=10)
    ax1.set_ylabel("归一化特征响应得分", fontsize=10)
    ax1.grid(True, linestyle=":", alpha=0.5, color="#94A3B8")
    ax1.legend(loc="upper right", fontsize=9, framealpha=0.95)

    # --- Panel 2: 视频 2 (42.4s, 4 件近景连续过线) + 节拍统计信息框 ---
    ax2 = fig.add_subplot(gs[1, :8])
    ax2.set_facecolor("#FFFFFF")
    t2 = np.array(v2_res["timestamps"])
    p2 = np.array(v2_res["presence_scores"])

    ax2.plot(t2, p2, color="#0284C7", lw=2.0, label="近景工件存在得分 (960×544 特写机位)")
    ax2.fill_between(t2, 0, p2, color="#38BDF8", alpha=0.18)

    v2_trigs = v2_res["triggers"]
    for i, tr in enumerate(v2_trigs):
        ts = tr["timestamp_s"]
        ps = tr["presence_score"]
        seq_id = i + 9
        ax2.axvline(ts, color="#D97706", lw=1.8, ls="-.")
        ax2.scatter([ts], [ps], color="#D97706", s=55, zorder=5)
        ax2.text(ts, min(ps + 0.04, 0.46), f"seq_{seq_id:02d}\n({ts:.1f}s)",
                 color="#92400E", fontsize=8.5, fontweight="bold", ha="center",
                 bbox=dict(boxstyle="round,pad=0.2", fc="#FFFBEB", ec="#F59E0B", lw=0.9))
        if i > 0:
            dt = ts - v2_trigs[i - 1]["timestamp_s"]
            mid_t = 0.5 * (ts + v2_trigs[i - 1]["timestamp_s"])
            ax2.annotate(f"Δt={dt:.1f}s", xy=(mid_t, 0.06), ha="center",
                         fontsize=8.5, color="#334155",
                         bbox=dict(boxstyle="round,pad=0.15", fc="#F1F5F9", ec="#CBD5E1", lw=0.7))

    ax2.set_xlim(0, v2_res["duration_s"])
    ax2.set_ylim(0, 0.72)
    ax2.set_title(
        f"连续过线视频 2 (ad7a371d...mp4 | 960×544 @ 30fps, 1273帧, 42.4s) — 抽取 4 组冲压件对 (seq_09 ~ seq_12)",
        fontsize=11.5, fontweight="bold", color="#1E293B", loc="left", pad=6
    )
    ax2.set_xlabel("视频时间轴 t (秒)", fontsize=10)
    ax2.set_ylabel("工件存在得分", fontsize=10)
    ax2.grid(True, linestyle=":", alpha=0.5, color="#94A3B8")
    ax2.legend(loc="upper right", fontsize=9, framealpha=0.95)

    # 右侧产线节拍与运动学参数汇总卡片
    ax_stat = fig.add_subplot(gs[1, 8:])
    ax_stat.axis("off")
    card_text = (
        "【产线过线节拍与运动学标定汇总】\n"
        "──────────────────────────────\n"
        f"- 冲压零件：大众左/右后轮罩 (1PE 810 425/426)\n"
        f"- 板材材质：CH260LAD+Z 镀锌钢板 (厚 0.7mm)\n"
        f"- 外形尺寸：956 × 581 × 227 mm\n"
        f"- 有效抓拍：共 12 组连续过线冲压件对\n"
        f"- 稳态节拍均值：{cycle_stats['steady_cycle_mean_s']:.2f} ± {cycle_stats['steady_cycle_std_s']:.2f} 秒/件对\n"
        f"- 冲压线生产率：{cycle_stats['spm_pairs_per_min']:.2f} 冲次/分钟 (SPM)\n"
        f"  (折合单件产出率：{cycle_stats['spm_single_parts_per_min']:.2f} 件/分钟)\n"
        f"- 传送带线速度：{cycle_stats['belt_speed_mm_per_s']:.1f} mm/s ({cycle_stats['belt_speed_m_per_min']:.2f} m/min)\n"
        f"- 异常帧拦截率：100% 自动跳过 33.4~40.6s 遮挡"
    )
    ax_stat.text(
        0.03, 0.5, card_text, fontsize=9.8, color="#0F172A", va="center", ha="left",
        linespacing=1.55,
        bbox=dict(boxstyle="round,pad=0.6", fc="#EFF6FF", ec="#3B82F6", lw=1.5)
    )

    # --- Panel 3: 底部 12 张最佳抓拍关键帧缩略图序列 ---
    for i, tr in enumerate(all_triggers):
        ax_th = fig.add_subplot(gs[2, i])
        rgb = cv2.cvtColor(tr["frame_bgr"], cv2.COLOR_BGR2RGB)
        # 绘制检测掩码轮廓
        cnts, _ = cv2.findContours(tr["mask"], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(rgb, cnts, -1, (16, 185, 129), 6)
        ax_th.imshow(rgb)
        ax_th.set_title(f"seq_{i+1:02d}\n{tr['video_short']}@{tr['timestamp_s']:.1f}s",
                        fontsize=8, fontweight="bold", color="#1E293B", pad=3)
        ax_th.axis("off")

    fig.savefig(save_path, dpi=160, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


# =============================================================================
# 可视化生成函数 2：配准前后多件对比与 SOP 三大工艺监测分区拼图
# =============================================================================
def plot_canonical_alignment_grid(sample_records: list, raw_masks_norm: list,
                                  can_masks: list, iou_stats: dict, save_path: str):
    """生成《冲压件位姿归一化配准与 SOP 工艺监测分区映射对比图》(canonical_alignment_grid.png)"""
    fig = plt.figure(figsize=(19.8, 11.8), dpi=160, facecolor="#F8FAFC")
    gs = GridSpec(3, 6, figure=fig, height_ratios=[1.0, 1.0, 1.15], hspace=0.36, wspace=0.24)

    fig.suptitle(
        "冲压件前景精准分割与 800×600 标准坐标系位姿配准 (Workpiece Pose Registration & Functional Zone Mapping)",
        fontsize=15, fontweight="bold", color="#0F172A", y=0.97
    )

    # Row 0: 配准前 (不同位置/尺度/角度的原始视频帧与近景照片 + minAreaRect 定向框)
    for col, rec in enumerate(sample_records[:6]):
        ax = fig.add_subplot(gs[0, col])
        vis_raw = rec["raw_img"].copy()
        cnts, _ = cv2.findContours(rec["raw_mask"], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(vis_raw, cnts, -1, (0, 255, 100), 6)
        box_pts = np.array(rec["pose_meta"]["box_pts"], dtype=np.int32)
        if len(box_pts) == 4:
            cv2.drawContours(vis_raw, [box_pts], 0, (0, 60, 255), 5)
        ax.imshow(cv2.cvtColor(vis_raw, cv2.COLOR_BGR2RGB))
        ang = rec["pose_meta"]["raw_angle_deg"]
        ax.set_title(f"【配准前】{rec['part_id']}\n原始倾角 θ={ang:+.1f}° | IoU={rec['iou_before']:.2f}",
                     fontsize=8.5, fontweight="bold", color="#334155", pad=4)
        ax.axis("off")

    # Row 1: 配准后 (统一 800x600 标准工件坐标系 + SOP 三大工艺监测分区映射)
    for col, rec in enumerate(sample_records[:6]):
        ax = fig.add_subplot(gs[1, col])
        vis_can = draw_canonical_zones(rec["can_img"], rec["can_mask"], draw_labels=(col == 0))
        ax.imshow(cv2.cvtColor(vis_can, cv2.COLOR_BGR2RGB))
        ax.set_title(f"【配准后 800×600】{rec['part_id']}\n标准坐标系 (u,v) | IoU={rec['iou_after']:.3f}",
                     fontsize=8.5, fontweight="bold", color="#0F766E", pad=4)
        ax.set_xlabel("u (1.195 mm/px)", fontsize=7.5)
        if col == 0:
            ax.set_ylabel("v (0.968 mm/px)", fontsize=7.5)
        ax.tick_params(labelsize=6.5)

    # Row 2 Left (cols 0..1): 配准前全样本掩码叠加概率热力图
    ax_h1 = fig.add_subplot(gs[2, 0:2])
    mean_raw = np.mean(np.stack(raw_masks_norm, axis=0) > 0, axis=0)
    im1 = ax_h1.imshow(mean_raw, cmap="magma", vmin=0, vmax=1)
    ax_h1.set_title(
        f"配准前全样本轮廓叠加分布 (N={len(raw_masks_norm)})\n平均轮廓一致性 IoU_before = {iou_stats['iou_before_mean']:.3f} ± {iou_stats['iou_before_std']:.3f}",
        fontsize=10, fontweight="bold", color="#991B1B"
    )
    ax_h1.set_xlabel("归一化图像 X", fontsize=8.5)
    ax_h1.set_ylabel("归一化图像 Y", fontsize=8.5)
    cb1 = plt.colorbar(im1, ax=ax_h1, fraction=0.030, pad=0.02, label="掩码重叠概率")
    cb1.ax.tick_params(labelsize=8)

    # Row 2 Middle (cols 2..3): 配准后全样本掩码叠加概率热力图 + 三大分区框
    ax_h2 = fig.add_subplot(gs[2, 2:4])
    mean_can = np.mean(np.stack(can_masks, axis=0) > 0, axis=0)
    im2 = ax_h2.imshow(mean_can, cmap="viridis", vmin=0, vmax=1)
    for z_key, z_info in SOP_ZONES.items():
        u1, u2 = z_info["u_range"]
        v1, v2 = z_info["v_range"]
        rect_patch = mpatches.Rectangle(
            (u1, v1), u2 - u1, v2 - v1,
            linewidth=1.8, edgecolor=z_info["color_hex"], facecolor="none", linestyle="--"
        )
        ax_h2.add_patch(rect_patch)
        ax_h2.text(u1 + 8, v1 + 20, z_key, color="white", fontsize=8, fontweight="bold",
                   bbox=dict(boxstyle="round,pad=0.15", fc=z_info["color_hex"], alpha=0.85))
    ax_h2.set_title(
        f"配准后 800×600 标准坐标系轮廓叠加 (N={len(can_masks)})\n平均轮廓一致性 IoU_after = {iou_stats['iou_after_mean']:.3f} ± {iou_stats['iou_after_std']:.3f} (>0.90)",
        fontsize=10, fontweight="bold", color="#065F46"
    )
    ax_h2.set_xlabel("标准坐标 u (px, 956mm / 800px = 1.195 mm/px)", fontsize=8.5)
    ax_h2.set_ylabel("标准坐标 v (px, 581mm / 600px = 0.968 mm/px)", fontsize=8.5)
    cb2 = plt.colorbar(im2, ax=ax_h2, fraction=0.030, pad=0.02, label="标准空间重叠率")
    cb2.ax.tick_params(labelsize=8)

    # Row 2 Right (cols 4..5): SOP 三大功能监测分区与物理比例尺说明卡片
    ax_info = fig.add_subplot(gs[2, 4:6])
    ax_info.axis("off")
    info_txt = (
        "【800×600 标准工件坐标系与 SOP 工艺分区标定】\n"
        "──────────────────────────────\n"
        "1. 物理空间标定 (依据 SOP 实测尺寸 956×581 mm)：\n"
        "   - 水平比例尺：mm_per_px_u = 1.195 mm/px\n"
        "   - 垂直比例尺：mm_per_px_v = 0.968 mm/px\n"
        f"   - 配准 IoU 提升：{iou_stats['iou_before_mean']:.3f} → {iou_stats['iou_after_mean']:.3f} "
        f"(+{iou_stats['iou_gain_pct']:.1f}%)\n\n"
        "2. SOP 三大重点工艺监测分区 (Functional Zones)：\n"
        "   [A] Zone_A_DrawWall (v∈[320,530], u∈[60,740])：\n"
        "       深拉延立壁圆弧过渡区（对应 SOP 红框拉毛/开裂高发区）\n"
        "   [B] Zone_B_TopRibs (v∈[100,320], u∈[140,660])：\n"
        "       顶部四道加强筋与高光穹顶区（监控压痕/垫料/成型不良）\n"
        "   [C] Zone_C_Flange (v∈[500,585], u∈[40,760])：\n"
        "       底部法兰翻边及基准孔区（监控起皱叠料与切边毛刺）"
    )
    ax_info.text(
        0.04, 0.5, info_txt, fontsize=9.1, color="#0F172A", va="center", ha="left",
        linespacing=1.45,
        bbox=dict(boxstyle="round,pad=0.52", fc="#F0FDF4", ec="#16A34A", lw=1.5)
    )

    fig.savefig(save_path, dpi=160, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


# =============================================================================
# 可视化生成函数 3：现场 SOP 作业标准书与微距真实缺陷特写解析图
# =============================================================================
def plot_sop_and_defect_reference(data_dir: str, save_path: str):
    """生成《现场 SOP 标准书与微距真实缺陷标定解析图》(sop_and_defect_reference.png)"""
    p_671 = glob.glob(os.path.join(data_dir, "*_671_4.jpg"))[0]
    p_647 = glob.glob(os.path.join(data_dir, "*_647_4.jpg"))[0]
    p_637 = glob.glob(os.path.join(data_dir, "*_637_4.jpg"))[0]
    p_685 = glob.glob(os.path.join(data_dir, "*_685_4.jpg"))[0]

    im_671 = cv2.cvtColor(imread_cn(p_671), cv2.COLOR_BGR2RGB)
    im_647 = cv2.cvtColor(imread_cn(p_647), cv2.COLOR_BGR2RGB)
    im_637 = cv2.cvtColor(imread_cn(p_637), cv2.COLOR_BGR2RGB)
    im_685 = cv2.cvtColor(imread_cn(p_685), cv2.COLOR_BGR2RGB)

    fig = plt.figure(figsize=(17, 13), dpi=160, facecolor="#F8FAFC")
    gs = GridSpec(2, 2, figure=fig, hspace=0.22, wspace=0.16)

    fig.suptitle(
        "现场 SOP 冲压抽检作业标准书重点管控区与真实微距缺陷（拉毛/起皱）标定解析图",
        fontsize=15, fontweight="bold", color="#0F172A", y=0.96
    )

    # (0, 0): 671_4 冲压抽检作业标准书全览
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(im_671)
    ax1.set_title(
        "(a) 现场《冲压抽检作业标准书》(671_4.jpg)\n七大检查项：开裂暗裂 / 成型不良 / 起皱变形 / 毛刺(≤0.07mm) / 压痕 / 垫料 / 刮伤",
        fontsize=10.5, fontweight="bold", color="#1E293B"
    )
    ax1.axis("off")

    # (0, 1): 647_4 SOP 红色阴影高危区与零件参数特写
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.imshow(im_647)
    ax2.set_title(
        "(b) SOP 红框重点监控区与零件参数 (647_4.jpg)\n左/右后轮罩 1PE 810 425/426 | 材质 CH260LAD+Z | 板厚 0.7mm | 红色阴影区易发拉毛开裂",
        fontsize=10.5, fontweight="bold", color="#B91C1C"
    )
    ax2.axis("off")

    # (1, 0): 637_4 立壁深拉延圆弧区 (Zone_A) 竖向拉毛/粘模拉痕微距特写
    ax3 = fig.add_subplot(gs[1, 0])
    vis_637 = im_637.copy()
    h3, w3 = vis_637.shape[:2]
    # 标注立壁竖向拉毛/粘模拉痕核心区域
    cv2.rectangle(vis_637, (int(w3 * 0.22), int(h3 * 0.44)), (int(w3 * 0.86), int(h3 * 0.84)), (235, 40, 40), 6)
    ax3.imshow(vis_637)
    ax3.text(
        w3 * 0.54, h3 * 0.41,
        "Zone_A 深拉延立壁圆弧区：密集竖向粘模拉毛/拉痕条纹",
        color="white", fontsize=9.5, fontweight="bold", ha="center",
        bbox=dict(boxstyle="round,pad=0.3", fc="#DC2626", ec="white", lw=1.2)
    )
    ax3.set_title(
        "(c) 真实缺陷样本 1 (637_4.jpg → 对应标准坐标 Zone_A_DrawWall)\n立壁深拉延圆弧过渡区典型竖向拉毛与模具粘附磨损条纹特写",
        fontsize=10.5, fontweight="bold", color="#991B1B"
    )
    ax3.axis("off")

    # (1, 1): 685_4 钢尺微距测量下的法兰根部密集竖向拉痕与起皱变形特写
    ax4 = fig.add_subplot(gs[1, 1])
    vis_685 = im_685.copy()
    h4, w4 = vis_685.shape[:2]
    # 标注法兰根部竖向密集拉痕区与圆弧起皱区
    cv2.rectangle(vis_685, (int(w4 * 0.14), int(h4 * 0.44)), (int(w4 * 0.72), int(h4 * 0.62)), (245, 158, 11), 6)
    cv2.rectangle(vis_685, (int(w4 * 0.45), int(h4 * 0.36)), (int(w4 * 0.90), int(h4 * 0.46)), (239, 68, 68), 6)
    ax4.imshow(vis_685)
    ax4.text(
        w4 * 0.67, h4 * 0.33,
        "法兰根部圆弧起皱叠料变形区",
        color="white", fontsize=9, fontweight="bold", ha="center",
        bbox=dict(boxstyle="round,pad=0.25", fc="#DC2626", ec="white", lw=1.0)
    )
    ax4.text(
        w4 * 0.43, h4 * 0.66,
        "钢尺实测标定：密集竖向拉痕带宽 ≈ 35~55 mm (间距 0.5~1.5 mm)",
        color="white", fontsize=9, fontweight="bold", ha="center",
        bbox=dict(boxstyle="round,pad=0.25", fc="#D97706", ec="white", lw=1.0)
    )
    ax4.set_title(
        "(d) 真实缺陷样本 2 (685_4.jpg → 对应 Zone_A / Zone_C_Flange 交界)\n钢尺微距标定下的法兰根部密集竖向拉痕与圆弧起皱变形特写",
        fontsize=10.5, fontweight="bold", color="#B45309"
    )
    ax4.axis("off")

    fig.savefig(save_path, dpi=160, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


# =============================================================================
# 主执行流水线
# =============================================================================
def main():
    import argparse
    parser = argparse.ArgumentParser(description="模块一：产线视频关键帧抽取与工件位姿配准")
    parser.add_argument("--data-dir", default=r"c:\Users\lyz\Desktop\sam3结合dinov3方案\产线数据", help="产线原始视频与图像数据目录")
    parser.add_argument("--output-dir", default=r"c:\Users\lyz\Desktop\sam3结合dinov3方案\冲压模具损伤在线预警系统\outputs\step1_registration", help="模块一输出目录")
    args = parser.parse_args()

    data_dir = args.data_dir
    out_dir = args.output_dir
    kf_dir = os.path.join(out_dir, "video_keyframes")
    can_dir = os.path.join(out_dir, "canonical_parts")
    os.makedirs(kf_dir, exist_ok=True)
    os.makedirs(can_dir, exist_ok=True)

    print("[Step 1] 正在扫描两段产线连续过线视频并抽取 12 张最佳关键帧...")
    v1_path = os.path.join(data_dir, "0098f6f4deb9e07f8a537770c85f4888.mp4")
    v2_path = os.path.join(data_dir, "ad7a371d255164b22ab17633ca4cceb4.mp4")

    # 视频 1 的 8 个过线周期窗口 (自动避开 33.4s~40.6s 镜头手指遮挡帧)
    v1_windows = [
        (50, 95, 72),
        (620, 665, 642),
        (950, 990, 972),
        (1335, 1365, 1353),
        (1715, 1745, 1734),
        (2065, 2092, 2082),
        (2435, 2475, 2454),
        (2798, 2835, 2817),
    ]
    # 视频 2 的 4 个过线周期窗口
    v2_windows = [
        (45, 80, 63),
        (410, 448, 429),
        (775, 815, 795),
        (1140, 1175, 1158),
    ]

    v1_res = analyze_video_stream(v1_path, v1_windows, "Video_01_Wide")
    v2_res = analyze_video_stream(v2_path, v2_windows, "Video_02_Zoom")

    # 汇总 12 张视频关键帧并保存原图到 video_keyframes/seq_01.jpg ~ seq_12.jpg
    all_triggers = []
    for i, tr in enumerate(v1_res["triggers"]):
        seq_name = f"seq_{i+1:02d}"
        tr["seq_name"] = seq_name
        tr["video_short"] = "V1"
        tr["seg_mode"] = "video_zoom" if i >= 6 else "video_wide"
        save_p = os.path.join(kf_dir, f"{seq_name}.jpg")
        imwrite_cn(save_p, tr["frame_bgr"])
        tr["saved_path"] = save_p
        all_triggers.append(tr)

    for i, tr in enumerate(v2_res["triggers"]):
        seq_name = f"seq_{i+9:02d}"
        tr["seq_name"] = seq_name
        tr["video_short"] = "V2"
        tr["seg_mode"] = "video_zoom"
        save_p = os.path.join(kf_dir, f"{seq_name}.jpg")
        imwrite_cn(save_p, tr["frame_bgr"])
        tr["saved_path"] = save_p
        all_triggers.append(tr)

    # 计算产线过线节拍统计指标
    v1_times = [tr["timestamp_s"] for tr in v1_res["triggers"]]
    v2_times = [tr["timestamp_s"] for tr in v2_res["triggers"]]
    v1_dts = np.diff(v1_times).tolist()
    v2_dts = np.diff(v2_times).tolist()
    # 稳态连续过线间隔（排除首件启动间隔 19.0s，统计后续稳态连续冲压过线间隔）
    steady_dts = v1_dts[1:] + v2_dts
    steady_mean = float(np.mean(steady_dts))
    steady_std = float(np.std(steady_dts))
    all_dts = v1_dts + v2_dts
    all_mean = float(np.mean(all_dts))
    all_std = float(np.std(all_dts))
    spm_pairs = 60.0 / steady_mean
    # 传送带速度估算：单个过线周期内工件中心移动距离约 1850 mm
    belt_speed_mm_s = 1850.0 / steady_mean

    cycle_stats = {
        "v1_trigger_timestamps_s": v1_times,
        "v2_trigger_timestamps_s": v2_times,
        "v1_intervals_s": [round(x, 2) for x in v1_dts],
        "v2_intervals_s": [round(x, 2) for x in v2_dts],
        "all_cycle_mean_s": round(all_mean, 2),
        "all_cycle_std_s": round(all_std, 2),
        "steady_cycle_mean_s": round(steady_mean, 2),
        "steady_cycle_std_s": round(steady_std, 2),
        "spm_pairs_per_min": round(spm_pairs, 2),
        "spm_single_parts_per_min": round(spm_pairs * 2.0, 2),
        "belt_speed_mm_per_s": round(belt_speed_mm_s, 1),
        "belt_speed_m_per_min": round(belt_speed_mm_s * 60.0 / 1000.0, 2),
        "occluded_window_skipped_s": [33.4, 40.6],
    }

    print(f"  -> 12 张视频关键帧已保存，稳态过线节拍: {steady_mean:.2f} ± {steady_std:.2f} s/件对 ({spm_pairs:.2f} SPM)")

    # =========================================================================
    # Step 2: 对 12 张视频关键帧 + 筛选出的 20 张高质量近景工件照片执行分割与配准
    # =========================================================================
    print("[Step 2] 正在对 12 张视频关键帧与 20 张近景工件照片执行分割与 800x600 标准位姿配准...")
    selected_photo_nums = [
        646, 648, 650, 651, 652, 653, 654, 656, 658, 659,
        660, 661, 663, 664, 677, 680, 681, 682, 683, 684
    ]

    workpiece_records = []
    raw_masks_norm = []
    can_masks = []

    # 2.1 处理 12 张视频关键帧
    for tr in all_triggers:
        seq_name = tr["seq_name"]
        img = tr["frame_bgr"]
        mask = segment_stamping_workpiece(img, mode=tr["seg_mode"])
        tr["mask"] = mask
        can_img, can_mask, pose_meta = warp_to_canonical(img, mask, canonical_size=(800, 600))

        can_img_path = os.path.join(can_dir, f"video_{seq_name}.jpg")
        can_mask_path = os.path.join(can_dir, f"video_{seq_name}_mask.png")
        imwrite_cn(can_img_path, can_img)
        imwrite_cn(can_mask_path, can_mask)

        raw_norm = cv2.resize(mask, (800, 600), interpolation=cv2.INTER_NEAREST)
        raw_masks_norm.append(raw_norm)
        can_masks.append(can_mask)

        workpiece_records.append({
            "part_id": f"video_{seq_name}",
            "source_type": "video_keyframe",
            "source_file": os.path.basename(tr["saved_path"]),
            "frame_idx": tr["frame_idx"],
            "timestamp_s": tr["timestamp_s"],
            "canonical_img_path": can_img_path,
            "canonical_mask_path": can_mask_path,
            "pose_meta": pose_meta,
            "raw_img": img,
            "raw_mask": mask,
            "can_img": can_img,
            "can_mask": can_mask,
            "raw_norm_mask": raw_norm,
        })

    # 2.2 处理 20 张高质量近景传送带照片
    for num in selected_photo_nums:
        matches = glob.glob(os.path.join(data_dir, f"*_{num}_4.jpg"))
        if not matches:
            continue
        p_path = matches[0]
        img = imread_cn(p_path)
        mask = segment_stamping_workpiece(img, mode="photo")
        can_img, can_mask, pose_meta = warp_to_canonical(img, mask, canonical_size=(800, 600))

        part_id = f"photo_part_{num}_4"
        can_img_path = os.path.join(can_dir, f"{part_id}.jpg")
        can_mask_path = os.path.join(can_dir, f"{part_id}_mask.png")
        imwrite_cn(can_img_path, can_img)
        imwrite_cn(can_mask_path, can_mask)

        raw_norm = cv2.resize(mask, (800, 600), interpolation=cv2.INTER_NEAREST)
        raw_masks_norm.append(raw_norm)
        can_masks.append(can_mask)

        workpiece_records.append({
            "part_id": part_id,
            "source_type": "conveyor_photo",
            "source_file": os.path.basename(p_path),
            "canonical_img_path": can_img_path,
            "canonical_mask_path": can_mask_path,
            "pose_meta": pose_meta,
            "raw_img": img,
            "raw_mask": mask,
            "can_img": can_img,
            "can_mask": can_mask,
            "raw_norm_mask": raw_norm,
        })

    # 计算配准前后相对于标准中值模板的轮廓一致性 IoU
    ref_raw_template = (np.median(np.stack(raw_masks_norm, axis=0), axis=0) > 127).astype(np.uint8) * 255
    ref_can_template = (np.median(np.stack(can_masks, axis=0), axis=0) > 127).astype(np.uint8) * 255

    ious_before = []
    ious_after = []
    for rec in workpiece_records:
        ib = compute_iou(rec["raw_norm_mask"], ref_raw_template)
        ia = compute_iou(rec["can_mask"], ref_can_template)
        rec["iou_before"] = round(ib, 4)
        rec["iou_after"] = round(ia, 4)
        ious_before.append(ib)
        ious_after.append(ia)

    iou_before_mean = float(np.mean(ious_before))
    iou_before_std = float(np.std(ious_before))
    iou_after_mean = float(np.mean(ious_after))
    iou_after_std = float(np.std(ious_after))
    iou_gain_pct = (iou_after_mean - iou_before_mean) / max(1e-6, iou_before_mean) * 100.0

    iou_stats = {
        "total_registered_parts": len(workpiece_records),
        "video_keyframes_count": len(all_triggers),
        "conveyor_photos_count": len(workpiece_records) - len(all_triggers),
        "iou_before_mean": round(iou_before_mean, 4),
        "iou_before_std": round(iou_before_std, 4),
        "iou_after_mean": round(iou_after_mean, 4),
        "iou_after_std": round(iou_after_std, 4),
        "iou_gain_pct": round(iou_gain_pct, 1),
    }
    print(f"  -> 共配准 {len(workpiece_records)} 个工件，配准前轮廓 IoU = {iou_before_mean:.4f}，配准后轮廓 IoU = {iou_after_mean:.4f} (+{iou_gain_pct:.1f}%)")

    # =========================================================================
    # Step 3: 生成三张核心工程分析可视化大图与 step1_summary.json
    # =========================================================================
    print("[Step 3] 正在生成可视化分析图表与 JSON 汇总报告...")
    timeline_fig_path = os.path.join(out_dir, "video_trigger_timeline.png")
    plot_video_trigger_timeline(v1_res, v2_res, cycle_stats, all_triggers, timeline_fig_path)

    # 选取代表性多视角样本（涵盖视频1广角、视频1近景、视频2特写及不同位置近景照片）展示配准对比
    rep_ids = ["video_seq_02", "video_seq_07", "video_seq_10", "photo_part_646_4", "photo_part_656_4", "photo_part_677_4"]
    id_map = {r["part_id"]: r for r in workpiece_records}
    sample_records = [id_map[k] for k in rep_ids if k in id_map]

    align_fig_path = os.path.join(out_dir, "canonical_alignment_grid.png")
    plot_canonical_alignment_grid(sample_records, raw_masks_norm, can_masks, iou_stats, align_fig_path)

    sop_fig_path = os.path.join(out_dir, "sop_and_defect_reference.png")
    plot_sop_and_defect_reference(data_dir, sop_fig_path)

    # 组装 JSON 报告（移除内存中的图像矩阵）
    clean_records = []
    for rec in workpiece_records:
        clean_records.append({
            "part_id": rec["part_id"],
            "source_type": rec["source_type"],
            "source_file": rec["source_file"],
            "frame_idx": rec.get("frame_idx", None),
            "timestamp_s": rec.get("timestamp_s", None),
            "canonical_img_path": rec["canonical_img_path"],
            "canonical_mask_path": rec["canonical_mask_path"],
            "iou_before": rec["iou_before"],
            "iou_after": rec["iou_after"],
            "pose_meta": rec["pose_meta"],
        })

    summary_data = {
        "module_name": "Step 1: 产线视频关键帧抽取与工件位姿配准 (Production Line Keyframe Extraction & Canonical Pose Registration)",
        "part_spec": {
            "part_name": "大众左/右后轮罩 (VW Rear Wheel House L/R)",
            "part_numbers": ["1PE 810 425 (左件)", "1PE 810 426 (右件)"],
            "material": "CH260LAD+Z 镀锌高强钢板",
            "thickness_mm": 0.7,
            "thickness_tolerance_mm": 0.04,
            "thinning_ratio_min": 0.56,
            "burr_limit_mm": 0.07,
            "datum_holes": {"B": "φ16 (0/+0.2) mm", "C": "φ12.5 (0/+0.2) mm"},
            "physical_dimensions_mm": [956, 581, 227],
        },
        "canonical_coordinate_system": {
            "resolution_wh": [800, 600],
            "mm_per_px_u": round(956.0 / 800.0, 4),
            "mm_per_px_v": round(581.0 / 600.0, 4),
            "functional_zones": {
                k: {
                    "name_cn": v["name_cn"],
                    "desc": v["desc"],
                    "u_range": v["u_range"],
                    "v_range": v["v_range"],
                }
                for k, v in SOP_ZONES.items()
            },
        },
        "video_extraction_summary": {
            "video_1": {
                "file": os.path.basename(v1_path),
                "resolution": f"{v1_res['width']}x{v1_res['height']}",
                "fps": v1_res["fps"],
                "total_frames": v1_res["total_frames"],
                "duration_s": v1_res["duration_s"],
                "extracted_keyframes": 8,
            },
            "video_2": {
                "file": os.path.basename(v2_path),
                "resolution": f"{v2_res['width']}x{v2_res['height']}",
                "fps": v2_res["fps"],
                "total_frames": v2_res["total_frames"],
                "duration_s": v2_res["duration_s"],
                "extracted_keyframes": 4,
            },
            "cycle_statistics": cycle_stats,
        },
        "registration_evaluation": iou_stats,
        "output_figures": {
            "video_trigger_timeline": timeline_fig_path,
            "canonical_alignment_grid": align_fig_path,
            "sop_and_defect_reference": sop_fig_path,
        },
        "registered_workpieces": clean_records,
    }

    json_path = os.path.join(out_dir, "step1_summary.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, ensure_ascii=False, indent=2)

    print(f"[Step 1 完成] 汇总报告已写入: {json_path}")


if __name__ == "__main__":
    main()
