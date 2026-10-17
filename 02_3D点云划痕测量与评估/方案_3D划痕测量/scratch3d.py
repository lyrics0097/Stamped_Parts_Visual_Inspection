"""scratch3d: 基于 3D 点云 / 深度图的表面缺陷 (划痕、凹坑) 无监督分割与宽度、深度测量。

设计目标: 不依赖任何缺陷训练数据; 与传感器无关 (线激光轮廓仪 / 结构光 / RealSense 都先转成点云再进来)。

核心思路 (全部是几何, 每一步都可解释):
    1. 参考坐标系: 对点云做 PCA, 主平面作为 (u, v) 平面, 法向为高度 h。
    2. 栅格化: 按点间距把 (u, v, h) 栅格化成高度图 H (mm), 每个格子取中位数高度; 空格子用最近邻填补。
    3. 去除工件本体形状: 用 Huber 加权的二次多项式曲面拟合 H (稳健, 缺陷点权重被压低), 得到 H_poly;
       H1 = H - H_poly 只剩 "局部起伏 + 缺陷 + 噪声"。
    4. 估计无损表面: 对 H1 做灰度形态学闭运算 (结构元直径 > 缺陷宽度), 凹坑/划痕被"填平", 得到 H_ref。
       残差 R = H_ref - H1 >= 0 就是每个格子的 "凹陷深度"。(凸起缺陷用开运算, 对称处理)
    5. 分割: 噪声水平 sigma 用 R 的 MAD 稳健估计; R > max(k*sigma, 绝对下限) 即缺陷; 形态学去噪 + 连通域 + 面积过滤。
    6. 测量: 每个缺陷 深度 = R 的 99 分位 (抗单点飞点), 宽度 = 骨架上距离变换 * 2 的中位/最大值, 长度 = 骨架长度,
       面积, 3D 质心, 主方向。
    7. 判定: 按 rules.json 中的 宽度 / 深度 / 长度 阈值给出 正常 / 异常。

输入: N x 3 (xyz, mm) 点云; 可选逐点标签用于评估。
"""
import json
import os

import cv2
import numpy as np
from scipy.spatial import cKDTree
from skimage.morphology import skeletonize


# ================================================================================================================
# 1-2. 参考坐标系 与 栅格化
# ================================================================================================================
def pca_frame(xyz):
    """返回 (质心 c, 3x3 旋转 R): 行 0,1 为平面内基 (u, v), 行 2 为法向 (h)。"""
    c = xyz.mean(axis=0)
    _, _, vt = np.linalg.svd(xyz - c, full_matrices=False)
    R = vt.copy()
    if np.linalg.det(R) < 0:
        R[2] *= -1
    return c, R


def estimate_spacing(xyz, n_sample=20000, seed=0):
    """点云中位最近邻距离 (mm), 用于确定栅格间距。"""
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(xyz), min(n_sample, len(xyz)), replace=False)
    tree = cKDTree(xyz)
    d, _ = tree.query(xyz[idx], k=2)
    return float(np.median(d[:, 1]))


def rasterize(uvh, spacing):
    """(u, v, h) -> 高度图。返回 dict: H (nan = 空), cell_index (每个点所属格子的展平索引), 原点, 尺寸。"""
    u, v, h = uvh[:, 0], uvh[:, 1], uvh[:, 2]
    u0, v0 = u.min(), v.min()
    W = int(np.ceil((u.max() - u0) / spacing)) + 1
    Hh = int(np.ceil((v.max() - v0) / spacing)) + 1
    iu = np.clip(((u - u0) / spacing).astype(np.int64), 0, W - 1)
    iv = np.clip(((v - v0) / spacing).astype(np.int64), 0, Hh - 1)
    flat = iv * W + iu
    # 每格中位数高度: 按格排序后分组
    order = np.argsort(flat, kind='stable')
    fs, hs = flat[order], h[order]
    starts = np.r_[0, np.flatnonzero(np.diff(fs)) + 1]
    ends = np.r_[starts[1:], len(fs)]
    H = np.full(W * Hh, np.nan)
    cnt = np.zeros(W * Hh, np.int32)
    for s, e in zip(starts, ends):
        H[fs[s]] = np.median(hs[s:e])
        cnt[fs[s]] = e - s
    H = H.reshape(Hh, W)
    return {'H': H, 'count': cnt.reshape(Hh, W), 'cell_index': flat, 'u0': u0, 'v0': v0, 'spacing': spacing, 'shape': (Hh, W)}


def fill_nan_nearest(H):
    """用最近有效格子的值填补空格子 (仅供形态学计算, 输出时空格仍标记为无效)。"""
    valid = ~np.isnan(H)
    if valid.all():
        return H.copy()
    yy, xx = np.nonzero(valid)
    tree = cKDTree(np.c_[yy, xx])
    ny, nx = np.nonzero(~valid)
    _, i = tree.query(np.c_[ny, nx], k=1)
    out = H.copy()
    out[ny, nx] = H[yy[i], xx[i]]
    return out


# ================================================================================================================
# 3. 稳健二次曲面拟合 (去除工件宏观形状)
# ================================================================================================================
def robust_poly2_fit(H, valid, iters=8, huber_k=1.5):
    """z = a + b u + c v + d u^2 + e uv + f v^2, IRLS-Huber 稳健拟合 (缺陷格子权重自动降低)。返回拟合曲面 (同 H 形状)。"""
    Hh, W = H.shape
    vv, uu = np.mgrid[0:Hh, 0:W]
    uu = (uu - W / 2) / max(W, 1)
    vv = (vv - Hh / 2) / max(Hh, 1)
    A_full = np.stack([np.ones_like(uu), uu, vv, uu ** 2, uu * vv, vv ** 2], axis=-1).reshape(-1, 6)
    m = valid.ravel()
    A, z = A_full[m], H.ravel()[m]
    w = np.ones(len(z))
    coef = np.zeros(6)
    for _ in range(iters):
        sw = np.sqrt(w)
        coef, *_ = np.linalg.lstsq(A * sw[:, None], z * sw, rcond=None)
        r = z - A @ coef
        s = 1.4826 * np.median(np.abs(r - np.median(r))) + 1e-9
        a = np.abs(r) / (huber_k * s)
        w = np.where(a <= 1, 1.0, 1.0 / a)
    return (A_full @ coef).reshape(Hh, W)


# ================================================================================================================
# 4-5. 无损表面估计 + 分割
# ================================================================================================================
def disk(diam_px):
    d = max(3, int(round(diam_px)) | 1)
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (d, d))


def segment_defects(H, spacing, se_mm=6.0, k_sigma=4.0, abs_min_mm=0.02, min_area_mm2=0.5, mode='pit'):
    """返回 dict: residual (mm, 凹陷为正), mask (uint8), labels, sigma, H_ref, valid。
    mode='pit' 检测凹陷 (划痕/凹坑), 'bump' 检测凸起, 'both' 两者并集。"""
    valid = ~np.isnan(H)
    Hf = fill_nan_nearest(H)
    H_poly = robust_poly2_fit(Hf, valid)
    H1 = Hf - H_poly
    se = disk(se_mm / spacing)
    # 3x3 中值滤波先去掉单点噪声尖峰: 否则闭运算 (局部最大) 会把噪声正尖峰当作无损表面, 深度系统性偏大 ~3 sigma
    H1f = cv2.medianBlur(H1.astype(np.float32), 3)
    res_pit = cv2.morphologyEx(H1f, cv2.MORPH_CLOSE, se) - H1f       # >=0, 凹陷深度
    res_bump = H1f - cv2.morphologyEx(H1f, cv2.MORPH_OPEN, se)       # >=0, 凸起高度
    if mode == 'pit':
        residual = res_pit
    elif mode == 'bump':
        residual = res_bump
    else:
        residual = np.maximum(res_pit, res_bump)
    residual = residual.astype(np.float64)
    residual[~valid] = 0
    # 噪声: 残差的 MAD (缺陷占少数, 中位数附近即噪声)
    r = residual[valid]
    sigma = 1.4826 * np.median(np.abs(r - np.median(r))) + 1e-9
    thr = max(k_sigma * sigma, abs_min_mm)
    mask = ((residual > thr) & valid).astype(np.uint8)
    k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k3)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k3)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    min_px = min_area_mm2 / spacing ** 2
    keep = np.zeros(n, bool)
    for i in range(1, n):
        keep[i] = stats[i, cv2.CC_STAT_AREA] >= min_px
    labels = np.where(keep[labels], labels, 0)
    mask = (labels > 0).astype(np.uint8) * 255
    return {'residual': residual, 'mask': mask, 'labels': labels, 'sigma': float(sigma), 'thr': float(thr),
            'H_ref': H_poly + cv2.morphologyEx(H1f, cv2.MORPH_CLOSE, se), 'H1': H1f, 'valid': valid}


# ================================================================================================================
# 6. 逐缺陷测量
# ================================================================================================================
def measure_components(seg, spacing, frame=None, raster=None):
    """对每个连通域测 深度 / 宽度 / 长度 / 面积 / 方向 / 3D 质心。返回列表 (按深度降序)。"""
    labels, residual = seg['labels'], seg['residual']
    out = []
    for lab in np.unique(labels):
        if lab == 0:
            continue
        comp = labels == lab
        r = residual[comp]
        ys, xs = np.nonzero(comp)
        # 深度: 99 分位 (抗飞点); 同时给最大值与平均
        depth_p99, depth_max, depth_mean = float(np.percentile(r, 99)), float(r.max()), float(r.mean())
        # 宽度: 骨架上的距离变换 * 2
        comp_u8 = comp.astype(np.uint8)
        dist = cv2.distanceTransform(comp_u8, cv2.DIST_L2, 5)
        skel = skeletonize(comp_u8 > 0)
        if skel.sum() >= 3:
            wd = dist[skel] * 2 * spacing
            width_med, width_max = float(np.median(wd)), float(wd.max())
            length = float(skel.sum() * spacing)
        else:
            width_med = width_max = float(dist.max() * 2 * spacing)
            length = float(max(np.ptp(ys), np.ptp(xs)) * spacing)
        rect = cv2.minAreaRect(np.c_[xs, ys].astype(np.float32))
        rl, rs = max(rect[1]) * spacing, min(rect[1]) * spacing
        # 长度不应小于外接矩形长边 (骨架在端点会短一点), 取二者最大
        length = max(length, rl)
        item = {'id': int(lab), 'area_mm2': float(comp.sum() * spacing ** 2),
                'depth_mm_p99': round(depth_p99, 4), 'depth_mm_max': round(depth_max, 4), 'depth_mm_mean': round(depth_mean, 4),
                'width_mm_median': round(width_med, 3), 'width_mm_max': round(width_max, 3),
                'length_mm': round(length, 2), 'rect_long_mm': round(rl, 2), 'rect_short_mm': round(rs, 2),
                'orientation_deg': round(float(rect[2]), 1), 'aspect_ratio': round(rl / max(rs, 1e-6), 2),
                'centroid_px': [round(float(xs.mean()), 1), round(float(ys.mean()), 1)]}
        if frame is not None and raster is not None:
            c, R = frame
            u = raster['u0'] + xs.mean() * spacing
            v = raster['v0'] + ys.mean() * spacing
            h = float(np.nanmedian(raster['H'][comp])) if np.isfinite(raster['H'][comp]).any() else 0.0
            item['centroid_xyz_mm'] = [round(float(x), 3) for x in (c + R.T @ np.array([u, v, h]))]
        out.append(item)
    out.sort(key=lambda d: -d['depth_mm_p99'])
    return out


# ================================================================================================================
# 7. 判定规则
# ================================================================================================================
DEFAULT_RULES = {
    'abnormal_if_any': {'depth_mm_p99_gt': 0.05, 'width_mm_median_gt': 0.5, 'length_mm_gt': 5.0},
    'scratch_if': {'aspect_ratio_ge': 3.0},
    'note': '任一条件超过阈值即判为异常缺陷; 细长 (长宽比>=3) 归为划痕, 否则归为凹坑/压痕。阈值需按工艺标准修改。'
}


def judge(items, rules=None):
    rules = rules or DEFAULT_RULES
    ab = rules['abnormal_if_any']
    for it in items:
        it['type'] = 'scratch' if it['aspect_ratio'] >= rules['scratch_if']['aspect_ratio_ge'] else 'dent'
        reasons = []
        if it['depth_mm_p99'] > ab['depth_mm_p99_gt']:
            reasons.append(f"depth {it['depth_mm_p99']:.3f} > {ab['depth_mm_p99_gt']}")
        if it['width_mm_median'] > ab['width_mm_median_gt']:
            reasons.append(f"width {it['width_mm_median']:.2f} > {ab['width_mm_median_gt']}")
        if it['length_mm'] > ab['length_mm_gt']:
            reasons.append(f"length {it['length_mm']:.1f} > {ab['length_mm_gt']}")
        it['abnormal'] = bool(reasons)
        it['reasons'] = reasons
    return items


# ================================================================================================================
# 一站式入口
# ================================================================================================================
def analyze_pointcloud(xyz, labels_gt=None, spacing=None, se_mm=6.0, k_sigma=4.0, abs_min_mm=0.02,
                       min_area_mm2=0.5, mode='pit', rules=None):
    """点云 -> 高度图 -> 分割 -> 测量 -> 判定。返回 dict (含中间结果, 便于可视化与评估)。"""
    c, R = pca_frame(xyz)
    uvh = (xyz - c) @ R.T
    if spacing is None:
        spacing = 1.5 * estimate_spacing(xyz)
    ras = rasterize(uvh, spacing)
    seg = segment_defects(ras['H'], spacing, se_mm, k_sigma, abs_min_mm, min_area_mm2, mode)
    items = judge(measure_components(seg, spacing, (c, R), ras), rules)
    res = {'spacing_mm': spacing, 'frame': (c, R), 'raster': ras, 'seg': seg, 'defects': items}
    if labels_gt is not None:
        res['eval'] = evaluate_points(seg['mask'], ras, labels_gt)
    return res


def evaluate_points(mask, raster, labels_gt):
    """把栅格预测映射回每个点, 与逐点标签比较 -> IoU / precision / recall。"""
    pred = (mask.ravel()[raster['cell_index']] > 0)
    gt = labels_gt > 0
    tp, fp, fn = np.sum(pred & gt), np.sum(pred & ~gt), np.sum(~pred & gt)
    return {'iou': float(tp / max(tp + fp + fn, 1)), 'precision': float(tp / max(tp + fp, 1)),
            'recall': float(tp / max(tp + fn, 1)), 'gt_points': int(gt.sum()), 'pred_points': int(pred.sum())}


def gt_depth_from_labels(res, labels_gt):
    """用真值标签区域算本算法残差的深度 (用于区分 "分割错" 和 "深度算错")。"""
    ras, seg = res['raster'], res['seg']
    gt_cells = np.zeros(ras['shape'][0] * ras['shape'][1], bool)
    gt_cells[ras['cell_index'][labels_gt > 0]] = True
    r = seg['residual'].ravel()[gt_cells]
    return float(np.percentile(r, 99)) if r.size else 0.0


def save_json(obj, path):
    def conv(o):
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
        raise TypeError(str(type(o)))
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=conv)
