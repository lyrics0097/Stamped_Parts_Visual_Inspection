"""冲压件自动检测 Demo (手持版): D455 深度 + DINOv3 外观识别 + SAM 自动蒙版 + 3D 平面测量 + 参数输出。

使用场景: 相机立着, 背景杂乱 (人 / 桌子 / 电脑 / 线缆), 操作者手拿工件伸到相机前。不需要干净台面。

工作原理:
    学习 (只需一次, 模型存盘, 之后启动自动加载; 多按几次 l 可从不同角度追加样本)
      1. 定格当前画面, 用深度高亮"最近物体簇"作提示, 用户拖框框住工件 -> SAM 蒙版。
      2. 用 DINOv3 (ViT-B/16) 提取整图 patch 特征, 蒙版内的 patch 特征作为工件"原型" (prototype) 保存。
         同时用翻转图自测, 自动确定相似度阈值 (工件 patch 与背景 patch 的相似度分界)。
    识别 (后台线程, 约 1 Hz)
      3. 每帧缩到 640x352 提 DINOv3 特征, 每个 patch 与原型做余弦相似度 -> 相似度热图 -> 阈值 -> 候选区域。
      4. 每个候选取"自身 + 周围一圈"的有效深度中位数 (黑色表面自身可能无深度, 周围通常是握着它的手),
         选深度最近的候选 —— 手拿工件伸向相机时, 工件一定比背景里长得像的东西近。
      5. 候选框连续稳定 N 次 => 自动触发 (也可按 s 手动触发)。
    分割
      6. 触发时按 800x448 重新精定位, 粗框作为 SAM 框提示 -> 精细蒙版; 再用深度剔除泄漏到背景的像素。
    测量 (真正的 3D)
      7. 蒙版内有效深度点 PCA 拟合工件平面 (法向量 + 倾斜角); 蒙版轮廓每个像素的视线与平面求交 =>
         即使黑色表面没有深度, 也能得到每个边界点的真实 3D 坐标; 在平面内做最小外接矩形 => 与倾斜无关的真实长宽。
         工件本身测不到深度时, 回退用握着它的手 / 最近簇的深度构造正对平面。
    输出
      8. 每次结果写入  桌面\\冲压件检测\\<时间戳>\\ , 并在  桌面\\冲压件检测\\results.csv  追加一行;
         工件模型存于  桌面\\冲压件检测\\model\\。

按键:  l = 学习 / 追加学习工件    r = 清空模型重学    s = 手动触发    a = 开/关自动触发    q / Esc = 退出

运行 (rad_env 环境):  python stamping_auto_inspect.py [--device cuda:0] [--no_auto]
"""
import argparse
import csv
import json
import os
import threading
import time
from datetime import datetime

import cv2
import numpy as np
import pyrealsense2 as rs
import torch
import torch.nn.functional as F

from dinov3.hub.backbones import load_dinov3_model
from realsense_sam_measure import (SAM_WEIGHT, _fill_holes, build_sam_predictor, crop_by_mask, sam_segment_box)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DINO_WEIGHT = os.path.join(os.path.dirname(SCRIPT_DIR), 'weights', 'dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth')
OUT_ROOT = os.path.join(os.environ['USERPROFILE'], 'Desktop', '冲压件检测')
MODEL_DIR = os.path.join(OUT_ROOT, 'model')
MODEL_PATH = os.path.join(MODEL_DIR, 'object_dino_model.npz')
WIN = 'Stamping Auto Inspect   [l] learn  [r] reset model  [s] trigger  [a] auto  [q] quit'
LEARN_WIN = 'Learn object: drag a box around the PART (not the hand), press Enter'

LIVE_HW = (352, 640)        # 后台识别分辩率 (H, W), CPU ~0.7 s
TRIG_HW = (448, 800)        # 触发时精定位分辩率, CPU ~1.2 s
PATCH = 16
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def imwrite_u(path, img, quality=95):
    """支持中文路径的图片写入 (cv2.imwrite 在 Windows 上不支持非 ASCII 路径)。"""
    ext = os.path.splitext(path)[1].lower()
    params = [cv2.IMWRITE_JPEG_QUALITY, quality] if ext in ('.jpg', '.jpeg') else []
    ok, buf = cv2.imencode(ext, img, params)
    if not ok:
        raise RuntimeError(f'编码失败: {path}')
    buf.tofile(path)


# ================================================================================================================
# 1. 深度: 最近物体簇 (用于学习时的提示、以及工件无深度时的回退)
# ================================================================================================================
def nearest_cluster(depth_m, min_z=0.20, band=0.15, min_area_ratio=0.003):
    """返回 (簇二值图, 外接框 xyxy, 簇深度中位数); 找不到返回 (None, None, None)。
    以有效深度的 1% 分位数为"最近距离" (抗飞点), 取其后 band 米内的像素, 形态学处理后取最大连通域。"""
    valid = depth_m > min_z
    if valid.sum() < 500:
        return None, None, None
    z_near = float(np.percentile(depth_m[valid], 1))
    near = (valid & (depth_m < z_near + band)).astype(np.uint8) * 255
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    near = cv2.morphologyEx(near, cv2.MORPH_OPEN, k)
    near = cv2.morphologyEx(near, cv2.MORPH_CLOSE, k, iterations=3)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(near, connectivity=8)
    if n <= 1:
        return None, None, None
    i = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    if stats[i, cv2.CC_STAT_AREA] < min_area_ratio * depth_m.size:
        return None, None, None
    x, y, w, h = stats[i, :4]
    cluster = (labels == i).astype(np.uint8) * 255
    return cluster, (int(x), int(y), int(x + w), int(y + h)), float(np.median(depth_m[cluster > 0]))


# ================================================================================================================
# 2. DINOv3 外观模型: 特征提取 / 学习 / 识别
# ================================================================================================================
class DinoMatcher:
    def __init__(self, device='cpu'):
        self.device = torch.device(device if device != 'cuda:0' or torch.cuda.is_available() else 'cpu')
        print(f'正在加载 DINOv3 ViT-B/16: {DINO_WEIGHT}')
        self.enc = load_dinov3_model('dinov3_vitb16', pretrained_weight_path=DINO_WEIGHT).to(self.device).eval()
        for p in self.enc.parameters():
            p.requires_grad = False
        self.protos = None        # [P, C] 工件原型特征 (已归一化)
        self.thr = 0.6            # 相似度阈值
        self.n_samples = 0
        self.lock = threading.Lock()

    @torch.no_grad()
    def features(self, color_bgr, hw):
        """整图 -> 归一化 patch 特征 [C, H/16, W/16]。"""
        img = cv2.resize(color_bgr, (hw[1], hw[0]), interpolation=cv2.INTER_AREA)
        x = torch.from_numpy(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).permute(2, 0, 1).float().unsqueeze(0) / 255.0
        x = ((x - IMAGENET_MEAN) / IMAGENET_STD).to(self.device)
        f = self.enc.get_intermediate_layers(x, n=[11], reshape=True, norm=True)[0][0]   # [C, h, w]
        return F.normalize(f, dim=0)

    def score_map(self, feat):
        """每个 patch 与所有原型的最大余弦相似度 -> [h, w] numpy。"""
        with self.lock:
            protos = self.protos
        if protos is None:
            return None
        sims = torch.einsum('pc,chw->phw', protos.to(feat.device), feat)
        return sims.max(dim=0).values.cpu().numpy()

    def learn(self, color_bgr, mask):
        """从 (图, 工件蒙版) 追加原型, 并用水平翻转图自测更新阈值。"""
        feat = self.features(color_bgr, TRIG_HW)                       # [C, h, w]
        h, w = feat.shape[1:]
        cov = cv2.resize(mask.astype(np.float32) / 255.0, (w, h), interpolation=cv2.INTER_AREA)
        obj = torch.from_numpy(cov >= 0.6).to(feat.device)
        if obj.sum() < 3:
            raise RuntimeError('工件太小, 覆盖不足 3 个 patch, 请靠近相机再学习')
        new = feat.permute(1, 2, 0)[obj]                                # [P, C]
        with self.lock:
            self.protos = new if self.protos is None else torch.cat([self.protos, new], 0)
            self.n_samples += 1
        # 自测: 翻转图上工件 patch 的相似度分布 vs 背景 patch 的相似度分布 -> 阈值取两者中点
        feat_f = self.features(cv2.flip(color_bgr, 1), TRIG_HW)
        sm = self.score_map(feat_f)
        cov_f = cv2.flip(cov, 1)
        pos, neg = sm[cov_f >= 0.6], sm[cov_f == 0]
        if pos.size and neg.size:
            thr = 0.5 * (np.percentile(pos, 20) + np.percentile(neg, 99.5))
            self.thr = float(np.clip(thr, 0.45, 0.8))
            print(f'[学习] 工件patch相似度 p20={np.percentile(pos, 20):.3f}  背景 p99.5={np.percentile(neg, 99.5):.3f}'
                  f'  -> 阈值 {self.thr:.3f}  (原型 {self.protos.shape[0]}, 样本 {self.n_samples})')
        return int(new.shape[0])

    def save(self):
        os.makedirs(MODEL_DIR, exist_ok=True)
        np.savez(MODEL_PATH, protos=self.protos.cpu().numpy(), thr=self.thr, n_samples=self.n_samples,
                 learned_at=datetime.now().isoformat(timespec='seconds'))

    def load(self):
        if not os.path.exists(MODEL_PATH):
            return False
        d = np.load(MODEL_PATH)
        self.protos = torch.from_numpy(d['protos']).to(self.device)
        self.thr, self.n_samples = float(d['thr']), int(d['n_samples'])
        print(f'[模型] 已加载工件模型: {MODEL_PATH}  (原型 {self.protos.shape[0]}, 样本 {self.n_samples}, 阈值 {self.thr:.3f})')
        return True

    def reset(self):
        with self.lock:
            self.protos, self.n_samples = None, 0
        if os.path.exists(MODEL_PATH):
            os.remove(MODEL_PATH)


def find_object(matcher, color_bgr, depth_m, hw, min_patches=2, max_area_ratio=0.3):
    """DINOv3 相似度 + 深度 联合识别。返回 (工件粗框 xyxy, 候选列表, 相似度热图)。

    候选 = 相似度热图 > 阈值 的连通域 (面积 >= min_patches 个 patch); 每个候选记录 平均相似度 与 深度
    (自身 + 周围一圈的有效深度中位数); 选 深度最近 的候选, 深度都无效时选相似度最高的。
    """
    H, W = color_bgr.shape[:2]
    feat = matcher.features(color_bgr, hw)
    sm = matcher.score_map(feat)
    if sm is None:
        return None, [], None
    hot = (sm > matcher.thr).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(hot, connectivity=8)
    sy, sx = H / sm.shape[0], W / sm.shape[1]                      # patch -> 原图像素
    cands = []
    ring = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
    for i in range(1, n):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < min_patches or area > max_area_ratio * sm.size:
            continue
        x, y, w, h = stats[i, :4]
        box = (int(x * sx), int(y * sy), int(min(W, (x + w) * sx)), int(min(H, (y + h) * sy)))
        comp = cv2.resize((labels == i).astype(np.uint8) * 255, (W, H), interpolation=cv2.INTER_NEAREST)
        x0, y0, x1, y1 = max(0, box[0] - 20), max(0, box[1] - 20), min(W, box[2] + 20), min(H, box[3] + 20)
        region = cv2.dilate(comp[y0:y1, x0:x1], ring) > 0
        dz = depth_m[y0:y1, x0:x1][region]
        dz = dz[dz > 0]
        z = float(np.median(dz)) if dz.size >= 30 else np.inf
        cands.append({'box': box, 'area': int(area), 'sim': float(sm[labels == i].mean()), 'z': z})
    if not cands:
        return None, cands, sm
    finite = [c for c in cands if np.isfinite(c['z'])]
    best = min(finite, key=lambda c: c['z']) if finite else max(cands, key=lambda c: c['sim'])
    cands.sort(key=lambda c: (c is not best, c['z'], -c['sim']))
    return best['box'], cands, sm


def box_iou(a, b):
    ix0, iy0, ix1, iy1 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix1 - ix0) * max(0, iy1 - iy0)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def learn_interactive(predictor, matcher, color_bgr, depth_m):
    """定格 -> 高亮最近簇 -> 用户框工件 -> SAM 蒙版 -> DINOv3 原型, 保存模型。"""
    cluster, cbox, _ = nearest_cluster(depth_m)
    show = color_bgr.copy()
    if cluster is not None:
        show[cluster > 0] = (0.6 * show[cluster > 0] + (0, 100, 0)).astype(np.uint8)
        cv2.rectangle(show, cbox[:2], cbox[2:], (0, 255, 0), 2)
    cv2.putText(show, 'Green = nearest (depth). Drag a box around the PART only, then Enter (c = cancel)',
                (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
    x, y, bw, bh = cv2.selectROI(LEARN_WIN, show, showCrosshair=False, fromCenter=False)
    cv2.destroyWindow(LEARN_WIN)
    cv2.waitKey(1)
    if bw < 4 or bh < 4:
        print('[学习] 未框选, 取消')
        return False
    box = (int(x), int(y), int(x + bw), int(y + bh))
    color_rgb = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB)
    mask, _, score = sam_segment_box(predictor, color_rgb, box)
    if (mask > 0).sum() < 200:
        print('[学习] SAM 蒙版为空, 取消')
        return False
    n_new = matcher.learn(color_bgr, mask)
    matcher.save()
    os.makedirs(MODEL_DIR, exist_ok=True)
    tag = datetime.now().strftime('%Y%m%d_%H%M%S')
    imwrite_u(os.path.join(MODEL_DIR, f'ref_{tag}_crop.jpg'), crop_by_mask(color_rgb, mask))
    imwrite_u(os.path.join(MODEL_DIR, f'ref_{tag}_full.jpg'), color_bgr)
    print(f'[学习] 已追加 {n_new} 个原型 (SAM score {score:.3f}), 模型保存至 {MODEL_PATH}')
    return True


# ================================================================================================================
# 3. 3D 测量: 平面拟合 + 视线-平面求交
# ================================================================================================================
def deproject(us, vs, zs, intr):
    return np.stack([(us - intr.ppx) / intr.fx * zs, (vs - intr.ppy) / intr.fy * zs, zs], axis=1)


def fit_plane(points):
    """PCA 平面拟合 -> (质心, 单位法向 (朝相机, z<0), 平面内基 e1 e2, 残差 RMS)。"""
    c = points.mean(axis=0)
    _, _, vt = np.linalg.svd(points - c, full_matrices=False)
    n = vt[2] if vt[2][2] < 0 else -vt[2]
    return c, n, vt[0], vt[1], float(np.sqrt((((points - c) @ n) ** 2).mean()))


def ransac_plane(points, iters=120, inlier_mm=4.0, seed=0):
    """RANSAC 平面拟合: 随机 3 点定平面, 统计 |距离| < inlier_mm 的内点, 取内点最多的平面再用 PCA 精修。
    返回 (质心, 法向, e1, e2, rms, 内点比例)。"""
    rng = np.random.default_rng(seed)
    N = len(points)
    best_inl, best_cnt = None, 0
    for _ in range(iters):
        i = rng.choice(N, 3, replace=False)
        p0, p1, p2 = points[i]
        n = np.cross(p1 - p0, p2 - p0)
        nn = np.linalg.norm(n)
        if nn < 1e-9:
            continue
        n /= nn
        dist = np.abs((points - p0) @ n)
        inl = dist < inlier_mm / 1000.0
        if inl.sum() > best_cnt:
            best_cnt, best_inl = inl.sum(), inl
    if best_inl is None or best_cnt < 3:
        c, n, e1, e2, rms = fit_plane(points)
        return c, n, e1, e2, rms, 0.0
    c, n, e1, e2, rms = fit_plane(points[best_inl])
    return c, n, e1, e2, rms, float(best_cnt / N)


def measure_3d(mask, depth_m, intr, z_fallback, max_tilt=60.0, max_rms_mm=5.0, min_inlier=0.5):
    """3D 测量。平面拟合必须通过质量检查 (内点比例 / 残差 / 倾角 / 短轴展幅), 否则退回"正对相机的平面":
    细长或黑色工件的深度点稀疏且噪声大, PCA 会把"短边方向"误当成法向, 给出荒谬的倾角。"""
    ys, xs = np.nonzero(mask)
    z = depth_m[ys, xs]
    ok = z > 0
    plane_src, quality = 'frontal_plane_at_object_depth', {}
    zz = z[ok]
    z_obj = float(np.median(zz)) if ok.sum() >= 50 else float(z_fallback)
    if ok.sum() < 50:
        plane_src = 'frontal_plane_at_hand_depth'
    fitted = False
    if ok.sum() >= 300:
        keep = np.abs(zz - np.median(zz)) < 0.05
        pts = deproject(xs[ok][keep].astype(np.float64), ys[ok][keep].astype(np.float64), zz[keep].astype(np.float64), intr)
        if len(pts) >= 50:
            c, n, e1, e2, rms, inl = ransac_plane(pts)
            tilt = float(np.degrees(np.arccos(min(1.0, abs(n[2])))))
            # 短轴展幅: 内点在 e2 方向的标准差; 必须显著大于残差, 否则平面方向不可信 (细条形工件)
            spread2 = float(np.std((pts - c) @ e2))
            quality = {'inlier_ratio': round(inl, 3), 'rms_mm': round(rms * 1000, 2), 'tilt_deg': round(tilt, 1),
                       'short_axis_spread_mm': round(spread2 * 1000, 1), 'n_points': int(len(pts))}
            if inl >= min_inlier and rms * 1000 <= max_rms_mm and tilt <= max_tilt and spread2 > 4 * max(rms, 0.002):
                fitted, plane_src = True, 'object_depth_ransac_pca'
                z_obj = float(c[2])
    if not fitted:
        c = deproject(np.array([xs.mean()]), np.array([ys.mean()]), np.array([z_obj]), intr)[0]
        n, e1, e2 = np.array([0.0, 0.0, -1.0]), np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0])
        rms = 0.0

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cnt = max(contours, key=cv2.contourArea).reshape(-1, 2).astype(np.float64)
    d = np.stack([(cnt[:, 0] - intr.ppx) / intr.fx, (cnt[:, 1] - intr.ppy) / intr.fy, np.ones(len(cnt))], axis=1)
    t = (n @ c) / (d @ n)                          # 视线与平面交点
    P = d * t[:, None]
    uv = np.stack([(P - c) @ e1, (P - c) @ e2], axis=1).astype(np.float32)
    rect3d = cv2.minAreaRect(uv)
    long_m, short_m = max(rect3d[1]), min(rect3d[1])
    tilt_deg = float(np.degrees(np.arccos(min(1.0, abs(n[2])))))

    rect_px = cv2.minAreaRect(cnt.astype(np.float32))
    long_px, short_px = max(rect_px[1]), min(rect_px[1])
    mm_per_px = z_obj / intr.fx * 1000.0
    return {
        'plane_source': plane_src, 'plane_fit_rms_mm': round(rms * 1000, 2), 'plane_fit_quality': quality,
        'plane_normal_camera_frame': [round(float(v), 4) for v in n],
        'plane_center_camera_frame_m': [round(float(v), 4) for v in c],
        'tilt_deg': round(tilt_deg, 1), 'object_plane_depth_m': round(z_obj, 4),
        'mm_per_pixel_frontal': round(mm_per_px, 5),
        'mm_per_pixel_along_plane_approx': round(mm_per_px / max(np.cos(np.radians(tilt_deg)), 0.3), 5),
        'min_area_rect_px': {'long_side_px': round(float(long_px), 1), 'short_side_px': round(float(short_px), 1),
                             'angle_deg': round(float(rect_px[2]), 1),
                             'center_xy': [round(float(rect_px[0][0]), 1), round(float(rect_px[0][1]), 1)]},
        'size_mm_pinhole_frontal': {'long_side': round(float(long_px) * mm_per_px, 1), 'short_side': round(float(short_px) * mm_per_px, 1)},
        'size_mm_3d_plane': {'long_side': round(long_m * 1000, 1), 'short_side': round(short_m * 1000, 1)},
        '_rect_px': rect_px,
    }


# ================================================================================================================
# 4. 触发后: 精定位 + SAM + 深度过滤 + 3D 测量 + 保存
# ================================================================================================================
def depth_colormap(depth_m, max_m=1.5):
    v = np.clip(depth_m / max_m * 255, 0, 255).astype(np.uint8)
    vis = cv2.applyColorMap(v, cv2.COLORMAP_JET)
    vis[depth_m <= 0] = 0
    return vis


def sim_heatmap(sm, W, H, thr):
    v = np.clip((sm - 0.2) / 0.7, 0, 1)
    hm = cv2.applyColorMap((v * 255).astype(np.uint8), cv2.COLORMAP_JET)
    hm[sm <= thr] //= 3
    return cv2.resize(hm, (W, H), interpolation=cv2.INTER_NEAREST)


def draw_overlay(color_bgr, mask, m3, box, info):
    vis = color_bgr.copy()
    layer = np.zeros_like(vis)
    layer[:] = (255, 144, 30)
    blend = cv2.addWeighted(vis, 0.55, layer, 0.45, 0)
    vis[mask > 0] = blend[mask > 0]
    cv2.drawContours(vis, [cv2.boxPoints(m3['_rect_px']).astype(np.int32)], 0, (0, 255, 0), 2)
    cv2.rectangle(vis, box[:2], box[2:], (0, 255, 255), 1)
    s3, sp = m3['size_mm_3d_plane'], m3['size_mm_pinhole_frontal']
    lines = [f"3D plane:  {s3['long_side']:.1f} x {s3['short_side']:.1f} mm   (tilt {m3['tilt_deg']:.0f} deg)",
             f"Pinhole :  {sp['long_side']:.1f} x {sp['short_side']:.1f} mm   Z={m3['object_plane_depth_m']:.3f} m",
             f"scale {m3['mm_per_pixel_frontal']:.4f} mm/px   SAM {info['sam_score']:.3f}   {info['run_id']}"]
    y = 30
    for t in lines:
        cv2.putText(vis, t, (15, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(vis, t, (15, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
        y += 30
    return vis


def inspect_and_save(predictor, matcher, color_bgr, depth_m, intr, trigger, coarse_box=None, z_hint=None, depth_tol=0.06):
    t0 = time.time()
    run_id = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_dir = os.path.join(OUT_ROOT, run_id)
    os.makedirs(out_dir, exist_ok=True)
    h, w = color_bgr.shape[:2]

    # ---- 触发帧上用更高分辨率重新精定位; 失败则用后台线程给的粗框
    box_hi, cands, sm = find_object(matcher, color_bgr, depth_m, TRIG_HW)
    obj_box = box_hi if box_hi is not None else coarse_box
    if obj_box is None:
        raise RuntimeError('触发帧中未识别到工件')
    z_cands = next((c['z'] for c in cands if np.isfinite(c['z'])), None) if cands else None
    z_ref0 = z_cands if z_cands is not None else z_hint
    cluster, _, z_cl = nearest_cluster(depth_m)
    if z_ref0 is None:
        z_ref0 = z_cl
    if z_ref0 is None:
        raise RuntimeError('没有任何有效深度可用')

    # ---- SAM 框提示 (patch 级粗框外扩半个 patch)
    pad = 10
    box = (max(0, obj_box[0] - pad), max(0, obj_box[1] - pad), min(w, obj_box[2] + pad), min(h, obj_box[3] + pad))
    color_rgb = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB)
    mask, _, score = sam_segment_box(predictor, color_rgb, box)

    # ---- 深度过滤: 蒙版内有深度且比工件远 depth_tol 以上的像素 => 泄漏到背景, 剔除
    zv = depth_m[(mask > 0) & (depth_m > 0)]
    z_ref = float(np.median(zv)) if zv.size >= 100 else z_ref0
    mask[(mask > 0) & (depth_m > 0) & (depth_m > z_ref + depth_tol)] = 0
    mask = _fill_holes(cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))))
    if (mask > 0).sum() < 50:
        raise RuntimeError('SAM 蒙版为空')

    m3 = measure_3d(mask, depth_m, intr, z_fallback=z_ref)
    ys, xs = np.nonzero(mask)
    mg = 10
    crop_bbox = [int(max(0, xs.min() - mg)), int(max(0, ys.min() - mg)), int(min(w, xs.max() + 1 + mg)), int(min(h, ys.max() + 1 + mg))]

    info = {
        'run_id': run_id, 'timestamp': datetime.now().isoformat(timespec='seconds'), 'trigger': trigger,
        'camera': 'Intel RealSense D455 (depth aligned to color)',
        'image_size': {'width': w, 'height': h},
        'intrinsics_color': {'fx': intr.fx, 'fy': intr.fy, 'ppx': intr.ppx, 'ppy': intr.ppy, 'model': str(intr.model), 'coeffs': list(intr.coeffs)},
        'recognition': {'method': 'DINOv3 ViT-B/16 patch similarity + nearest depth', 'sim_threshold': round(matcher.thr, 3),
                        'candidates': [{'box': list(c['box']), 'sim': round(c['sim'], 3), 'z_m': (round(c['z'], 3) if np.isfinite(c['z']) else None)} for c in cands[:5]],
                        'chosen_box_xyxy': list(obj_box)},
        'nearest_cluster_depth_m': (round(z_cl, 4) if z_cl is not None else None),
        'object_depth_valid_pixels': int(zv.size),
        'sam_weight': SAM_WEIGHT, 'sam_box_prompt_xyxy': list(box), 'sam_score': round(float(score), 4),
        'mask_area_px': int((mask > 0).sum()),
        'crop_bbox_in_full_image_xyxy': crop_bbox,
        'measurement_3d': {k: v for k, v in m3.items() if not k.startswith('_')},
        'scratch_width_conversion': {'formula': 'scratch_width_mm = scratch_width_px * mm_per_pixel',
                                     'mm_per_pixel': m3['mm_per_pixel_frontal'],
                                     'note': '若工件倾斜, 沿倾斜方向像素被压缩, 可用 mm_per_pixel_along_plane_approx 或按平面法向逐点修正'},
        'files': {'color': 'color.png', 'depth_m_npy': 'depth_m.npy', 'depth_vis': 'depth_vis.png', 'similarity': 'similarity.jpg',
                  'mask': 'mask.png', 'crop': 'crop.jpg', 'overlay': 'overlay.jpg'},
        'notes': ['mask.png 与 color.png 同尺寸, 255 = 工件', 'crop.jpg 与原图 1:1 像素, 位置见 crop_bbox_in_full_image_xyxy',
                  'size_mm_3d_plane 为倾斜校正后的尺寸 (推荐); size_mm_pinhole_frontal 为正对假设下的尺寸'],
    }

    imwrite_u(os.path.join(out_dir, 'color.png'), color_bgr)
    np.save(os.path.join(out_dir, 'depth_m.npy'), depth_m)
    imwrite_u(os.path.join(out_dir, 'depth_vis.png'), depth_colormap(depth_m))
    if sm is not None:
        imwrite_u(os.path.join(out_dir, 'similarity.jpg'), cv2.addWeighted(color_bgr, 0.4, sim_heatmap(sm, w, h, matcher.thr), 0.6, 0))
    imwrite_u(os.path.join(out_dir, 'mask.png'), mask)
    imwrite_u(os.path.join(out_dir, 'crop.jpg'), crop_by_mask(color_rgb, mask, mg))
    overlay = draw_overlay(color_bgr, mask, m3, box, info)
    imwrite_u(os.path.join(out_dir, 'overlay.jpg'), overlay)
    with open(os.path.join(out_dir, 'info.json'), 'w', encoding='utf-8') as f:
        json.dump(info, f, ensure_ascii=False, indent=2)

    csv_path = os.path.join(OUT_ROOT, 'results.csv')
    new = not os.path.exists(csv_path)
    with open(csv_path, 'a', newline='', encoding='utf-8-sig') as f:
        wr = csv.writer(f)
        if new:
            wr.writerow(['run_id', 'trigger', 'Z_m', 'plane_source', 'tilt_deg', 'fx', 'mm_per_px', 'long_px', 'short_px',
                         'long_mm_3d', 'short_mm_3d', 'long_mm_pinhole', 'short_mm_pinhole', 'sam_score', 'folder'])
        wr.writerow([run_id, trigger, m3['object_plane_depth_m'], m3['plane_source'], m3['tilt_deg'], f'{intr.fx:.2f}',
                     m3['mm_per_pixel_frontal'], m3['min_area_rect_px']['long_side_px'], m3['min_area_rect_px']['short_side_px'],
                     m3['size_mm_3d_plane']['long_side'], m3['size_mm_3d_plane']['short_side'],
                     m3['size_mm_pinhole_frontal']['long_side'], m3['size_mm_pinhole_frontal']['short_side'], f'{score:.3f}', out_dir])

    s3 = m3['size_mm_3d_plane']
    print(f'[检测] {run_id}  Z={m3["object_plane_depth_m"]:.3f} m ({m3["plane_source"]}, tilt {m3["tilt_deg"]:.0f}°)  '
          f'3D尺寸 {s3["long_side"]} x {s3["short_side"]} mm  scale={m3["mm_per_pixel_frontal"]:.4f} mm/px  '
          f'SAM={score:.3f}  ({time.time() - t0:.1f}s)\n        -> {out_dir}')
    return info, overlay


# ================================================================================================================
# 5. 后台识别线程
# ================================================================================================================
class Detector(threading.Thread):
    """持续对"最新一帧"做 DINOv3 识别 (~1 Hz), 结果放在 self.result。"""

    def __init__(self, matcher):
        super().__init__(daemon=True)
        self.matcher = matcher
        self.frame = None            # (color_bgr, depth_m)
        self.result = None           # (box, cands, sm, t)
        self.cv = threading.Condition()
        self.running = True

    def submit(self, color_bgr, depth_m):
        with self.cv:
            self.frame = (color_bgr, depth_m)
            self.cv.notify()

    def run(self):
        while self.running:
            with self.cv:
                while self.frame is None and self.running:
                    self.cv.wait(0.2)
                if not self.running:
                    break
                color_bgr, depth_m = self.frame
                self.frame = None
            try:
                box, cands, sm = find_object(self.matcher, color_bgr, depth_m, LIVE_HW)
                self.result = (box, cands, sm, time.time())
            except Exception as e:
                print(f'[识别] 线程出错: {e}')

    def stop(self):
        self.running = False
        with self.cv:
            self.cv.notify()


# ================================================================================================================
# 6. 主循环
# ================================================================================================================
def main():
    ap = argparse.ArgumentParser(description='冲压件自动检测 Demo (手持 / DINOv3 + 深度)')
    ap.add_argument('--device', default='cuda:0' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--width', type=int, default=1280)
    ap.add_argument('--height', type=int, default=720)
    ap.add_argument('--fps', type=int, default=30)
    ap.add_argument('--stable_hits', type=int, default=3, help='连续多少次识别结果稳定后自动触发 (每次 ~1 s)')
    ap.add_argument('--no_auto', action='store_true', help='关闭自动触发, 仅按 s 触发')
    ap.add_argument('--relearn', action='store_true', help='忽略已保存模型, 启动后重新学习')
    args = ap.parse_args()

    os.makedirs(OUT_ROOT, exist_ok=True)
    print(f'[输出] 所有结果将保存到: {OUT_ROOT}')
    predictor = build_sam_predictor(args.device)
    matcher = DinoMatcher(args.device)
    if args.relearn:
        matcher.reset()
    if not matcher.load():
        print('[模型] 尚无工件模型: 请手拿工件伸到相机前, 按 l 学习')
    detector = Detector(matcher)
    detector.start()

    pipeline, config = rs.pipeline(), rs.config()
    config.enable_stream(rs.stream.color, args.width, args.height, rs.format.rgb8, args.fps)
    config.enable_stream(rs.stream.depth, args.width, args.height, rs.format.z16, args.fps)
    profile = pipeline.start(config)
    try:
        depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
        align = rs.align(rs.stream.color)
        spatial = rs.spatial_filter()                 # 空间滤波 + 补洞, 让黑色表面多一些有效深度
        spatial.set_option(rs.option.holes_fill, 2)
        for _ in range(30):
            pipeline.wait_for_frames()

        cv2.namedWindow(WIN, cv2.WINDOW_AUTOSIZE)
        auto = not args.no_auto
        prev_box, hits, armed, last_seen_t = None, 0, True, 0
        last_overlay, last_overlay_t = None, 0
        show_heat = False

        while True:
            frames = align.process(pipeline.wait_for_frames(timeout_ms=5000))
            c, d = frames.get_color_frame(), frames.get_depth_frame()
            if not c or not d:
                continue
            d = spatial.process(d)
            color_bgr = cv2.cvtColor(np.asanyarray(c.get_data()), cv2.COLOR_RGB2BGR)
            depth_m = np.asanyarray(d.get_data()).astype(np.float32) * depth_scale
            h, w = depth_m.shape
            if matcher.protos is not None:
                detector.submit(color_bgr, depth_m)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), 27) or cv2.getWindowProperty(WIN, cv2.WND_PROP_VISIBLE) < 1:
                break
            if key == ord('a'):
                auto = not auto
                print(f'[设置] 自动触发 {"开" if auto else "关"}')
            if key == ord('h'):
                show_heat = not show_heat
            if key == ord('r'):
                matcher.reset()
                detector.result = None
                print('[模型] 已清空, 请按 l 重新学习')
            if key == ord('l'):
                learn_interactive(predictor, matcher, color_bgr, depth_m)
                detector.result = None
                prev_box, hits = None, 0
                continue

            # ---- 取后台识别结果, 判稳定 / 触发
            res = detector.result
            box = None
            if res is not None and time.time() - res[3] < 2.5:
                box, cands, sm, t_res = res
                if box is not None and t_res != last_seen_t:      # 每个新识别结果只计一次
                    last_seen_t = t_res
                    hits = hits + 1 if (prev_box and box_iou(prev_box, box) > 0.6) else 1
                    prev_box = box
            if box is None:
                prev_box, hits = None, 0
                if res is not None and time.time() - res[3] > 1.5:
                    armed = True                                  # 工件离开视野 => 重新武装
            trigger = None
            if auto and armed and box is not None and hits >= args.stable_hits:
                trigger = 'auto'
            if key == ord('s'):
                trigger = 'manual'

            if trigger:
                intr = c.profile.as_video_stream_profile().get_intrinsics()
                z_hint = next((cd['z'] for cd in cands if np.isfinite(cd['z'])), None) if box is not None else None
                try:
                    _, last_overlay = inspect_and_save(predictor, matcher, color_bgr, depth_m.copy(), intr, trigger, box, z_hint)
                    last_overlay_t = time.time()
                except Exception as e:
                    print(f'[检测] 失败: {e}')
                armed, hits = False, 0
                detector.result = None
                for _ in range(5):
                    pipeline.poll_for_frames()

            # ---- 预览
            if last_overlay is not None and time.time() - last_overlay_t < 4:
                show = last_overlay
            else:
                show = color_bgr.copy()
                if show_heat and res is not None and res[2] is not None:
                    show = cv2.addWeighted(show, 0.5, sim_heatmap(res[2], w, h, matcher.thr), 0.5, 0)
                if box is not None:
                    for cd in cands[1:6]:
                        cv2.rectangle(show, cd['box'][:2], cd['box'][2:], (160, 160, 160), 1)
                    col = (0, 255, 255) if armed else (0, 255, 0)
                    cv2.rectangle(show, box[:2], box[2:], col, 2)
                    zt = f"{cands[0]['z']:.2f} m" if np.isfinite(cands[0]['z']) else 'z n/a'
                    cv2.putText(show, f"PART sim {cands[0]['sim']:.2f} {zt}  stable {hits}/{args.stable_hits}" + ('' if armed else '  (done, remove part)'),
                                (box[0], max(35, box[1] - 24)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2, cv2.LINE_AA)
                status = 'NO MODEL: press l to learn' if matcher.protos is None else ('AUTO' if auto else 'MANUAL (s)')
                cv2.putText(show, f'[{status}]  l=learn r=reset s=trigger a=auto h=heatmap q=quit', (15, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
                dv = cv2.resize(depth_colormap(depth_m), (w // 4, h // 4))
                show[-dv.shape[0]:, -dv.shape[1]:] = dv
            cv2.imshow(WIN, show)
    finally:
        detector.stop()
        pipeline.stop()
        cv2.destroyAllWindows()
        print('[相机] 数据流已关闭')


if __name__ == '__main__':
    main()
