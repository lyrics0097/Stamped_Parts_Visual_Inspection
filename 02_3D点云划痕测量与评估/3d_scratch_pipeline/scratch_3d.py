"""无标准件的 3D 点云划痕检测与宽/深测量。

输入: N x 3 点云, 单位 mm。不需要 CAD / 良品对齐。
适用: 工业 3D 相机拍摄的冲压件/钣金局部表面。

算法:
    1. PCA 把点云摊成高度图 (u, v, h)。
    2. 多尺度开/闭运算: 凹痕和凸起都转成正残差。核宽随板面尺寸变化,
       窄划痕和十几毫米的压痕都能留下。
    3. MAD 阈值 + 连通域得到每条划痕。
    4. 在残差剖面上测:
         深度 = 剖面峰值
         宽度 = 半高全宽 (FWHM)
         长度 = 长轴 1%–99%
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import List, Optional, Tuple

import numpy as np
from scipy import ndimage
from sklearn.neighbors import NearestNeighbors


@dataclass
class Scratch:
    id: int
    n_points: int
    length_mm: float
    width_mm: float
    depth_mm: float
    volume_mm3: float
    center_xyz: Tuple[float, float, float]
    abnormal: bool
    reason: str


def estimate_spacing(xyz: np.ndarray, n_sample: int = 6000) -> float:
    n = min(len(xyz), n_sample)
    if n < 3:
        return 0.2
    rng = np.random.default_rng(0)
    idx = rng.choice(len(xyz), n, replace=False) if len(xyz) > n else np.arange(n)
    nn = NearestNeighbors(n_neighbors=2, algorithm="auto").fit(xyz)
    dist, _ = nn.kneighbors(xyz[idx])
    return float(max(np.median(dist[:, 1]), 1e-4))


def _fit_plane(pts: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    c = pts.mean(axis=0)
    _, _, vt = np.linalg.svd(pts - c, full_matrices=False)
    return c, vt[-1]


def _frame(xyz: np.ndarray):
    c, n = _fit_plane(xyz)
    tmp = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    e1 = np.cross(n, tmp)
    e1 /= np.linalg.norm(e1) + 1e-12
    e2 = np.cross(n, e1)
    u = (xyz - c) @ e1
    v = (xyz - c) @ e2
    h = (xyz - c) @ n
    return u, v, h, c, n, e1, e2


def _detrend_quadratic(u: np.ndarray, v: np.ndarray, h: np.ndarray) -> np.ndarray:
    """去掉大尺度二次曲面, 剩下局部沟槽。两遍最小二乘, 丢掉偏离最大的 30%。"""
    A = np.column_stack([np.ones(len(u)), u, v, u * u, v * v, u * v])
    coef, *_ = np.linalg.lstsq(A, h, rcond=None)
    r = h - A @ coef
    keep = np.abs(r) <= np.percentile(np.abs(r), 70)
    if keep.sum() >= 20:
        coef, *_ = np.linalg.lstsq(A[keep], h[keep], rcond=None)
    return h - (A @ coef)


def _raster(u: np.ndarray, v: np.ndarray, h: np.ndarray, pitch: float):
    umin, vmin = float(u.min()), float(v.min())
    ui = np.floor((u - umin) / pitch).astype(np.int32)
    vi = np.floor((v - vmin) / pitch).astype(np.int32)
    hgt, wdt = int(vi.max()) + 1, int(ui.max()) + 1
    if hgt * wdt > 5_000_000:
        pitch *= np.sqrt(hgt * wdt / 5_000_000)
        return _raster(u, v, h, pitch)
    acc = np.zeros((hgt, wdt), np.float64)
    cnt = np.zeros((hgt, wdt), np.float64)
    np.add.at(acc, (vi, ui), h)
    np.add.at(cnt, (vi, ui), 1.0)
    grid = np.divide(acc, cnt, out=np.full_like(acc, np.nan), where=cnt > 0)
    return grid, cnt, umin, vmin, pitch, ui, vi


def _morph_mag(g: np.ndarray, pitch: float, kernel_mm: float) -> np.ndarray:
    k = int(np.ceil(kernel_mm / max(pitch, 1e-6)))
    k = max(5, k | 1)
    dent = ndimage.grey_closing(g, size=(k, k)) - g
    bump = g - ndimage.grey_opening(g, size=(k, k))
    return np.maximum(dent, bump)


def _surface_residual(grid: np.ndarray, pitch: float, kernels_mm: Tuple[float, ...]) -> np.ndarray:
    """凹痕用闭运算、凸起用开运算，多尺度取较大偏差。"""
    valid = np.isfinite(grid)
    fill = float(np.nanmedian(grid)) if valid.any() else 0.0
    g = np.where(valid, grid, fill)
    mag = np.zeros_like(g)
    for kernel_mm in kernels_mm:
        mag = np.maximum(mag, _morph_mag(g, pitch, kernel_mm))
    mag[~valid] = 0.0
    # 只去掉零件外轮廓。网格比点距更细时内部会有空洞, 不能把空洞邻域也清掉。
    interior = ndimage.binary_erosion(valid, iterations=2)
    if interior.sum() > 0.55 * valid.sum():
        mag[valid & ~interior] = 0.0
    return mag


def _kernel_set(u: np.ndarray, v: np.ndarray, kernel_mm: Optional[float]) -> Tuple[float, ...]:
    if kernel_mm is not None and kernel_mm > 0:
        return (float(kernel_mm),)
    plate = min(float(u.max() - u.min()), float(v.max() - v.min()))
    # 6 mm 抓窄划痕, 18/30 mm 抓更宽的压痕; 核必须小于板面, 否则整板会被抹平
    return tuple(k for k in (6.0, 18.0, 30.0) if k < 0.62 * plate) or (max(4.0, 0.35 * plate),)


def _robust_threshold(vals: np.ndarray, k_sigma: float, floor_mm: float) -> float:
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return floor_mm
    med = float(np.median(vals))
    mad = float(np.median(np.abs(vals - med))) * 1.4826
    return max(med + k_sigma * mad, floor_mm)


def _profile_fwhm(y: np.ndarray, depth: np.ndarray, n_bin: int = 40) -> Tuple[float, float]:
    if len(y) < 6:
        return float(np.max(depth) if len(depth) else 0.0), 0.0
    lo, hi = np.percentile(y, [1, 99])
    if hi - lo < 1e-6:
        return float(np.max(depth)), 0.0
    edges = np.linspace(lo, hi, n_bin + 1)
    prof, yc = [], []
    for i in range(n_bin):
        m = (y >= edges[i]) & (y < edges[i + 1])
        if m.any():
            prof.append(float(np.percentile(depth[m], 90)))
            yc.append(0.5 * (edges[i] + edges[i + 1]))
    if len(prof) < 3:
        return float(np.max(depth)), float(hi - lo)
    prof = np.asarray(prof)
    yc = np.asarray(yc)
    peak = float(prof.max())
    above = np.where(prof >= 0.5 * max(peak, 1e-9))[0]
    width = float(yc[above[-1]] - yc[above[0]]) if len(above) else float(hi - lo)
    return peak, width


def _measure_on_residual(
    u: np.ndarray,
    v: np.ndarray,
    res: np.ndarray,
    idx: np.ndarray,
) -> dict:
    uu, vv, rr = u[idx], v[idx], res[idx]
    pts2 = np.stack([uu, vv], axis=1)
    mu = pts2.mean(0)
    _, _, vt = np.linalg.svd(pts2 - mu, full_matrices=False)
    # vt 行是主方向; 保证第 0 轴是长轴
    proj = (pts2 - mu) @ vt.T
    if np.std(proj[:, 0]) < np.std(proj[:, 1]):
        vt = vt[::-1]
        proj = proj[:, ::-1]
    xs, ys = proj[:, 0], proj[:, 1]

    x_lo, x_hi = np.percentile(xs, [10, 90])
    cuts = np.linspace(x_lo, x_hi, 8)
    sl = max(0.45, 0.08 * max(x_hi - x_lo, 1.0))
    slice_w, slice_d = [], []
    for xc in cuts:
        sm = np.abs(xs - xc) <= sl
        if sm.sum() < 8:
            continue
        pk, w = _profile_fwhm(ys[sm], rr[sm])
        if pk > 1e-6 and w > 1e-4:
            slice_d.append(pk)
            slice_w.append(w)
    if slice_d:
        depth = float(np.median(slice_d))
        width = float(np.median(slice_w))
    else:
        depth, width = _profile_fwhm(ys, rr)

    # 闭运算把噪声峰值抬进参考面。用沟槽两侧肩部残差做基线。
    if len(u) > len(idx):
        others = np.ones(len(u), dtype=bool)
        others[idx] = False
        rel = np.stack([u[others] - mu[0], v[others] - mu[1]], axis=1) @ vt.T
        half_len = max(float(np.percentile(xs, 95) - np.percentile(xs, 5)) * 0.5, 1.0)
        y0 = 0.55 * max(width, 0.25)
        y1 = y0 + 2.2
        band = (np.abs(rel[:, 0]) < half_len) & (np.abs(rel[:, 1]) >= y0) & (np.abs(rel[:, 1]) <= y1)
        if int(band.sum()) >= 40:
            base = float(np.median(res[np.where(others)[0][band]]))
            depth = max(0.0, depth - max(base, 0.0))

    core = rr >= 0.25 * max(depth, 1e-9)
    if core.sum() >= 6:
        length = float(np.percentile(xs[core], 99) - np.percentile(xs[core], 1))
    else:
        length = float(np.percentile(xs, 99) - np.percentile(xs, 1))
    area = float(max(length * width * 0.6, 0.0))
    return dict(length_mm=length, width_mm=width, depth_mm=depth, volume_mm3=area * depth)


def crop_around_points(xyz: np.ndarray, seed_idx: np.ndarray, margin_mm: float = 10.0) -> np.ndarray:
    seed = xyz[seed_idx]
    lo, hi = seed.min(axis=0), seed.max(axis=0)
    center = 0.5 * (lo + hi)
    half = 0.5 * (hi - lo) + margin_mm
    m = np.all((xyz >= center - half) & (xyz <= center + half), axis=1)
    return np.where(m)[0]


def _detect_tiled(xyz, u, v, **kw) -> Tuple[List[Scratch], np.ndarray, float]:
    """曲面起伏大时按局部小块检测, 避免整件投影把成形面当成缺陷。"""
    tile, margin = 42.0, 10.0
    pred = np.zeros(len(xyz), dtype=np.int32)
    scratches: List[Scratch] = []
    thrs = []
    sid = 0
    u0, v0 = float(u.min()), float(v.min())
    nu = int(np.ceil((u.max() - u0) / tile))
    nv = int(np.ceil((v.max() - v0) / tile))
    for iu in range(max(nu, 1)):
        for iv in range(max(nv, 1)):
            ua, ub = u0 + iu * tile, u0 + (iu + 1) * tile
            va, vb = v0 + iv * tile, v0 + (iv + 1) * tile
            inner = (u >= ua) & (u < ub) & (v >= va) & (v < vb)
            sel_m = (u >= ua - margin) & (u < ub + margin) & (v >= va - margin) & (v < vb + margin)
            sel = np.where(sel_m)[0]
            if inner.sum() < 200 or len(sel) < 400:
                continue
            sc, pr, thr = detect_scratches(xyz[sel], _allow_tile=False, **kw)
            thrs.append(thr)
            for s in sc:
                local = np.where(pr == s.id)[0]
                keep_local = local[inner[sel[local]]]
                if len(keep_local) < max(12, kw.get("min_points", 20) // 2):
                    continue
                gi = sel[keep_local]
                gi = gi[pred[gi] == 0]
                if len(gi) < max(12, kw.get("min_points", 20) // 2):
                    continue
                sid += 1
                pred[gi] = sid
                scratches.append(Scratch(
                    id=sid, n_points=int(len(gi)),
                    length_mm=s.length_mm, width_mm=s.width_mm, depth_mm=s.depth_mm,
                    volume_mm3=s.volume_mm3, center_xyz=s.center_xyz,
                    abnormal=s.abnormal, reason=s.reason,
                ))
    scratches.sort(key=lambda s: s.depth_mm, reverse=True)
    thr = float(np.median(thrs)) if thrs else kw.get("floor_mm", 0.02)
    return scratches, pred, thr


def detect_scratches(
    xyz: np.ndarray,
    k_nn: Optional[int] = None,          # 兼容旧参数, 忽略
    k_sigma: float = 3.4,
    floor_mm: float = 0.020,
    dbscan_eps: Optional[float] = None,  # 兼容旧参数, 忽略
    min_points: int = 20,
    width_abn: float = 0.30,
    depth_abn: float = 0.05,
    suppress_boundary: bool = True,      # 兼容旧参数: 网格边缘已抑制
    kernel_mm: Optional[float] = None,
    _allow_tile: bool = True,
) -> Tuple[List[Scratch], np.ndarray, float]:
    """返回 (划痕列表, 每点标签 0/1..K, 残差阈值 mm)。"""
    xyz = np.asarray(xyz, dtype=np.float64)
    if xyz.ndim != 2 or xyz.shape[1] < 3:
        raise ValueError("xyz 必须是 N x 3+")
    xyz = xyz[:, :3]
    spacing = estimate_spacing(xyz)
    pitch = float(np.clip(spacing, 0.12, 0.45))

    u, v, h, c, n, e1, e2 = _frame(xyz)
    h = _detrend_quadratic(u, v, h)
    grid = cnt = None
    umin = vmin = 0.0
    ui = vi = None
    for _ in range(6):
        grid, cnt, umin, vmin, pitch, ui, vi = _raster(u, v, h, pitch)
        if np.isfinite(grid).mean() >= 0.55 or pitch >= 0.85:
            break
        pitch = min(0.85, pitch * 1.4)
    rough = float(np.percentile(h, 95) - np.percentile(h, 5))
    extent = max(float(u.max() - u.min()), float(v.max() - v.min()))
    if _allow_tile and rough > 2.5 and extent > 60 and len(xyz) > 8000:
        return _detect_tiled(
            xyz, u, v, k_sigma=k_sigma, floor_mm=floor_mm, min_points=min_points,
            width_abn=width_abn, depth_abn=depth_abn, kernel_mm=kernel_mm,
        )
    res_g = _surface_residual(grid, pitch, _kernel_set(u, v, kernel_mm))
    res = res_g[vi, ui]

    valid_res = res[res > 0]
    thr = _robust_threshold(valid_res if valid_res.size else res, k_sigma=k_sigma, floor_mm=floor_mm)
    cand = res > thr

    pred = np.zeros(len(xyz), dtype=np.int32)
    scratches: List[Scratch] = []
    if cand.sum() < min_points:
        return scratches, pred, thr

    bin_g = res_g > thr
    lab_g, nlab = ndimage.label(bin_g)
    sid = 0
    for lab in range(1, nlab + 1):
        pix = lab_g == lab
        if pix.sum() < max(6, min_points // 3):
            continue
        di = np.where(lab_g[vi, ui] == lab)[0]
        if len(di) < min_points:
            continue
        # 丢掉比阈值高不了多少的噪声碎块
        if float(np.percentile(res[di], 80)) < 1.15 * thr:
            continue
        m = _measure_on_residual(u, v, res, di)
        abn = (m["width_mm"] >= width_abn) or (m["depth_mm"] >= depth_abn)
        reason = []
        if m["width_mm"] >= width_abn:
            reason.append(f"宽{m['width_mm']:.3f}>={width_abn}")
        if m["depth_mm"] >= depth_abn:
            reason.append(f"深{m['depth_mm']:.3f}>={depth_abn}")
        sid += 1
        pred[di] = sid
        scratches.append(Scratch(
            id=sid, n_points=int(len(di)),
            length_mm=round(m["length_mm"], 3), width_mm=round(m["width_mm"], 3),
            depth_mm=round(m["depth_mm"], 3), volume_mm3=round(m["volume_mm3"], 3),
            center_xyz=tuple(round(float(x), 3) for x in xyz[di].mean(axis=0)),
            abnormal=abn, reason=("; ".join(reason) if reason else "未超标"),
        ))
    scratches.sort(key=lambda s: s.depth_mm, reverse=True)
    return scratches, pred, thr


def iou_binary(gt: np.ndarray, pred: np.ndarray) -> float:
    g, p = gt.astype(bool), pred.astype(bool)
    inter = np.logical_and(g, p).sum()
    union = np.logical_or(g, p).sum()
    return float(inter / union) if union else 1.0


def match_scratch_by_gt(scratches: List[Scratch], pred: np.ndarray, gt: np.ndarray):
    if not scratches:
        return None, 0.0
    g = gt.astype(bool)
    best_iou, best = -1.0, scratches[0]
    for sc in scratches:
        pmask = pred == sc.id
        inter = np.logical_and(g, pmask).sum()
        union = np.logical_or(g, pmask).sum()
        ci = inter / union if union else 0.0
        if ci > best_iou:
            best_iou, best = ci, sc
    if best_iou <= 0:
        best = max(scratches, key=lambda x: x.n_points * max(x.depth_mm, 0.01))
    return best, float(max(best_iou, 0.0))


def to_json(scratches: List[Scratch], extra: Optional[dict] = None) -> dict:
    out = {
        "n_scratches": len(scratches),
        "n_abnormal": sum(s.abnormal for s in scratches),
        "scratches": [asdict(s) for s in scratches],
    }
    if extra:
        out.update(extra)
    return out
