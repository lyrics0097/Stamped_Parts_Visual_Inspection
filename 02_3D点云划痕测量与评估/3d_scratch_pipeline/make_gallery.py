"""生成「高度原图 | 真值 | 检测结果」放大对照图。"""
from __future__ import annotations

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from scratch_3d import _detrend_quadratic, _frame, _raster, detect_scratches, match_scratch_by_gt


def _zoom_box(ui, vi, mask, shape, pad=18):
    if mask.any():
        ys, xs = np.where(mask) if mask.ndim == 2 else (vi[mask], ui[mask])
        r0, r1 = int(ys.min()), int(ys.max())
        c0, c1 = int(xs.min()), int(xs.max())
    else:
        r0, r1, c0, c1 = 0, shape[0] - 1, 0, shape[1] - 1
    r0 = max(0, r0 - pad)
    c0 = max(0, c0 - pad)
    r1 = min(shape[0] - 1, r1 + pad)
    c1 = min(shape[1] - 1, c1 + pad)
    return r0, r1, c0, c1


def render_one(xyz, gt, pred, title, path, measured=""):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    u, v, h, *_ = _frame(xyz)
    h = _detrend_quadratic(u, v, h)
    pitch = 0.25
    grid, _, _, _, pitch, ui, vi = _raster(u, v, h, pitch)
    finite = np.isfinite(grid)
    show = np.where(finite, grid, np.nan)
    gt_img = np.zeros(grid.shape, np.uint8)
    pred_img = np.zeros(grid.shape, np.uint8)
    if gt is not None:
        gt_img[vi[gt > 0], ui[gt > 0]] = 1
    pred_img[vi[pred > 0], ui[pred > 0]] = 1
    focus = (gt_img > 0) | (pred_img > 0)
    r0, r1, c0, c1 = _zoom_box(ui, vi, focus, grid.shape)
    sl = (slice(r0, r1 + 1), slice(c0, c1 + 1))

    fig, ax = plt.subplots(1, 3, figsize=(12.2, 4.4))
    panels = [
        (show[sl], "原图（高度，越亮越凸）", "viridis"),
    ]
    im0 = ax[0].imshow(show[sl], cmap="viridis", origin="lower")
    ax[0].set_title("原图（高度）")
    fig.colorbar(im0, ax=ax[0], fraction=0.046, pad=0.04, label="mm")

    for a, mask, name, color in (
        (ax[1], gt_img[sl], "真值缺陷", "#ff2a2a"),
        (ax[2], pred_img[sl], "检测结果", "#ff2a2a"),
    ):
        a.imshow(show[sl], cmap="gray", origin="lower")
        overlay = np.zeros(mask.shape + (4,))
        overlay[mask > 0] = (1.0, 0.15, 0.1, 0.85)
        a.imshow(overlay, origin="lower")
        a.set_title(name)
    for a in ax:
        a.set_xticks([])
        a.set_yticks([])
    fig.suptitle(title + (("\n" + measured) if measured else ""), fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def main():
    out = os.path.join(HERE, "benchmark_out", "对照图")
    os.makedirs(out, exist_ok=True)
    syn = os.path.join(HERE, "synthetic_data")
    for name in ["syn_01.npz", "syn_03.npz", "syn_04.npz", "syn_07.npz"]:
        z = np.load(os.path.join(syn, name))
        xyz, gt = z["xyz"], z["label"]
        sc, pred, _ = detect_scratches(xyz, k_sigma=3.2, floor_mm=0.018, min_points=18)
        s, _ = match_scratch_by_gt(sc, pred, gt)
        measured = ""
        if s is not None:
            measured = (
                f"真值 宽{float(z['gt_width_mm']):.2f} 深{float(z['gt_depth_mm']):.2f} 长{float(z['gt_length_mm']):.1f} mm"
                f"    测得 宽{s.width_mm:.2f} 深{s.depth_mm:.2f} 长{s.length_mm:.2f} mm"
            )
        render_one(xyz, gt, pred, f"合成划痕 {name}", os.path.join(out, name.replace('.npz', '.png')), measured)
        print("wrote", name)

    imt = os.path.join(os.path.dirname(HERE), "datasets", "imt_albi")
    csv_map = {
        "plaque_g2Metric-SPIDER-scan_completD3_labels.npy": 0.37,
        "plaque_g2Metric-SPIDER-scan_completD5_labels.npy": 0.53,
        "plaque_g2Metric-SPIDER-scan_completD6_labels.npy": 0.41,
        "plaque_MoireView-SPIDER-scan_completD3_labels.npy": 0.92,
        "tole_metallique-SPIDER-scan_completD7_labels.npy": 1.99,
    }
    for name, gt_d in csv_map.items():
        a = np.load(os.path.join(imt, name))
        xyz, gt = a[:, :3], (a[:, -1] > 0).astype(np.int32)
        sc, pred, _ = detect_scratches(xyz, floor_mm=0.03, min_points=30)
        s, _ = match_scratch_by_gt(sc, pred, gt)
        measured = f"参考深度 {gt_d:.2f} mm"
        if s is not None:
            measured += f"    测得深度 {s.depth_mm:.2f} mm   宽 {s.width_mm:.2f} mm   长 {s.length_mm:.2f} mm"
        short = name.replace("_labels.npy", "")
        render_one(xyz, gt, pred, short, os.path.join(out, short + ".png"), measured)
        print("wrote", short)


if __name__ == "__main__":
    main()
