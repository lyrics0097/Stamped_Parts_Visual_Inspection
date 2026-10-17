"""生成带真值的合成冲压件曲面 + 划痕点云 (模拟结构光 3D 相机)。

曲面: 轻度弯曲的金属板 (二次曲面), 叠加高斯噪声。
划痕: 沿曲线挖一条高斯截面的沟槽, 真值宽/深已知。
"""
from __future__ import annotations

import json
import os

import numpy as np


def make_plate_with_scratch(
    width_mm: float = 80.0,
    height_mm: float = 50.0,
    pitch: float = 0.15,
    scratch_width: float = 0.80,
    scratch_depth: float = 0.25,
    scratch_length: float = 28.0,
    noise_sigma: float = 0.012,
    seed: int = 0,
) -> dict:
    rng = np.random.default_rng(seed)
    xs = np.arange(-width_mm / 2, width_mm / 2, pitch)
    ys = np.arange(-height_mm / 2, height_mm / 2, pitch)
    xx, yy = np.meshgrid(xs, ys)
    # 轻微弯曲 + 倾斜, 模拟冲压件局部曲面
    zz = 0.0012 * xx ** 2 - 0.0004 * yy ** 2 + 0.02 * xx
    # 划痕中心线: y = 0.08 * x (略斜)
    a = 0.08
    dist = np.abs(yy - a * xx) / np.sqrt(1 + a ** 2)
    along = (xx + a * yy) / np.sqrt(1 + a ** 2)
    in_len = np.abs(along) < scratch_length / 2
    # 高斯截面: 半高全宽 ≈ scratch_width
    sigma_w = scratch_width / (2 * np.sqrt(2 * np.log(2)))
    groove = scratch_depth * np.exp(-0.5 * (dist / sigma_w) ** 2)
    groove = np.where(in_len, groove, 0.0)
    zz = zz - groove
    zz = zz + rng.normal(0, noise_sigma, zz.shape)

    xyz = np.stack([xx.ravel(), yy.ravel(), zz.ravel()], axis=1)
    # 标签: 沟槽深度超过 20% 真值深度 且 在长度范围内
    label = ((groove > 0.20 * scratch_depth) & in_len).ravel().astype(np.int32)
    return {
        "xyz": xyz.astype(np.float64),
        "label": label,
        "gt": {
            "width_mm": scratch_width,
            "depth_mm": scratch_depth,
            "length_mm": scratch_length,
            "noise_sigma_mm": noise_sigma,
            "pitch_mm": pitch,
        },
    }


def write_suite(out_dir: str, n: int = 8) -> str:
    os.makedirs(out_dir, exist_ok=True)
    catalog = []
    specs = [
        (0.40, 0.08, 18.0, 1),
        (0.60, 0.12, 22.0, 2),
        (0.80, 0.20, 28.0, 3),
        (1.20, 0.35, 32.0, 4),
        (0.50, 0.06, 16.0, 5),
        (1.00, 0.18, 24.0, 6),
        (0.70, 0.45, 20.0, 7),
        (1.50, 0.25, 36.0, 8),
    ]
    for i, (w, d, L, seed) in enumerate(specs[:n], 1):
        rec = make_plate_with_scratch(scratch_width=w, scratch_depth=d, scratch_length=L, seed=seed)
        path = os.path.join(out_dir, f"syn_{i:02d}.npz")
        np.savez_compressed(path, xyz=rec["xyz"], label=rec["label"], **{f"gt_{k}": v for k, v in rec["gt"].items()})
        catalog.append({"file": os.path.basename(path), **rec["gt"]})
    meta = os.path.join(out_dir, "catalog.json")
    with open(meta, "w", encoding="utf-8") as f:
        json.dump(catalog, f, ensure_ascii=False, indent=2)
    return meta


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    write_suite(os.path.join(here, "synthetic_data"))
    print("synthetic_data written")
