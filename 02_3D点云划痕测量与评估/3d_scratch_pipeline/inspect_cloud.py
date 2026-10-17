"""对工业 3D 相机点云做划痕检测并导出结果。

支持: .npy / .npz / .txt / .xyz / .ply(ascii)
单位必须是毫米。

用法:
    python inspect_cloud.py --input part.npy --out_dir 结果
    python inspect_cloud.py --input part.ply --out_dir 结果 --floor_mm 0.02
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scratch_3d import detect_scratches, iou_binary, to_json


def load_xyz(path: str):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".npy":
        a = np.load(path)
        lab = (a[:, -1] > 0).astype(np.int32) if a.ndim == 2 and a.shape[1] >= 7 else None
        return a[:, :3], lab
    if ext == ".npz":
        d = np.load(path)
        return d["xyz"], d["label"] if "label" in d.files else None
    if ext == ".ply":
        return _load_ply_xyz(path), None
    a = np.loadtxt(path)
    return a[:, :3], None


def _load_ply_xyz(path: str) -> np.ndarray:
    try:
        from plyfile import PlyData
        ply = PlyData.read(path)
        v = ply["vertex"]
        return np.stack([v["x"], v["y"], v["z"]], axis=1).astype(np.float64)
    except Exception:
        pass
    xs, ys, zs, header = [], [], [], True
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if header:
                if line.strip().lower() == "end_header":
                    header = False
                continue
            parts = line.split()
            if len(parts) >= 3:
                xs.append(float(parts[0]))
                ys.append(float(parts[1]))
                zs.append(float(parts[2]))
    if not xs:
        raise ValueError(f"无法解析 PLY: {path}")
    return np.stack([xs, ys, zs], axis=1)


def _save_topview(xyz, pred, path, title=""):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(xyz[pred == 0, 0], xyz[pred == 0, 1], s=0.3, c="#bbbbbb", linewidths=0)
    ax.scatter(xyz[pred > 0, 0], xyz[pred > 0, 1], s=1.4, c="#d62728", linewidths=0)
    ax.set_aspect("equal")
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--out_dir", default="")
    ap.add_argument("--floor_mm", type=float, default=0.025)
    ap.add_argument("--width_abn", type=float, default=0.30)
    ap.add_argument("--depth_abn", type=float, default=0.05)
    args = ap.parse_args()
    xyz, lab = load_xyz(args.input)
    scratches, pred, thr = detect_scratches(
        xyz, floor_mm=args.floor_mm, width_abn=args.width_abn, depth_abn=args.depth_abn
    )
    extra = {
        "file": os.path.abspath(args.input),
        "n_points": int(len(xyz)),
        "residual_thr_mm": float(thr),
    }
    if lab is not None:
        extra["iou_vs_gt"] = iou_binary(lab, pred)
    report = to_json(scratches, extra)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.out_dir:
        os.makedirs(args.out_dir, exist_ok=True)
        jpath = os.path.join(args.out_dir, "result.json")
        with open(jpath, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        _save_topview(xyz, pred, os.path.join(args.out_dir, "overlay.png"),
                      title=os.path.basename(args.input))
        print(f"saved -> {jpath}")


if __name__ == "__main__":
    main()
