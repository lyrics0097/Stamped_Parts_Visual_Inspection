"""对单个 npy / npz / txt 点云做划痕检测, 打印结果并可选保存 JSON。"""
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
        xyz, lab = a[:, :3], (a[:, -1] > 0).astype(np.int32) if a.shape[1] >= 7 else None
        return xyz, lab
    if ext == ".npz":
        d = np.load(path)
        return d["xyz"], d["label"] if "label" in d.files else None
    a = np.loadtxt(path)
    return a[:, :3], None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npy", required=True)
    ap.add_argument("--json_out", default="")
    args = ap.parse_args()
    xyz, lab = load_xyz(args.npy)
    scratches, pred, thr = detect_scratches(xyz)
    extra = {"file": args.npy, "n_points": int(len(xyz)), "residual_thr_mm": thr}
    if lab is not None:
        extra["iou_vs_gt"] = iou_binary(lab, pred)
    report = to_json(scratches, extra)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
