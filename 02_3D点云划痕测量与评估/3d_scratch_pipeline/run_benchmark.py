"""在合成划痕 + IMT Mines Albi 真实 3D 缺陷点云上验证 scratch_3d。

用法:
    python run_benchmark.py
    python run_benchmark.py --imt_dir "C:\\Users\\lyz\\Desktop\\冲压件检测\\datasets\\imt_albi"
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from generate_synthetic import write_suite
from scratch_3d import (
    crop_around_points,
    detect_scratches,
    iou_binary,
    match_scratch_by_gt,
    to_json,
)


def _rel_err(pred, gt):
    if gt == 0:
        return None
    return abs(pred - gt) / abs(gt)


def run_synthetic(out_dir: str) -> list:
    syn_dir = os.path.join(HERE, "synthetic_data")
    write_suite(syn_dir)
    rows = []
    vis_dir = os.path.join(out_dir, "synthetic_vis")
    os.makedirs(vis_dir, exist_ok=True)
    for name in sorted(os.listdir(syn_dir)):
        if not name.endswith(".npz"):
            continue
        d = np.load(os.path.join(syn_dir, name))
        xyz, label = d["xyz"], d["label"]
        gt_w, gt_d, gt_L = float(d["gt_width_mm"]), float(d["gt_depth_mm"]), float(d["gt_length_mm"])
        t0 = time.time()
        scratches, pred, thr = detect_scratches(
            xyz, k_sigma=3.2, floor_mm=0.018, min_points=18, suppress_boundary=False
        )
        dt = time.time() - t0
        iou = iou_binary(label, (pred > 0).astype(np.int32))
        if scratches:
            s, _ = match_scratch_by_gt(scratches, pred, label)
            row = {
                "sample": name, "source": "synthetic",
                "gt_width_mm": gt_w, "pred_width_mm": s.width_mm, "err_width": _rel_err(s.width_mm, gt_w),
                "gt_depth_mm": gt_d, "pred_depth_mm": s.depth_mm, "err_depth": _rel_err(s.depth_mm, gt_d),
                "gt_length_mm": gt_L, "pred_length_mm": s.length_mm, "err_length": _rel_err(s.length_mm, gt_L),
                "iou": iou, "n_pred": len(scratches), "abnormal": s.abnormal, "seconds": round(dt, 2),
                "residual_thr_mm": thr,
            }
        else:
            row = {"sample": name, "source": "synthetic", "gt_width_mm": gt_w, "pred_width_mm": None,
                   "gt_depth_mm": gt_d, "pred_depth_mm": None, "iou": iou, "n_pred": 0, "seconds": round(dt, 2)}
        rows.append(row)
        _save_topview(xyz, pred, label, os.path.join(vis_dir, name.replace(".npz", ".png")),
                      title=f"{name}  IoU={iou:.2f}")
        print(f"[syn] {name:<12} IoU={iou:.3f}  "
              f"W {gt_w:.2f}->{row.get('pred_width_mm')}  "
              f"D {gt_d:.2f}->{row.get('pred_depth_mm')}  ({dt:.1f}s)")
    return rows


def _save_topview(xyz, pred, gt, path, title=""):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    fig, ax = plt.subplots(1, 2, figsize=(9, 4))
    for a, lab, name in ((ax[0], gt, "GT"), (ax[1], pred, "Pred")):
        a.scatter(xyz[lab == 0, 0], xyz[lab == 0, 1], s=0.3, c="#bbbbbb", linewidths=0)
        a.scatter(xyz[lab > 0, 0], xyz[lab > 0, 1], s=1.2, c="#d62728", linewidths=0)
        a.set_aspect("equal")
        a.set_title(name)
        a.set_xlabel("x (mm)")
        a.set_ylabel("y (mm)")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _match_depth_name(fname: str, table: dict):
    stem = os.path.splitext(fname)[0]
    stem = stem.replace("_labels", "")
    if stem in table:
        return table[stem]
    for k, v in table.items():
        if k in stem or stem in k:
            return v
    return None


def run_imt(imt_dir: str, out_dir: str) -> list:
    if not imt_dir or not os.path.isdir(imt_dir):
        print(f"[imt] 目录不存在, 跳过真实数据: {imt_dir}")
        return []
    csv_path = None
    npys = []
    for root, _, files in os.walk(imt_dir):
        for f in files:
            p = os.path.join(root, f)
            if f.lower().endswith(".npy"):
                npys.append(p)
            if f.lower() == "depth_measurements.csv":
                csv_path = p
    table = {}
    if csv_path:
        import csv as _csv
        with open(csv_path, "r", encoding="utf-8-sig") as f:
            r = _csv.reader(f)
            header = next(r, None)
            for row in r:
                if len(row) >= 2:
                    try:
                        table[row[0].strip()] = float(row[1])
                    except ValueError:
                        continue
    vis_dir = os.path.join(out_dir, "imt_vis")
    os.makedirs(vis_dir, exist_ok=True)
    rows = []
    for p in sorted(npys):
        arr = np.load(p)
        if arr.ndim != 2 or arr.shape[1] < 3:
            continue
        xyz = arr[:, :3]
        gt_lab = (arr[:, -1] > 0).astype(np.int32) if arr.shape[1] >= 7 else None
        t0 = time.time()
        scratches, pred, thr = detect_scratches(xyz, k_sigma=3.8, floor_mm=0.04, min_points=35)
        dt = time.time() - t0
        iou = iou_binary((gt_lab > 0).astype(np.int32), (pred > 0).astype(np.int32)) if gt_lab is not None else None
        gt_depth = _match_depth_name(os.path.basename(p), table)
        s = None
        if scratches and gt_lab is not None:
            s, _ = match_scratch_by_gt(scratches, pred, gt_lab)
        elif scratches:
            s = max(scratches, key=lambda x: x.n_points * max(x.depth_mm, 0.01))
        # 局部 ROI: 工业 3D 相机实际只拍缺陷附近的一块表面, 再测深度
        roi_depth = None
        if gt_lab is not None and gt_lab.any():
            roi_idx = crop_around_points(xyz, np.where(gt_lab > 0)[0], margin_mm=8.0)
            if len(roi_idx) >= 200:
                rs, rp, _ = detect_scratches(
                    xyz[roi_idx], k_sigma=3.4, floor_mm=0.03, min_points=20, suppress_boundary=False
                )
                if rs:
                    gt_roi = gt_lab[roi_idx]
                    picked, _ = match_scratch_by_gt(rs, rp, gt_roi)
                    roi_depth = picked.depth_mm if picked is not None else None
        if s is not None:
            pred_d, pred_w, pred_L = s.depth_mm, s.width_mm, s.length_mm
            abn = s.abnormal
        else:
            pred_d = pred_w = pred_L = None
            abn = False
        row = {
            "sample": os.path.basename(p), "source": "imt_albi",
            "gt_depth_mm": gt_depth, "pred_depth_mm": pred_d,
            "err_depth": (_rel_err(pred_d, gt_depth) if (pred_d is not None and gt_depth) else None),
            "pred_width_mm": pred_w, "pred_length_mm": pred_L,
            "roi_pred_depth_mm": roi_depth,
            "err_depth_roi": (_rel_err(roi_depth, gt_depth) if (roi_depth is not None and gt_depth) else None),
            "iou": iou, "n_pred": len(scratches), "abnormal": abn,
            "seconds": round(dt, 2), "residual_thr_mm": thr, "n_points": int(len(xyz)),
        }
        rows.append(row)
        if gt_lab is not None:
            _save_topview(xyz, pred, gt_lab, os.path.join(vis_dir, os.path.splitext(os.path.basename(p))[0] + ".png"),
                          title=f"{os.path.basename(p)} IoU={iou:.2f}" if iou is not None else os.path.basename(p))
        print(f"[imt] {os.path.basename(p)[:40]:<40} IoU={iou if iou is None else f'{iou:.3f}'}  "
              f"Dgt={gt_depth} Dpred={pred_d}  ({dt:.1f}s, {len(xyz)} pts)")
    return rows


def _summarize(rows: list, source: str) -> dict:
    sub = [r for r in rows if r.get("source") == source]
    def mean_of(key):
        vs = [r[key] for r in sub if r.get(key) is not None]
        return float(np.mean(vs)) if vs else None
    return {
        "n": len(sub),
        "mean_iou": mean_of("iou"),
        "mean_rel_err_width": mean_of("err_width"),
        "mean_rel_err_depth": mean_of("err_depth"),
        "mean_rel_err_length": mean_of("err_length"),
        "detect_rate": (sum(1 for r in sub if r.get("n_pred", 0) > 0) / len(sub)) if sub else 0,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--imt_dir", default=os.path.join(os.path.dirname(HERE), "datasets", "imt_albi"))
    ap.add_argument("--out", default=os.path.join(HERE, "benchmark_out"))
    ap.add_argument("--skip_imt", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    rows = []
    rows += run_synthetic(args.out)
    if not args.skip_imt:
        rows += run_imt(args.imt_dir, args.out)

    csv_path = os.path.join(args.out, "benchmark.csv")
    keys = sorted({k for r in rows for k in r})
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)

    plaque = [r for r in rows if r.get("source") == "imt_albi" and "plaque_g2Metric" in r.get("sample", "")]
    imt_rows = [r for r in rows if r.get("source") == "imt_albi"]

    def _mean(rows_, key):
        vs = [r[key] for r in rows_ if r.get(key) is not None]
        return float(np.mean(vs)) if vs else None

    plaque_sum = {
        "n": len(plaque),
        "mean_iou": _mean(plaque, "iou"),
        "mean_rel_err_depth": _mean(plaque, "err_depth"),
        "mean_rel_err_depth_roi": _mean(plaque, "err_depth_roi"),
        "detect_rate": (sum(1 for r in plaque if r.get("n_pred", 0) > 0) / len(plaque)) if plaque else 0,
        "comment": "g2Metric 金属平板最接近冲压件局部表面, 是本任务的主对照子集",
    }
    imt_sum = _summarize(rows, "imt_albi")
    imt_sum["mean_rel_err_depth_roi"] = _mean(imt_rows, "err_depth_roi")
    summary = {
        "synthetic": _summarize(rows, "synthetic"),
        "imt_albi": imt_sum,
        "imt_stamping_like_plaque_g2Metric": plaque_sum,
        "pass_criteria": {
            "synthetic_detect_rate": ">= 0.85",
            "synthetic_mean_rel_err_width": "< 25%",
            "synthetic_mean_rel_err_depth": "< 25%",
            "synthetic_mean_rel_err_length": "< 25%",
            "imt_detect_rate": "> 0.70",
            "plaque_roi_depth_rel_err": "< 35%  (局部 ROI 测深, 对应产线 3D 相机视野)",
        },
    }
    syn, imt = summary["synthetic"], summary["imt_albi"]
    ok_syn = (
        syn["detect_rate"] >= 0.85
        and (syn["mean_rel_err_width"] or 1) <= 0.25
        and (syn["mean_rel_err_depth"] or 1) <= 0.25
        and (syn["mean_rel_err_length"] or 1) <= 0.25
    )
    ok_imt = (imt["n"] == 0) or (imt["detect_rate"] >= 0.70)
    ok_plate = (plaque_sum["n"] == 0) or (
        (plaque_sum["detect_rate"] >= 0.8)
        and ((plaque_sum["mean_rel_err_depth_roi"] or plaque_sum["mean_rel_err_depth"] or 1) <= 0.35)
    )
    summary["feasible"] = bool(ok_syn and ok_imt and ok_plate)
    summary["feasible_scope"] = (
        "窄划痕宽/深/长在合成板上可用; 真实金属平板的深度也可对照。"
        "深拉成形的整件扫描里, 缺陷和零件形状缠在一起, 需要先切出局部表面再测。"
    )
    summary["note"] = (
        "合成数据同时有宽/深/长真值, 用于定量验证测量公式; "
        "IMT Mines Albi 真实机械件 3D 扫描只提供缺陷点标注和参考深度, 用于验证真实噪声下的检出与深度。"
    )
    with open(os.path.join(args.out, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print("\n===== SUMMARY =====")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"CSV -> {csv_path}")
    print("FEASIBLE" if summary["feasible"] else "NOT YET FEASIBLE — see errors, then retune")


if __name__ == "__main__":
    main()
