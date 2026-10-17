#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Continuous-scratch detection and line-based measurement on NEU-SEG.

A continuous scratch is one connected stroke at least 60 px long with
length / width >= 5. Only those count, in the annotation and in the output.
A prediction that lands on a shorter or stubbier annotated scratch is ignored
instead of being counted as a false alarm.

On top of the scratch-only UNet (results_scratch/scratch_unet.pt):
  * two parallel scratches fused into one blob are split apart again,
  * faint ends are recovered by walking outward along the stroke while the
    network still gives some evidence,
  * width can be read from the grey-level cross-section instead of the mask.
Every parameter is chosen on the validation split; test is only reported.
"""

import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse
import json

import cv2
import numpy as np
import torch

import detect_scratch as ds
from detect_and_measure import ROOT, read_split, write_csv
from measure_defects import render_measurement, write_image

MIN_LENGTH = 60.0
MIN_ASPECT = 5.0
MATCH_IOU = 0.3
OUT_DIR = os.path.join(ROOT, "results_continuous")
BASELINE = {"threshold": 0.8, "min_area": 60, "split": False, "ext_t": None, "ext_miss": 0, "width": "mask"}


def is_continuous(record):
    return record["length_px"] >= MIN_LENGTH and record["length_px"] / max(record["width_px"], 1.0) >= MIN_ASPECT


# ---------------------------------------------------------------- stroke geometry


def stroke_axis(part):
    ys, xs = np.nonzero(part)
    points = np.stack([xs, ys], axis=1).astype(np.float64)
    center = points.mean(axis=0)
    if len(points) < 3:
        direction = np.array([1.0, 0.0])
    else:
        _, vectors = np.linalg.eigh(np.cov((points - center).T))
        direction = vectors[:, -1]
    normal = np.array([-direction[1], direction[0]])
    proj = (points - center) @ direction
    perp = (points - center) @ normal
    halfwidth = max(1.0, float(np.percentile(np.abs(perp), 90)))
    return {
        "center": center, "dir": direction, "normal": normal,
        "pmin": float(proj.min()), "pmax": float(proj.max()),
        "proj": proj, "perp": perp, "halfwidth": halfwidth,
    }


def elongated(part, ratio=3.0):
    axis = stroke_axis(part)
    return (axis["pmax"] - axis["pmin"] + 1.0) >= ratio * 2.0 * axis["halfwidth"]


# ---------------------------------------------------------------- split fused parallel scratches


def _side_by_side(parts, max_angle=12.0):
    axes = [stroke_axis(p) for p in parts]
    cos_limit = np.cos(np.deg2rad(max_angle))
    for i in range(len(axes)):
        for j in range(i + 1, len(axes)):
            a, b = axes[i], axes[j]
            if abs(float(a["dir"] @ b["dir"])) < cos_limit:
                continue
            offset = abs(float((b["center"] - a["center"]) @ a["normal"]))
            if offset > max(2.0 * a["halfwidth"], 2.0 * b["halfwidth"], 4.0):
                return True
    return False


def split_parallel(parts, prob, min_area, levels=(0.9, 0.95, 0.98)):
    result = []
    for part in parts:
        pieces = None
        for level in levels:
            core = ((prob >= level) & part).astype(np.uint8)
            candidates = [p for p in ds.components(core, max(10, min_area // 2)) if elongated(p)]
            if len(candidates) >= 2 and _side_by_side(candidates):
                pieces = candidates
                break
        if pieces is None:
            result.append(part)
            continue
        distances = np.stack([
            cv2.distanceTransform((~p).astype(np.uint8), cv2.DIST_L2, 3) for p in pieces
        ])
        owner = distances.argmin(axis=0)
        for index in range(len(pieces)):
            piece = part & (owner == index)
            if piece.sum() >= min_area:
                result.append(piece)
    return result


# ---------------------------------------------------------------- extend faint ends


def extend_ends(parts, prob, t_ext, max_miss, max_steps=400):
    height, width = prob.shape
    labels = np.zeros(prob.shape, dtype=np.int32)
    for index, part in enumerate(parts, start=1):
        labels[part] = index
    extended = []
    for index, part in enumerate(parts, start=1):
        axis = stroke_axis(part)
        if (axis["pmax"] - axis["pmin"] + 1.0) < 3.0 * 2.0 * axis["halfwidth"]:
            extended.append(part)
            continue
        radius = int(np.ceil(axis["halfwidth"])) + 1
        thickness = max(1, int(round(2.0 * axis["halfwidth"])))
        canvas = np.zeros(prob.shape, dtype=np.uint8)
        for sign, start in ((-1.0, axis["pmin"]), (1.0, axis["pmax"])):
            step = axis["dir"] * sign
            origin = axis["center"] + axis["dir"] * start
            current = origin.copy()
            last_good = origin.copy()
            misses = 0
            for _ in range(max_steps):
                current = current + step
                if not (0 <= current[0] < width and 0 <= current[1] < height):
                    break
                best_value, best_offset, blocked = -1.0, 0.0, False
                for offset in range(-radius, radius + 1):
                    x, y = current + axis["normal"] * offset
                    ix, iy = int(round(x)), int(round(y))
                    if not (0 <= ix < width and 0 <= iy < height):
                        continue
                    owner = labels[iy, ix]
                    if owner and owner != index:
                        blocked = True
                    if prob[iy, ix] > best_value:
                        best_value, best_offset = float(prob[iy, ix]), float(offset)
                if blocked:
                    break
                if best_value >= t_ext:
                    current = current + axis["normal"] * float(np.clip(best_offset, -1.0, 1.0))
                    last_good = current.copy()
                    misses = 0
                else:
                    misses += 1
                    if misses > max_miss:
                        break
            if np.linalg.norm(last_good - origin) >= 1.0:
                cv2.line(
                    canvas,
                    (int(round(origin[0])), int(round(origin[1]))),
                    (int(round(last_good[0])), int(round(last_good[1]))),
                    1, thickness,
                )
        grown = part | ((canvas > 0) & (labels == 0))
        extended.append(grown)
    return extended


# ---------------------------------------------------------------- width from grey-level cross-section


def profile_width(gray, part, mask_width, min_contrast=4.0):
    axis = stroke_axis(part)
    span = axis["pmax"] - axis["pmin"]
    if span < 10:
        return None
    radius = max(6.0, 2.5 * mask_width)
    offsets = np.arange(-radius, radius + 1e-6, 0.25, dtype=np.float64)
    gray_f = gray.astype(np.float32)
    widths = []
    for t in np.arange(axis["pmin"] + 0.1 * span, axis["pmax"] - 0.1 * span + 1e-6, 2.0):
        near = np.abs(axis["proj"] - t) < 1.0
        if not near.any():
            continue
        base = axis["center"] + axis["dir"] * t + axis["normal"] * float(axis["perp"][near].mean())
        xs = (base[0] + axis["normal"][0] * offsets).astype(np.float32).reshape(1, -1)
        ys = (base[1] + axis["normal"][1] * offsets).astype(np.float32).reshape(1, -1)
        profile = cv2.remap(gray_f, xs, ys, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)[0].astype(np.float64)
        tail = max(2, len(profile) // 5)
        background = float(np.median(np.concatenate([profile[:tail], profile[-tail:]])))
        middle = len(profile) // 2
        reach = max(2, int(round(mask_width / 0.25)))
        core = profile[max(0, middle - reach): middle + reach + 1]
        sign = 1.0 if core.mean() >= background else -1.0
        signal = sign * (profile - background)
        window = slice(max(0, middle - reach), min(len(profile), middle + reach + 1))
        peak = window.start + int(np.argmax(signal[window]))
        amplitude = signal[peak]
        if amplitude < min_contrast:
            continue
        half = amplitude / 2.0
        left = peak
        while left > 0 and signal[left - 1] >= half:
            left -= 1
        right = peak
        while right < len(signal) - 1 and signal[right + 1] >= half:
            right += 1
        if left == 0 or right == len(signal) - 1:
            continue
        left_pos = offsets[left - 1] + (half - signal[left - 1]) / (signal[left] - signal[left - 1]) * 0.25
        right_pos = offsets[right] + (signal[right] - half) / (signal[right] - signal[right + 1]) * 0.25
        widths.append(right_pos - left_pos)
    if len(widths) < 3:
        return None
    return float(np.median(widths))


# ---------------------------------------------------------------- pipeline


def any_scratch(record):
    return True


def detect(prob, cfg):
    mask = ds.threshold_mask(prob, cfg["threshold"], cfg["min_area"])
    parts = ds.components(mask, cfg["min_area"])
    if cfg["split"]:
        parts = split_parallel(parts, prob, cfg["min_area"])
    if cfg["ext_t"] is not None:
        parts = extend_ends(parts, prob, cfg["ext_t"], cfg["ext_miss"])
    found = []
    for part in parts:
        record = ds.measure(part)
        if record is not None:
            found.append((part, record))
    return found


def final_width(gray, part, record, mode, calib):
    if mode == "mask":
        return record["width_px"]
    measured = profile_width(gray, part, record["width_px"])
    if measured is None:
        return record["width_px"]
    if mode == "fwhm":
        return measured
    slope, intercept = calib
    return max(1.0, slope * measured + intercept)


def _iou(a, b):
    union = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum()) / float(union + 1e-6)


def cache_split(model, device, split):
    cached = []
    for stem in read_split(split):
        gray = ds.load_gray(stem)
        truth = ds.load_truth(stem)
        item = {"stem": stem, "gray": gray, "gt": ground_truth(truth)}
        if model is not None:
            item["prob"] = ds.predict_prob(model, gray, device)
        cached.append(item)
    return cached


def ground_truth(truth):
    gt = []
    for part in ds.components(truth):
        record = ds.measure(part)
        if record is not None:
            gt.append((part, record))
    return gt


def assess(cached, cfg, calib=None, keep=False, predicate=is_continuous):
    return score_found([(item, detect(item["prob"], cfg)) for item in cached], keep=keep, predicate=predicate)


def score_found(items_found, keep=False, predicate=is_continuous):
    """Score predictions against annotation, counting only scratches that satisfy predicate.

    items_found: list of (item, found) where found is a list of (mask, record).
    """
    tp = fp = fn = 0
    pairs_out = []
    rows = []
    for item, found in items_found:
        found = [(p, r) for p, r in found if predicate(r)]
        gt_cont = [(p, r) for p, r in item["gt"] if predicate(r)]
        gt_ignore = [p for p, r in item["gt"] if not predicate(r)]
        pairs = ds.match([p for p, _ in gt_cont], [p for p, _ in found])
        matched_pred = {pred_index for _, pred_index, _ in pairs}
        tp += len(pairs)
        fn += len(gt_cont) - len(pairs)
        for index, (part, _) in enumerate(found):
            if index in matched_pred:
                continue
            area = part.sum()
            ignored = any(
                _iou(part, g) >= MATCH_IOU or np.logical_and(part, g).sum() >= 0.5 * area for g in gt_ignore
            )
            if not ignored:
                fp += 1
        for gt_index, pred_index, iou in pairs:
            gt_part, gt_record = gt_cont[gt_index]
            pred_part, pred_record = found[pred_index]
            pairs_out.append({
                "stem": item["stem"], "iou": iou, "gray": item["gray"],
                "pred_part": pred_part, "pred": pred_record, "gt": gt_record,
            })
        if keep:
            rows.append((item, found))
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    result = {
        "tp": tp, "fp": fp, "fn": fn,
        "precision": precision, "recall": recall,
        "f1": 2 * precision * recall / max(precision + recall, 1e-6),
    }
    if pairs_out:
        length_err = np.array([abs(p["pred"]["length_px"] - p["gt"]["length_px"]) for p in pairs_out])
        result.update({
            "length_err_mean": float(length_err.mean()),
            "length_err_median": float(np.median(length_err)),
            "length_err_p90": float(np.percentile(length_err, 90)),
        })
    return result, pairs_out, rows


def width_stats(pairs, mode, calib=None):
    errors, relative = [], []
    for pair in pairs:
        width = final_width(pair["gray"], pair["pred_part"], pair["pred"], mode, calib)
        error = abs(width - pair["gt"]["width_px"])
        errors.append(error)
        relative.append(error / max(pair["gt"]["width_px"], 1.0))
    errors = np.array(errors)
    return {
        "width_err_mean": float(errors.mean()),
        "width_err_median": float(np.median(errors)),
        "width_err_p90": float(np.percentile(errors, 90)),
        "width_rel_median": float(np.median(relative)),
        "width_within_1px": float((errors <= 1).mean()),
        "width_within_2px": float((errors <= 2).mean()),
    }


def fit_calibration(pairs):
    measured, truth = [], []
    for pair in pairs:
        value = profile_width(pair["gray"], pair["pred_part"], pair["pred"]["width_px"])
        if value is not None:
            measured.append(value)
            truth.append(pair["gt"]["width_px"])
    slope, intercept = np.polyfit(np.array(measured), np.array(truth), 1)
    return float(slope), float(intercept)


def line(name, result):
    print(
        "%-34s 精确率 %.3f 召回率 %.3f F1 %.3f | 长度误差 平均 %.2f 中位 %.2f P90 %.1f"
        % (name, result["precision"], result["recall"], result["f1"],
           result.get("length_err_mean", float("nan")), result.get("length_err_median", float("nan")),
           result.get("length_err_p90", float("nan")))
    )


def clearly_better(new, ref):
    """Validation differences below these margins are within split-to-split noise."""
    if new["f1"] >= ref["f1"] + 0.01:
        return True
    return new["f1"] >= ref["f1"] - 0.002 and new["length_err_mean"] <= ref["length_err_mean"] - 1.0


def tune(val):
    base_result, _, _ = assess(val, BASELINE)
    line("验证 基线", base_result)
    chosen = (BASELINE, base_result)
    for threshold in (0.7, 0.8, 0.9):
        for split in (False, True):
            cfg = dict(BASELINE, threshold=threshold, split=split)
            result, _, _ = assess(val, cfg)
            line("验证 阈值%.1f 拆并排=%s" % (threshold, split), result)
            if clearly_better(result, chosen[1]):
                chosen = (cfg, result)
    for ext_t in (0.5, 0.3, 0.15):
        for ext_miss in (2, 5):
            trial = dict(chosen[0], ext_t=ext_t, ext_miss=ext_miss)
            trial_result, _, _ = assess(val, trial)
            line("验证 延伸 阈值%.2f 容忍%d" % (ext_t, ext_miss), trial_result)
            if clearly_better(trial_result, chosen[1]):
                chosen = (trial, trial_result)
    cfg, result = chosen
    _, pairs, _ = assess(val, cfg)
    calib = fit_calibration(pairs)
    widths = {mode: width_stats(pairs, mode, calib) for mode in ("mask", "fwhm", "fwhm_cal")}
    for mode, stats in widths.items():
        print("验证 宽度方法 %-8s 平均 %.2f 中位 %.2f P90 %.2f" % (
            mode, stats["width_err_mean"], stats["width_err_median"], stats["width_err_p90"]))
    width_mode = "mask"
    for mode in ("fwhm", "fwhm_cal"):
        if widths[mode]["width_err_mean"] <= widths[width_mode]["width_err_mean"] - 0.2:
            width_mode = mode
    cfg = dict(cfg, width=width_mode)
    print("验证集选定 %s  灰度标定 %.3f*x%+.3f" % (json.dumps(cfg), calib[0], calib[1]))
    return cfg, calib, {"val": result, "val_width": widths[cfg["width"]]}


def summarize(cached, cfg, calib):
    result, pairs, rows = assess(cached, cfg, calib, keep=True)
    return finish_summary(result, pairs, rows, cfg["width"], calib)


def finish_summary(result, pairs, rows, width_mode="mask", calib=None):
    result.update(width_stats(pairs, width_mode, calib))
    length_err = np.array([abs(p["pred"]["length_px"] - p["gt"]["length_px"]) for p in pairs])
    result["length_within_5px"] = float((length_err <= 5).mean())
    result["length_within_10px"] = float((length_err <= 10).mean())
    return result, pairs, rows


def main():
    parser = argparse.ArgumentParser(description="连续划痕检测与长宽测量（不重训）")
    parser.add_argument("--weights-dir", default="results_scratch")
    parser.add_argument("--vis", type=int, default=12)
    args = parser.parse_args()
    os.makedirs(os.path.join(OUT_DIR, "vis"), exist_ok=True)
    ds.OUT_DIR = os.path.join(ROOT, args.weights_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ds.load_model(device)
    val = cache_split(model, device, "val")
    test = cache_split(model, device, "test")

    cfg, calib, val_info = tune(val)
    baseline, _, _ = summarize(test, BASELINE, calib)
    optimized, pairs, rows = summarize(test, cfg, calib)
    line("测试 基线", baseline)
    line("测试 优化后", optimized)
    for name, result in (("基线", baseline), ("优化后", optimized)):
        print("测试 %s 宽度误差 平均 %.2f 中位 %.2f P90 %.2f 相对中位 %.2f  1px内 %.0f%% 2px内 %.0f%%" % (
            name, result["width_err_mean"], result["width_err_median"], result["width_err_p90"],
            result["width_rel_median"], 100 * result["width_within_1px"], 100 * result["width_within_2px"]))

    size_rows = []
    saved = 0
    for item, found in rows:
        records = []
        for index, (part, record) in enumerate(found, start=1):
            record = dict(record)
            record["width_px"] = round(final_width(item["gray"], part, record, cfg["width"], calib), 2)
            record["defect_id"] = index
            records.append(record)
            size_rows.append({
                "image": item["stem"], "scratch_id": index,
                "length_px": record["length_px"], "width_px": record["width_px"],
                "center_x": record["center_x"], "center_y": record["center_y"],
            })
        if saved < args.vis and records:
            write_image(os.path.join(OUT_DIR, "vis", item["stem"] + "_continuous.png"),
                        render_measurement(item["gray"], records))
            saved += 1
    error_rows = []
    for pair in pairs:
        width = final_width(pair["gray"], pair["pred_part"], pair["pred"], cfg["width"], calib)
        error_rows.append({
            "image": pair["stem"], "iou": round(pair["iou"], 3),
            "gt_length_px": pair["gt"]["length_px"], "pred_length_px": pair["pred"]["length_px"],
            "gt_width_px": pair["gt"]["width_px"], "pred_width_px": round(width, 2),
            "length_err_px": round(abs(pair["pred"]["length_px"] - pair["gt"]["length_px"]), 2),
            "width_err_px": round(abs(width - pair["gt"]["width_px"]), 2),
        })
    write_csv(os.path.join(OUT_DIR, "continuous_sizes.csv"), size_rows)
    write_csv(os.path.join(OUT_DIR, "continuous_size_errors.csv"), error_rows)
    with open(os.path.join(OUT_DIR, "continuous_summary.json"), "w", encoding="utf-8") as handle:
        json.dump({
            "definition": {"min_length_px": MIN_LENGTH, "min_aspect": MIN_ASPECT},
            "cfg": cfg, "width_calibration": calib, **val_info,
            "test_baseline": baseline, "test_optimized": optimized,
        }, handle, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
