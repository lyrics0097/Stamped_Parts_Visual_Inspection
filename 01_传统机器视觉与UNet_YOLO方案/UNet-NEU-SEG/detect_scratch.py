#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Scratch-only segmentation on NEU-SEG, then measure each scratch's length and width.

Inclusion and patch pixels are treated as background, so those images teach the
model what is not a scratch. On a shared GPU the process stays under 4GB.

Post-processing (all chosen on the validation split, never on test):
  * test-time augmentation over the 8 flips/rotations,
  * probability threshold and minimum area,
  * bridging collinear pieces of one scratch that the network broke apart.
"""

import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse
import json
import time

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from detect_and_measure import (
    IMAGE_DIR,
    MASK_DIR,
    ROOT,
    TinyUNet,
    get_device,
    read_split,
    write_csv,
)
from measure_defects import measure_index_mask, render_measurement, write_image

SCRATCH = 3
MATCH_IOU = 0.3
OUT_DIR = os.path.join(ROOT, "results_scratch")


def paths():
    return {
        "weight": os.path.join(OUT_DIR, "scratch_unet.pt"),
        "config": os.path.join(OUT_DIR, "scratch_config.json"),
        "summary": os.path.join(OUT_DIR, "scratch_summary.json"),
    }


class ScratchDataset(Dataset):
    def __init__(self, stems, train=False):
        self.stems = stems
        self.train = train

    def __len__(self):
        return len(self.stems)

    def __getitem__(self, index):
        stem = self.stems[index]
        image = load_gray(stem).astype(np.float32) / 255.0
        mask = load_truth(stem)
        if self.train:
            if np.random.rand() < 0.5:
                image = np.fliplr(image)
                mask = np.fliplr(mask)
            if np.random.rand() < 0.5:
                image = np.flipud(image)
                mask = np.flipud(mask)
            turns = np.random.randint(0, 4)
            if turns:
                image = np.rot90(image, turns)
                mask = np.rot90(mask, turns)
            if np.random.rand() < 0.5:
                gain = np.random.uniform(0.8, 1.2)
                bias = np.random.uniform(-0.08, 0.08)
                image = np.clip(image * gain + bias, 0.0, 1.0)
        image = torch.from_numpy(np.ascontiguousarray(image, dtype=np.float32)).unsqueeze(0)
        target = torch.from_numpy(np.ascontiguousarray(mask.astype(np.int64)))
        return image, target


def load_gray(stem):
    return np.array(Image.open(os.path.join(IMAGE_DIR, stem + ".jpg")).convert("L"))


def load_truth(stem):
    mask = np.array(Image.open(os.path.join(MASK_DIR, stem + ".png")))
    if mask.ndim == 3:
        mask = mask[:, :, 0]
    return (mask == SCRATCH).astype(np.uint8)


def scratch_weight(stems):
    positive = 0
    total = 0
    for stem in stems:
        mask = load_truth(stem)
        positive += int(mask.sum())
        total += mask.size
    ratio = positive / max(total, 1)
    weight = float(np.clip((1.0 - ratio) / max(ratio, 1e-6), 1.0, 20.0)) ** 0.5
    return torch.tensor([1.0, weight], dtype=torch.float32), ratio


def dice_loss(logits, target):
    prob = torch.softmax(logits, dim=1)[:, 1]
    target = target.float()
    inter = (prob * target).sum(dim=(1, 2))
    denom = prob.sum(dim=(1, 2)) + target.sum(dim=(1, 2))
    return (1.0 - (2.0 * inter + 1.0) / (denom + 1.0)).mean()


# ---------------------------------------------------------------- inference


def _dihedral(x, k):
    """The 8 flips/rotations of an NCHW tensor; k in 0..7."""
    if k >= 4:
        x = torch.flip(x, dims=[3])
    return torch.rot90(x, k % 4, dims=[2, 3])


def _undo_dihedral(x, k):
    x = torch.rot90(x, -(k % 4), dims=[2, 3])
    if k >= 4:
        x = torch.flip(x, dims=[3])
    return x


@torch.inference_mode()
def predict_prob(model, gray, device, tta=False):
    tensor = torch.from_numpy(gray.astype(np.float32) / 255.0).unsqueeze(0).unsqueeze(0).to(device)
    if not tta:
        return torch.softmax(model(tensor), dim=1)[0, 1].detach().cpu().numpy()
    batch = torch.cat([_dihedral(tensor, k) for k in range(8)], dim=0)
    prob = torch.softmax(model(batch), dim=1)[:, 1:2]
    restored = torch.stack([_undo_dihedral(prob[k:k + 1], k)[0, 0] for k in range(8)])
    return restored.mean(dim=0).detach().cpu().numpy()


def threshold_mask(prob, threshold, min_area):
    binary = (prob >= threshold).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    cleaned = np.zeros_like(binary)
    for label_id in range(1, count):
        if int(stats[label_id, cv2.CC_STAT_AREA]) >= min_area:
            cleaned[labels == label_id] = 1
    return cleaned


def _segment_axis(part):
    ys, xs = np.nonzero(part)
    points = np.stack([xs, ys], axis=1).astype(np.float64)
    center = points.mean(axis=0)
    if len(points) < 3:
        direction = np.array([1.0, 0.0])
    else:
        _, vectors = np.linalg.eigh(np.cov((points - center).T))
        direction = vectors[:, -1]
    projection = (points - center) @ direction
    ends = (center + direction * projection.min(), center + direction * projection.max())
    rect = cv2.minAreaRect(points.astype(np.float32))
    thickness = max(1.0, min(rect[1]))
    return center, direction, ends, thickness


def bridge_gaps(binary, max_gap, max_angle=12.0, max_offset=6.0):
    """Join collinear scratch pieces whose facing ends are within max_gap pixels."""
    if max_gap <= 0:
        return binary
    parts = components(binary)
    if len(parts) < 2:
        return binary
    axes = [_segment_axis(part) for part in parts]
    cos_limit = np.cos(np.deg2rad(max_angle))
    bridged = binary.copy()
    for i in range(len(parts)):
        center_i, dir_i, ends_i, thick_i = axes[i]
        for j in range(i + 1, len(parts)):
            center_j, dir_j, ends_j, thick_j = axes[j]
            if abs(float(dir_i @ dir_j)) < cos_limit:
                continue
            best = None
            for end_i in ends_i:
                for end_j in ends_j:
                    gap = float(np.linalg.norm(end_i - end_j))
                    if best is None or gap < best[0]:
                        best = (gap, end_i, end_j)
            gap, end_i, end_j = best
            if gap > max_gap:
                continue
            normal = np.array([-dir_i[1], dir_i[0]])
            offset = abs(float((center_j - center_i) @ normal))
            if offset > max(max_offset, thick_i, thick_j):
                continue
            thickness = int(round(min(thick_i, thick_j)))
            cv2.line(
                bridged,
                tuple(int(round(v)) for v in end_i),
                tuple(int(round(v)) for v in end_j),
                1,
                max(1, thickness),
            )
    return bridged


def postprocess(prob, cfg):
    mask = threshold_mask(prob, cfg["threshold"], cfg["min_area"])
    return bridge_gaps(mask, cfg.get("gap", 0))


def components(binary, min_area=10):
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary.astype(np.uint8), connectivity=8)
    return [labels == i for i in range(1, count) if int(stats[i, cv2.CC_STAT_AREA]) >= min_area]


def measure(part):
    mask = np.zeros(part.shape, dtype=np.uint8)
    mask[part] = SCRATCH
    records = measure_index_mask(mask, min_area=1)
    return records[0] if records else None


def match(truth_parts, pred_parts):
    used = set()
    pairs = []
    for gt_index, gt in enumerate(truth_parts):
        best_index, best_iou = -1, 0.0
        for index, pred in enumerate(pred_parts):
            if index in used:
                continue
            union = np.logical_or(gt, pred).sum()
            iou = float(np.logical_and(gt, pred).sum()) / float(union + 1e-6)
            if iou > best_iou:
                best_index, best_iou = index, iou
        if best_index >= 0 and best_iou >= MATCH_IOU:
            used.add(best_index)
            pairs.append((gt_index, best_index, best_iou))
    return pairs


def assess(cached, cfg, with_sizes=False):
    tp = fp = fn = 0
    inter = union = 0
    length_err = []
    width_err = []
    key = "tta" if cfg.get("tta") else "plain"
    for item in cached:
        truth = item["truth"]
        pred = postprocess(item[key], cfg)
        inter += int(np.logical_and(truth, pred).sum())
        union += int(np.logical_or(truth, pred).sum())
        truth_parts = components(truth)
        pred_parts = components(pred, cfg["min_area"])
        pairs = match(truth_parts, pred_parts)
        tp += len(pairs)
        fn += len(truth_parts) - len(pairs)
        fp += len(pred_parts) - len(pairs)
        if with_sizes:
            for gt_index, pred_index, _ in pairs:
                gt = measure(truth_parts[gt_index])
                pr = measure(pred_parts[pred_index])
                length_err.append(abs(pr["length_px"] - gt["length_px"]))
                width_err.append(abs(pr["width_px"] - gt["width_px"]))
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    result = {
        "tp": tp, "fp": fp, "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / max(precision + recall, 1e-6),
        "pixel_iou": inter / max(union, 1),
    }
    if with_sizes and length_err:
        result["length_err_mean"] = float(np.mean(length_err))
        result["length_err_median"] = float(np.median(length_err))
        result["width_err_mean"] = float(np.mean(width_err))
        result["width_err_median"] = float(np.median(width_err))
    return result


# ---------------------------------------------------------------- training


def train(epochs, batch_size, width):
    os.makedirs(OUT_DIR, exist_ok=True)
    device = get_device()
    if device.type == "cuda" and batch_size > 2:
        print("共享显卡，batch 从 %d 降到 2" % batch_size)
        batch_size = 2
    torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))
    train_stems = read_split("train")
    val_stems = read_split("val")
    weights, ratio = scratch_weight(train_stems)
    print("训练集 %d，验证集 %d，batch %d，宽度 %d，划痕像素占比 %.2f%%，划痕类权重 %.2f"
          % (len(train_stems), len(val_stems), batch_size, width, ratio * 100, float(weights[1])))
    train_loader = DataLoader(ScratchDataset(train_stems, train=True), batch_size=batch_size, shuffle=True, num_workers=2)
    # batch 只有 2，BatchNorm 的运行统计量不稳，改用不依赖 batch 的 GroupNorm。
    model = TinyUNet(n_classes=2, norm="group", width=width).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    criterion = nn.CrossEntropyLoss(weight=weights.to(device))
    best_iou = -1.0
    for epoch in range(1, epochs + 1):
        model.train()
        started = time.perf_counter()
        total_loss = 0.0
        seen = 0
        for images, masks in train_loader:
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, masks) + dice_loss(logits, masks)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach()) * images.size(0)
            seen += images.size(0)
        scheduler.step()
        val_iou = pixel_iou(model, val_stems, device)
        print("epoch %d/%d  loss %.4f  验证划痕IoU %.3f  %.0fs"
              % (epoch, epochs, total_loss / max(seen, 1), val_iou, time.perf_counter() - started))
        if val_iou > best_iou:
            best_iou = val_iou
            torch.save({"state_dict": model.state_dict(), "val_iou": val_iou, "width": width}, paths()["weight"])
    print("最好验证划痕IoU %.3f" % best_iou)


@torch.inference_mode()
def pixel_iou(model, stems, device, threshold=0.5):
    model.eval()
    inter = 0
    union = 0
    for stem in stems:
        truth = load_truth(stem).astype(bool)
        pred = predict_prob(model, load_gray(stem), device) >= threshold
        inter += int(np.logical_and(truth, pred).sum())
        union += int(np.logical_or(truth, pred).sum())
    return inter / max(union, 1)


def load_model(device):
    checkpoint = torch.load(paths()["weight"], map_location="cpu", weights_only=False)
    model = TinyUNet(n_classes=2, norm="group", width=checkpoint.get("width", 16))
    model.load_state_dict(checkpoint["state_dict"])
    return model.to(device).eval()


# ---------------------------------------------------------------- tuning / test


def cache_probs(model, stems, device):
    cached = []
    for stem in stems:
        gray = load_gray(stem)
        cached.append({
            "stem": stem,
            "gray": gray,
            "truth": load_truth(stem),
            "plain": predict_prob(model, gray, device),
            "tta": predict_prob(model, gray, device, tta=True),
        })
    return cached


def tune(model, device):
    cached = cache_probs(model, read_split("val"), device)
    best = None
    for tta in (False, True):
        for threshold in (0.4, 0.5, 0.6, 0.7, 0.8, 0.9):
            for min_area in (30, 60, 100, 150):
                cfg = {"tta": tta, "threshold": threshold, "min_area": min_area, "gap": 0}
                result = assess(cached, cfg)
                if best is None or result["f1"] > best[1]["f1"]:
                    best = (cfg, result)
    base_cfg, base = best
    print("验证集 检测参数 TTA=%s 阈值 %.1f 最小面积 %d  精确率 %.3f 召回率 %.3f F1 %.3f"
          % (base_cfg["tta"], base_cfg["threshold"], base_cfg["min_area"],
             base["precision"], base["recall"], base["f1"]))
    # 拼接距离按长度误差选，F1 最多允许掉 0.01。
    chosen = None
    for gap in (0, 10, 20, 30, 45):
        cfg = dict(base_cfg, gap=gap)
        result = assess(cached, cfg, with_sizes=True)
        print("  拼接距离 %2d  F1 %.3f  长度误差 平均 %.2f 中位 %.2f  宽度误差 平均 %.2f"
              % (gap, result["f1"], result["length_err_mean"], result["length_err_median"], result["width_err_mean"]))
        if result["f1"] < base["f1"] - 0.01:
            continue
        if chosen is None or result["length_err_mean"] < chosen[1]["length_err_mean"]:
            chosen = (cfg, result)
    cfg, result = chosen
    print("验证集选定 %s" % json.dumps(cfg))
    with open(paths()["config"], "w", encoding="utf-8") as handle:
        json.dump({"cfg": cfg, "val": result}, handle, ensure_ascii=False, indent=2)
    return cfg


def evaluate(cfg, vis_limit=12):
    device = get_device()
    model = load_model(device)
    cached = cache_probs(model, read_split("test"), device)
    vis_dir = os.path.join(OUT_DIR, "vis")
    os.makedirs(vis_dir, exist_ok=True)
    key = "tta" if cfg.get("tta") else "plain"
    pred_rows = []
    error_rows = []
    saved = 0
    for item in cached:
        stem, gray, truth = item["stem"], item["gray"], item["truth"]
        pred = postprocess(item[key], cfg)
        truth_parts = components(truth)
        pred_parts = components(pred, cfg["min_area"])
        pred_records = [measure(part) for part in pred_parts]
        for index, record in enumerate(pred_records):
            record["defect_id"] = index + 1
            pred_rows.append({
                "image": stem,
                "scratch_id": index + 1,
                "length_px": record["length_px"],
                "width_px": record["width_px"],
                "area_px": record["area_px"],
                "center_x": record["center_x"],
                "center_y": record["center_y"],
            })
        for gt_index, pred_index, iou in match(truth_parts, pred_parts):
            gt = measure(truth_parts[gt_index])
            pr = pred_records[pred_index]
            error_rows.append({
                "image": stem,
                "iou": round(iou, 3),
                "gt_length_px": gt["length_px"],
                "pred_length_px": pr["length_px"],
                "gt_width_px": gt["width_px"],
                "pred_width_px": pr["width_px"],
                "length_err_px": round(abs(pr["length_px"] - gt["length_px"]), 2),
                "width_err_px": round(abs(pr["width_px"] - gt["width_px"]), 2),
            })
        if saved < vis_limit and truth.any() and pred_records:
            write_image(os.path.join(vis_dir, stem + "_scratch.png"), render_measurement(gray, pred_records))
            saved += 1
    write_csv(os.path.join(OUT_DIR, "scratch_sizes.csv"), pred_rows)
    write_csv(os.path.join(OUT_DIR, "scratch_size_errors.csv"), error_rows)
    summary = assess(cached, cfg)
    length_err = np.array([row["length_err_px"] for row in error_rows], dtype=np.float64)
    width_err = np.array([row["width_err_px"] for row in error_rows], dtype=np.float64)
    gt_length = np.array([row["gt_length_px"] for row in error_rows], dtype=np.float64)
    gt_width = np.array([row["gt_width_px"] for row in error_rows], dtype=np.float64)
    summary.update({
        "images": len(cached),
        "cfg": cfg,
        "length_err_median": float(np.median(length_err)),
        "length_err_mean": float(length_err.mean()),
        "length_err_p90": float(np.percentile(length_err, 90)),
        "length_rel_median": float(np.median(length_err / np.maximum(gt_length, 1))),
        "length_within_5px": float((length_err <= 5).mean()),
        "length_within_10px": float((length_err <= 10).mean()),
        "width_err_median": float(np.median(width_err)),
        "width_err_mean": float(width_err.mean()),
        "width_err_p90": float(np.percentile(width_err, 90)),
        "width_rel_median": float(np.median(width_err / np.maximum(gt_width, 1))),
        "width_within_2px": float((width_err <= 2).mean()),
    })
    with open(paths()["summary"], "w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    print(
        "测试图 %d  像素IoU %.3f  标注 %d 预测 %d  对上 %d 误报 %d 漏检 %d  精确率 %.3f 召回率 %.3f"
        % (summary["images"], summary["pixel_iou"], summary["tp"] + summary["fn"],
           summary["tp"] + summary["fp"], summary["tp"], summary["fp"], summary["fn"],
           summary["precision"], summary["recall"])
    )
    print(
        "长度误差 中位 %.2fpx 平均 %.2fpx P90 %.2fpx  宽度误差 中位 %.2fpx 平均 %.2fpx P90 %.2fpx"
        % (summary["length_err_median"], summary["length_err_mean"], summary["length_err_p90"],
           summary["width_err_median"], summary["width_err_mean"], summary["width_err_p90"])
    )
    return summary


def main():
    global OUT_DIR
    parser = argparse.ArgumentParser(description="只检测划痕并测量长宽")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--width", type=int, default=16)
    parser.add_argument("--out-dir", default="results_scratch")
    parser.add_argument("--skip-train", action="store_true")
    args = parser.parse_args()
    OUT_DIR = os.path.join(ROOT, args.out_dir)
    os.makedirs(OUT_DIR, exist_ok=True)
    if not args.skip_train:
        train(args.epochs, args.batch_size, args.width)
    device = get_device()
    cfg = tune(load_model(device), device)
    evaluate(cfg)


if __name__ == "__main__":
    main()
