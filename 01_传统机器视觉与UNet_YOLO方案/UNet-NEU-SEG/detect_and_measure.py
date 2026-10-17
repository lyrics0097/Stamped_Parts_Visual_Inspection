#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Detect NEU-SEG defects from a raw image, then measure length and width.

On a shared GPU the process caps itself at 4GB and uses batch size 2, so it
can run beside SAM3 / DINOv3 without taking the rest of the card.
"""

import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse
import csv
import time

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from measure_defects import (
    CLASS_NAMES,
    measure_index_mask,
    render_measurement,
    write_image,
)

ROOT = os.path.dirname(os.path.abspath(__file__))
IMAGE_DIR = os.path.join(ROOT, "data_wenjian", "JPEGImages")
MASK_DIR = os.path.join(ROOT, "data_wenjian", "SegmentationClass")
SPLIT_DIR = os.path.join(ROOT, "data_wenjian", "ImageSets", "Segmentation")
OUT_DIR = os.path.join(ROOT, "results_detect")
WEIGHT_PATH = os.path.join(OUT_DIR, "tiny_unet.pt")
MAX_VRAM_GB = 4.0
DEVICE = None


def get_device():
    """Use CUDA when present, and never reserve more than 4GB."""
    global DEVICE
    if DEVICE is not None:
        return DEVICE
    if not torch.cuda.is_available():
        print("CUDA 不可用，使用 CPU")
        DEVICE = torch.device("cpu")
        return DEVICE
    free_bytes, total_bytes = torch.cuda.mem_get_info()
    total_gb = total_bytes / (1024 ** 3)
    free_gb = free_bytes / (1024 ** 3)
    cap_gb = min(MAX_VRAM_GB, max(1.0, free_gb - 1.0))
    torch.cuda.set_per_process_memory_fraction(cap_gb / total_gb, 0)
    print(
        "GPU %s  总显存 %.1fGB  空闲 %.1fGB  本进程上限 %.1fGB"
        % (torch.cuda.get_device_name(0), total_gb, free_gb, cap_gb)
    )
    DEVICE = torch.device("cuda")
    return DEVICE


def make_norm(kind, channels):
    if kind == "group":
        return nn.GroupNorm(8, channels)
    return nn.BatchNorm2d(channels)


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, norm="batch"):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            make_norm(norm, out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            make_norm(norm, out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class TinyUNet(nn.Module):
    """Compact 4-class segmenter for 200x200 steel images."""

    def __init__(self, n_classes=4, norm="batch", width=16):
        super().__init__()
        c1, c2, c3, c4 = width, width * 2, width * 4, width * 6
        self.enc1 = ConvBlock(1, c1, norm)
        self.enc2 = ConvBlock(c1, c2, norm)
        self.enc3 = ConvBlock(c2, c3, norm)
        self.bottleneck = ConvBlock(c3, c4, norm)
        self.pool = nn.MaxPool2d(2)
        self.up3 = nn.ConvTranspose2d(c4, c3, 2, stride=2)
        self.dec3 = ConvBlock(c3 * 2, c3, norm)
        self.up2 = nn.ConvTranspose2d(c3, c2, 2, stride=2)
        self.dec2 = ConvBlock(c2 * 2, c2, norm)
        self.up1 = nn.ConvTranspose2d(c2, c1, 2, stride=2)
        self.dec1 = ConvBlock(c1 * 2, c1, norm)
        self.head = nn.Conv2d(c1, n_classes, 1)

    def forward(self, x):
        skip1 = self.enc1(x)
        skip2 = self.enc2(self.pool(skip1))
        skip3 = self.enc3(self.pool(skip2))
        x = self.bottleneck(self.pool(skip3))
        x = self.dec3(torch.cat([self.up3(x), skip3], dim=1))
        x = self.dec2(torch.cat([self.up2(x), skip2], dim=1))
        x = self.dec1(torch.cat([self.up1(x), skip1], dim=1))
        return self.head(x)


class NeuDataset(Dataset):
    def __init__(self, stems, train=False):
        self.stems = stems
        self.train = train

    def __len__(self):
        return len(self.stems)

    def __getitem__(self, index):
        stem = self.stems[index]
        image = np.array(Image.open(os.path.join(IMAGE_DIR, stem + ".jpg")).convert("L"), dtype=np.float32)
        mask = np.array(Image.open(os.path.join(MASK_DIR, stem + ".png")))
        if mask.ndim == 3:
            mask = mask[:, :, 0]
        image = image / 255.0
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
        image = torch.from_numpy(np.ascontiguousarray(image)).unsqueeze(0)
        mask = torch.from_numpy(np.ascontiguousarray(mask.astype(np.int64)))
        return image, mask


def read_split(name):
    path = os.path.join(SPLIT_DIR, name + ".txt")
    stems = []
    missing = 0
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            stem = line.strip()
            if not stem:
                continue
            image_path = os.path.join(IMAGE_DIR, stem + ".jpg")
            mask_path = os.path.join(MASK_DIR, stem + ".png")
            if os.path.isfile(image_path) and os.path.isfile(mask_path):
                stems.append(stem)
            else:
                missing += 1
    if missing:
        print("跳过 %d 个缺失样本 (%s)" % (missing, name))
    return stems


def class_weights(stems):
    counts = np.zeros(4, dtype=np.float64)
    for stem in stems:
        mask = np.array(Image.open(os.path.join(MASK_DIR, stem + ".png")))
        if mask.ndim == 3:
            mask = mask[:, :, 0]
        counts += np.bincount(mask.ravel(), minlength=4)[:4]
    weights = counts.sum() / np.maximum(counts, 1.0)
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32)


def foreground_miou(confusion):
    scores = []
    for class_id in (1, 2, 3):
        intersection = confusion[class_id, class_id]
        union = confusion[class_id, :].sum() + confusion[:, class_id].sum() - intersection
        scores.append(float(intersection) / float(union + 1e-6))
    return float(np.mean(scores)), scores


@torch.inference_mode()
def evaluate_miou(model, loader, device):
    model.eval()
    confusion = np.zeros((4, 4), dtype=np.int64)
    for images, masks in loader:
        images = images.to(device, non_blocking=True)
        predicted = model(images).argmax(dim=1).detach().cpu().numpy()
        target = masks.numpy()
        valid = (target >= 0) & (target < 4)
        index = 4 * target[valid] + predicted[valid]
        confusion += np.bincount(index, minlength=16).reshape(4, 4)
    return foreground_miou(confusion)


def train(epochs, batch_size):
    os.makedirs(OUT_DIR, exist_ok=True)
    device = get_device()
    if device.type == "cuda" and batch_size > 2:
        print("共享显卡，batch 从 %d 降到 2" % batch_size)
        batch_size = 2
    torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))
    train_stems = read_split("train")
    val_stems = read_split("val")
    print("训练集 %d，验证集 %d，batch %d" % (len(train_stems), len(val_stems), batch_size))
    weights = class_weights(train_stems)
    print("类别权重", [round(float(v), 3) for v in weights])
    train_loader = DataLoader(NeuDataset(train_stems, train=True), batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(NeuDataset(val_stems, train=False), batch_size=batch_size, shuffle=False, num_workers=0)
    model = TinyUNet().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss(weight=weights.to(device))
    best_score = -1.0
    for epoch in range(1, epochs + 1):
        model.train()
        if epoch == int(epochs * 0.7):
            for group in optimizer.param_groups:
                group["lr"] = 1e-4
        total_loss = 0.0
        seen = 0
        started = time.perf_counter()
        for images, masks in train_loader:
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(images), masks)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach()) * images.size(0)
            seen += images.size(0)
        score, class_scores = evaluate_miou(model, val_loader, device)
        elapsed = time.perf_counter() - started
        print(
            "epoch %d/%d  loss %.4f  前景mIoU %.3f  各类 %s  %.0fs"
            % (epoch, epochs, total_loss / max(seen, 1), score, [round(v, 3) for v in class_scores], elapsed)
        )
        if score > best_score:
            best_score = score
            torch.save({"state_dict": model.state_dict(), "fg_miou": score}, WEIGHT_PATH)
    print("最好验证前景mIoU %.3f，权重 %s" % (best_score, WEIGHT_PATH))
    return best_score


def load_model():
    if not os.path.isfile(WEIGHT_PATH):
        raise FileNotFoundError("还没有训练权重: %s" % WEIGHT_PATH)
    device = get_device()
    model = TinyUNet()
    checkpoint = torch.load(WEIGHT_PATH, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()
    return model


def segment_image(model, gray):
    device = next(model.parameters()).device
    tensor = torch.from_numpy(gray.astype(np.float32) / 255.0).unsqueeze(0).unsqueeze(0)
    tensor = tensor.to(device)
    with torch.inference_mode():
        predicted = model(tensor).argmax(dim=1)[0].detach().cpu().numpy().astype(np.uint8)
    return clean_small_components(predicted)


def clean_small_components(mask, min_area=10):
    cleaned = np.zeros_like(mask)
    for class_id in CLASS_NAMES:
        binary = (mask == class_id).astype(np.uint8)
        count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
        for label_id in range(1, count):
            if int(stats[label_id, cv2.CC_STAT_AREA]) >= min_area:
                cleaned[labels == label_id] = class_id
    return cleaned


def component_masks(mask):
    found = []
    for class_id in CLASS_NAMES:
        binary = (mask == class_id).astype(np.uint8)
        count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
        for label_id in range(1, count):
            if int(stats[label_id, cv2.CC_STAT_AREA]) < 10:
                continue
            found.append((class_id, labels == label_id))
    return found


def measure_component(class_id, component):
    mask = np.zeros(component.shape, dtype=np.uint8)
    mask[component] = class_id
    records = measure_index_mask(mask, min_area=10)
    return records[0] if records else None


def evaluate_detection(vis_limit=8):
    model = load_model()
    stems = read_split("test")
    os.makedirs(os.path.join(OUT_DIR, "vis"), exist_ok=True)
    confusion = np.zeros((4, 4), dtype=np.int64)
    pred_rows = []
    error_rows = []
    matched = 0
    missed = 0
    for stem_index, stem in enumerate(stems):
        gray = np.array(Image.open(os.path.join(IMAGE_DIR, stem + ".jpg")).convert("L"))
        truth = np.array(Image.open(os.path.join(MASK_DIR, stem + ".png")))
        if truth.ndim == 3:
            truth = truth[:, :, 0]
        predicted = segment_image(model, gray)
        valid = (truth >= 0) & (truth < 4)
        index = 4 * truth[valid].astype(np.int64) + predicted[valid].astype(np.int64)
        confusion += np.bincount(index, minlength=16).reshape(4, 4)
        pred_records = measure_index_mask(predicted)
        for record in pred_records:
            pred_rows.append({
                "image": stem,
                "class_name": record["class_name"],
                "length_px": record["length_px"],
                "width_px": record["width_px"],
                "area_px": record["area_px"],
            })
        used = set()
        pred_parts = component_masks(predicted)
        for class_id, truth_part in component_masks(truth):
            truth_measure = measure_component(class_id, truth_part)
            if truth_measure is None:
                continue
            best_iou = 0.0
            best_index = -1
            for index, (pred_class, pred_part) in enumerate(pred_parts):
                if pred_class != class_id or index in used:
                    continue
                intersection = np.logical_and(truth_part, pred_part).sum()
                union = np.logical_or(truth_part, pred_part).sum()
                iou = float(intersection) / float(union + 1e-6)
                if iou > best_iou:
                    best_iou = iou
                    best_index = index
            if best_index < 0 or best_iou < 0.3:
                missed += 1
                continue
            used.add(best_index)
            pred_measure = measure_component(class_id, pred_parts[best_index][1])
            if pred_measure is None:
                missed += 1
                continue
            matched += 1
            error_rows.append({
                "image": stem,
                "class_name": CLASS_NAMES[class_id],
                "iou": round(best_iou, 3),
                "gt_length_px": truth_measure["length_px"],
                "pred_length_px": pred_measure["length_px"],
                "gt_width_px": truth_measure["width_px"],
                "pred_width_px": pred_measure["width_px"],
                "length_err_px": round(abs(pred_measure["length_px"] - truth_measure["length_px"]), 2),
                "width_err_px": round(abs(pred_measure["width_px"] - truth_measure["width_px"]), 2),
            })
        if stem_index < vis_limit and pred_records:
            preview = render_measurement(gray, pred_records)
            write_image(os.path.join(OUT_DIR, "vis", stem + "_detect.png"), preview)
    score, class_scores = foreground_miou(confusion)
    write_csv(os.path.join(OUT_DIR, "pred_sizes.csv"), pred_rows)
    write_csv(os.path.join(OUT_DIR, "size_errors.csv"), error_rows)
    length_errors = [float(row["length_err_px"]) for row in error_rows]
    width_errors = [float(row["width_err_px"]) for row in error_rows]
    summary = {
        "images": len(stems),
        "fg_miou": score,
        "class_iou": class_scores,
        "matched": matched,
        "missed": missed,
        "length_mae": float(np.mean(length_errors)) if length_errors else None,
        "width_mae": float(np.mean(width_errors)) if width_errors else None,
        "length_median": float(np.median(length_errors)) if length_errors else None,
        "width_median": float(np.median(width_errors)) if width_errors else None,
    }
    print(
        "测试图 %d  前景mIoU %.3f  各类 %s  匹配 %d  漏检 %d  长度MAE %.2fpx  宽度MAE %.2fpx"
        % (
            summary["images"],
            summary["fg_miou"],
            [round(v, 3) for v in summary["class_iou"]],
            summary["matched"],
            summary["missed"],
            summary["length_mae"] or 0,
            summary["width_mae"] or 0,
        )
    )
    return summary


def write_csv(path, rows):
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description="从钢材原图检测缺陷并测量长宽")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--skip-train", action="store_true")
    args = parser.parse_args()
    if not args.skip_train:
        train(args.epochs, args.batch_size)
    evaluate_detection()


if __name__ == "__main__":
    main()
