#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Convert NEU-SEG scratch masks to an Ultralytics segmentation dataset.

Only class 3 (scratch) is kept, as YOLO class 0. Images without scratches get
an empty label file so they act as negatives. The train / val / test split is
the original NEU-SEG one.
"""

import argparse
import os
import shutil

import cv2
import numpy as np

import detect_scratch as ds
from detect_and_measure import IMAGE_DIR, ROOT, read_split


def mask_to_polygons(binary, epsilon):
    polygons = []
    for part in ds.components(binary):
        contours, _ = cv2.findContours(part.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        contour = max(contours, key=cv2.contourArea)
        contour = cv2.approxPolyDP(contour, epsilon, True).reshape(-1, 2)
        if len(contour) >= 3:
            polygons.append(contour)
    return polygons


def rasterize(polygons, shape):
    canvas = np.zeros(shape, dtype=np.uint8)
    for polygon in polygons:
        cv2.fillPoly(canvas, [polygon.astype(np.int32)], 1)
    return canvas


def main():
    parser = argparse.ArgumentParser(description="NEU-SEG 划痕标注转 YOLO 分割格式")
    parser.add_argument("--out", default="yolo_scratch")
    parser.add_argument("--epsilon", type=float, default=0.7)
    args = parser.parse_args()
    out = os.path.join(ROOT, args.out)
    ious, width_diffs = [], []
    counts = {}
    for split in ("train", "val", "test"):
        image_dir = os.path.join(out, "images", split)
        label_dir = os.path.join(out, "labels", split)
        os.makedirs(image_dir, exist_ok=True)
        os.makedirs(label_dir, exist_ok=True)
        positives = 0
        stems = read_split(split)
        for stem in stems:
            shutil.copyfile(os.path.join(IMAGE_DIR, stem + ".jpg"), os.path.join(image_dir, stem + ".jpg"))
            truth = ds.load_truth(stem)
            height, width = truth.shape
            polygons = mask_to_polygons(truth, args.epsilon)
            lines = []
            for polygon in polygons:
                coords = polygon.astype(np.float64) / np.array([width, height])
                lines.append("0 " + " ".join("%.5f" % v for v in coords.ravel()))
            with open(os.path.join(label_dir, stem + ".txt"), "w", encoding="utf-8") as handle:
                handle.write("\n".join(lines) + ("\n" if lines else ""))
            if polygons:
                positives += 1
                redrawn = rasterize(polygons, truth.shape)
                inter = np.logical_and(truth, redrawn).sum()
                union = np.logical_or(truth, redrawn).sum()
                ious.append(inter / max(union, 1))
                for part in ds.components(truth):
                    original = ds.measure(part)
                    back = ds.measure(part & (redrawn > 0))
                    if original and back:
                        width_diffs.append(back["width_px"] - original["width_px"])
        counts[split] = (len(stems), positives)
    with open(os.path.join(out, "data.yaml"), "w", encoding="utf-8") as handle:
        handle.write("path: %s\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n  0: scratch\n"
                     % out.replace("\\", "/"))
    for split, (total, positives) in counts.items():
        print("%s: %d 张，其中含划痕 %d 张" % (split, total, positives))
    width_diffs = np.array(width_diffs)
    print("多边形回画 与原标注 IoU 中位 %.3f 最低 %.3f；宽度变化 中位 %+.2fpx 平均 %+.2fpx"
          % (np.median(ious), np.min(ious), np.median(width_diffs), width_diffs.mean()))


if __name__ == "__main__":
    main()
