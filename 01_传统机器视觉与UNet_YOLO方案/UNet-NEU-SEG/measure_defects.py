#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Measure length and width of each NEU-SEG defect instance.

Class index masks use:
    0 background, 1 inclusion (夹渣), 2 patches (斑块), 3 scratches (划痕).

Length and width are the long and short sides of the minimum rotated
rectangle, counted in whole pixels. A curved scratch uses its centerline
length when that is clearly longer than the rectangle. NEU-SEG has no
millimeter calibration, so results are in pixels unless a scale is given.
"""

import argparse
import csv
import heapq
import os
from collections import defaultdict

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

CLASS_NAMES = {
    1: "夹渣",
    2: "斑块",
    3: "划痕",
}
CLASS_COLORS = {
    1: (255, 0, 0),
    2: (0, 80, 255),
    3: (0, 180, 0),
}
_NEIGHBORS = (
    (-1, -1, 2 ** 0.5), (-1, 0, 1.0), (-1, 1, 2 ** 0.5),
    (0, -1, 1.0), (0, 1, 1.0),
    (1, -1, 2 ** 0.5), (1, 0, 1.0), (1, 1, 2 ** 0.5),
)


def read_image(path):
    """Read an image without going through OpenCV's path decoder.

    OpenCV on Windows cannot open paths that contain non-ASCII characters.
    """
    with Image.open(path) as image:
        return np.array(image)


def write_image(path, array):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    Image.fromarray(array).save(path)


def _skeleton(binary):
    image = (binary.astype(np.uint8) * 255)
    skeleton = np.zeros_like(image)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    while cv2.countNonZero(image):
        eroded = cv2.erode(image, element)
        opened = cv2.dilate(eroded, element)
        skeleton = cv2.bitwise_or(skeleton, cv2.subtract(image, opened))
        image = eroded
    return skeleton > 0


def _longest_path(skeleton):
    points = list(zip(*np.nonzero(skeleton)))
    if not points:
        return [], 0.0
    index = {point: i for i, point in enumerate(points)}
    adjacency = [[] for _ in points]
    for i, (y, x) in enumerate(points):
        for dy, dx, weight in _NEIGHBORS:
            other = index.get((y + dy, x + dx))
            if other is not None and other > i:
                adjacency[i].append((other, weight))
                adjacency[other].append((i, weight))

    def farthest(source):
        distance = [float("inf")] * len(points)
        previous = [-1] * len(points)
        distance[source] = 0.0
        heap = [(0.0, source)]
        while heap:
            current_distance, node = heapq.heappop(heap)
            if current_distance != distance[node]:
                continue
            for neighbor, weight in adjacency[node]:
                candidate = current_distance + weight
                if candidate < distance[neighbor]:
                    distance[neighbor] = candidate
                    previous[neighbor] = node
                    heapq.heappush(heap, (candidate, neighbor))
        reachable = [i for i, value in enumerate(distance) if value < float("inf")]
        end = max(reachable, key=lambda i: distance[i])
        return end, distance, previous

    # Two Dijkstra passes give the diameter of a tree-shaped skeleton.
    start, _, _ = farthest(0)
    end, distance, previous = farthest(start)
    path = []
    cursor = end
    while cursor != -1:
        path.append(points[cursor])
        cursor = previous[cursor]
    path.reverse()
    return path, float(distance[end])


def _rotated_box(component):
    ys, xs = np.nonzero(component)
    if len(xs) == 0:
        return 0.0, 0.0, None
    points = np.stack([xs, ys], axis=1).astype(np.float32)
    rect = cv2.minAreaRect(points)
    width, height = rect[1]
    return float(max(width, height)), float(min(width, height)), rect


def measure_index_mask(mask, min_area=10, mm_per_pixel=None):
    """Return one record per defect connected component."""
    if mask.ndim == 3:
        raise ValueError("需要单通道类别索引图，不能是彩色掩膜")
    records = []
    defect_id = 0
    for class_id, class_name in CLASS_NAMES.items():
        binary = (mask == class_id).astype(np.uint8)
        if binary.max() == 0:
            continue
        count, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
        for label_id in range(1, count):
            area = int(stats[label_id, cv2.CC_STAT_AREA])
            if area < min_area:
                continue
            component = labels == label_id
            box_length, box_width, rect = _rotated_box(component)
            distance = cv2.distanceTransform(component.astype(np.uint8), cv2.DIST_L2, 5)
            path, path_length = _longest_path(_skeleton(component))
            if path:
                radii = distance[tuple(np.array(path).T)]
                stroke_length = float(path_length + radii[0] + radii[-1])
                stroke_width = float(np.median(radii) * 2.0)
            else:
                stroke_length = box_length
                stroke_width = box_width
            # OpenCV 量的是像素中心间距，整像素宽高要再加 1。
            rect_length = box_length + 1.0
            rect_width = box_width + 1.0
            curved = class_id == 3 and stroke_length > rect_length * 1.25
            if curved:
                length_px = stroke_length
                width_px = max(stroke_width - 1.0, 1.0)
            else:
                length_px = rect_length
                width_px = rect_width
            if length_px < width_px:
                length_px, width_px = width_px, length_px
            defect_id += 1
            center_x, center_y = centroids[label_id]
            record = {
                "defect_id": defect_id,
                "class_id": class_id,
                "class_name": class_name,
                "area_px": area,
                "length_px": round(length_px, 2),
                "width_px": round(width_px, 2),
                "box_length_px": round(box_length, 2),
                "box_width_px": round(box_width, 2),
                "center_x": round(float(center_x), 1),
                "center_y": round(float(center_y), 1),
                "rect": rect,
            }
            if mm_per_pixel is not None:
                record["length_mm"] = round(length_px * mm_per_pixel, 3)
                record["width_mm"] = round(width_px * mm_per_pixel, 3)
            records.append(record)
    return records


def format_size_summary(records):
    if not records:
        return "尺寸: 无"
    parts = [
        f"{item['class_name']}#{item['defect_id']} 长{item['length_px']:.1f}px 宽{item['width_px']:.1f}px"
        for item in records
    ]
    return "尺寸: " + "；".join(parts)


def _font(size):
    for path in (
        r"C:\Windows\Fonts\msyh.ttc",
        r"C:\Windows\Fonts\simhei.ttf",
        r"C:\Windows\Fonts\arial.ttf",
    ):
        if os.path.isfile(path):
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def render_measurement(image, records, scale=3):
    if image.ndim == 2:
        canvas = np.stack([image, image, image], axis=-1)
    else:
        canvas = image[:, :, :3].copy()
    if canvas.dtype != np.uint8:
        canvas = canvas.astype(np.uint8)
    height, width = canvas.shape[:2]
    canvas = cv2.resize(canvas, (width * scale, height * scale), interpolation=cv2.INTER_NEAREST)
    for item in records:
        color = CLASS_COLORS[item["class_id"]]
        if item["rect"] is not None:
            box = cv2.boxPoints(item["rect"]) * scale
            box = np.round(box).astype(np.int32)
            cv2.polylines(canvas, [box], True, color, 1, cv2.LINE_AA)
        x = int(item["center_x"] * scale)
        y = int(item["center_y"] * scale)
        cv2.circle(canvas, (x, y), 2, color, -1)
    pil_image = Image.fromarray(canvas)
    draw = ImageDraw.Draw(pil_image)
    font = _font(14)
    for index, item in enumerate(records):
        text = f"{item['class_name']} 长{item['length_px']:.1f}px 宽{item['width_px']:.1f}px"
        draw.text((4, 4 + index * 18), text, fill=CLASS_COLORS[item["class_id"]], font=font)
    return np.array(pil_image)


def measure_directory(mask_dir, image_dir, out_dir, min_area=10, mm_per_pixel=None, vis_per_class=4):
    os.makedirs(out_dir, exist_ok=True)
    vis_dir = os.path.join(out_dir, "vis")
    os.makedirs(vis_dir, exist_ok=True)
    mask_names = sorted(
        name for name in os.listdir(mask_dir)
        if name.lower().endswith(".png")
    )
    csv_path = os.path.join(out_dir, "defect_sizes.csv")
    saved = defaultdict(int)
    image_count = 0
    defect_count = 0
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "image", "defect_id", "class_id", "class_name", "area_px",
            "length_px", "width_px", "box_length_px", "box_width_px",
            "center_x", "center_y", "length_mm", "width_mm",
        ])
        writer.writeheader()
        for name in mask_names:
            mask = read_image(os.path.join(mask_dir, name))
            if mask.ndim == 3:
                mask = mask[:, :, 0]
            records = measure_index_mask(mask, min_area=min_area, mm_per_pixel=mm_per_pixel)
            image_count += 1
            defect_count += len(records)
            stem = os.path.splitext(name)[0]
            for item in records:
                row = {key: item.get(key, "") for key in writer.fieldnames}
                row["image"] = stem
                row.pop("rect", None)
                writer.writerow(row)
            if not records:
                continue
            classes_here = {item["class_id"] for item in records}
            if not any(saved[class_id] < vis_per_class for class_id in classes_here):
                continue
            image = _paired_image(image_dir, stem)
            if image is None:
                image = np.zeros((*mask.shape[:2], 3), dtype=np.uint8)
            preview = render_measurement(image, records)
            write_image(os.path.join(vis_dir, stem + "_measure.png"), preview)
            for class_id in classes_here:
                saved[class_id] += 1
    return {
        "images": image_count,
        "defects": defect_count,
        "csv": csv_path,
        "vis_dir": vis_dir,
    }


def _paired_image(image_dir, stem):
    if not image_dir:
        return None
    for ext in (".jpg", ".jpeg", ".png", ".bmp"):
        path = os.path.join(image_dir, stem + ext)
        if os.path.isfile(path):
            image = read_image(path)
            if image.ndim == 2:
                return np.stack([image, image, image], axis=-1)
            return image[:, :, :3]
    return None


def _self_test():
    mask = np.zeros((80, 80), dtype=np.uint8)
    mask[20:23, 10:50] = 3  # 40 x 3 horizontal scratch
    mask[40:55, 40:55] = 2  # 15 x 15 patch
    records = measure_index_mask(mask, min_area=5)
    by_name = {item["class_name"]: item for item in records}
    scratch = by_name["划痕"]
    patch = by_name["斑块"]
    if not (39 <= scratch["length_px"] <= 41 and 2.5 <= scratch["width_px"] <= 3.5):
        raise AssertionError("划痕尺寸异常: %s" % scratch)
    if not (14.5 <= patch["length_px"] <= 15.5 and 14.5 <= patch["width_px"] <= 15.5):
        raise AssertionError("斑块尺寸异常: %s" % patch)
    print("self-test ok", scratch["length_px"], scratch["width_px"], patch["length_px"], patch["width_px"])


def main():
    parser = argparse.ArgumentParser(description="测量 NEU-SEG 缺陷的像素长宽")
    parser.add_argument("--mask-dir", default="data_wenjian/SegmentationClass")
    parser.add_argument("--image-dir", default="data_wenjian/JPEGImages")
    parser.add_argument("--out-dir", default="results_measure")
    parser.add_argument("--min-area", type=int, default=10)
    parser.add_argument("--mm-per-pixel", type=float, default=None)
    parser.add_argument("--vis-per-class", type=int, default=4)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        _self_test()
        return
    summary = measure_directory(
        args.mask_dir,
        args.image_dir,
        args.out_dir,
        min_area=args.min_area,
        mm_per_pixel=args.mm_per_pixel,
        vis_per_class=args.vis_per_class,
    )
    print(
        "完成: %d 张图, %d 条缺陷, 表格 %s, 预览 %s"
        % (summary["images"], summary["defects"], summary["csv"], summary["vis_dir"])
    )


if __name__ == "__main__":
    main()
