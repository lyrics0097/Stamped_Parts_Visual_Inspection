#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Score a trained YOLO26-seg scratch model with the same rules as the UNet baseline.

Each YOLO instance mask is one scratch; its length and width come from the
same measurement code. The confidence threshold is picked on val, and test is
reported for both the continuous-scratch rule and the all-scratch rule, next
to the UNet baseline evaluated by identical code.
"""

import argparse
import json
import os

import numpy as np
import torch

import continuous_scratch as cs
import detect_scratch as ds
from detect_and_measure import ROOT, write_csv
from measure_defects import render_measurement, write_image
from train_yolo_scratch import cap_gpu_memory

MIN_AREA = 60


def predict_instances(model, items, imgsz):
    for item in items:
        rgb = np.repeat(item["gray"][:, :, None], 3, axis=2)
        result = model.predict(rgb, imgsz=imgsz, conf=0.05, retina_masks=True, max_det=50, verbose=False)[0]
        instances = []
        if result.masks is not None:
            masks = result.masks.data.cpu().numpy() > 0.5
            confs = result.boxes.conf.cpu().numpy()
            for mask, conf in zip(masks, confs):
                cleaned = np.zeros_like(mask)
                for part in ds.components(mask.astype(np.uint8)):
                    cleaned |= part
                if cleaned.sum() >= MIN_AREA:
                    instances.append((cleaned, float(conf)))
        item["instances"] = instances


def found_at(item, conf):
    found = []
    for mask, score in item["instances"]:
        if score < conf:
            continue
        record = ds.measure(mask)
        if record is not None:
            found.append((mask, record))
    return found


def score(items, conf, predicate, keep=False):
    return cs.score_found([(item, found_at(item, conf)) for item in items], keep=keep, predicate=predicate)


def main():
    parser = argparse.ArgumentParser(description="按与 UNet 相同的规则评估 YOLO 划痕分割")
    parser.add_argument("--weights", default="runs_yolo/yolo26s_seg/weights/best.pt")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--out-dir", default="results_yolo26s_seg")
    parser.add_argument("--vis", type=int, default=12)
    args = parser.parse_args()
    out_dir = os.path.join(ROOT, args.out_dir)
    os.makedirs(os.path.join(out_dir, "vis"), exist_ok=True)
    cap_gpu_memory()
    from ultralytics import YOLO

    model = YOLO(args.weights)
    val = cs.cache_split(None, None, "val")
    test = cs.cache_split(None, None, "test")
    predict_instances(model, val, args.imgsz)
    predict_instances(model, test, args.imgsz)

    best = None
    for conf in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7):
        result, _, _ = score(val, conf, cs.is_continuous)
        cs.line("验证 YOLO 置信度 %.1f" % conf, result)
        key = (round(result["f1"], 3), -result.get("length_err_mean", 1e9))
        if best is None or key > best[0]:
            best = (key, conf)
    conf = best[1]
    print("验证集选定置信度 %.1f" % conf)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds.OUT_DIR = os.path.join(ROOT, "results_scratch")
    unet = ds.load_model(device)
    for item in test:
        item["prob"] = ds.predict_prob(unet, item["gray"], device)

    report = {"conf": conf, "weights": args.weights}
    for rule, predicate in (("continuous", cs.is_continuous), ("all", cs.any_scratch)):
        yolo = cs.finish_summary(*score(test, conf, predicate, keep=True))
        unet_result = cs.finish_summary(*cs.assess(test, cs.BASELINE, keep=True, predicate=predicate))
        report[rule] = {"yolo": yolo[0], "unet": unet_result[0]}
        for name, (result, _, _) in (("UNet", unet_result), ("YOLO26s-seg", yolo)):
            cs.line("测试[%s] %s" % (rule, name), result)
            print("    宽度误差 平均 %.2f 中位 %.2f P90 %.2f  2px内 %.0f%%" % (
                result["width_err_mean"], result["width_err_median"], result["width_err_p90"],
                100 * result["width_within_2px"]))
        if rule == "continuous":
            _, pairs, rows = yolo
            size_rows, error_rows, saved = [], [], 0
            for item, found in rows:
                records = []
                for index, (_, record) in enumerate(found, start=1):
                    record = dict(record, defect_id=index)
                    records.append(record)
                    size_rows.append({"image": item["stem"], "scratch_id": index,
                                      "length_px": record["length_px"], "width_px": record["width_px"]})
                if saved < args.vis and records:
                    write_image(os.path.join(out_dir, "vis", item["stem"] + "_yolo.png"),
                                render_measurement(item["gray"], records))
                    saved += 1
            for pair in pairs:
                error_rows.append({
                    "image": pair["stem"], "iou": round(pair["iou"], 3),
                    "gt_length_px": pair["gt"]["length_px"], "pred_length_px": pair["pred"]["length_px"],
                    "gt_width_px": pair["gt"]["width_px"], "pred_width_px": pair["pred"]["width_px"],
                    "length_err_px": round(abs(pair["pred"]["length_px"] - pair["gt"]["length_px"]), 2),
                    "width_err_px": round(abs(pair["pred"]["width_px"] - pair["gt"]["width_px"]), 2),
                })
            write_csv(os.path.join(out_dir, "yolo_sizes.csv"), size_rows)
            write_csv(os.path.join(out_dir, "yolo_size_errors.csv"), error_rows)
    with open(os.path.join(out_dir, "yolo_summary.json"), "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
