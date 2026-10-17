#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Train YOLO26-seg on the scratch-only dataset, capped at 4GB of a shared GPU."""

import argparse
import os

import torch

MAX_VRAM_GB = 4.0


def cap_gpu_memory():
    if not torch.cuda.is_available():
        return
    free_bytes, total_bytes = torch.cuda.mem_get_info()
    cap_gb = min(MAX_VRAM_GB, max(1.0, free_bytes / 1024 ** 3 - 1.0))
    torch.cuda.set_per_process_memory_fraction(cap_gb / (total_bytes / 1024 ** 3), 0)
    print("GPU 空闲 %.1fGB，本进程上限 %.1fGB" % (free_bytes / 1024 ** 3, cap_gb))


def main():
    parser = argparse.ArgumentParser(description="训练 YOLO26-seg 划痕分割")
    parser.add_argument("--weights", default="weights/yolo26s-seg.pt")
    parser.add_argument("--data", default="yolo_scratch/data.yaml")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--name", default="yolo26s_seg")
    parser.add_argument("--resume", default="", help="从某次训练的 last.pt 接着训")
    args = parser.parse_args()
    if args.batch > 2:
        print("共享显卡，batch 从 %d 降到 2" % args.batch)
        args.batch = 2
    cap_gpu_memory()
    from ultralytics import YOLO

    if args.resume:
        # 续训沿用检查点里保存的全部训练参数（batch、轮数、输出目录）。
        YOLO(args.resume).train(resume=True)
        return
    model = YOLO(args.weights)
    model.train(
        data=args.data,
        imgsz=args.imgsz,
        epochs=args.epochs,
        batch=args.batch,
        device=0,
        workers=2,
        amp=True,
        plots=False,
        patience=20,
        fliplr=0.5,
        flipud=0.5,
        seed=0,
        # 相对路径会被 Ultralytics 放到 runs/segment/ 下面，评估脚本就找不到权重。
        project=os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs_yolo"),
        name=args.name,
        exist_ok=True,
    )


if __name__ == "__main__":
    main()
