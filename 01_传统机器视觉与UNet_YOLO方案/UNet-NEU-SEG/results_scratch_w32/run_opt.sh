#!/bin/bash
cd /root/autodl-tmp/neu-seg-lw
PY=/root/miniconda3/bin/python
echo "=== step1: existing model + TTA + bridging ==="
$PY -u detect_scratch.py --skip-train --out-dir results_scratch_opt
echo "=== step2: width 32, 60 epochs ==="
$PY -u detect_scratch.py --width 32 --epochs 60 --batch-size 2 --out-dir results_scratch_w32
echo "=== ALL DONE ==="
