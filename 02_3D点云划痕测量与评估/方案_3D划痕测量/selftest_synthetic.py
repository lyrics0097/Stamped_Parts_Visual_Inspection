"""合成数据自测: 已知真值的划痕 + 凹坑, 检验 scratch3d 的分割与测量精度。

场景: 60x40 mm 微弯钢板 (曲率半径 2~3 m), 点间距 0.05 mm, 高度噪声 5 um (相当于高端线激光轮廓仪),
      一条划痕 (宽 0.40 mm, 深 0.080 mm, 长 20 mm, 斜 30 度), 一个凹坑 (直径 2 mm, 深 0.150 mm)。
      点云被随机旋转 / 平移 / 打乱, 模拟任意扫描姿态。

用法: python selftest_synthetic.py [--noise_um 5] [--spacing 0.05]
"""
import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import scratch3d as S


def make_scene(spacing=0.05, noise_um=5.0, seed=1):
    rng = np.random.default_rng(seed)
    u, v = np.meshgrid(np.arange(0, 60, spacing), np.arange(0, 40, spacing))
    u, v = u.ravel(), v.ravel()
    h = (u - 30) ** 2 / (2 * 2000) + (v - 20) ** 2 / (2 * 3000)                # 宏观弯曲
    th = np.radians(30)
    d = -(u - 30) * np.sin(th) + (v - 20) * np.cos(th)
    along = (u - 30) * np.cos(th) + (v - 20) * np.sin(th)
    scratch = (np.abs(d) < 0.2) & (np.abs(along) < 10)
    h -= np.where(scratch, 0.08 * np.clip(1 - (d / 0.2) ** 2, 0, 1), 0)
    r2 = (u - 45) ** 2 + (v - 10) ** 2
    dent = r2 < 1.0
    h -= np.where(dent, 0.15 * np.clip(1 - r2, 0, 1), 0)
    h += rng.normal(0, noise_um / 1000.0, h.shape)
    gt = (scratch | dent).astype(int)
    xyz = np.c_[u, v, h]
    ang = np.radians(25)
    Rm = np.array([[np.cos(ang), 0, np.sin(ang)], [0, 1, 0], [-np.sin(ang), 0, np.cos(ang)]])
    xyz = xyz @ Rm.T + [100, -50, 300]
    perm = rng.permutation(len(xyz))
    return xyz[perm], gt[perm]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--noise_um', type=float, default=5.0)
    ap.add_argument('--spacing', type=float, default=0.05)
    args = ap.parse_args()
    xyz, gt = make_scene(args.spacing, args.noise_um)
    t = time.time()
    res = S.analyze_pointcloud(xyz, gt, se_mm=6.0)
    print(f"time {time.time() - t:.1f}s  grid {res['spacing_mm']:.4f} mm  sigma {res['seg']['sigma']:.4f}  thr {res['seg']['thr']:.4f}")
    print('eval', {k: round(v, 3) if isinstance(v, float) else v for k, v in res['eval'].items()})
    for d in res['defects']:
        print({k: d[k] for k in ('type', 'depth_mm_p99', 'depth_mm_max', 'width_mm_median', 'length_mm', 'area_mm2', 'abnormal')})
    print('truth: scratch depth 0.080 width 0.40 length 20.0 | dent depth 0.150 width ~2.0')


if __name__ == '__main__':
    main()
