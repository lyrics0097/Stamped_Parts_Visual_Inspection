"""在 IMT Mines Albi 表面缺陷 3D 数据集上评估 scratch3d: 分割 IoU + 深度测量误差 (对照参考深度)。

用法:
    python evaluate_imt_dataset.py --data <解压后的数据目录> --out <结果目录> [--se_mm 6] [--k_sigma 4]

输出:
    <out>/per_sample.csv        每个样本: IoU / P / R, 预测深度 vs 参考深度, 宽度, 长度
    <out>/summary.json          汇总: 平均 IoU, 深度 MAE / 相对误差 / 相关系数, 检出率
    <out>/fig_<sample>.png      每个样本 4 联图: 高度图 / 残差 / 真值 / 预测
    <out>/report.md             可读报告
"""
import argparse
import csv
import glob
import json
import os
import sys
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import scratch3d as S


def load_depth_csv(path):
    ref = {}
    with open(path, newline='', encoding='utf-8-sig') as f:
        for row in csv.reader(f):
            if len(row) < 2:
                continue
            try:
                ref[row[0].strip()] = float(row[1])
            except ValueError:
                continue                      # 表头
    return ref


def match_ref(name, ref):
    """样本文件名 -> 参考深度 (文件名去掉 _labels 后与 csv 首列做包含匹配)。"""
    key = name.replace('_labels', '')
    if key in ref:
        return ref[key]
    for k, v in ref.items():
        if k in key or key in k:
            return v
    return None


def fig4(res, labels_gt, title, path):
    ras, seg = res['raster'], res['seg']
    H = ras['H']
    gt = np.zeros(ras['shape'][0] * ras['shape'][1], np.uint8)
    gt[ras['cell_index'][labels_gt > 0]] = 1
    gt = gt.reshape(ras['shape'])
    fig, ax = plt.subplots(1, 4, figsize=(18, 4.5))
    im0 = ax[0].imshow(H, cmap='viridis'); ax[0].set_title('height map (mm)'); plt.colorbar(im0, ax=ax[0], fraction=0.046)
    im1 = ax[1].imshow(seg['residual'], cmap='magma', vmin=0, vmax=max(seg['thr'] * 4, 1e-3)); ax[1].set_title(f"pit residual (mm)  thr={seg['thr']:.3f}"); plt.colorbar(im1, ax=ax[1], fraction=0.046)
    ax[2].imshow(gt, cmap='gray'); ax[2].set_title('GT defect')
    ax[3].imshow(seg['mask'], cmap='gray'); ax[3].set_title('predicted defect')
    for a in ax:
        a.axis('off')
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--se_mm', type=float, default=6.0)
    ap.add_argument('--k_sigma', type=float, default=4.0)
    ap.add_argument('--abs_min_mm', type=float, default=0.02)
    ap.add_argument('--min_area_mm2', type=float, default=0.5)
    ap.add_argument('--mode', default='pit', choices=['pit', 'bump', 'both'])
    ap.add_argument('--max_points', type=int, default=1500000, help='点数超过则随机下采样 (速度)')
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    files = sorted(glob.glob(os.path.join(args.data, '**', '*.npy'), recursive=True))
    csvs = glob.glob(os.path.join(args.data, '**', '*.csv'), recursive=True)
    ref = load_depth_csv(csvs[0]) if csvs else {}
    print(f'{len(files)} 个样本, {len(ref)} 条参考深度')

    rows = []
    for p in files:
        name = os.path.splitext(os.path.basename(p))[0]
        t0 = time.time()
        arr = np.load(p)
        if len(arr) > args.max_points:
            arr = arr[np.random.default_rng(0).choice(len(arr), args.max_points, replace=False)]
        xyz, lab = arr[:, :3], arr[:, 6].astype(int)
        res = S.analyze_pointcloud(xyz, lab, se_mm=args.se_mm, k_sigma=args.k_sigma, abs_min_mm=args.abs_min_mm,
                                   min_area_mm2=args.min_area_mm2, mode=args.mode)
        ev = res['eval']
        defects = res['defects']
        # 与真值重叠最大的预测缺陷作为"该样本的缺陷"
        ras, seg = res['raster'], res['seg']
        gt_cells = np.zeros(ras['shape'][0] * ras['shape'][1], bool)
        gt_cells[ras['cell_index'][lab > 0]] = True
        gt_cells = gt_cells.reshape(ras['shape'])
        best, best_ov = None, 0
        for d in defects:
            comp = seg['labels'] == d['id']
            ov = int((comp & gt_cells).sum())
            if ov > best_ov:
                best, best_ov = d, ov
        depth_ref = match_ref(name, ref)
        depth_pred = best['depth_mm_p99'] if best else 0.0
        depth_on_gt = S.gt_depth_from_labels(res, lab)
        row = {'sample': name, 'points': len(arr), 'spacing_mm': round(res['spacing_mm'], 4),
               'sigma_mm': round(seg['sigma'], 4), 'thr_mm': round(seg['thr'], 4),
               'iou': round(ev['iou'], 3), 'precision': round(ev['precision'], 3), 'recall': round(ev['recall'], 3),
               'n_pred_defects': len(defects), 'detected': best is not None,
               'depth_ref_mm': depth_ref, 'depth_pred_mm': round(depth_pred, 4), 'depth_on_gt_region_mm': round(depth_on_gt, 4),
               'width_mm': best['width_mm_median'] if best else None, 'length_mm': best['length_mm'] if best else None,
               'type': best['type'] if best else None, 'abnormal': best['abnormal'] if best else None,
               'time_s': round(time.time() - t0, 1)}
        rows.append(row)
        fig4(res, lab, f"{name}  IoU={ev['iou']:.2f}  depth pred={depth_pred:.3f} ref={depth_ref}", os.path.join(args.out, f'fig_{name}.png'))
        print(f"{name:<55} IoU={ev['iou']:.2f} P={ev['precision']:.2f} R={ev['recall']:.2f}  depth pred={depth_pred:.3f} "
              f"(onGT {depth_on_gt:.3f}) ref={depth_ref}  n={len(defects)}  {row['time_s']}s")

    with open(os.path.join(args.out, 'per_sample.csv'), 'w', newline='', encoding='utf-8-sig') as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)

    real = [r for r in rows if r['depth_ref_mm'] is not None]
    err = np.array([r['depth_pred_mm'] - r['depth_ref_mm'] for r in real])
    err_gt = np.array([r['depth_on_gt_region_mm'] - r['depth_ref_mm'] for r in real])
    refs = np.array([r['depth_ref_mm'] for r in real])
    preds = np.array([r['depth_pred_mm'] for r in real])
    summary = {
        'n_samples': len(rows), 'n_with_ref_depth': len(real),
        'detection_rate': float(np.mean([r['detected'] for r in rows])),
        'mean_iou': float(np.mean([r['iou'] for r in rows])), 'median_iou': float(np.median([r['iou'] for r in rows])),
        'mean_precision': float(np.mean([r['precision'] for r in rows])), 'mean_recall': float(np.mean([r['recall'] for r in rows])),
        'depth_MAE_mm': float(np.mean(np.abs(err))) if len(err) else None,
        'depth_median_abs_err_mm': float(np.median(np.abs(err))) if len(err) else None,
        'depth_mean_rel_err': float(np.mean(np.abs(err) / np.maximum(refs, 1e-6))) if len(err) else None,
        'depth_pearson_r': float(np.corrcoef(preds, refs)[0, 1]) if len(err) > 2 else None,
        'depth_MAE_on_gt_region_mm': float(np.mean(np.abs(err_gt))) if len(err_gt) else None,
        'ref_depth_range_mm': [float(refs.min()), float(refs.max())] if len(refs) else None,
        'params': vars(args),
    }
    with open(os.path.join(args.out, 'summary.json'), 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # 深度散点图
    if len(real) > 2:
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.scatter(refs, preds, c='tab:blue', label='pred (matched defect)')
        ax.scatter(refs, [r['depth_on_gt_region_mm'] for r in real], c='tab:orange', marker='x', label='pred on GT region')
        m = max(refs.max(), preds.max()) * 1.1
        ax.plot([0, m], [0, m], 'k--', lw=1)
        ax.set_xlabel('reference depth (mm)'); ax.set_ylabel('predicted depth (mm)'); ax.legend(); ax.grid(alpha=0.3)
        ax.set_title(f"depth: MAE={summary['depth_MAE_mm']:.3f} mm  r={summary['depth_pearson_r']:.2f}")
        fig.tight_layout(); fig.savefig(os.path.join(args.out, 'depth_scatter.png'), dpi=120); plt.close(fig)

    with open(os.path.join(args.out, 'report.md'), 'w', encoding='utf-8') as f:
        f.write('# scratch3d 在 IMT Mines Albi 数据集上的评估\n\n')
        f.write(f"- 样本数: {summary['n_samples']} (有参考深度: {summary['n_with_ref_depth']})\n")
        f.write(f"- 检出率: {summary['detection_rate']:.1%}\n")
        f.write(f"- 分割 IoU: 均值 {summary['mean_iou']:.3f} / 中位 {summary['median_iou']:.3f}; P={summary['mean_precision']:.3f} R={summary['mean_recall']:.3f}\n")
        if summary['depth_MAE_mm'] is not None:
            f.write(f"- 深度 MAE: {summary['depth_MAE_mm']:.3f} mm (中位 {summary['depth_median_abs_err_mm']:.3f}), 相对误差 {summary['depth_mean_rel_err']:.1%}, 相关系数 r={summary['depth_pearson_r']:.3f}\n")
            f.write(f"- 参考深度范围: {summary['ref_depth_range_mm'][0]:.3f} ~ {summary['ref_depth_range_mm'][1]:.3f} mm\n")
        f.write(f"- 参数: se_mm={args.se_mm}, k_sigma={args.k_sigma}, abs_min_mm={args.abs_min_mm}, mode={args.mode}\n\n")
        f.write('| sample | IoU | P | R | depth_pred | depth_ref | width | length | type |\n|---|---|---|---|---|---|---|---|---|\n')
        for r in rows:
            f.write(f"| {r['sample']} | {r['iou']:.2f} | {r['precision']:.2f} | {r['recall']:.2f} | {r['depth_pred_mm']:.3f} | {r['depth_ref_mm']} | {r['width_mm']} | {r['length_mm']} | {r['type']} |\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
