"""PoC: RAD heat-map peak -> SAM point prompt -> refined defect mask.

Pipeline
    1. Load a RAD anomaly map (.npy float map, or a grey-scale image) produced for `--image`.
    2. Threshold it (relative to its own range, or an absolute value such as the tau of normal images), keep the
       connected hot region that contains the global maximum, and take its centroid as the prompt point.
    3. Map that point from RAD's 256x256 grid (Resize(512) + CenterCrop(448) view) back to original pixel coordinates.
    4. Run SAM (Ultralytics ViT-B, same weights as preprocess_sam.py) with the single positive point,
       multimask_output=True, and pick the candidate whose IoU with the hot region is highest -- this prevents SAM
       from returning the whole object instead of the defect.
    5. Save the binary defect mask, a 4-panel overlay, and print IoU against ground truth when available.

Example
    python rad_peak_sam_refine.py \
        --image   ../wild_experiment/sam_cleaned/test/defect/001.png \
        --heatmap ./saved_results_mvtec/exp_compare/heatmaps/B_sam/defect_001.npy
"""
import argparse
import json
import os

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from preprocess_sam import build_predictor

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RAD_RESIZE, RAD_CROP = 512, 448  # must match get_data_transforms(512, 448) used by RAD


# ----------------------------------------------------------------------------------------------------------------
# 1. heat map I/O
# ----------------------------------------------------------------------------------------------------------------
def load_heatmap(path):
    if path.lower().endswith('.npy'):
        return np.load(path).astype(np.float32)
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(path)
    if img.ndim == 3:  # expects a grey-scale export; a colour-mapped (jet) png is NOT recoverable exactly
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    scale = 65535.0 if img.dtype == np.uint16 else 255.0
    return img.astype(np.float32) / scale


# ----------------------------------------------------------------------------------------------------------------
# 2. hot-region + prompt point on the RAD grid
# ----------------------------------------------------------------------------------------------------------------
def find_hot_region(amap, rel_thr=0.6, abs_thr=None):
    """Return (binary hot region, centroid (row, col), argmax (row, col), threshold used)."""
    vmin, vmax = float(amap.min()), float(amap.max())
    thr = abs_thr if abs_thr is not None else vmin + rel_thr * (vmax - vmin)
    thr = min(thr, vmax - 1e-6)  # guarantee the peak survives
    binary = (amap >= thr).astype(np.uint8)

    peak = np.unravel_index(int(np.argmax(amap)), amap.shape)
    n, labels = cv2.connectedComponents(binary, connectivity=8)
    region = (labels == labels[peak]).astype(np.uint8)

    # score-weighted centroid of the hot region; fall back to the peak if it leaves the region
    ys, xs = np.nonzero(region)
    wts = amap[ys, xs] - thr + 1e-6
    cy, cx = float((ys * wts).sum() / wts.sum()), float((xs * wts).sum() / wts.sum())
    if region[int(round(cy)), int(round(cx))] == 0:
        cy, cx = float(peak[0]), float(peak[1])
    return region, (cy, cx), peak, thr


# ----------------------------------------------------------------------------------------------------------------
# 3. coordinate mapping  RAD grid  ->  original image
# ----------------------------------------------------------------------------------------------------------------
def grid_to_original(rc, grid_hw, orig_hw):
    """(row, col) on the RAD map -> (x, y) pixel in the original image."""
    r, c = rc
    gh, gw = grid_hw
    H, W = orig_hw
    off = (RAD_RESIZE - RAD_CROP) / 2.0
    y_res = (r + 0.5) * RAD_CROP / gh + off          # position inside the 512x512 resized view
    x_res = (c + 0.5) * RAD_CROP / gw + off
    return x_res * W / RAD_RESIZE, y_res * H / RAD_RESIZE


def grid_map_to_original(arr, orig_hw, pad_value=0.0, interpolation=cv2.INTER_LINEAR):
    """Resample a map defined on the RAD grid onto the full original image (the un-cropped border gets pad_value)."""
    H, W = orig_hw
    crop = cv2.resize(arr.astype(np.float32), (RAD_CROP, RAD_CROP), interpolation=interpolation)
    off = (RAD_RESIZE - RAD_CROP) // 2
    full = np.full((RAD_RESIZE, RAD_RESIZE), pad_value, np.float32)
    full[off:off + RAD_CROP, off:off + RAD_CROP] = crop
    return cv2.resize(full, (W, H), interpolation=interpolation)


# ----------------------------------------------------------------------------------------------------------------
# 4. SAM with a single point prompt
# ----------------------------------------------------------------------------------------------------------------
def sam_point_segment(predictor, img_bgr, point_xy, hot_full, max_area=0.5):
    h, w = img_bgr.shape[:2]
    predictor.set_image(img_bgr)
    res = predictor(points=[[float(point_xy[0]), float(point_xy[1])]], labels=[1], multimask_output=True)[0]
    predictor.reset_image()

    masks = res.masks.data.cpu().numpy() > 0.5
    scores = res.boxes.conf.cpu().numpy()
    hot = hot_full > 0.5
    cands = []
    for m, s in zip(masks, scores):
        m = cv2.resize(m.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST) > 0
        inter, union = np.logical_and(m, hot).sum(), np.logical_or(m, hot).sum()
        cands.append({'mask': m, 'sam_score': float(s), 'area': float(m.mean()),
                      'iou_hot': float(inter / max(union, 1))})
    ok = [c for c in cands if c['area'] <= max_area] or cands
    best = max(ok, key=lambda c: c['iou_hot'])
    return best, cands


# ----------------------------------------------------------------------------------------------------------------
def find_gt(image_path):
    p = os.path.abspath(image_path).replace('\\', '/')
    parts = p.split('/')
    if len(parts) >= 3 and parts[-3] == 'test' and parts[-2] != 'good':
        stem = os.path.splitext(parts[-1])[0]
        gt = '/'.join(parts[:-3] + ['ground_truth', parts[-2], stem + '_mask.png'])
        if os.path.exists(gt):
            return gt
    return None


def main():
    ap = argparse.ArgumentParser(description='RAD heat-map peak -> SAM point prompt -> defect mask')
    ap.add_argument('--image', required=True, help='the test image that RAD scored')
    ap.add_argument('--heatmap', required=True, help='RAD anomaly map: .npy (preferred) or grey png')
    ap.add_argument('--sam_weight', default=r'C:\Users\lyz\Desktop\sam3\sam_b.pt')
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--rel_thr', type=float, default=0.6,
                    help='hot-region threshold as a fraction of the map range (ignored if --abs_thr given)')
    ap.add_argument('--abs_thr', type=float, default=None,
                    help='absolute threshold, e.g. tau = max score over normal images')
    ap.add_argument('--out_dir', default='./saved_results_mvtec/exp_peak_sam')
    args = ap.parse_args()

    img = cv2.imread(args.image, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(args.image)
    H, W = img.shape[:2]
    amap = load_heatmap(args.heatmap)

    # --- 2. hot region and prompt point on the RAD grid
    region, (cy, cx), peak, thr = find_hot_region(amap, args.rel_thr, args.abs_thr)
    # --- 3. to original coordinates
    px, py = grid_to_original((cy, cx), amap.shape, (H, W))
    peak_x, peak_y = grid_to_original(peak, amap.shape, (H, W))
    hot_full = grid_map_to_original(region, (H, W), 0.0, cv2.INTER_NEAREST)
    amap_full = grid_map_to_original(amap, (H, W), float(amap.min()))

    # --- 4. SAM point prompt
    predictor = build_predictor(args.sam_weight, args.device)
    best, cands = sam_point_segment(predictor, img, (px, py), hot_full)
    mask = (best['mask'].astype(np.uint8)) * 255

    # --- 5. outputs
    os.makedirs(args.out_dir, exist_ok=True)
    tag = os.path.splitext(os.path.basename(args.heatmap))[0]
    mask_path = os.path.join(args.out_dir, f'{tag}_defect_mask.png')
    cv2.imwrite(mask_path, mask)

    info = {
        'image': os.path.abspath(args.image), 'heatmap': os.path.abspath(args.heatmap),
        'map_shape': list(amap.shape), 'map_min_max': [float(amap.min()), float(amap.max())],
        'threshold': float(thr), 'hot_region_pixels_on_grid': int(region.sum()),
        'peak_grid_rc': [int(peak[0]), int(peak[1])], 'peak_orig_xy': [round(peak_x, 1), round(peak_y, 1)],
        'prompt_grid_rc': [round(cy, 2), round(cx, 2)], 'prompt_orig_xy': [round(px, 1), round(py, 1)],
        'sam_candidates': [{k: round(v, 4) for k, v in c.items() if k != 'mask'} for c in cands],
        'chosen': {k: round(v, 4) for k, v in best.items() if k != 'mask'},
        'mask_area_ratio': float(mask.mean() / 255), 'mask_path': os.path.abspath(mask_path),
    }

    gt_path = find_gt(args.image)
    gt = None
    if gt_path:
        gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE) > 127
        m = mask > 0
        inter, union = np.logical_and(m, gt).sum(), np.logical_or(m, gt).sum()
        hot_inter, hot_union = np.logical_and(hot_full > 0.5, gt).sum(), np.logical_or(hot_full > 0.5, gt).sum()
        info.update({
            'gt': gt_path, 'prompt_inside_gt': bool(gt[int(round(py)), int(round(px))]),
            'iou_hot_region_vs_gt': float(hot_inter / max(hot_union, 1)),
            'iou_sam_mask_vs_gt': float(inter / max(union, 1)),
            'sam_mask_precision': float(inter / max(m.sum(), 1)), 'sam_mask_recall': float(inter / max(gt.sum(), 1)),
        })

    # overlay figure
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    norm = (amap_full - amap.min()) / (amap.max() - amap.min() + 1e-8)
    heat = plt.get_cmap('jet')(norm)[..., :3]
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.8))
    axes[0].imshow(rgb); axes[0].set_title('Test image')
    axes[1].imshow(0.5 * rgb / 255 + 0.5 * heat)
    axes[1].contour(hot_full, levels=[0.5], colors='white', linewidths=1.2)
    axes[1].plot(px, py, 'w*', ms=16, mec='black'); axes[1].plot(peak_x, peak_y, 'r.', ms=8)
    axes[1].set_title(f'RAD heat map + hot region (thr={thr:.3f})\nstar = point prompt, red dot = argmax')
    sam_vis = rgb.copy().astype(np.float32)
    sam_vis[mask > 0] = 0.4 * sam_vis[mask > 0] + 0.6 * np.array([255, 0, 0])
    axes[2].imshow(sam_vis.astype(np.uint8)); axes[2].plot(px, py, 'w*', ms=16, mec='black')
    axes[2].set_title(f'SAM point-prompt mask\niou(hot)={best["iou_hot"]:.2f}  area={best["area"]:.3f}  '
                      f'sam_conf={best["sam_score"]:.2f}')
    axes[3].imshow(mask, cmap='gray')
    if gt is not None:
        axes[3].contour(gt.astype(np.uint8), levels=[0.5], colors='lime', linewidths=1.5)
        axes[3].set_title(f'Final defect mask (green = GT)\nIoU vs GT: hot-region {info["iou_hot_region_vs_gt"]:.3f}'
                          f'  ->  SAM {info["iou_sam_mask_vs_gt"]:.3f}')
    else:
        axes[3].set_title('Final defect mask')
    for ax in axes:
        ax.axis('off')
    plt.tight_layout()
    overlay_path = os.path.join(args.out_dir, f'{tag}_overlay.png')
    plt.savefig(overlay_path, dpi=130); plt.close()
    info['overlay_path'] = os.path.abspath(overlay_path)

    with open(os.path.join(args.out_dir, f'{tag}_info.json'), 'w', encoding='utf-8') as f:
        json.dump(info, f, indent=2, ensure_ascii=False)

    print(f'[heatmap] {amap.shape} range=[{amap.min():.4f}, {amap.max():.4f}] thr={thr:.4f} '
          f'hot-region={int(region.sum())} grid px')
    print(f'[prompt ] grid (r,c)=({cy:.1f},{cx:.1f}) -> original (x,y)=({px:.1f},{py:.1f})   '
          f'argmax -> ({peak_x:.1f},{peak_y:.1f})')
    print(f'[sam    ] candidates (conf, area, iou_hot): '
          f'{[(round(c["sam_score"], 2), round(c["area"], 3), round(c["iou_hot"], 2)) for c in cands]}')
    print(f'[sam    ] chosen: conf={best["sam_score"]:.3f} area={best["area"]:.3f} iou_hot={best["iou_hot"]:.3f}')
    if gt is not None:
        print(f'[gt     ] prompt inside GT: {info["prompt_inside_gt"]} | IoU vs GT: hot-region '
              f'{info["iou_hot_region_vs_gt"]:.3f} -> SAM mask {info["iou_sam_mask_vs_gt"]:.3f} '
              f'(precision {info["sam_mask_precision"]:.3f}, recall {info["sam_mask_recall"]:.3f})')
    print(f'[saved  ] {mask_path}\n[saved  ] {overlay_path}')


if __name__ == '__main__':
    main()
