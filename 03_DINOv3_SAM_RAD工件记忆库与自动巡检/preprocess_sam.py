"""SAM foreground purification for RAD: black out the background and tightly crop the main object.

For every image under <src>/{train,test}/<type>/:
    1. SAM (Ultralytics, ViT-B) is prompted with a single positive point at the image center (the gimbal keeps the
       target centered). multimask_output=True yields sub-part / part / whole candidates; the largest candidate with a
       good predicted IoU and a sane area is kept, reduced to the component under the prompt, and its holes are filled.
    2. Pixels outside the mask are set to pure black RGB(0, 0, 0).
    3. The image is cropped to the mask's bounding box, then padded with black to a square in which the object spans
       `--fill` of the side, so RAD's Resize(512)+CenterCrop(448) never cuts the object.
Ground-truth masks under <src>/ground_truth/ receive the identical crop/pad so pixel metrics stay valid.
The output mirrors the <src> layout under <dst>; overlays for inspection are written to <dst>_debug/.
"""
import argparse
import glob
import json
import os
import shutil
import time

import cv2
import numpy as np
from ultralytics.models.sam import Predictor as SAMPredictor

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
IMG_EXT = ('.png', '.jpg', '.jpeg', '.bmp')


def build_predictor(weight, device):
    overrides = dict(conf=0.0, task='segment', mode='predict', imgsz=1024, model=weight,
                     device=device, save=False, verbose=False)
    return SAMPredictor(overrides=overrides)


def fill_holes(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    filled = np.zeros_like(mask)
    cv2.drawContours(filled, contours, -1, 255, -1)
    return filled


def component_at(mask, pt):
    n, labels = cv2.connectedComponents(mask, connectivity=8)
    lab = labels[pt[1], pt[0]]
    if lab == 0:
        areas = [(labels == i).sum() for i in range(1, n)]
        lab = 1 + int(np.argmax(areas)) if areas else 0
    return (labels == lab).astype(np.uint8) * 255 if lab else mask


def solidity(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return 0.0
    c = max(contours, key=cv2.contourArea)
    hull = cv2.contourArea(cv2.convexHull(c))
    return float(cv2.contourArea(c) / hull) if hull > 0 else 0.0


def segment_main_object(predictor, img_bgr, max_area=0.6, min_score=0.7, min_solidity=0.93):
    h, w = img_bgr.shape[:2]
    pt = (w // 2, h // 2)
    predictor.set_image(img_bgr)
    res = predictor(points=[list(pt)], labels=[1], multimask_output=True)[0]
    predictor.reset_image()

    masks = res.masks.data.cpu().numpy() > 0.5
    scores = res.boxes.conf.cpu().numpy()
    k = max(3, int(0.015 * min(h, w)) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    cands = []
    for m, s in zip(masks, scores):
        m = cv2.resize(m.astype(np.uint8) * 255, (w, h), interpolation=cv2.INTER_NEAREST)
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, kernel)
        m = fill_holes(component_at(m, pt))
        cands.append({'mask': m, 'score': float(s), 'area': float((m > 0).mean()), 'solidity': solidity(m)})

    ok = [c for c in cands
          if c['score'] >= min_score and c['area'] <= max_area and c['solidity'] >= min_solidity]
    best = max(ok, key=lambda c: c['area']) if ok else max(cands, key=lambda c: c['score'] * c['solidity'])
    return best, cands, pt


def crop_square(arr, bbox, fill, pad_value=0):
    x0, y0, x1, y1 = bbox
    crop = arr[y0:y1, x0:x1]
    ch, cw = crop.shape[:2]
    side = int(np.ceil(max(ch, cw) / fill))
    out = np.full((side, side) + arr.shape[2:], pad_value, arr.dtype)
    oy, ox = (side - ch) // 2, (side - cw) // 2
    out[oy:oy + ch, ox:ox + cw] = crop
    return out


def main():
    parser = argparse.ArgumentParser(description='SAM background removal + bbox crop for RAD')
    parser.add_argument('--src', default=os.path.join(ROOT_DIR, 'wild_experiment', 'raw'))
    parser.add_argument('--dst', default=os.path.join(ROOT_DIR, 'wild_experiment', 'sam_cleaned'))
    parser.add_argument('--sam_weight', default=r'C:\Users\lyz\Desktop\sam3\sam_b.pt')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--fill', type=float, default=0.85, help='object side / output side after padding')
    args = parser.parse_args()

    debug_dir = args.dst.rstrip('/\\') + '_debug'
    for d in (args.dst, debug_dir):
        if os.path.isdir(d):
            shutil.rmtree(d)
        os.makedirs(d)

    predictor = build_predictor(args.sam_weight, args.device)
    records = {}

    img_paths = sorted(p for split in ('train', 'test')
                       for p in glob.glob(os.path.join(args.src, split, '*', '*'))
                       if p.lower().endswith(IMG_EXT))
    for p in img_paths:
        rel = os.path.relpath(p, args.src)
        img = cv2.imread(p, cv2.IMREAD_COLOR)
        t0 = time.time()
        best, cands, pt = segment_main_object(predictor, img)
        mask = best['mask']

        ys, xs = np.nonzero(mask)
        bbox = (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)
        cleaned = img.copy()
        cleaned[mask == 0] = (0, 0, 0)
        out = crop_square(cleaned, bbox, args.fill)

        dst = os.path.join(args.dst, rel)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        cv2.imwrite(dst, out)

        overlay = img.copy()
        overlay[mask > 0] = (0.5 * overlay[mask > 0] + (0, 127, 0)).astype(np.uint8)
        cv2.rectangle(overlay, bbox[:2], bbox[2:], (0, 255, 255), 3)
        cv2.circle(overlay, pt, 9, (0, 0, 255), -1)
        dbg = os.path.join(debug_dir, rel.replace(os.sep, '_'))
        cv2.imwrite(dbg, overlay)

        records[rel.replace(os.sep, '/')] = {
            'bbox_xyxy': bbox, 'score': best['score'], 'area_ratio': best['area'], 'solidity': best['solidity'],
            'candidates': [{'score': round(c['score'], 4), 'area': round(c['area'], 4),
                            'solidity': round(c['solidity'], 4)} for c in cands],
        }
        print(f'[sam] {rel:<24} score={best["score"]:.3f} area={best["area"]:.3f} sol={best["solidity"]:.3f} '
              f'cands(score,area,sol)={[(round(c["score"], 2), round(c["area"], 3), round(c["solidity"], 2)) for c in cands]} '
              f'-> {out.shape[1]}x{out.shape[0]}  ({time.time() - t0:.1f}s)')

    gt_root = os.path.join(args.src, 'ground_truth')
    for g in sorted(glob.glob(os.path.join(gt_root, '*', '*_mask.png'))):
        defect = os.path.basename(os.path.dirname(g))
        stem = os.path.basename(g)[:-len('_mask.png')]
        img_rel = next((k for k in records if k.startswith(f'test/{defect}/{stem}.')), None)
        if img_rel is None:
            raise RuntimeError(f'no test image for GT {g}')
        gt = cv2.imread(g, cv2.IMREAD_GRAYSCALE)
        out = crop_square(gt, records[img_rel]['bbox_xyxy'], args.fill)
        gdst = os.path.join(args.dst, 'ground_truth', defect, os.path.basename(g))
        os.makedirs(os.path.dirname(gdst), exist_ok=True)
        cv2.imwrite(gdst, out)
        kept = (out > 127).sum() / max((gt > 127).sum(), 1)
        print(f'[gt ] ground_truth/{defect}/{os.path.basename(g)}  defect pixels kept after crop: {kept:.1%}')

    with open(os.path.join(args.dst, 'sam_meta.json'), 'w', encoding='utf-8') as f:
        json.dump(records, f, indent=2)
    print(f'[done] {len(records)} images -> {args.dst}   overlays -> {debug_dir}')


if __name__ == '__main__':
    main()
