"""Build a small "in-the-wild" anomaly dataset: MVTec bottles composited onto cluttered real-world backgrounds.

Backgrounds are cluttered scene photos fetched from Wikimedia Commons (fallback: picsum.photos, then procedural
clutter). Each bottle is cut out of its white MVTec backdrop and pasted at a jittered position/scale, so the object
is never pixel-aligned across images. Defect GT masks undergo the exact same transform.

Output layout (MVTec-style, directly consumable by RAD with --data_path ../wild_experiment --item_list raw):
    ../wild_experiment/raw/train/good/*.png
    ../wild_experiment/raw/test/good/*.png
    ../wild_experiment/raw/test/defect/*.png
    ../wild_experiment/raw/ground_truth/defect/*_mask.png
"""
import argparse
import io
import json
import os
import random
import shutil
import time

import cv2
import numpy as np
import requests
from PIL import Image

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)

CANVAS = 768
UA = {'User-Agent': 'RAD-wild-experiment/1.0 (research use)'}
COMMONS_API = 'https://commons.wikimedia.org/w/api.php'
QUERIES = [
    'cluttered desk', 'messy workbench tools', 'flea market table', 'junk shop interior',
    'cluttered kitchen counter', 'hardware store shelves', 'messy garage', 'tangled cables',
    'electronics repair bench', 'antique shop clutter', 'cluttered street', 'messy office desk',
    'second-hand market stall', 'workshop clutter', 'cluttered shelf', 'scrap yard', 'toolbox tools',
    'bazaar stall goods',
]


def fetch_commons_backgrounds(n, min_side=700):
    urls, seen = [], set()
    for q in QUERIES:
        if len(urls) >= n:
            break
        params = {
            'action': 'query', 'format': 'json', 'generator': 'search', 'gsrnamespace': 6,
            'gsrsearch': f'filetype:bitmap {q}', 'gsrlimit': 8, 'prop': 'imageinfo',
            'iiprop': 'url|size|mime', 'iiurlwidth': 1024,
        }
        try:
            pages = requests.get(COMMONS_API, params=params, headers=UA, timeout=20).json()['query']['pages']
        except Exception as e:
            print(f'[bg] commons query "{q}" failed: {e}')
            continue
        for page in sorted(pages.values(), key=lambda p: p.get('index', 0)):
            info = page.get('imageinfo', [{}])[0]
            if info.get('mime') != 'image/jpeg' or min(info.get('width', 0), info.get('height', 0)) < min_side:
                continue
            url = info.get('thumburl') or info.get('url')
            if url and url not in seen:
                seen.add(url)
                urls.append((url, page['title'], q))
                break  # one image per query keeps scene diversity high
    return urls


def download_image(url, retries=4):
    for attempt in range(retries):
        r = requests.get(url, headers=UA, timeout=30)
        if r.status_code == 429 and attempt < retries - 1:
            time.sleep(5 * (attempt + 1))
            continue
        r.raise_for_status()
        return Image.open(io.BytesIO(r.content)).convert('RGB')


def procedural_clutter(rng):
    img = np.full((CANVAS, CANVAS, 3), rng.integers(40, 200, 3), np.uint8)
    for _ in range(160):
        color = tuple(int(c) for c in rng.integers(0, 256, 3))
        kind = rng.integers(3)
        p = tuple(int(v) for v in rng.integers(0, CANVAS, 2))
        if kind == 0:
            cv2.circle(img, p, int(rng.integers(10, 90)), color, -1)
        elif kind == 1:
            q = tuple(int(v) for v in rng.integers(0, CANVAS, 2))
            cv2.rectangle(img, p, q, color, -1)
        else:
            q = tuple(int(v) for v in rng.integers(0, CANVAS, 2))
            cv2.line(img, p, q, color, int(rng.integers(2, 12)))
    noise = rng.normal(0, 12, img.shape)
    return Image.fromarray(np.clip(img + noise, 0, 255).astype(np.uint8))


def square_resize(img):
    w, h = img.size
    s = min(w, h)
    img = img.crop(((w - s) // 2, (h - s) // 2, (w - s) // 2 + s, (h - s) // 2 + s))
    return img.resize((CANVAS, CANVAS), Image.BICUBIC)


def get_backgrounds(n, bg_dir, rng):
    os.makedirs(bg_dir, exist_ok=True)
    cached = [os.path.join(bg_dir, f'bg_{i:02d}.jpg') for i in range(n)]
    if all(os.path.exists(p) for p in cached):
        print(f'[bg] reusing {n} cached backgrounds in {bg_dir}')
        return [Image.open(p).convert('RGB') for p in cached]
    credits, bgs = [], []

    for url, title, q in fetch_commons_backgrounds(n + 6):
        if len(bgs) >= n:
            break
        try:
            img = square_resize(download_image(url))
        except Exception as e:
            print(f'[bg] download failed ({q}): {e}')
            continue
        bgs.append(img)
        credits.append({'source': 'wikimedia_commons', 'title': title, 'query': q, 'url': url})
        print(f'[bg] commons: {title}')
        time.sleep(1.5)

    seed = 10
    while len(bgs) < n and seed < 40:
        url = f'https://picsum.photos/seed/rad{seed}/1024/1024'
        seed += 1
        try:
            bgs.append(square_resize(download_image(url)))
            credits.append({'source': 'picsum', 'url': url})
            print(f'[bg] picsum: {url}')
        except Exception as e:
            print(f'[bg] picsum failed: {e}')
            break

    while len(bgs) < n:
        bgs.append(procedural_clutter(rng))
        credits.append({'source': 'procedural'})
        print('[bg] procedural clutter')

    for i, bg in enumerate(bgs):
        bg.save(os.path.join(bg_dir, f'bg_{i:02d}.jpg'), quality=95)
    with open(os.path.join(bg_dir, 'credits.json'), 'w', encoding='utf-8') as f:
        json.dump(credits, f, indent=2, ensure_ascii=False)
    return bgs


def bottle_alpha(rgb):
    """Soft alpha of the bottle against the white MVTec backdrop."""
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    fg = (gray < 215).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(fg, 8)
    fg = (labels == 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])).astype(np.uint8) * 255
    contours, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    filled = np.zeros_like(fg)
    cv2.drawContours(filled, contours, -1, 255, -1)
    filled = cv2.erode(filled, np.ones((5, 5), np.uint8))
    return cv2.GaussianBlur(filled, (9, 9), 0).astype(np.float32) / 255.0


def add_artificial_defect(rgb, rng):
    """Paint an obvious crack + chipped region on the bottle rim; returns (image, binary GT)."""
    h, w = rgb.shape[:2]
    out = rgb.copy()
    gt = np.zeros((h, w), np.uint8)
    cx, cy, r = w // 2, h // 2, int(0.33 * w)
    ang = rng.uniform(0, 2 * np.pi)
    pts = []
    for t in np.linspace(0, 1, 9):
        a = ang + t * 0.9
        rr = r + rng.uniform(-25, 25)
        pts.append([int(cx + rr * np.cos(a)), int(cy + rr * np.sin(a))])
    pts = np.array(pts, np.int32)
    cv2.polylines(out, [pts], False, (235, 235, 225), 30)
    cv2.polylines(gt, [pts], False, 255, 30)
    chip_c = (int(cx + r * np.cos(ang + 0.45)), int(cy + r * np.sin(ang + 0.45)))
    cv2.ellipse(out, chip_c, (90, 55), np.degrees(ang), 0, 360, (150, 60, 30), -1)
    cv2.ellipse(gt, chip_c, (90, 55), np.degrees(ang), 0, 360, 255, -1)
    return out, gt


def composite(bg, rgb, alpha, gt, rng):
    size = int(CANVAS * rng.uniform(0.52, 0.62))
    rgb_s = cv2.resize(rgb, (size, size), interpolation=cv2.INTER_AREA)
    a_s = cv2.resize(alpha, (size, size), interpolation=cv2.INTER_LINEAR)[..., None]
    max_off = int(0.12 * CANVAS)
    x0 = (CANVAS - size) // 2 + int(rng.integers(-max_off, max_off + 1))
    y0 = (CANVAS - size) // 2 + int(rng.integers(-max_off, max_off + 1))

    canvas = np.asarray(bg, np.float32).copy()
    roi = canvas[y0:y0 + size, x0:x0 + size]
    canvas[y0:y0 + size, x0:x0 + size] = a_s * rgb_s + (1 - a_s) * roi
    out = Image.fromarray(np.clip(canvas, 0, 255).astype(np.uint8))

    gt_canvas = None
    if gt is not None:
        gt_canvas = np.zeros((CANVAS, CANVAS), np.uint8)
        gt_s = cv2.resize(gt, (size, size), interpolation=cv2.INTER_NEAREST)
        gt_canvas[y0:y0 + size, x0:x0 + size] = np.where(gt_s > 127, 255, 0)
    return out, gt_canvas, (x0, y0, size)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mvtec_bottle', default=os.path.join(ROOT_DIR, 'mvtec_anomaly_detection', 'bottle'))
    parser.add_argument('--out_root', default=os.path.join(ROOT_DIR, 'wild_experiment'))
    parser.add_argument('--n_train', type=int, default=8)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    random.seed(args.seed)
    src = args.mvtec_bottle
    raw = os.path.join(args.out_root, 'raw')
    if os.path.isdir(raw):
        shutil.rmtree(raw)

    # (split, defect_dir, source_image, source_gt or 'artificial' or None)
    plan = [('train', 'good', os.path.join(src, 'train', 'good', f'{i:03d}.png'), None) for i in range(args.n_train)]
    plan += [('test', 'good', os.path.join(src, 'test', 'good', f'{i:03d}.png'), None) for i in range(3)]
    plan += [
        ('test', 'defect', os.path.join(src, 'test', 'broken_large', '000.png'),
         os.path.join(src, 'ground_truth', 'broken_large', '000_mask.png')),
        ('test', 'defect', os.path.join(src, 'test', 'contamination', '000.png'),
         os.path.join(src, 'ground_truth', 'contamination', '000_mask.png')),
        ('test', 'defect', os.path.join(src, 'train', 'good', '100.png'), 'artificial'),
    ]

    bgs = get_backgrounds(len(plan), os.path.join(args.out_root, 'backgrounds'), rng)
    order = rng.permutation(len(bgs))

    meta, counters = [], {}
    for k, (split, defect, img_path, gt_src) in enumerate(plan):
        rgb = np.asarray(Image.open(img_path).convert('RGB'))
        alpha = bottle_alpha(rgb)
        gt = None
        if gt_src == 'artificial':
            rgb, gt = add_artificial_defect(rgb, rng)
        elif gt_src is not None:
            gt = np.asarray(Image.open(gt_src).convert('L'))

        out, gt_canvas, place = composite(bgs[order[k]], rgb, alpha, gt, rng)
        idx = counters.get((split, defect), 0)
        counters[(split, defect)] = idx + 1
        name = f'{idx:03d}.png'
        dst = os.path.join(raw, split, defect, name)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        out.save(dst)
        if gt_canvas is not None:
            gdst = os.path.join(raw, 'ground_truth', defect, f'{idx:03d}_mask.png')
            os.makedirs(os.path.dirname(gdst), exist_ok=True)
            Image.fromarray(gt_canvas).save(gdst)
        meta.append({'file': f'{split}/{defect}/{name}', 'source': os.path.relpath(img_path, ROOT_DIR),
                     'defect_source': gt_src if gt_src in (None, 'artificial') else os.path.relpath(gt_src, ROOT_DIR),
                     'background': f'bg_{order[k]:02d}.jpg', 'x0': place[0], 'y0': place[1], 'size': place[2]})
        print(f'[raw] {split}/{defect}/{name}  <- {os.path.basename(os.path.dirname(img_path))}/{os.path.basename(img_path)}'
              f'{"  (+artificial defect)" if gt_src == "artificial" else ""}')

    with open(os.path.join(raw, 'meta.json'), 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f'[done] {len(plan)} images -> {raw}')


if __name__ == '__main__':
    main()
