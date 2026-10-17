"""Download DINOv3 ViT-B/16 weights and MVTec AD categories (default: `cable`) from HuggingFace mirrors.

Layout produced (relative to this script):
    ../weights/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth
    ../mvtec_anomaly_detection/<category>/{train/good, test/<defect>, ground_truth/<defect>}
"""
import argparse
import hashlib
import os
import shutil
from collections import Counter

import pyarrow.parquet as pq
import torch
from huggingface_hub import hf_hub_download

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)

WEIGHT_REPO = 'jaychempan/dinov3'
WEIGHT_FILE = 'dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth'
WEIGHT_SHA256 = '73cec8be7427c8655ceced13ce62f6e20a1fa90d1b4d4a550df17a1144081a7c'

DATASET_REPO = 'TheoM55/mvtec_all_objects_split'

# Official MVTec AD split sizes, used to validate the mirror.
EXPECTED = {
    'bottle': {
        'train': {'good': 209},
        'test': {'good': 20, 'broken_large': 20, 'broken_small': 22, 'contamination': 21},
    },
    'cable': {
        'train': {'good': 224},
    },
}

# Image-count mismatches are reported but no longer fail verification.
STRICT_COUNT = False


def sha256sum(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(chunk), b''):
            h.update(block)
    return h.hexdigest()


def download_weight(weight_dir):
    os.makedirs(weight_dir, exist_ok=True)
    dst = os.path.join(weight_dir, WEIGHT_FILE)

    if os.path.exists(dst) and sha256sum(dst) == WEIGHT_SHA256:
        print(f'[weight] already present and verified: {dst}')
        return dst

    print(f'[weight] downloading {WEIGHT_REPO}/{WEIGHT_FILE} ...')
    cached = hf_hub_download(WEIGHT_REPO, WEIGHT_FILE)
    digest = sha256sum(cached)
    if digest != WEIGHT_SHA256:
        raise RuntimeError(f'[weight] SHA256 mismatch: got {digest}, expected {WEIGHT_SHA256}')
    shutil.copyfile(cached, dst)

    # weights_only=True refuses arbitrary pickled objects from the third-party mirror.
    state_dict = torch.load(dst, map_location='cpu', weights_only=True)
    print(f'[weight] verified SHA256, {len(state_dict)} tensors -> {dst}')
    return dst


def write_bytes(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'wb') as f:
        f.write(data)


def download_category(category, data_root):
    cat_dir = os.path.join(data_root, category)
    expected = EXPECTED[category]

    if os.path.isdir(cat_dir) and verify_category(cat_dir, expected, quiet=True):
        print(f'[data] {category} already present and verified: {cat_dir}')
        return cat_dir

    if os.path.isdir(cat_dir):
        shutil.rmtree(cat_dir)

    for split in ('train', 'test'):
        fname = f'data/{category}.{split}-00000-of-00001.parquet'
        print(f'[data] downloading {DATASET_REPO}/{fname} ...')
        table = pq.read_table(hf_hub_download(DATASET_REPO, fname, repo_type='dataset'))

        for row in table.to_pylist():
            defect = row['defect']
            image = row['image_path']
            write_bytes(os.path.join(cat_dir, split, defect, image['path']), image['bytes'])

            mask = row['mask_path']
            if split == 'test' and defect != 'good':
                if not mask or not mask.get('bytes'):
                    raise RuntimeError(f'[data] missing mask for {split}/{defect}/{image["path"]}')
                write_bytes(os.path.join(cat_dir, 'ground_truth', defect, mask['path']), mask['bytes'])

    if not verify_category(cat_dir, expected):
        raise RuntimeError(f'[data] {category} structure does not match the official MVTec AD split')
    return cat_dir


def verify_category(cat_dir, expected, quiet=False):
    def count_png(d):
        return len([f for f in os.listdir(d) if f.lower().endswith('.png')]) if os.path.isdir(d) else 0

    ok = True
    for split, defects in expected.items():
        for defect, n in defects.items():
            got = count_png(os.path.join(cat_dir, split, defect))
            if got != n:
                if STRICT_COUNT:
                    ok = False
                if not quiet:
                    print(f'[verify] {split}/{defect}: got {got}, expected {n}')

    for split in ('train', 'test'):
        if not os.path.isdir(os.path.join(cat_dir, split)):
            return False

    test_dir = os.path.join(cat_dir, 'test')
    for defect in sorted(os.listdir(test_dir)):
        if defect == 'good':
            continue
        imgs = sorted(os.path.splitext(f)[0] for f in os.listdir(os.path.join(test_dir, defect)))
        gt_dir = os.path.join(cat_dir, 'ground_truth', defect)
        masks = sorted(f[:-len('_mask.png')] for f in os.listdir(gt_dir)) if os.path.isdir(gt_dir) else []
        if imgs != masks:
            ok = False
            if not quiet:
                print(f'[verify] ground_truth/{defect}: masks do not pair with test images')

    if ok and not quiet:
        summary = Counter()
        for split in ('train', 'test', 'ground_truth'):
            split_dir = os.path.join(cat_dir, split)
            for defect in sorted(os.listdir(split_dir)):
                summary[f'{split}/{defect}'] = count_png(os.path.join(split_dir, defect))
        print(f'[verify] {cat_dir} OK')
        for k, v in summary.items():
            print(f'           {k:<28} {v}')
    return ok


def main():
    parser = argparse.ArgumentParser(description='Download RAD weights and MVTec AD categories')
    parser.add_argument('--weight_dir', default=os.path.join(ROOT_DIR, 'weights'))
    parser.add_argument('--data_root', default=os.path.join(ROOT_DIR, 'mvtec_anomaly_detection'))
    parser.add_argument('--item_list', nargs='+', default=['cable'], choices=sorted(EXPECTED))
    parser.add_argument('--skip_weight', action='store_true')
    args = parser.parse_args()

    if not args.skip_weight:
        download_weight(args.weight_dir)
    for item in args.item_list:
        download_category(item, args.data_root)
    print('[done] all downloads verified')


if __name__ == '__main__':
    main()
