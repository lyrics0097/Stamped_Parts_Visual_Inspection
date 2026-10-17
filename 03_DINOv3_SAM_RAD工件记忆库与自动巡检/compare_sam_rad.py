"""Side-by-side comparison of RAD anomaly maps: raw cluttered input (group A) vs SAM-purified input (group B).

Anomaly maps are recomputed with the exact algorithm of rad_mvtec_visa_3dadam.py (multi-layer, position-aware
patch-KNN over the top-k CLS neighbours, bilinear upsampling, 5x5 Gaussian smoothing). Image/pixel AUROC over each
full test set are recomputed first so they can be checked against the RAD logs.

Per defect sample:
    recall@tau  share of GT pixels above tau = max anomaly score over all test/good images of the same group
                (i.e. how much of the defect stands out above the background noise seen on normal images)
    FA@R50      share of non-GT pixels that also fire when the threshold is lowered to catch 50% of the defect
                (threshold = median GT-pixel score); this is the false-alarm area an operator would face
    top-1% prec share of the hottest 1% pixels that fall inside GT, plus whether the peak lies inside GT
"""
import argparse
import os

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch.nn import functional as F
from torch.utils.data import DataLoader

from dataset import get_data_transforms, MVTecDataset
from dinov3.hub.backbones import load_dinov3_model
from rad_mvtec_visa_3dadam import tensor_to_np
from utils import get_gaussian_kernel

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)


def neighbour_table(h, w, radius, device):
    table = []
    for j in range(h * w):
        r, c = divmod(j, w)
        idx = [rr * w + cc
               for rr in range(max(0, r - radius), min(h - 1, r + radius) + 1)
               for cc in range(max(0, c - radius), min(w - 1, c + radius) + 1)]
        idx += [j] * ((2 * radius + 1) ** 2 - len(idx))  # duplicates do not change the max
        table.append(idx)
    return torch.tensor(table, device=device)


@torch.no_grad()
def compute_maps(encoder, bank, loader, k_image, pos_radius, resize_mask, device):
    layers, cls_banks, patch_banks = bank['layers'], bank['cls_banks'], bank['patch_banks']
    cls_global = F.normalize(cls_banks[-1].to(device), dim=-1)
    patch_banks = [F.normalize(pb.to(device), dim=-1) for pb in patch_banks]
    gauss = get_gaussian_kernel(kernel_size=5, sigma=1.0).to(device)
    table = None

    out = {'maps': [], 'gts': [], 'labels': [], 'paths': [], 'imgs': []}
    for img, gt, label, paths in loader:
        img = img.to(device)
        inter = encoder.get_intermediate_layers(img, n=layers, return_class_token=True, reshape=False, norm=True)
        sim_img = F.normalize(inter[-1][1], dim=-1) @ cls_global.t()
        topk = sim_img.topk(k_image, dim=-1).indices
        B, L, C = inter[0][0].shape
        h = w = int(L ** 0.5)
        if table is None:
            table = neighbour_table(h, w, pos_radius, device)

        scores = torch.zeros(B, L, device=device)
        for li, (patch_tok, _) in enumerate(inter):
            q = F.normalize(patch_tok, dim=-1)
            for b in range(B):
                neigh = patch_banks[li][topk[b]][:, table]           # [k, L, P, C]
                sim = torch.einsum('lc,klpc->lkp', q[b], neigh)
                scores[b] += (1.0 - sim.flatten(1).max(dim=1).values) / len(layers)

        amap = F.interpolate(scores.view(B, 1, h, w), size=resize_mask, mode='bilinear', align_corners=False)
        amap = gauss(amap)
        gt = F.interpolate(gt.to(device), size=resize_mask, mode='nearest').bool()

        out['maps'].append(amap[:, 0].cpu().numpy())
        out['gts'].append(gt[:, 0].cpu().numpy())
        out['labels'].append(label.numpy())
        out['paths'] += list(paths)
        out['imgs'] += [tensor_to_np(x) for x in img.cpu()]
    for k in ('maps', 'gts', 'labels'):
        out[k] = np.concatenate(out[k])
    return out


def image_scores(maps, max_ratio):
    flat = maps.reshape(len(maps), -1)
    k = int(flat.shape[1] * max_ratio)
    return np.sort(flat, axis=1)[:, ::-1][:, :k].mean(axis=1) if k > 0 else flat.max(axis=1)


def sample_stats(amap, gt, tau):
    top = amap >= np.quantile(amap, 0.99)
    peak = np.unravel_index(np.argmax(amap), amap.shape)
    thr50 = float(np.median(amap[gt]))
    return {
        'recall': (amap > tau)[gt].mean(),
        'thr50': thr50,
        'fa_r50': (amap >= thr50)[~gt].mean(),
        'top1_precision': gt[top].mean(),
        'peak_in_gt': bool(gt[peak]),
        'pixel_auroc': roc_auc_score(gt.ravel(), amap.ravel()),
    }


def overlay(img, amap, vmin, vmax):
    norm = np.clip((amap - vmin) / (vmax - vmin + 1e-8), 0, 1)
    norm = cv2.resize(norm, img.shape[1::-1], interpolation=cv2.INTER_LINEAR)
    heat = plt.get_cmap('jet')(norm)[..., :3]
    return 0.5 * img + 0.5 * heat


def draw_gt(ax, gt, shape):
    g = cv2.resize(gt.astype(np.uint8), shape[1::-1], interpolation=cv2.INTER_NEAREST)
    ax.contour(g, levels=[0.5], colors='lime', linewidths=1.5)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default=os.path.join(ROOT_DIR, 'wild_experiment'))
    parser.add_argument('--sam_bank', default='./bank/wild_sam_bank.pth')
    parser.add_argument('--raw_bank', default='./bank/wild_raw_bank.pth',
                        help='optional fair control A\' (raw bank + raw test); skipped if missing')
    parser.add_argument('--encoder_name', default='dinov3_vitb16')
    parser.add_argument('--encoder_weight', default='../weights/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth')
    parser.add_argument('--k_image', type=int, default=8)
    parser.add_argument('--pos_radius', type=int, default=1)
    parser.add_argument('--resize_mask', type=int, default=256)
    parser.add_argument('--max_ratio', type=float, default=0.01)
    parser.add_argument('--sample', default='000', help='defect sample id shown in the main side-by-side figure')
    parser.add_argument('--out_dir', default='./saved_results_mvtec/exp_compare')
    args = parser.parse_args()

    torch.manual_seed(1)
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    os.makedirs(args.out_dir, exist_ok=True)
    encoder = load_dinov3_model(args.encoder_name, pretrained_weight_path=args.encoder_weight).to(device).eval()
    data_tf, gt_tf = get_data_transforms(512, 448)

    groups = [('A  raw  (SAM bank)', 'raw', args.sam_bank, 'A_raw'),
              ('B  SAM  (SAM bank)', 'sam_cleaned', args.sam_bank, 'B_sam')]
    if args.raw_bank and os.path.exists(args.raw_bank):
        groups.append(("A' raw  (raw bank)", 'raw', args.raw_bank, 'Ap_raw'))

    res = {}
    for name, item, bank_path, key in groups:
        ds = MVTecDataset(os.path.join(args.data_path, item), data_tf, gt_tf, 'test')
        loader = DataLoader(ds, batch_size=4, shuffle=False, num_workers=0)
        bank = torch.load(bank_path, map_location='cpu')
        r = compute_maps(encoder, bank, loader, args.k_image, args.pos_radius, args.resize_mask, device)
        r['i_auroc'] = roc_auc_score(r['labels'], image_scores(r['maps'], args.max_ratio))
        r['p_auroc'] = roc_auc_score(r['gts'].ravel(), r['maps'].ravel())
        r['tau'] = r['maps'][r['labels'] == 0].max()
        res[name] = r

        # dump raw anomaly maps (float32 .npy, RAD 256x256 grid of the Resize(512)+CenterCrop(448) view) and a
        # per-image-normalised grey png, for downstream consumers such as rad_peak_sam_refine.py
        hm_dir = os.path.join(args.out_dir, 'heatmaps', key)
        os.makedirs(hm_dir, exist_ok=True)
        for amap, p in zip(r['maps'], r['paths']):
            p = p.replace('\\', '/')
            stem = f'{p.split("/")[-2]}_{os.path.splitext(os.path.basename(p))[0]}'
            np.save(os.path.join(hm_dir, stem + '.npy'), amap.astype(np.float32))
            grey = (255 * (amap - amap.min()) / (amap.max() - amap.min() + 1e-8)).astype(np.uint8)
            cv2.imwrite(os.path.join(hm_dir, stem + '.png'), grey)
        with open(os.path.join(hm_dir, 'tau.txt'), 'w') as f:
            f.write(f'{r["tau"]:.6f}\n')

    print('\n' + '=' * 96)
    print('Sanity check (should match RAD logs):')
    for name, r in res.items():
        print(f'  {name:<20} I-AUROC={r["i_auroc"]:.4f}  P-AUROC={r["p_auroc"]:.4f}  '
              f'tau(max normal score)={r["tau"]:.4f}')

    defect_ids = sorted(os.path.splitext(os.path.basename(p))[0]
                        for p in res[groups[0][0]]['paths'] if os.sep + 'defect' + os.sep in p or '/defect/' in p)
    print('\nPer defect sample (tau = max score over normal test images; FA@R50 = background area firing at 50% '
          'defect recall):')
    print(f'  {"sample":<8}{"group":<22}{"recall@tau":>11}{"FA@R50":>10}{"top1%-prec":>12}{"peak-in-GT":>12}'
          f'{"px-AUROC":>10}')
    stats = {}
    for sid in defect_ids:
        for name, r in res.items():
            i = next(k for k, p in enumerate(r['paths']) if p.replace('\\', '/').endswith(f'/defect/{sid}.png'))
            s = sample_stats(r['maps'][i], r['gts'][i], r['tau'])
            stats[(sid, name)] = (i, s)
            print(f'  {sid:<8}{name:<22}{s["recall"]:>11.1%}{s["fa_r50"]:>10.2%}{s["top1_precision"]:>12.1%}'
                  f'{"yes" if s["peak_in_gt"] else "NO":>12}{s["pixel_auroc"]:>10.4f}')

    def plot_rows(sids, names, path):
        fig, axes = plt.subplots(len(sids), 2 * len(names), figsize=(4.2 * len(names) * 2, 4.2 * len(sids)),
                                 squeeze=False)
        for row, sid in enumerate(sids):
            maps = [res[n]['maps'][stats[(sid, n)][0]] for n in names]
            vmin, vmax = min(m.min() for m in maps), max(m.max() for m in maps)
            for col, n in enumerate(names):
                i, s = stats[(sid, n)]
                img, gt = res[n]['imgs'][i], res[n]['gts'][i]
                ax_in, ax_map = axes[row, 2 * col], axes[row, 2 * col + 1]
                ax_in.imshow(img)
                draw_gt(ax_in, gt, img.shape)
                ax_in.set_title(f'{n.split("(")[0].strip()} input  [defect {sid}]', fontsize=11)
                ax_map.imshow(overlay(img, maps[col], vmin, vmax))
                draw_gt(ax_map, gt, img.shape)
                ax_map.contour(cv2.resize((maps[col] >= s['thr50']).astype(np.uint8), img.shape[1::-1],
                                          interpolation=cv2.INTER_NEAREST), levels=[0.5], colors='white',
                               linewidths=1.0, linestyles='dashed')
                ax_map.set_title(f'{n}\nFA@R50 {s["fa_r50"]:.1%} | top1% prec {s["top1_precision"]:.0%} | '
                                 f'px-AUROC {s["pixel_auroc"]:.3f}', fontsize=10)
            for ax in axes[row]:
                ax.axis('off')
        fig.suptitle('RAD anomaly map: raw cluttered input vs SAM-purified input  '
                     '(shared colour scale per row; green = GT, white dashed = alarms at 50% defect recall)',
                     fontsize=12)
        plt.tight_layout()
        plt.savefig(path, dpi=130)
        plt.close()

    main_path = os.path.join(args.out_dir, f'compare_defect_{args.sample}.png')
    plot_rows([args.sample], [groups[0][0], groups[1][0]], main_path)
    all_path = os.path.join(args.out_dir, 'compare_all_defects.png')
    plot_rows(defect_ids, [n for n, _, _, _ in groups], all_path)

    a, b = stats[(args.sample, groups[0][0])][1], stats[(args.sample, groups[1][0])][1]
    print('\n' + '-' * 96)
    print(f'Selected defect sample {args.sample}:  A (raw) -> B (SAM)')
    print(f'  normal-noise ceiling tau : {res[groups[0][0]]["tau"]:.4f} -> {res[groups[1][0]]["tau"]:.4f}')
    print(f'  defect recall @ tau  : {a["recall"]:.1%} -> {b["recall"]:.1%}')
    print(f'  false alarms @ R50   : {a["fa_r50"]:.2%} -> {b["fa_r50"]:.2%}  (background area firing)')
    print(f'  top-1% precision     : {a["top1_precision"]:.1%} -> {b["top1_precision"]:.1%}')
    print(f'  peak inside GT       : {a["peak_in_gt"]} -> {b["peak_in_gt"]}')
    print(f'  per-image P-AUROC    : {a["pixel_auroc"]:.4f} -> {b["pixel_auroc"]:.4f}')
    print(f'\n[saved] {os.path.abspath(main_path)}\n[saved] {os.path.abspath(all_path)}')
    print('=' * 96)


if __name__ == '__main__':
    main()
