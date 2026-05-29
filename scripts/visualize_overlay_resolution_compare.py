#!/usr/bin/env python3
"""Side-by-side overlay: stored 224×224 vs model-input 64×64.

For each NPZ in --input-dir, render two panels:
  Left  — stored 224×224 (what the H5 holds)
  Right — 64×64 (what the model sees after the loader's Resize)

The 64×64 version uses cv2.INTER_AREA (the same downsampling pytorch's
Resize uses for shrinking). Both binary masks composited with the same
overlay colors as visualize_overlay.py.
"""
import argparse
import glob
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def composite(masks_dict, h):
    rgb = np.ones((h, h, 3), dtype=np.float32)
    def tint(rgb, m, color, a):
        m = (m > 0.5)[..., None].astype(np.float32)
        col = np.array(color, dtype=np.float32)
        return rgb*(1 - a*m) + a*m*col
    def solid(rgb, m, color):
        m = (m > 0.5)[..., None].astype(np.float32)
        col = np.array(color, dtype=np.float32)
        return rgb*(1 - m) + m*col
    rgb = tint(rgb, masks_dict.get('robot_region', np.zeros((h,h))), (0.4,0.65,0.95), 0.30)
    rgb = tint(rgb, masks_dict.get('goal_region',  np.zeros((h,h))), (0.45,0.85,0.45), 0.30)
    rgb = solid(rgb, masks_dict.get('static', np.zeros((h,h))), (0.2,0.2,0.2))
    movable = masks_dict.get('movable', np.zeros((h,h)))
    tgt_obj = masks_dict.get('tgt_obj', np.zeros((h,h)))
    rgb = solid(rgb, np.clip(movable - tgt_obj, 0, 1), (0.65,0.65,0.65))
    rgb = solid(rgb, tgt_obj, (0.20,0.40,0.95))
    rgb = tint(rgb, masks_dict.get('mask_a1', np.zeros((h,h))), (0.95,0.20,0.20), 0.7)
    if masks_dict.get('mask_a2') is not None and (masks_dict['mask_a2'] > 0.5).sum() > 0:
        rgb = tint(rgb, masks_dict['mask_a2'], (0.98,0.55,0.10), 0.5)
    return rgb


def extract(d, key, ch=0):
    arr = d.get(key)
    if arr is None: return None
    return arr[..., ch] if arr.ndim == 3 else arr


def render(npz_path, out_path):
    d = np.load(npz_path)
    raw = {
        'static':       extract(d, 'local_static'),
        'movable':      extract(d, 'local_movable'),
        'tgt_obj':      extract(d, 'local_target_object'),
        'robot_region': extract(d, 'local_robot_region'),
        'goal_region':  extract(d, 'local_goal_sample_region'),
        'mask_a1':      extract(d, 'local_goal_mask_a1'),
        'mask_a2':      extract(d, 'local_goal_mask_a2'),
    }
    H_full = raw['static'].shape[0]

    # Downsample each binary mask to 64 (the same path the loader uses).
    raw64 = {}
    for k, m in raw.items():
        if m is None: raw64[k] = None; continue
        small = cv2.resize(m.astype(np.float32), (64, 64), interpolation=cv2.INTER_AREA)
        raw64[k] = small

    rgb_full = composite(raw, H_full)
    rgb_64   = composite(raw64, 64)

    crop_m = float(d.get('local_crop_size_meters', [-1])[0]) if 'local_crop_size_meters' in d else -1
    sd = int(d['solution_depth'][0]) if 'solution_depth' in d else -1
    ep = str(d['episode_id'][0]) if 'episode_id' in d else Path(npz_path).stem
    xml = Path(str(d['xml_file'][0])).name if 'xml_file' in d else ""

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11, 5.4))
    axL.imshow(rgb_full, origin='lower')
    axL.set_title(f"stored {H_full}×{H_full}\n(2mm wavefront → {H_full}px)", fontsize=9)
    axL.set_xticks([]); axL.set_yticks([])
    axR.imshow(rgb_64, origin='lower', interpolation='nearest')
    axR.set_title(f"model input 64×64\n(after Resize INTER_AREA)", fontsize=9)
    axR.set_xticks([]); axR.set_yticks([])

    plt.suptitle(f"{ep}\nsolution_depth={sd}  crop={crop_m:.2f} m  ({xml})", fontsize=9)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    in_dir = Path(args.input_dir); out_dir = Path(args.output_dir)
    npzs = sorted(in_dir.glob("**/*.npz"))
    if args.limit > 0: npzs = npzs[:args.limit]
    print(f"Rendering {len(npzs)} side-by-side comparisons")
    for i, p in enumerate(npzs, 1):
        rel = p.relative_to(in_dir)
        out_png = out_dir / rel.with_suffix(".compare.png")
        try:
            render(p, out_png)
            print(f"  [{i}/{len(npzs)}] {rel}")
        except Exception as e:
            print(f"  [{i}/{len(npzs)}] FAIL: {e}")


if __name__ == "__main__":
    main()
