#!/usr/bin/env python3
"""Three-panel overlay: full env (global) | 224×224 local crop | 64×64 local crop.

For each NPZ in --input-dir, render three panels showing how the same scene
looks at progressively reduced resolution / scope.
"""
import argparse
import glob
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def composite(masks_dict, h, has_a2_color=True):
    rgb = np.ones((h, h, 3), dtype=np.float32)
    def tint(rgb, m, color, a):
        if m is None: return rgb
        m = (m > 0.5)[..., None].astype(np.float32)
        col = np.array(color, dtype=np.float32)
        return rgb*(1 - a*m) + a*m*col
    def solid(rgb, m, color):
        if m is None: return rgb
        m = (m > 0.5)[..., None].astype(np.float32)
        col = np.array(color, dtype=np.float32)
        return rgb*(1 - m) + m*col
    rgb = tint(rgb, masks_dict.get('robot_region'), (0.4,0.65,0.95), 0.30)
    rgb = tint(rgb, masks_dict.get('goal_region'),  (0.45,0.85,0.45), 0.30)
    rgb = solid(rgb, masks_dict.get('static'), (0.2,0.2,0.2))
    movable = masks_dict.get('movable')
    tgt_obj = masks_dict.get('tgt_obj')
    if movable is not None and tgt_obj is not None:
        non_tgt = np.clip(movable - tgt_obj, 0, 1)
    else:
        non_tgt = movable
    rgb = solid(rgb, non_tgt, (0.65,0.65,0.65))
    rgb = solid(rgb, tgt_obj, (0.20,0.40,0.95))
    rgb = tint(rgb, masks_dict.get('mask_a1'), (0.95,0.20,0.20), 0.7)
    if has_a2_color and masks_dict.get('mask_a2') is not None and (masks_dict['mask_a2'] > 0.5).sum() > 0:
        rgb = tint(rgb, masks_dict['mask_a2'], (0.98,0.55,0.10), 0.5)
    return rgb


def extract(d, key, ch=0):
    arr = d.get(key)
    if arr is None: return None
    return arr[..., ch] if arr.ndim == 3 else arr


def render(npz_path, out_path):
    d = np.load(npz_path)
    # Local masks
    local = {
        'static':       extract(d, 'local_static'),
        'movable':      extract(d, 'local_movable'),
        'tgt_obj':      extract(d, 'local_target_object'),
        'robot_region': extract(d, 'local_robot_region'),
        'goal_region':  extract(d, 'local_goal_sample_region'),
        'mask_a1':      extract(d, 'local_goal_mask_a1'),
        'mask_a2':      extract(d, 'local_goal_mask_a2'),
    }
    # Global masks (full env view)
    glb = {
        'static':       extract(d, 'static'),
        'movable':      extract(d, 'movable'),
        'tgt_obj':      extract(d, 'target_object'),
        'robot_region': extract(d, 'robot_region'),
        'goal_region':  extract(d, 'goal_sample_region'),
        'mask_a1':      extract(d, 'goal_mask_a1') if d.get('goal_mask_a1') is not None else extract(d, 'target_goal'),
        'mask_a2':      extract(d, 'goal_mask_a2'),
    }
    H_local = local['static'].shape[0]
    H_glb = glb['static'].shape[0] if glb['static'] is not None else H_local

    # Downsample local to 64
    local_64 = {}
    for k, m in local.items():
        if m is None: local_64[k] = None; continue
        local_64[k] = cv2.resize(m.astype(np.float32), (64, 64), interpolation=cv2.INTER_AREA)

    rgb_glb   = composite(glb, H_glb)
    rgb_local = composite(local, H_local)
    rgb_64    = composite(local_64, 64)

    crop_m = float(d.get('local_crop_size_meters', [-1])[0]) if 'local_crop_size_meters' in d else -1
    sd = int(d['solution_depth'][0]) if 'solution_depth' in d else -1
    xml = Path(str(d['xml_file'][0])).name if 'xml_file' in d else ""

    fig, axes = plt.subplots(1, 3, figsize=(15, 5.5))
    axes[0].imshow(rgb_glb, origin='lower')
    axes[0].set_title(f"global (full env)\n{H_glb}×{H_glb}", fontsize=10)
    axes[1].imshow(rgb_local, origin='lower')
    axes[1].set_title(f"local crop stored\n{H_local}×{H_local}  ({crop_m:.2f} m)", fontsize=10)
    axes[2].imshow(rgb_64, origin='lower', interpolation='nearest')
    axes[2].set_title(f"model input\n64×64  (after INTER_AREA)", fontsize=10)
    for ax in axes: ax.set_xticks([]); ax.set_yticks([])

    ep = Path(npz_path).stem
    plt.suptitle(f"{ep}   solution_depth={sd}   {xml}", fontsize=10)
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
    print(f"Rendering {len(npzs)} three-panel views")
    for i, p in enumerate(npzs, 1):
        rel = p.relative_to(in_dir)
        out_png = out_dir / rel.with_suffix(".threepanel.png")
        try:
            render(p, out_png)
            print(f"  [{i}/{len(npzs)}] {rel}")
        except Exception as e:
            print(f"  [{i}/{len(npzs)}] FAIL: {e}")


if __name__ == "__main__":
    main()
