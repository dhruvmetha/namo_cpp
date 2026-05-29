#!/usr/bin/env python3
"""Render dual-crop NPZs (wide 1.2m + tight 0.5m) as PNG panels.

For each NPZ:
  Panel 1: global env view (224x224)
  Panel 2: wide crop (1.2 m) — mask supervision, shows goal_mask_a1 + a2
  Panel 3: tight crop (0.5 m) — SE(2) supervision context
  Panel 4: 64x64 wide (model input view, INTER_AREA)
  Panel 5: 64x64 tight (model input view, INTER_AREA)
SE(2) target + edge/depth indices annotated underneath.
"""
import argparse
import glob
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def composite(masks, h):
    rgb = np.ones((h, h, 3), dtype=np.float32)

    def tint(rgb, m, color, a):
        if m is None:
            return rgb
        m = (m > 0.5)[..., None].astype(np.float32)
        col = np.array(color, dtype=np.float32)
        return rgb * (1 - a * m) + a * m * col

    def solid(rgb, m, color):
        if m is None:
            return rgb
        m = (m > 0.5)[..., None].astype(np.float32)
        col = np.array(color, dtype=np.float32)
        return rgb * (1 - m) + m * col

    rgb = tint(rgb, masks.get('robot_region'), (0.40, 0.65, 0.95), 0.30)
    rgb = tint(rgb, masks.get('goal_region'),  (0.45, 0.85, 0.45), 0.30)
    rgb = solid(rgb, masks.get('static'), (0.2, 0.2, 0.2))
    movable = masks.get('movable')
    tgt_obj = masks.get('tgt_obj')
    if movable is not None and tgt_obj is not None:
        non_tgt = np.clip(movable - tgt_obj, 0, 1)
    else:
        non_tgt = movable
    rgb = solid(rgb, non_tgt, (0.65, 0.65, 0.65))
    rgb = solid(rgb, tgt_obj, (0.20, 0.40, 0.95))
    rgb = tint(rgb, masks.get('mask_a1'), (0.95, 0.20, 0.20), 0.7)
    a2 = masks.get('mask_a2')
    if a2 is not None and (a2 > 0.5).sum() > 0:
        rgb = tint(rgb, a2, (0.98, 0.55, 0.10), 0.5)
    return rgb


def ex(d, key):
    a = d.get(key)
    if a is None:
        return None
    return a[..., 0] if a.ndim == 3 else a


def collect(d, prefix):
    """Pull mask channels for a given prefix (e.g. 'local_wide', 'local_tight', or '' for global).

    Action mask (mask_a1): prefer the explicit goal_mask_a1 key, else fall back
    to target_goal — the tight crop only stores target_goal (which is the same
    content: action[0]'s push target). See visualizer.py:1261 vs 1270.
    """
    if prefix:
        p = prefix + '_'
        return {
            'static':       ex(d, p + 'static'),
            'movable':      ex(d, p + 'movable'),
            'tgt_obj':      ex(d, p + 'target_object'),
            'robot_region': ex(d, p + 'robot_region'),
            'goal_region':  ex(d, p + 'goal_sample_region'),
            'mask_a1':      ex(d, p + 'goal_mask_a1') if (p + 'goal_mask_a1') in d.files else ex(d, p + 'target_goal'),
            'mask_a2':      ex(d, p + 'goal_mask_a2'),
        }
    else:
        return {
            'static':       ex(d, 'static'),
            'movable':      ex(d, 'movable'),
            'tgt_obj':      ex(d, 'target_object'),
            'robot_region': ex(d, 'robot_region'),
            'goal_region':  ex(d, 'goal_sample_region'),
            'mask_a1':      ex(d, 'goal_mask_a1') if 'goal_mask_a1' in d.files else ex(d, 'target_goal'),
            'mask_a2':      ex(d, 'goal_mask_a2'),
        }


def downsample(masks, size):
    out = {}
    for k, m in masks.items():
        if m is None:
            out[k] = None
        else:
            out[k] = cv2.resize(m.astype(np.float32), (size, size), interpolation=cv2.INTER_AREA)
    return out


def render(npz_path, out_path):
    d = np.load(npz_path)

    has_global = 'static' in d.files  # batch_collection --local-only skips these
    wide = collect(d, 'local_wide')
    tight = collect(d, 'local_tight')
    glb = collect(d, '') if has_global else None

    rgb_wide = composite(wide, 224)
    rgb_tight = composite(tight, 224)
    rgb_wide_64 = composite(downsample(wide, 64), 64)
    rgb_tight_64 = composite(downsample(tight, 64), 64)
    rgb_glb = composite(glb, 224) if glb is not None else None

    wide_m = float(d.get('local_wide_crop_size_meters', [-1])[0])
    tight_m = float(d.get('local_tight_crop_size_meters', [-1])[0])
    sd = int(d['solution_depth'][0]) if 'solution_depth' in d else -1
    xml = Path(str(d['xml_file'][0])).name if 'xml_file' in d else ""
    ep = Path(npz_path).stem

    se2_a1 = d['se2_target_a1'] if 'se2_target_a1' in d else np.zeros(3)
    se2_a2 = d['se2_target_a2'] if 'se2_target_a2' in d else np.zeros(3)
    ei1 = int(d['edge_idx_a1'][0]) if 'edge_idx_a1' in d else -1
    ei2 = int(d['edge_idx_a2'][0]) if 'edge_idx_a2' in d else -1
    di1 = int(d['depth_idx_a1'][0]) if 'depth_idx_a1' in d else -1
    di2 = int(d['depth_idx_a2'][0]) if 'depth_idx_a2' in d else -1

    if rgb_glb is not None:
        fig = plt.figure(figsize=(18, 6))
        gs = fig.add_gridspec(1, 5, width_ratios=[1.0, 1.0, 1.0, 0.5, 0.5])
        ax_glb = fig.add_subplot(gs[0])
        ax_glb.imshow(rgb_glb, origin='lower')
        ax_glb.set_title("global (224x224)", fontsize=10)
        ax_glb.set_xticks([]); ax_glb.set_yticks([])
        ax_w, ax_t, ax_w64, ax_t64 = [fig.add_subplot(gs[i]) for i in (1, 2, 3, 4)]
    else:
        fig = plt.figure(figsize=(15, 6))
        gs = fig.add_gridspec(1, 4, width_ratios=[1.0, 1.0, 0.5, 0.5])
        ax_w, ax_t, ax_w64, ax_t64 = [fig.add_subplot(gs[i]) for i in (0, 1, 2, 3)]

    ax_w.imshow(rgb_wide, origin='lower')
    ax_w.set_title(f"local_wide ({wide_m:.2f} m)\n224x224 stored", fontsize=10)
    ax_w.set_xticks([]); ax_w.set_yticks([])

    ax_t.imshow(rgb_tight, origin='lower')
    ax_t.set_title(f"local_tight ({tight_m:.2f} m)\n224x224 stored", fontsize=10)
    ax_t.set_xticks([]); ax_t.set_yticks([])

    ax_w64.imshow(rgb_wide_64, origin='lower', interpolation='nearest')
    ax_w64.set_title("wide @ 64x64", fontsize=9)
    ax_w64.set_xticks([]); ax_w64.set_yticks([])

    ax_t64.imshow(rgb_tight_64, origin='lower', interpolation='nearest')
    ax_t64.set_title("tight @ 64x64", fontsize=9)
    ax_t64.set_xticks([]); ax_t64.set_yticks([])

    se2_str = (f"se2_a1=({se2_a1[0]:+.3f},{se2_a1[1]:+.3f},{se2_a1[2]:+.3f})  "
               f"edge={ei1} depth={di1}")
    if ei2 >= 0:
        se2_str += (f"   se2_a2=({se2_a2[0]:+.3f},{se2_a2[1]:+.3f},{se2_a2[2]:+.3f})  "
                    f"edge={ei2} depth={di2}")
    fig.suptitle(f"{ep}  sol_depth={sd}  {xml}\n{se2_str}", fontsize=9)
    plt.tight_layout(rect=(0, 0, 1, 0.93))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=110, bbox_inches='tight')
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--limit", type=int, default=20)
    args = ap.parse_args()
    in_dir = Path(args.input_dir)
    out_dir = Path(args.output_dir)
    npzs = sorted(in_dir.glob("**/*.npz"))
    if args.limit > 0:
        npzs = npzs[:args.limit]
    print(f"Rendering {len(npzs)} dual-crop views")
    for i, p in enumerate(npzs, 1):
        rel = p.relative_to(in_dir)
        out_png = out_dir / rel.with_suffix(".dualcrop.png")
        try:
            render(p, out_png)
            print(f"  [{i}/{len(npzs)}] {rel}")
        except Exception as e:
            print(f"  [{i}/{len(npzs)}] FAIL: {e}")


if __name__ == "__main__":
    main()
