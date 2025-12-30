#!/usr/bin/env python3
"""Check if Diffusion and Hybrid runs match (region goals, ML goals, primitives, skill calls)."""

import pickle
import os
import sys
import numpy as np

def check_match(diffusion_dir: str, hybrid_dir: str):
    if not os.path.exists(diffusion_dir):
        print(f"Diffusion dir not found: {diffusion_dir}")
        return
    if not os.path.exists(hybrid_dir):
        print(f"Hybrid dir not found: {hybrid_dir}")
        return

    common_files = sorted([
        f for f in os.listdir(diffusion_dir)
        if f.endswith('.pkl') and 'env_' in f and f in os.listdir(hybrid_dir)
    ])

    print(f"Common files: {len(common_files)}")

    goals_match = 0
    ml_match = 0
    prims_match = 0
    skills_match = 0
    both_success = 0
    total = 0

    mismatches = []

    for fname in common_files:
        with open(os.path.join(diffusion_dir, fname), 'rb') as f:
            d = pickle.load(f)
        with open(os.path.join(hybrid_dir, fname), 'rb') as f:
            h = pickle.load(f)

        d_ep = d['episode_results'][0]
        h_ep = h['episode_results'][0]
        total += 1

        file_goals_match = False
        file_ml_match = False
        file_prims_match = False

        # Check region goals
        d_goals = d_ep.get('algorithm_stats', {}).get('region_goals_sampled', [])
        h_goals = h_ep.get('algorithm_stats', {}).get('region_goals_sampled', [])
        if d_goals and h_goals and len(d_goals) == len(h_goals):
            if all(abs(dg[0]-hg[0]) < 0.01 and abs(dg[1]-hg[1]) < 0.01 for dg, hg in zip(d_goals, h_goals)):
                goals_match += 1
                file_goals_match = True

        # Check ML goals
        d_ml = d_ep.get('algorithm_stats', {}).get('ml_goals_raw', [])
        h_ml = h_ep.get('algorithm_stats', {}).get('ml_goals_raw', [])
        if d_ml and h_ml and len(d_ml) == len(h_ml):
            if all(abs(dm['x']-hm['x']) < 0.001 and abs(dm['y']-hm['y']) < 0.001 for dm, hm in zip(d_ml, h_ml)):
                ml_match += 1
                file_ml_match = True

        # Check aligned primitives order
        d_prims = d_ep.get('algorithm_stats', {}).get('aligned_primitives', [])
        h_prims = h_ep.get('algorithm_stats', {}).get('aligned_primitives', [])
        if d_prims and h_prims and len(d_prims) == len(h_prims):
            if all(dp['edge_idx'] == hp['edge_idx'] and dp['depth_idx'] == hp['depth_idx']
                   for dp, hp in zip(d_prims, h_prims)):
                prims_match += 1
                file_prims_match = True

        # Check skill calls
        d_skills = d_ep.get('algorithm_stats', {}).get('skill_calls_before_success', -1)
        h_skills = h_ep.get('algorithm_stats', {}).get('skill_calls_before_success', -1)
        if d_ep.get('success') and h_ep.get('success'):
            both_success += 1
            if d_skills == h_skills:
                skills_match += 1

        # Track mismatches
        if not (file_goals_match and file_ml_match and file_prims_match):
            mismatches.append({
                'file': fname,
                'goals': file_goals_match,
                'ml': file_ml_match,
                'prims': file_prims_match,
                'd_skills': d_skills,
                'h_skills': h_skills,
            })

        # Detailed analysis: when both succeed but skill calls differ
        if d_ep.get('success') and h_ep.get('success') and d_skills != h_skills:
            d_action = d_ep.get('action_sequence', [{}])[0]
            h_action = h_ep.get('action_sequence', [{}])[0]
            d_target = d_action.get('target', (None, None, None))
            h_target = h_action.get('target', (None, None, None))

            # Find depth of winning action in D's primitives
            d_depth = None
            h_depth = None
            for p in d_prims:
                if abs(p['x'] - d_target[0]) < 0.01 and abs(p['y'] - d_target[1]) < 0.01:
                    d_depth = p['depth_idx']
                    break
            for p in d_prims:  # Use same primitives for H since they should match
                if abs(p['x'] - h_target[0]) < 0.01 and abs(p['y'] - h_target[1]) < 0.01:
                    h_depth = p['depth_idx']
                    break

            print(f"\n{fname}: D={d_skills} H={h_skills} skill calls")
            print(f"  D action: ({d_target[0]:.3f}, {d_target[1]:.3f}) depth={d_depth}")
            print(f"  H action: ({h_target[0]:.3f}, {h_target[1]:.3f}) depth={h_depth} {'(NOT ML-aligned!)' if h_depth is None else ''}")

    print(f"\n{'='*50}")
    print(f"Match stats (out of {total}):")
    print(f"  Region goals match: {goals_match}/{total} {'✓' if goals_match == total else '✗'}")
    print(f"  ML goals match:     {ml_match}/{total} {'✓' if ml_match == total else '✗'}")
    print(f"  Primitives match:   {prims_match}/{total} {'✓' if prims_match == total else '✗'}")
    print(f"  Both succeed:       {both_success}/{total}")
    print(f"  Skill calls match:  {skills_match}/{both_success} {'✓' if skills_match == both_success else '✗'}")

    if mismatches:
        print(f"\nMismatches ({len(mismatches)}):")
        for m in mismatches[:5]:
            print(f"  {m['file']}: goals={m['goals']}, ml={m['ml']}, prims={m['prims']}, skills D={m['d_skills']} H={m['h_skills']}")

if __name__ == "__main__":
    base = "/common/users/dm1487/namo_data/outputs/cropped_diffusion_crossattn_balanced_solutions/2025-12-26/12-32-32"

    if len(sys.argv) > 1:
        subdir = sys.argv[1]
    else:
        subdir = "modular_data_1spring-303-desktop-08"

    diffusion_dir = f"{base}/results_32samples_int5_fasterbfs_seed42_test/{subdir}"
    hybrid_dir = f"{base}/results_32samples_int5_fasterbfs_seed42_hybrid_test/{subdir}"

    print(f"Diffusion: {diffusion_dir}")
    print(f"Hybrid:    {hybrid_dir}")
    check_match(diffusion_dir, hybrid_dir)
