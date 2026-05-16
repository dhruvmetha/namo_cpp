#!/usr/bin/env python3
"""Offline ML-goal-model prediction harness.

For each (xml, neighbour_label, object_id) instance in a ground-truth F
characterization pkl, replays the initial state, calls GoalInferenceModel.infer,
aligns the resulting SE(2) samples to primitive slots, and writes
{instance_key, ml_aligned: [(edge_idx, depth_idx, vote_count, x, y, theta)]}.

This does NOT run the planner. It only does inference + alignment, so it can
be scored against the GT F directly with no exploration noise.

Usage:
    python ml_prediction_offline.py \\
        --gt-dir /common/users/dm1487/.../1_push_exhaustive_full/modular_data_rlab7 \\
        --ml-model /common/users/.../cropped_diffusion_crossattn_2push/2025-12-16/05-36-44 \\
        --out /common/users/.../ml_preds.pkl \\
        --samples 32 --seed 42 --sampler ddim --num-steps 5
"""
from __future__ import annotations

import argparse
import os
import pickle
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Allow importing the namo python package from a non-package script.
NAMO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(NAMO_ROOT / "python"))

import namo_rl
from namo.strategies.primitive_goal_strategy import (
    MLPrimitiveGoalStrategy,
    PrimitiveGoalStrategy,
)


def _instance_key(ep: Dict[str, Any]) -> Optional[Tuple[str, str, str]]:
    stats = ep.get("algorithm_stats") or {}
    if not isinstance(stats, dict):
        return None
    obj = stats.get("chosen_object_id")
    region = stats.get("neighbour_region_label")
    xml = ep.get("xml_file")
    if not obj or not region or not xml:
        return None
    return (xml, region, obj)


def load_gt_instances(gt_dir: str) -> List[Dict[str, Any]]:
    """Dedupe (xml, region, object) instances across pkls; keep first trial_log."""
    out: List[Dict[str, Any]] = []
    seen = set()
    pkl_files = sorted(Path(gt_dir).glob("*_results.pkl"))
    for pkl in pkl_files:
        try:
            d = pickle.load(open(pkl, "rb"))
        except Exception as e:
            print(f"  skipped {pkl.name}: {e}", flush=True)
            continue
        for ep in d.get("episode_results") or []:
            k = _instance_key(ep)
            if k is None or k in seen:
                continue
            stats = ep["algorithm_stats"]
            tlog = stats.get("primitive_trial_log")
            if not tlog:
                continue
            seen.add(k)
            out.append({
                "key": k,
                "xml": k[0],
                "region": k[1],
                "object": k[2],
                "robot_goal": ep.get("robot_goal"),
                "trial_log": tlog,
                "src_pkl": pkl.name,
            })
    return out


def run_ml_for_instance(strategy: MLPrimitiveGoalStrategy,
                        env: namo_rl.RLEnvironment,
                        inst: Dict[str, Any]) -> Dict[str, Any]:
    """One ML inference + alignment, returns serialisable record."""
    env.reset()
    if inst.get("robot_goal") is not None:
        try:
            rg = inst["robot_goal"]
            env.set_robot_goal(float(rg[0]), float(rg[1]),
                               float(rg[2]) if len(rg) > 2 else 0.0)
        except Exception:
            pass
    state = env.get_full_state()

    # We need primitive_goals to know which slots exist and to alignment-vote.
    # MLPrimitiveGoalStrategy.generate_goals returns a List[List[Goal]] of shape
    # [edges][depths], where None means "no ML votes for that slot." We don't
    # need the structured return — we'll pull alignment_info directly.
    strategy._last_alignment_info = None
    t0 = time.time()
    try:
        _ = strategy.generate_goals(
            object_id=inst["object"],
            state=state,
            env=env,
            max_goals=0,  # use strategy default (== samples)
            region_goals_sampled=None,
        )
    except Exception as e:
        return {
            "key": inst["key"],
            "ok": False,
            "error": f"{type(e).__name__}: {e}",
            "duration_s": time.time() - t0,
            "ml_aligned": [],
            "ml_samples_raw": [],
        }

    stats = strategy.get_last_goal_stats()
    ml_aligned = []
    for p in stats.get("aligned_primitives", []):
        if p.get("edge_idx") is None or p.get("depth_idx") is None:
            continue
        ml_aligned.append({
            "edge_idx": int(p["edge_idx"]),
            "depth_idx": int(p["depth_idx"]),
            "votes": int(p.get("votes", 0)),
            "x": float(p["x"]) if p.get("x") is not None else None,
            "y": float(p["y"]) if p.get("y") is not None else None,
            "theta": float(p["theta"]) if p.get("theta") is not None else None,
        })
    raw = [{"x": float(g["x"]), "y": float(g["y"]), "theta": float(g["theta"])}
           for g in stats.get("ml_goals_raw", [])]
    return {
        "key": inst["key"],
        "ok": True,
        "duration_s": time.time() - t0,
        "ml_aligned": ml_aligned,
        "ml_samples_raw": raw,
        "alignment_meta": {
            "total_ml_goals": stats.get("ml_goals_generated", 0),
            "total_aligned_slots": stats.get("ml_goals_aligned", 0),
            "reachable_edges_count": stats.get("reachable_edges_count", 0),
            "reachable_edges": stats.get("reachable_edges", []),
        },
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--gt-dir", required=True,
                   help="Directory of *_results.pkl files (F char GT)")
    p.add_argument("--ml-model", required=True,
                   help="Hydra output dir containing the diffusion model")
    p.add_argument("--out", required=True, help="Output pkl path")
    p.add_argument("--config-file", default="config/namo_config_complete_skill15.yaml")
    p.add_argument("--primitive-data-dir", default="data")
    p.add_argument("--samples", type=int, default=32)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda")
    p.add_argument("--sampler", default="ddim",
                   help="Sampler override (ddim, ddpm, euler, midpoint, rk4, dopri5)")
    p.add_argument("--num-steps", type=int, default=5,
                   help="Diffusion / flow-matching integration steps")
    p.add_argument("--pos-tol", type=float, default=0.2)
    p.add_argument("--ang-tol", type=float, default=0.2)
    p.add_argument("--ang-weight", type=float, default=1.0)
    p.add_argument("--k-nearest", type=int, default=1)
    p.add_argument("--max-matches", type=int, default=9999)
    p.add_argument("--limit", type=int, default=None,
                   help="Stop after this many instances (smoke testing)")
    p.add_argument("--filter-manifest", default=None,
                   help="If given, only score instances whose xml is in this manifest")
    args = p.parse_args()

    instances = load_gt_instances(args.gt_dir)
    print(f"Loaded {len(instances)} dedup instances from {args.gt_dir}", flush=True)

    if args.filter_manifest:
        with open(args.filter_manifest) as f:
            allowed = {ln.strip().split("\t")[0] for ln in f
                       if ln.strip() and not ln.startswith("#")}
        before = len(instances)
        instances = [i for i in instances if i["xml"] in allowed]
        print(f"Filtered by manifest: {before} -> {len(instances)}", flush=True)

    if args.limit:
        instances = instances[: args.limit]
        print(f"Limited to {len(instances)} instances", flush=True)

    # Load the goal inference model once
    print(f"Loading GoalInferenceModel from {args.ml_model}", flush=True)
    from sage_learning.goal_inference_model import GoalInferenceModel
    goal_model = GoalInferenceModel(
        model_path=args.ml_model,
        device=args.device,
        sampler_method=args.sampler,
        num_steps=args.num_steps,
    )

    strategy = MLPrimitiveGoalStrategy(
        goal_model_path=args.ml_model,
        primitive_data_dir=args.primitive_data_dir,
        samples=args.samples,
        device=args.device,
        match_position_tolerance=args.pos_tol,
        match_angle_tolerance=args.ang_tol,
        angle_weight=args.ang_weight,
        max_matches=args.max_matches,
        k_nearest=args.k_nearest,
        seed=args.seed,
        preloaded_model=goal_model,
        verbose=False,
    )

    results = []
    last_xml = None
    env = None
    t_start = time.time()
    for idx, inst in enumerate(instances):
        try:
            if inst["xml"] != last_xml:
                # New env — recreate. RLEnvironment is cheap (~ms).
                env = namo_rl.RLEnvironment(inst["xml"], args.config_file, False)
                last_xml = inst["xml"]
            rec = run_ml_for_instance(strategy, env, inst)
        except Exception as e:
            traceback.print_exc()
            rec = {
                "key": inst["key"],
                "ok": False,
                "error": f"{type(e).__name__}: {e}",
                "duration_s": 0.0,
                "ml_aligned": [],
            }
        rec["xml"] = inst["xml"]
        rec["region"] = inst["region"]
        rec["object"] = inst["object"]
        rec["src_pkl"] = inst["src_pkl"]
        results.append(rec)
        if (idx + 1) % 25 == 0 or idx == len(instances) - 1:
            elapsed = time.time() - t_start
            rate = (idx + 1) / max(elapsed, 1e-3)
            eta = (len(instances) - idx - 1) / max(rate, 1e-6)
            ok_n = sum(1 for r in results if r["ok"])
            print(f"[{idx+1}/{len(instances)}] ok={ok_n} "
                  f"rate={rate:.2f}/s elapsed={elapsed:.0f}s eta={eta:.0f}s",
                  flush=True)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump({
            "ml_model": args.ml_model,
            "gt_dir": args.gt_dir,
            "samples": args.samples,
            "seed": args.seed,
            "sampler": args.sampler,
            "num_steps": args.num_steps,
            "pos_tol": args.pos_tol,
            "ang_tol": args.ang_tol,
            "k_nearest": args.k_nearest,
            "results": results,
        }, f)
    print(f"\nWrote {len(results)} predictions to {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
