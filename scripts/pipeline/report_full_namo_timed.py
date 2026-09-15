#!/usr/bin/env python3
"""Report the timed Full NAMO frozen400 runs after the goal-picture fix.

HY5U seeds 1-3 ran on this repo's native Amarel build. The only timed Random is
Tri-An's five seeds from his 2026-09-12 campaign: same CPU model, but his container
build, so its seconds are not directly comparable with ours. The calibration job reran
Random seed 7000 on his first 20 scenes with our build. Scenes that reach the same
result in the same number of simulator calls give the speed ratio between the builds,
and the "scaled" Random rows multiply Tri-An's times by it, as an estimate only.

Scoring is eval_full_namo_walltime.summarize: a failed run takes infinite time.

Usage:
  python scripts/pipeline/report_full_namo_timed.py --ours <campaign>/hy5u/raw \
      --theirs <tri-an campaign>/timing/raw \
      --calibration <campaign>/random_calibration/raw/random_s7000/shard_0000 --out report_timed.json
"""

import argparse
import importlib.util
import json
from pathlib import Path

spec = importlib.util.spec_from_file_location("walltime", Path(__file__).with_name("eval_full_namo_walltime.py"))
walltime = importlib.util.module_from_spec(spec)
spec.loader.exec_module(walltime)

CPU = "Intel(R) Xeon(R) Platinum 8358 CPU @ 2.60GHz"
TIERS = ("easy", "medium", "hard", "unresolved")
HY5U = ("HY5U_s1", "HY5U_s2", "HY5U_s3")
RANDOM = ("random_s7000", "random_s8000", "random_s9000", "random_s10000", "random_s11000")


def load_arm(folder):
    rows = [row for path in sorted(folder.glob("shard_*/outcomes.jsonl")) for row in walltime.read_rows(path)]
    if len(rows) != 400 or len({r["geometry_id"] for r in rows}) != 400 or {r["cpu_model"] for r in rows} != {CPU}:
        raise RuntimeError(f"{folder}: need 400 distinct scenes on {CPU}")
    return rows


def calibration(ours_folder, theirs_folder):
    ours = {r["geometry_id"]: r for r in walltime.read_rows(ours_folder / "outcomes.jsonl")}
    theirs = {r["geometry_id"]: r for r in walltime.read_rows(theirs_folder / "outcomes.jsonl")}
    if set(ours) != set(theirs):
        raise RuntimeError("calibration shards hold different scenes")
    same = [g for g in theirs if (ours[g]["solved"], ours[g]["total_calls"]) == (theirs[g]["solved"], theirs[g]["total_calls"])]
    per_scene = [ours[g]["t_wall"] / theirs[g]["t_wall"] for g in same if theirs[g]["t_wall"] > 0.5]
    return {"scenes": len(theirs), "same_result_and_calls": len(same),
            "different": {g: {"ours": [ours[g]["solved"], ours[g]["total_calls"]],
                              "theirs": [theirs[g]["solved"], theirs[g]["total_calls"]]} for g in theirs if g not in same},
            "ratio": sum(ours[g]["t_wall"] for g in same) / sum(theirs[g]["t_wall"] for g in same),
            "per_scene_ratio_min": min(per_scene), "per_scene_ratio_max": max(per_scene), "per_scene_count": len(per_scene)}


def tier_summaries(arms):
    """summarize() per arm, per tier, plus solved within 60 s."""
    out = {}
    for name, rows in arms.items():
        out[name] = {}
        for tier in TIERS:
            chosen = [r for r in rows if r["difficulty"] == tier]
            out[name][tier] = dict(walltime.summarize(chosen),
                                   solve_at_60s=sum(r["solved"] and r["t_wall"] <= 60 for r in chosen) / len(chosen))
    return out


def span(values, fmt):
    finite = [v for v in values if v is not None]
    if len(finite) < len(values):
        return "not reached" if not finite else f"{fmt(min(finite))} to not reached"
    low, high = fmt(min(finite)), fmt(max(finite))
    return low if low == high else f"{low} to {high}"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ours", required=True, type=Path)
    parser.add_argument("--theirs", required=True, type=Path)
    parser.add_argument("--calibration", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    calib = calibration(args.calibration, args.theirs / "random_s7000" / args.calibration.name)
    ours = {name: load_arm(args.ours / name) for name in HY5U}
    theirs = {name: load_arm(args.theirs / name) for name in RANDOM}
    tiers = {r["geometry_id"]: r["difficulty"] for r in ours["HY5U_s1"]}
    if any({r["geometry_id"]: r["difficulty"] for r in rows} != tiers for rows in [*ours.values(), *theirs.values()]):
        raise RuntimeError("arms disagree on scenes or tiers")
    scaled = {name: [dict(r, t_wall=r["t_wall"] * calib["ratio"]) for r in rows] for name, rows in theirs.items()}
    report = {"calibration": calib, "hy5u": tier_summaries(ours), "random_tri_an_build": tier_summaries(theirs),
              "random_scaled_estimate": tier_summaries(scaled),
              "hy5u_scoring_share_of_time": {name: sum(r["t_score"] for r in rows) / sum(r["t_wall"] for r in rows)
                                             for name, rows in ours.items()}}
    args.out.write_text(json.dumps(report, indent=1) + "\n")

    print(f"Calibration: {calib['same_result_and_calls']}/{calib['scenes']} scenes same result and calls; "
          f"our build takes {calib['ratio']:.3f} of Tri-An's time "
          f"(per scene {calib['per_scene_ratio_min']:.3f} to {calib['per_scene_ratio_max']:.3f}); different: {calib['different']}")
    print("HY5U share of time spent scoring:", {k: round(v, 3) for k, v in report["hy5u_scoring_share_of_time"].items()})
    print("\n| tier | arm | solved /100 | median seconds to solve | solved within 1 s | within 5 s | within 60 s |")
    print("|---|---|---|---|---|---|---|")
    groups = (("HY5U seeds 1-3, our build", report["hy5u"]), ("Random 5 seeds, Tri-An's build", report["random_tri_an_build"]),
              (f"Random 5 seeds, scaled x{calib['ratio']:.3f} (estimate)", report["random_scaled_estimate"]))
    for tier in TIERS:
        for label, summaries in groups:
            cells = [summaries[name][tier] for name in summaries]
            print(f"| {tier} | {label} | {span([c['solved'] for c in cells], str)} "
                  f"| {span([c['median_time_until_success'] for c in cells], lambda v: f'{v:.1f}')} | "
                  + " | ".join(span([100 * c[key] for c in cells], lambda v: f"{v:.0f}")
                               for key in ("solve_at_1s", "solve_at_5s", "solve_at_60s")) + " |")


if __name__ == "__main__":
    main()
