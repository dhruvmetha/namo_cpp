"""Per-generation report row + pre-registered kill-signal checks.

Kill signals (card ## Kill signals):
  1. gen-1, pre-training: hard-episode positive coverage < 50%  -> collection redesign.
  2. end of gen-1: held-out hard-2push greedy open (@max) < 35%  -> forecast falsified.
  3. whole approach: hard-tier unique-solve coverage FLAT across two consecutive gens.

check_kill_signals prints PASS/FAIL explicitly and returns the machine-readable verdicts;
it never raises (the orchestrator gates on the numbers).
"""
from typing import Optional
import json


HARD_COVERAGE_MIN = 0.50          # signal 1
HARD_2PUSH_GREEDY_MIN = 35.0      # signal 2 (percent, open@eval_max)


def check_kill_signals(generation: int, buffer_stats: dict, eval_report: dict,
                       eval_max_k: int, prev_hard_unique: Optional[int] = None) -> dict:
    out = {}

    # --- signal 1: hard-episode positive coverage (gen-1 gate) ---
    cov = buffer_stats.get("hard_positive_coverage")
    if cov is not None:
        ok = cov >= HARD_COVERAGE_MIN
        out["signal1_hard_coverage"] = {
            "value": round(cov, 3), "threshold": HARD_COVERAGE_MIN,
            "status": "PASS" if ok else "FAIL",
            "active": generation >= 1}

    # --- signal 2: held-out hard-2push greedy open@max (end of gen-1) ---
    cell = eval_report.get("2push/hard", {})
    g = cell.get(f"open@{eval_max_k}")
    if g is not None:
        ok = g >= HARD_2PUSH_GREEDY_MIN
        out["signal2_hard2push_greedy"] = {
            "value": g, "threshold": HARD_2PUSH_GREEDY_MIN,
            "status": "PASS" if ok else "FAIL",
            "active": generation >= 1}

    # --- signal 3: hard unique-solve coverage flat vs previous generation ---
    hard_unique = buffer_stats.get("unique_solves_by_tier", {}).get("hard", 0)
    if prev_hard_unique is not None:
        flat = hard_unique <= prev_hard_unique
        out["signal3_hard_unique_flat"] = {
            "value": hard_unique, "prev": prev_hard_unique,
            "status": "FAIL" if flat else "PASS",
            "note": "compare across TWO consecutive gens before acting"}
    return out


def build_report_row(generation: int, cfg_arm: str, buffer_stats: dict, eval_report: dict,
                     kills: dict, extra: Optional[dict] = None) -> dict:
    return {
        "generation": generation, "arm": cfg_arm,
        "buffer": buffer_stats, "greedy_open_by_tier_horizon": eval_report,
        "kill_signals": kills, **(extra or {}),
    }


def print_report_row(row: dict) -> None:
    print("=" * 78)
    print(f" GENERATION {row['generation']}  (arm {row['arm']})")
    print("=" * 78)
    print(" buffer composition:")
    b = row["buffer"]
    print(f"   unique solves by tier : {b.get('unique_solves_by_tier')}")
    print(f"   episodes w/ solve     : {b.get('episodes_with_solve_by_tier')}")
    print(f"   hard positive coverage: {b.get('hard_positive_coverage')}")
    print(f"   fail records (V-head) : {b.get('n_fail_records')}")
    print(" greedy open@k  (horizon/difficulty):")
    for k in sorted(row["greedy_open_by_tier_horizon"]):
        print(f"   {k:<14} {row['greedy_open_by_tier_horizon'][k]}")
    if row.get("setup_ranking_by_tier_horizon"):
        print(" setup-hit@k  (USER headline — finishable setup in policy top-k; horizon/difficulty):")
        for k in sorted(row["setup_ranking_by_tier_horizon"]):
            print(f"   {k:<14} {row['setup_ranking_by_tier_horizon'][k]}")
    print(" kill signals:")
    for name, v in row["kill_signals"].items():
        print(f"   {name:<26} {v}")
    print("=" * 78, flush=True)


def save_report_row(row: dict, path: str) -> None:
    import os
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(row, f, indent=2)
