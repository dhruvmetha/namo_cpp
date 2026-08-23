#!/usr/bin/env python3
"""Policy mode (zero search) on fixed-physics v3, by difficulty x horizon, beside the registered search rows.

Reads eval_reactive_argmax `--leaf-out` shards for every arm, bins each episode into the SAME tiers the
registered search evaluation used (1push: bin_of(solve_rate) over onepush_v3.json; 2push: the `division`
field), and reports open@k as mean +/- sample SD across the three seeds of each arm.

The comparison this exists for: policy open@k costs exactly k simulator calls, and search solve@k costs at
most k, so the two sit on one budget axis with no new search runs. The search columns come straight from
the registered per-seed aggregate.json files, never recomputed here.

    python scripts/rl_loop/agg_policy_v3.py --policy-root $NAMO_SCRATCH/eval/policy_v3_20260822 \
        --search-root $NAMO_SCRATCH/eval/fixed_physics_v3_20260821/full --out <dir>
"""
import argparse
import glob
import json
import os
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
for _p in (f"{REPO}/python", f"{REPO}/scripts", f"{REPO}/scripts/rl_loop"):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from namo import eval_sets  # noqa: E402
from agg_testset_reactive import load_divisions  # noqa: E402  - same tier rule, one implementation

KS = (1, 2, 3, 5, 10)          # replaced in main() from --cuts
TIERS = ("easy", "medium", "hard", "all")
LEGS = {
    "1push": {"dir": "1push_policy", "search_dir": "1push", "tiers": lambda: eval_sets.ONEPUSH},
    "2push": {"dir": "2push_policy", "search_dir": "2push", "tiers": lambda: eval_sets.DIVISIONS},
}


def arm_open_rates(leaf_dir, div):
    """One arm, one leg -> {tier: {k: open@k}} plus the episode counts and the unmatched count."""
    bins = defaultdict(lambda: {"n": 0, "opened": defaultdict(int)})
    n_nomatch = 0
    seen = set()
    for path in sorted(glob.glob(os.path.join(leaf_dir, "shard_*.jsonl"))):
        with open(path) as fh:
            for line in fh:
                if not line.strip():
                    continue
                row = json.loads(line)
                key = (row["xml"], row["object_id"], row.get("region"))
                if key in seen:                      # a stale shard from a different --start/--end split
                    raise RuntimeError(f"duplicate episode {key} in {leaf_dir}: mixed shard counts")
                seen.add(key)
                tier = div.get(key)
                if tier is None:
                    n_nomatch += 1
                    continue
                for t in (tier, "all"):
                    bins[t]["n"] += 1
                    if row["opened_at"]:
                        bins[t]["opened"][row["opened_at"]] += 1
    out = {}
    for tier, b in bins.items():
        out[tier] = {"n": b["n"],
                     **{k: round(100.0 * sum(b["opened"][i] for i in range(1, k + 1)) / max(1, b["n"]), 1)
                        for k in KS}}
    return out, n_nomatch


def band(values):
    """mean +/- sample SD across seeds, formatted the way the registered tables are."""
    if not values:
        return "-"
    if len(values) == 1:
        return f"{values[0]:.1f}"
    return f"{st.mean(values):.1f}±{st.stdev(values):.1f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy-root", required=True)
    ap.add_argument("--search-root", default="", help="registered v3 search aggregates; omit to skip those columns")
    ap.add_argument("--model-arms", default="HY5U_s1,HY5U_s2,HY5U_s3")
    ap.add_argument("--random-arms", default="rand_s7000,rand_s8000,rand_s9000")
    ap.add_argument("--search-model-arms", default="HY5U_s1,HY5U_s2,HY5U_s3")
    ap.add_argument("--search-random-arms", default="random_s7000,random_s8000,random_s9000")
    ap.add_argument("--out", required=True)
    ap.add_argument("--cuts", default="1,2,3,5,10",
                    help="simulator budgets to report. NEVER quote a cut above the run's own depth cap: "
                         "a K=10 rollout's open@30 is just its open@10 wearing a bigger number.")
    a = ap.parse_args()
    global KS
    KS = tuple(int(x) for x in a.cuts.split(","))
    os.makedirs(a.out, exist_ok=True)
    report = {"policy_root": a.policy_root, "search_root": a.search_root, "ks": list(KS), "legs": {}}
    lines = []

    for leg, spec in LEGS.items():
        div = load_divisions(str(spec["tiers"]()))
        per_arm = {}
        for group, arms in (("model", a.model_arms), ("random", a.random_arms)):
            for arm in arms.split(","):
                leaf_dir = os.path.join(a.policy_root, arm, spec["dir"])
                rates, nomatch = arm_open_rates(leaf_dir, div)
                if nomatch:
                    raise RuntimeError(f"{leg} {arm}: {nomatch} leaf rows matched no tier record")
                per_arm[arm] = {"group": group, "rates": rates}

        search = {}
        if a.search_root:
            for group, arms in (("model", a.search_model_arms), ("random", a.search_random_arms)):
                for arm in arms.split(","):
                    agg = json.load(open(os.path.join(a.search_root, arm, "aggregate.json")))
                    search[arm] = {"group": group, "rates": agg[spec["search_dir"]]}

        report["legs"][leg] = {"policy": per_arm, "search_source": a.search_root or None}
        lines.append(f"\n### {leg} — policy (zero search) vs the registered search, same population, same tiers\n")
        head = "| tier | n | " + " | ".join(f"open@{k} HY5U / random" for k in KS) + " |"
        lines.append(head)
        lines.append("|---|---:|" + "---:|" * len(KS))
        for tier in TIERS:
            ns = [per_arm[x]["rates"].get(tier, {}).get("n", 0) for x in per_arm]
            cells = []
            for k in KS:
                mod = [per_arm[x]["rates"][tier][k] for x in per_arm
                       if per_arm[x]["group"] == "model" and tier in per_arm[x]["rates"]]
                rnd = [per_arm[x]["rates"][tier][k] for x in per_arm
                       if per_arm[x]["group"] == "random" and tier in per_arm[x]["rates"]]
                cells.append(f"{band(mod)} / {band(rnd)}")
            lines.append(f"| {tier} | {max(ns)} | " + " | ".join(cells) + " |")

        if search:
            lines.append(f"\n**Search at the same budget** (solve@k, registered `fixed_physics_v3_20260821`)\n")
            lines.append("| tier | " + " | ".join(f"solve@{k} HY5U / random" for k in KS) + " |")
            lines.append("|---|" + "---:|" * len(KS))
            for tier in TIERS:
                cells = []
                for k in KS:
                    mod = [search[x]["rates"][tier][f"solve@{k}"] for x in search
                           if search[x]["group"] == "model" and f"solve@{k}" in search[x]["rates"][tier]]
                    rnd = [search[x]["rates"][tier][f"solve@{k}"] for x in search
                           if search[x]["group"] == "random" and f"solve@{k}" in search[x]["rates"][tier]]
                    cells.append(f"{band(mod)} / {band(rnd)}")
                lines.append(f"| {tier} | " + " | ".join(cells) + " |")
            report["legs"][leg]["search"] = search

    md = "\n".join(lines)
    with open(os.path.join(a.out, "policy_v3.md"), "w") as fh:
        fh.write(md + "\n")
    json.dump(report, open(os.path.join(a.out, "policy_v3.json"), "w"), indent=1)
    print(md)


if __name__ == "__main__":
    main()
