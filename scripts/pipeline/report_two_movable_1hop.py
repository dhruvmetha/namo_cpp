#!/usr/bin/env python3
"""Split a two-movable single-hop run by whether the doorway needs one block or two.

The aggregate over all 227 scenes is not the result and should not be quoted on its own. 177 of them
have a door a single block opens, which is the problem HY5U was trained on, and they would dominate
any total. The 50 that need both blocks are the new thing, and the question is whether the model's
edge over random survives there or collapses to chance.

Door type comes from `multi_object_edges` in the region snapshot, written by the movable-blob pass:
a marked robot-goal pair means no single block opens it.

  python scripts/pipeline/report_two_movable_1hop.py --run-dir <dir with HY5U_s*/ and rand_s*/> \
      --meta <hy5u_1hop_meta.json>
"""
import argparse
import json
import os
import statistics as st
from collections import defaultdict


def load_arm(root):
    rows = {}
    for name, solved in (("solved.jsonl", True), ("unsolved.jsonl", False)):
        p = os.path.join(root, name)
        if not os.path.exists(p):
            continue
        for line in open(p):
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            rows[r["xml_path"]] = (solved, r)
    return rows


def scene_key(xml):
    return "/".join(xml.split("/")[-4:-1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--meta", required=True)
    a = ap.parse_args()

    meta = {r["scene"]: r for r in json.load(open(a.meta))}
    arms = sorted(d for d in os.listdir(a.run_dir)
                  if os.path.isdir(os.path.join(a.run_dir, d)))

    per_arm = {}
    for arm in arms:
        rows = load_arm(os.path.join(a.run_dir, arm))
        buckets = defaultdict(lambda: {"n": 0, "solved": 0, "sims": []})
        for xml, (ok, r) in rows.items():
            k = scene_key(xml)
            m = meta.get(k) or meta.get("/".join(k.split("/")[-2:]))
            if m is None:
                buckets["UNMATCHED"]["n"] += 1
                continue
            b = buckets["two blocks needed" if m["needs_two_blocks"] else "one block opens it"]
            b["n"] += 1
            if ok:
                b["solved"] += 1
                # solvability_runner names it simulation_budget_used_total. Read it by name rather
                # than guessing across aliases: a silently missing field would print "-" for median
                # sims and the run would look like it produced no timing at all.
                if "simulation_budget_used_total" in r:
                    b["sims"].append(r["simulation_budget_used_total"])
                else:
                    b["no_sim_field"] = b.get("no_sim_field", 0) + 1
        per_arm[arm] = buckets

    print(f"run: {a.run_dir}\n")
    for label in ("one block opens it", "two blocks needed"):
        print(f"--- {label}")
        print(f"{'arm':14} {'solved':>12} {'rate':>8} {'median sims':>12}")
        for arm in arms:
            b = per_arm[arm].get(label)
            if not b or not b["n"]:
                continue
            rate = 100.0 * b["solved"] / b["n"]
            med = f"{st.median(b['sims']):.0f}" if b["sims"] else "-"
            print(f"{arm:14} {b['solved']:>5}/{b['n']:<6} {rate:>7.1f}% {med:>12}")
        model = [per_arm[x][label] for x in arms if x.startswith("HY5U") and label in per_arm[x]]
        rand = [per_arm[x][label] for x in arms if x.startswith("rand") and label in per_arm[x]]
        if model and rand:
            mm = st.mean([100.0 * b["solved"] / b["n"] for b in model])
            rr = st.mean([100.0 * b["solved"] / b["n"] for b in rand])
            print(f"{'model - random':14} {'':>12} {mm - rr:>+7.1f} pt\n")
    un = sum(per_arm[x].get("UNMATCHED", {}).get("n", 0) for x in arms)
    if un:
        print(f"⛔ {un} rows did not match the meta file; the split above is incomplete")
    miss = sum(b.get("no_sim_field", 0) for x in arms for b in per_arm[x].values())
    if miss:
        print(f"⛔ {miss} solved rows carried no simulation count; the sims column is incomplete")


main()
