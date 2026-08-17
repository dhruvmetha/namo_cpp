#!/usr/bin/env python3
"""Attach the timed campaign's per-problem results to the scene gallery.

Writes timing.json next to scenes.json: for every card, how long each arm took on that ONE problem,
as mean +/- sample SD across the campaign's three seeds, plus the speed-up. The gallery reads it to
sort scenes by speed-up and to show the numbers beside the picture of the problem they came from.

Source: aquaman/round0/eval_walltime4k -- HY5U s1-3 against uniform random s7000/8000/9000, budget
4000, every shard on a whole exclusive single-generation node, single-threaded. Seconds are only
comparable because of that protocol; do not mix them with times from any other campaign.

An unsolved run spent the entire budget without an answer, so its time is a LOWER bound; those
problems are marked censored and the gallery says so rather than quietly averaging them in.

    python scripts/viz/build_scene_timing.py --out $NAMO_SCRATCH/viz/scenes
"""
import argparse
import glob
import json
import os
import statistics as st

CAMPAIGN = "aquaman/round0/eval_walltime4k"
ARMS = {"model": ["HY5U_s1", "HY5U_s2", "HY5U_s3"],
        "rand": ["rand_s7000", "rand_s8000", "rand_s9000"]}
LEGS = {"1push": "1push_hmax2", "2push": "2push"}


def suf(p, n=5):
    return "/".join(str(p).rstrip("/").split("/")[-n:])


def load(scratch, arm, leg):
    out = {}
    for f in glob.glob(os.path.join(scratch, CAMPAIGN, arm, LEGS[leg], "shard_*.jsonl")):
        for line in open(f):
            r = json.loads(line)
            out[(suf(r["xml"]), r["object_id"])] = r
    return out


def stats(v):
    return [round(st.mean(v), 3), round(st.stdev(v), 3) if len(v) > 1 else 0.0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="gallery data root (holds scenes.json)")
    ap.add_argument("--scratch", default=os.environ.get("NAMO_SCRATCH"))
    a = ap.parse_args()

    index = json.load(open(os.path.join(a.out, "scenes.json")))
    # The card file name is the gallery's key; join to the campaign on (xml suffix, object_id).
    cards = {}
    for row in index["cards"]:
        card = json.load(open(os.path.join(a.out, "cards", row["file"])))
        cards[row["file"]] = (row["horizon"], suf(card["meta"]["xml"]), row["object_id"])

    runs = {leg: {arm: [load(a.scratch, d, leg) for d in dirs] for arm, dirs in ARMS.items()}
            for leg in LEGS}

    out, missing = {}, 0
    for fn, (leg, x, obj) in cards.items():
        got = {arm: [d.get((x, obj)) for d in runs[leg][arm]] for arm in ARMS}
        if any(r is None for rs in got.values() for r in rs):
            missing += 1
            continue
        rec = {}
        for arm, rs in got.items():
            rec[arm] = stats([r["t_wall"] for r in rs])
            rec[arm + "_sims"] = stats([float(r["sims"]) for r in rs])
            rec[arm + "_solved"] = sum(bool(r["solved"]) for r in rs)
        # Speed-up from the seed-mean of each arm -- the quantity the two rows above show, so the
        # page never displays a ratio its own numbers cannot reproduce. (The campaign's canonical
        # statistic pairs seed-to-seed first; it differs slightly and lives in the analysis, not here.)
        rec["up"] = round(rec["rand"][0] / rec["model"][0], 3) if rec["model"][0] else None
        rec["saved_pct"] = round(100 * (1 - rec["model"][0] / rec["rand"][0]), 1) if rec["rand"][0] else None
        rec["censored"] = rec["model_solved"] < 3 or rec["rand_solved"] < 3
        out[fn] = rec

    json.dump({"schema_version": 1, "campaign": CAMPAIGN, "seeds": ARMS, "cards": out},
              open(os.path.join(a.out, "timing.json"), "w"))
    print(f"timing.json: {len(out)} cards timed, {missing} without a campaign row")


if __name__ == "__main__":
    main()
