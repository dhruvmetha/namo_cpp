#!/usr/bin/env python3
"""Bucket eval_bestfirst leaf-jsonl results by difficulty tier (easy/medium/hard).

Joins each episode (xml, object_id, region) to its PER-EPISODE `division` in
pure2push_divisions.json. Difficulty is per-episode (data invariant) — never inherited from the room.
Usage:
  python tier_breakdown.py --divisions <pure2push_divisions.json> --leaf a.jsonl [b.jsonl ...]
"""
import json, os, argparse
from collections import defaultdict


def load_div(path):
    div = json.load(open(path))
    lut = {}
    for xml, recs in div.items():
        rp = os.path.realpath(xml)
        for r in recs:
            k = (r["object_id"], r.get("region"))
            lut[(xml, *k)] = r["division"]
            lut[(rp, *k)] = r["division"]
    return lut


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--divisions", required=True)
    ap.add_argument("--leaf", required=True, nargs="+")
    a = ap.parse_args()
    lut = load_div(a.divisions)
    for leaf in a.leaf:
        rows = [json.loads(l) for l in open(leaf)]
        agg = defaultdict(lambda: {"n": 0, "solved": 0, "sims_solved": 0, "sims_all": 0})
        matched = 0
        for r in rows:
            div = (lut.get((r["xml"], r["object_id"], r.get("region")))
                   or lut.get((os.path.realpath(r["xml"]), r["object_id"], r.get("region"))))
            if div is None:
                div = "UNMATCHED"
            else:
                matched += 1
            b = agg[div]
            b["n"] += 1
            b["solved"] += int(r["solved"])
            b["sims_all"] += r["sims"]
            if r["solved"]:
                b["sims_solved"] += r["sims"]
        print(f"\n### {os.path.basename(leaf)}   matched {matched}/{len(rows)} = {100*matched/max(len(rows),1):.0f}%")
        print(f"{'tier':10s}{'n':>5s}{'solve%':>8s}{'sims_to_solve':>15s}{'sims_all':>10s}")
        order = ["easy", "medium", "hard", "UNMATCHED"]
        tot = {"n": 0, "solved": 0}
        for tier in order + [t for t in agg if t not in order]:
            if tier not in agg:
                continue
            b = agg[tier]
            sr = 100 * b["solved"] / max(b["n"], 1)
            sts = b["sims_solved"] / max(b["solved"], 1)
            sa = b["sims_all"] / max(b["n"], 1)
            print(f"{tier:10s}{b['n']:5d}{sr:8.1f}{sts:15.2f}{sa:10.2f}")
            if tier != "UNMATCHED":
                tot["n"] += b["n"]; tot["solved"] += b["solved"]
        print(f"{'ALL':10s}{tot['n']:5d}{100*tot['solved']/max(tot['n'],1):8.1f}")


if __name__ == "__main__":
    main()
