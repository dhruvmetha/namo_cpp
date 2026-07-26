#!/usr/bin/env python3
"""Build manifest.json for viz/search: arms, episodes with difficulty tiers, and index rows."""
import argparse
import glob
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
for _p in (f"{REPO}/python", f"{REPO}/scripts"):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from namo import eval_sets  # noqa: E402
from viz.index_metrics import rank_of_best_green, top1_truth  # noqa: E402
from viz.trace_schema import episode_filename  # noqa: E402


def index_row(trace, gt, tier):
    xml = trace["meta"]["xml"]
    oid = trace["meta"]["object_id"]
    root = next(b for b in trace["boards"] if b["depth"] == 0)
    pool = root["pool"]
    if gt is None:
        rank, top1 = None, "unknown"
    else:
        openers = {(e, d) for e, d in gt["root"]["openers"]}
        setups = {(e, d) for e, d in gt["root"]["setups"]}
        rank = rank_of_best_green(pool, openers | setups)
        top1 = top1_truth(pool, openers, setups)
    return {"key": episode_filename(xml, oid)[:-len(".json")],
            "xml": xml, "object_id": oid, "tier": tier,
            "solved": trace["result"]["solved"], "sims": trace["result"]["sims"],
            "rank_best_green": rank, "top1": top1, "has_gt": gt is not None}


def _tiers():
    # pure2push_divisions.json is keyed by xml path -> list of per-object entries
    # (one entry per (xml, object_id) pair, with "object_id" and "division" fields),
    # NOT {tier: [entries]}. Join on realpath since the same basename recurs across
    # many different scene directories.
    div = json.load(open(eval_sets.DIVISIONS))
    out = {}
    for xml, entries in div.items():
        rp = os.path.realpath(xml)
        for e in entries:
            out[(rp, e["object_id"])] = e["division"]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True, help="dir holding trace/<model>/<strategy>/ and gt/")
    a = ap.parse_args()
    tiers = _tiers()
    arms, index, episodes = [], {}, {}
    for tdir in sorted(glob.glob(os.path.join(a.data_root, "trace", "*", "*"))):
        strategy = os.path.basename(tdir)
        model = os.path.basename(os.path.dirname(tdir))
        arm = f"{model}|{strategy}"
        arms.append({"model": model, "strategy": strategy, "dir": f"trace/{model}/{strategy}"})
        rows = []
        for tf in sorted(glob.glob(os.path.join(tdir, "*.json"))):
            trace = json.load(open(tf))
            gtf = os.path.join(a.data_root, "gt", os.path.basename(tf))
            gt = json.load(open(gtf)) if os.path.exists(gtf) else None
            tier = tiers.get((os.path.realpath(trace["meta"]["xml"]), trace["meta"]["object_id"]), "unknown")
            row = index_row(trace, gt, tier)
            rows.append(row)
            episodes[row["key"]] = {"xml": row["xml"], "object_id": row["object_id"],
                                    "tier": tier, "has_gt": row["has_gt"]}
        index[arm] = rows
        print(f"{arm}: {len(rows)} episodes")
    json.dump({"arms": arms, "episodes": episodes, "index": index},
              open(os.path.join(a.data_root, "manifest.json"), "w"))


if __name__ == "__main__":
    main()
