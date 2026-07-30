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


# Difficulty from EXHAUSTIVE ground truth, on the project's usual fixed cuts. The shipped
# pure2push_divisions.json instead cuts on the MANIFEST's setup count, which is ~1/3 complete and
# undercounts worst on the sparsest episodes -- so its "hard" tier largely means "poorly labelled".
# Same cut values, honest denominator; kept as a SEPARATE field so existing reporting is untouched.
def tier_from_gt(setup_pct):
    if setup_pct is None:
        return None
    return "hard" if setup_pct < 5 else ("medium" if setup_pct < 30 else "easy")


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
    h = (gt or {}).get("hardness") or {}
    return {"key": episode_filename(xml, oid)[:-len(".json")],
            "setup_hardness_pct": h.get("setup_hardness_pct"),
            "tier_gt": tier_from_gt(h.get("setup_hardness_pct")),
            "finish_hardness_mean": h.get("finish_hardness_mean"),
            "finish_hardness_sd": h.get("finish_hardness_sd"),
            "n_setups": h.get("n_setups"),
            "xml": xml, "object_id": oid, "tier": tier,
            "solved": trace["result"]["solved"], "sims": trace["result"]["sims"],
            "rank_best_green": rank, "top1": top1, "has_gt": gt is not None}


def _tier_lookup(div):
    # div is pure2push_divisions.json's parsed form: xml path -> list of
    # per-object entries (one entry per (xml, object_id) pair, with
    # "object_id" and "division" fields), NOT {tier: [entries]}. Join on
    # realpath since the same basename recurs across many different scene
    # directories.
    out = {}
    for xml, entries in div.items():
        rp = os.path.realpath(xml)
        for e in entries:
            out[(rp, e["object_id"])] = e["division"]
    return out


def _tiers():
    return _tier_lookup(json.load(open(eval_sets.DIVISIONS)))


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
            trace_key = (os.path.realpath(trace["meta"]["xml"]), trace["meta"]["object_id"])
            if trace_key not in tiers:
                continue
            gtf = os.path.join(a.data_root, "gt", os.path.basename(tf))
            gt = json.load(open(gtf)) if os.path.exists(gtf) else None
            tier = tiers[trace_key]
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
