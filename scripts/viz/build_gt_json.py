#!/usr/bin/env python3
"""Join testset_gt.h5 into per-episode green sets for the search visualization.

GT is used as a BADGE, never as a number: value_target 1.0 = opener, 0.9 = setup whose subtree
contained a verified win. Everything else is not green. Label semantics: build_rung2_h5.py:95-113.

testset_gt.h5 roots 981 of the 1018 manifest episodes (build-version drift recorded in
docs/experiments/eval_set_registry.md). Uncovered episodes are listed in _coverage.json, never faked."""
import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np

REPO = Path(__file__).resolve().parents[2]
for _p in (f"{REPO}/python", f"{REPO}/scripts"):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from namo import eval_sets  # noqa: E402
from viz.trace_schema import episode_filename  # noqa: E402


def _s(v):
    return v.decode() if isinstance(v, bytes) else str(v)


def green_sets_from_grid(value_target):
    openers = set(zip(*np.where(value_target == 1.0)))
    setups = set(zip(*np.where(value_target == 0.9)))
    return {(int(e), int(d)) for e, d in openers}, {(int(e), int(d)) for e, d in setups}


def _pairs(s):
    return sorted([int(e), int(d)] for e, d in s)


def _hardness(rows, kinds, vt, rm, pe, pd, root_i):
    """How sparse are the RIGHT pushes, as a percentage of what the search must sift through.

    setup_hardness_pct = GT setups at the root / reachable pushes at the root.
    Then one figure per true setup: its finish board's GT openers / that board's reachable pushes.
    Higher = a larger share of pushes work = an EASIER episode; the name follows the user's wording.
    The finish numerator is never zero by construction -- a root push is labelled a setup precisely
    because its subtree contained a win -- so this measures sparsity, not existence."""
    reach = float(rm[root_i].sum())
    _, setups = green_sets_from_grid(vt[root_i])
    out = {"setup_hardness_pct": round(100.0 * len(setups) / reach, 3) if reach else None,
           "n_setups": len(setups)}
    fin = []
    for i in rows:
        if kinds[i] == "root" or (int(pe[i]), int(pd[i])) not in setups:
            continue
        r2 = float(rm[i].sum())
        if not r2:
            continue
        fo, _ = green_sets_from_grid(vt[i])
        fin.append(100.0 * len(fo) / r2)
    if fin:
        m = sum(fin) / len(fin)
        out["finish_hardness_mean"] = round(m, 3)
        out["finish_hardness_sd"] = round((sum((x - m) ** 2 for x in fin) / len(fin)) ** 0.5, 3)
        out["n_finish_boards"] = len(fin)
    else:
        out["finish_hardness_mean"] = out["finish_hardness_sd"] = None
        out["n_finish_boards"] = 0
    return out


def _build_doc(rows, kinds, vt, pe, pd, rm=None):
    root_rows = [i for i in rows if kinds[i] == "root"]
    if not root_rows:
        return None
    o, s = green_sets_from_grid(vt[root_rows[0]])
    doc = {"root": {"openers": _pairs(o), "setups": _pairs(s)}, "finish": {}}
    if rm is not None:
        doc["hardness"] = _hardness(rows, kinds, vt, rm, pe, pd, root_rows[0])
    for i in rows:
        if kinds[i] == "root":
            continue
        fo, fs = green_sets_from_grid(vt[i])
        doc["finish"][f"{int(pe[i])}_{int(pd[i])}"] = {"openers": _pairs(fo), "setups": _pairs(fs)}
    return doc


def build_episode_gt(h5, xml, object_id):
    xmls = [_s(v) for v in h5["xml"][:]]
    objs = [_s(v) for v in h5["object_id"][:]]
    kinds = [_s(v) for v in h5["node_kind"][:]]
    rows = [i for i in range(len(xmls))
            if os.path.realpath(xmls[i]) == os.path.realpath(xml) and objs[i] == object_id]
    vt = h5["value_target"]
    pe, pd = h5["parent_edge"][:], h5["parent_depth"][:]
    return _build_doc(rows, kinds, vt, pe, pd, h5["r_mask"])


def build_all(h5, key, out_dir):
    # index once: (realpath, object_id) -> row ids, so the pass is linear not quadratic
    xmls = [os.path.realpath(_s(v)) for v in h5["xml"][:]]
    objs = [_s(v) for v in h5["object_id"][:]]
    kinds = [_s(v) for v in h5["node_kind"][:]]
    vt, pe, pd, rm = h5["value_target"], h5["parent_edge"][:], h5["parent_depth"][:], h5["r_mask"]
    by_ep = defaultdict(list)
    for i in range(len(xmls)):
        by_ep[(xmls[i], objs[i])].append(i)
    covered, uncovered = 0, []
    for xml, recs in key.items():
        rp = os.path.realpath(xml)
        for rec in recs:
            oid = rec["object_id"]
            rows = by_ep.get((rp, oid), [])
            doc = _build_doc(rows, kinds, vt, pe, pd, rm)
            if doc is None:
                uncovered.append([xml, oid])
                continue
            json.dump(doc, open(os.path.join(out_dir, episode_filename(xml, oid)), "w"))
            covered += 1
    return covered, uncovered


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--h5", default=str(eval_sets.TWOPUSH_GT_H5))
    ap.add_argument("--key", default=str(eval_sets.PURE2PUSH))
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    key = json.load(open(a.key))
    with h5py.File(a.h5, "r") as f:
        covered, uncovered = build_all(f, key, a.out_dir)
    json.dump({"covered": covered, "uncovered": uncovered},
              open(os.path.join(a.out_dir, "_coverage.json"), "w"))
    print(f"covered={covered} uncovered={len(uncovered)}")


if __name__ == "__main__":
    main()
