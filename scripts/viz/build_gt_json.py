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


def _build_doc(rows, kinds, vt, pe, pd):
    root_rows = [i for i in rows if kinds[i] == "root"]
    if not root_rows:
        return None
    o, s = green_sets_from_grid(vt[root_rows[0]])
    doc = {"root": {"openers": _pairs(o), "setups": _pairs(s)}, "finish": {}}
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
    return _build_doc(rows, kinds, vt, pe, pd)


def build_all(h5, key, out_dir):
    # index once: (realpath, object_id) -> row ids, so the pass is linear not quadratic
    xmls = [os.path.realpath(_s(v)) for v in h5["xml"][:]]
    objs = [_s(v) for v in h5["object_id"][:]]
    kinds = [_s(v) for v in h5["node_kind"][:]]
    vt, pe, pd = h5["value_target"], h5["parent_edge"][:], h5["parent_depth"][:]
    by_ep = defaultdict(list)
    for i in range(len(xmls)):
        by_ep[(xmls[i], objs[i])].append(i)
    covered, uncovered = 0, []
    for xml, recs in key.items():
        rp = os.path.realpath(xml)
        for rec in recs:
            oid = rec["object_id"]
            rows = by_ep.get((rp, oid), [])
            doc = _build_doc(rows, kinds, vt, pe, pd)
            if doc is None:
                uncovered.append([xml, oid])
                continue
            json.dump(doc, open(os.path.join(out_dir, episode_filename(xml, oid)), "w"))
            covered += 1
    return covered, uncovered


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    key = json.load(open(eval_sets.PURE2PUSH))
    with h5py.File(eval_sets.TWOPUSH_GT_H5, "r") as f:
        covered, uncovered = build_all(f, key, a.out_dir)
    json.dump({"covered": covered, "uncovered": uncovered},
              open(os.path.join(a.out_dir, "_coverage.json"), "w"))
    print(f"covered={covered} uncovered={len(uncovered)}")


if __name__ == "__main__":
    main()
