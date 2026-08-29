#!/usr/bin/env python3
"""Recount the qualifying easy/med/hard x 1push/2push grid over every exhaustive sweep dir.

Qualifies (project_multimovable_selection_criterion): a scene counts only when a SOLVING push also
makes the movables touch.
  1push -> some cell kind=="opener" with non-empty movable_collisions
  2push -> some cell kind=="setup" with contact on the setup OR on its winning finish

Difficulty uses the solve rate OF THAT DOMAIN, which is the same convention as the rest of the
project rather than a new one. build_episode_validsets.py:79 defines the canonical 1-push
solve_rate as len(valid)/len(log) over a depth-1 enumeration, so a setup is not in the numerator
and not in the denominator. Binning the 1-push domain by the depth-2 rate instead was wrong and it
emptied the hard/1push cell on its own: over 1549 scenes the lowest depth-2 rate any opener-bearing
scene reached was 0.0765, above the 0.05 hard cut, because a scene with a working single push always
carries enough setups to lift the combined rate. Read against openers/cells, 188 of those same
scenes are hard.
  1push tier <- openers / cells
  2push tier <- (openers + setups) / cells      both solve within a 2-push horizon
Cuts are eval_common.bin_of, unchanged: hard <0.05, med <0.30, easy >=0.30.
"""
import json, sys, glob, os, collections

def tier(sr):
    return "hard" if sr < 0.05 else ("med" if sr < 0.30 else "easy")

def main(dirs):
    grid = collections.defaultdict(set)
    tot = collections.Counter()
    rows, seen, keep = [], set(), {}
    for d in dirs:
        for f in glob.glob(os.path.join(d, "*.json")):
            try:
                rec = json.load(open(f))
            except Exception:
                continue
            cells = rec.get("cells") or []
            if not cells:
                continue
            # Dedupe on the SCENE, not on the sweep filename. Two pull dirs can hold two copies of
            # one Amarel run under different names, which is how v4_exh2_pull and v4_snap_exh2_pull
            # double-counted 88 scenes and inflated every cell of the grid.
            key = rec["xml"]
            if key in seen:
                continue
            seen.add(key)
            n_op = sum(1 for c in cells if c["kind"] == "opener")
            n_su = sum(1 for c in cells if c["kind"] == "setup")
            t1 = tier(n_op / len(cells))
            t2 = tier((n_op + n_su) / len(cells))
            tot["scenes"] += 1
            if n_op:
                tot[t1 + "_has1"] += 1
            if n_su:
                tot[t2 + "_has2"] += 1
            hit1 = [c for c in cells if c["kind"] == "opener" and c.get("movable_collisions")]
            hit2 = [c for c in cells if c["kind"] == "setup"
                    and (c.get("movable_collisions") or c.get("finish_movable_collisions"))]
            # `key` is the path of the box the sweep RAN on, so an Amarel-labelled scene points at
            # /scratch/... which does not exist on CS. Record the pool-relative path too; that is the
            # half that travels, and manifest_to_scenes.py rebases on it.
            parts = key.split("/")
            rec_out = {"xml": key, "relpath": "/".join(parts[-4:]),
                       "scene": "/".join(parts[-4:-1]), "family": parts[-3],
                       "sr_1push": round(n_op / len(cells), 4),
                       "sr_2push": round((n_op + n_su) / len(cells), 4),
                       "tier_1push": t1, "tier_2push": t2,
                       "n_openers": n_op, "n_setups": n_su, "n_cells": len(cells),
                       "n_openers_contact": len(hit1), "n_setups_contact": len(hit2)}
            if hit1:
                grid[(t1, "1push")].add(key)
            if hit2:
                grid[(t2, "2push")].add(key)
            if hit1 or hit2:
                keep[key] = rec_out
            rows.append((rec_out["scene"], t1, t2, rec_out["sr_1push"], rec_out["sr_2push"],
                         n_op, n_su, len(cells), key))
    print(f"scenes with cells: {tot['scenes']}")
    print(f"{'tier':6} {'1push':>13} {'2push':>13}   qualifying/has-any-solve")
    for t in ("easy", "med", "hard"):
        print(f"{t:6} {len(grid[(t,'1push')]):>6}/{tot[t+'_has1']:<6} "
              f"{len(grid[(t,'2push')]):>6}/{tot[t+'_has2']:<6}")
    out = os.environ.get("ROWS_OUT")
    if out:
        with open(out, "w") as fh:
            fh.write("scene\ttier1\ttier2\tsr1\tsr2\tn_open\tn_setup\tn_cells\txml\n")
            for r in rows:
                fh.write("\t".join(str(x) for x in r) + "\n")
        print(f"rows -> {out}")
    man = os.environ.get("MANIFEST_OUT")
    if man:
        # Order inside a cell round-robins over generator family, so if the next step takes the
        # head of a list it gets a spread rather than a run of near-identical layouts out of one
        # seed. Same reason select_real_scene_tiers.py spreads instead of taking first-N.
        cells_out = {}
        for (t, h), xmls in sorted(grid.items()):
            byfam = collections.defaultdict(list)
            for x in sorted(xmls):
                byfam[keep[x]["family"]].append(keep[x])
            order, fams = [], sorted(byfam)
            while any(byfam[f] for f in fams):
                for f in fams:
                    if byfam[f]:
                        order.append(byfam[f].pop(0))
            cells_out[f"{t}_{h}"] = order
        json.dump({"n_scenes_labelled": tot["scenes"],
                   "criterion": "a push that opens the region AND makes the movables touch",
                   "relpath_root": "the real_buildable_2mov pool root on whichever box you are on; "
                                   "`xml` is the absolute path of the box that labelled it and is "
                                   "wrong everywhere else",
                   "tier_rule": {"1push": "openers/cells", "2push": "(openers+setups)/cells",
                                 "cuts": "hard <0.05, med <0.30, easy >=0.30"},
                   "counts": {k: len(v) for k, v in cells_out.items()},
                   "cells": cells_out}, open(man, "w"), indent=1)
        print(f"manifest -> {man}")

main(sys.argv[1:])
