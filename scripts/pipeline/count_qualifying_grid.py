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
    rows, seen = [], set()
    for d in dirs:
        for f in glob.glob(os.path.join(d, "*.json")):
            key = os.path.basename(d).replace("_exh2_pull", "") + "/" + os.path.basename(f)
            if key in seen:
                continue
            seen.add(key)
            try:
                rec = json.load(open(f))
            except Exception:
                continue
            cells = rec.get("cells") or []
            if not cells:
                continue
            n_op = sum(1 for c in cells if c["kind"] == "opener")
            n_su = sum(1 for c in cells if c["kind"] == "setup")
            t1 = tier(n_op / len(cells))
            t2 = tier((n_op + n_su) / len(cells))
            tot["scenes"] += 1
            if n_op:
                tot[t1 + "_has1"] += 1
            if n_su:
                tot[t2 + "_has2"] += 1
            if any(c["kind"] == "opener" and c.get("movable_collisions") for c in cells):
                grid[(t1, "1push")].add(key)
            if any(c["kind"] == "setup" and (c.get("movable_collisions") or c.get("finish_movable_collisions"))
                   for c in cells):
                grid[(t2, "2push")].add(key)
            rows.append((key, t1, t2, round(n_op / len(cells), 4),
                         round((n_op + n_su) / len(cells), 4), n_op, n_su, len(cells),
                         rec.get("xml", "")))
    print(f"scenes with cells: {tot['scenes']}")
    print(f"{'tier':6} {'1push':>13} {'2push':>13}   qualifying/has-any-solve")
    for t in ("easy", "med", "hard"):
        print(f"{t:6} {len(grid[(t,'1push')]):>6}/{tot[t+'_has1']:<6} "
              f"{len(grid[(t,'2push')]):>6}/{tot[t+'_has2']:<6}")
    out = os.environ.get("ROWS_OUT")
    if out:
        with open(out, "w") as fh:
            fh.write("key\ttier1\ttier2\tsr1\tsr2\tn_open\tn_setup\tn_cells\txml\n")
            for r in rows:
                fh.write("\t".join(str(x) for x in r) + "\n")
        print(f"rows -> {out}")

main(sys.argv[1:])
