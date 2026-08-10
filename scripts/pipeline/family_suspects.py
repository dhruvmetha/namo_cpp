#!/usr/bin/env python3
"""Suspect-episode identifier for the family corpus escalation pass. [card § collection]

Reads the BASE-pass H5, scores every CHILD board with the given checkpoint(s), and emits the
episodes where the model is being fooled — the burial criterion applied to training rooms:

    suspect episode := any child cell's score >= the episode's best VERIFIED setup score

Those episodes get the exhaustive (proven-label) escalation pass; everything else keeps its
capped censored labels (crowd-rival role, dirt-tolerant — measured). Episodes with no verified
setup are skipped (no duel exists there). A `--audit-frac` random sample is ALWAYS included
regardless of model opinion (selection-bias insurance + capped-label error meter).

Output: a manifest (one xml path per line, deduped) consumable by family_collect.slurm, plus the
suspect-rate census the cost model turns on.

Usage:
  python scripts/pipeline/family_suspects.py --h5 <base_pass.h5> \
      --ckpt <best1.ckpt> [--ckpt <best2.ckpt>] --out suspects_manifest.txt [--audit-frac 0.02]
"""
import argparse
import random
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
for _p in (str(REPO), str(REPO / "python"), str(REPO / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import h5py
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5", required=True)
    ap.add_argument("--ckpt", action="append", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--audit-frac", type=float, default=0.02)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()

    import importlib.util
    spec = importlib.util.spec_from_file_location("eval_auc", str(REPO / "scripts/eval_auc.py"))
    ea = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ea)

    with h5py.File(a.h5, "r") as f:
        xml = np.array([x.decode() if isinstance(x, bytes) else str(x) for x in f["xml"][:]])
        obj = np.array([x.decode() if isinstance(x, bytes) else str(x) for x in f["object_id"][:]])
        is_root = f["is_root"][:] > 0.5
        reach = f["r_mask"][:] > 0.5
        exact = (f["value_mask"][:] > 0.5) & reach
        tgt = f["value_target"][:]

    episodes = defaultdict(list)
    for i, key in enumerate(zip(xml.tolist(), obj.tolist())):
        episodes[key].append(i)

    # max over ckpts: a child is suspect if ANY current model is fooled by it
    scores = None
    for ck in a.ckpt:
        v = ea.score_h5(ck, a.h5, a.device)
        scores = v if scores is None else np.maximum(scores, v)

    suspects, n_duel, n_skip = set(), 0, 0
    for key, rows in episodes.items():
        roots = [i for i in rows if is_root[i]]
        setup_scores = [float(scores[i][exact[i] & (np.abs(tgt[i] - 0.5) < 0.05) |
                                        (exact[i] & (np.abs(tgt[i] - 0.9) < 0.05))].max())
                        for i in roots
                        if (exact[i] & ((np.abs(tgt[i] - 0.5) < 0.05) | (np.abs(tgt[i] - 0.9) < 0.05))).any()]
        if not setup_scores:
            n_skip += 1
            continue
        n_duel += 1
        best = max(setup_scores)
        for i in rows:
            if is_root[i] or not reach[i].any():
                continue
            if float(scores[i][reach[i]].max()) >= best:
                suspects.add(key[0])
                break

    audit = set(random.Random(0).sample(sorted(set(xml.tolist())),
                                        max(1, int(a.audit_frac * len(set(xml.tolist()))))))
    out_rooms = sorted(suspects | audit)
    Path(a.out).write_text("\n".join(out_rooms) + "\n")
    print(f"[suspects] duel-episodes={n_duel} no-setup-skipped={n_skip} "
          f"suspect-rooms={len(suspects)} (+audit {len(audit - suspects)}) -> {a.out}")
    print(f"[suspects] suspect rate among duel-episodes: {len(suspects) / max(n_duel, 1):.1%}")


if __name__ == "__main__":
    main()
