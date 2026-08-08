#!/usr/bin/env python3
"""Aquaman-0 rebuild — write bootstrapped two-sided targets onto colossus-block capped root cells.

EXP-2026-08-02-bootstrap-value-loop, round 0 step 2. Zero sims.

Inputs
  DEPLOY = d20_plus_setup_only.h5   (257,409 rows; 26,023 colossus setup roots, amav=1)
  SOURCE = colossus0_d20_plus_200k.h5 (431,386 rows; holds the 157,310 colossus CHILD boards
           that were dropped from DEPLOY — the boards a guess must look at)

Per colossus root board (amav=1, is_root=1) in DEPLOY:
  1. LINK: children = SOURCE rows, same (xml, object_id), is_root=0, amav=1. Each child's
     target_object_state is the post-push pose; predicted pose per root cell = root pose +
     denormalized action_motion (dx*0.5m, dy*0.5m, dtheta*pi, world frame per H5 attrs).
     Child -> nearest capped root cell; gate pos<=GATE_POS m, |wrapped dyaw|<=GATE_YAW rad;
     one child per cell (nearest wins, rest logged). Children matching EXACT cells are skipped
     (verified setups stay verified).
  2. GUESS: V-hat(child) = mean of top-5 raw E[bin] theta0 scores over the child's UNTRIED
     reachable cells (r_mask & ~value_mask). Child with no untried cells -> no guess (cell
     keeps its mute ceiling; nothing to look at).
  3. WRITE (into a fresh copy of DEPLOY): value_target = min(0.81, 0.9*V-hat),
     ceiling_mask = 0 (two-sided now), guess_mask = 1. Everything else byte-identical.
     New dataset guess_mask (N,60,5) uint8: trainer weights guessed cells at 0.5.

Every count (linkable children, gate failures, ambiguities, guessed cells, target histogram)
is printed and saved — the match rate IS a result of this step.
"""
import json
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np

REPO = Path(__file__).resolve().parents[2]
for _p in (REPO / "python", REPO / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from eval_auc import score_h5  # noqa: E402  cached (ckpt, h5) -> (N,60,5) raw E[bin]

R3 = Path("/common/users/dm1487/scratch_namo/curriculum2/beast/round3")
DEPLOY = R3 / "h5/d20_plus_setup_only.h5"
SOURCE = R3 / "h5/colossus0_d20_plus_200k.h5"
CKPT = R3 / "models/d20_plus_setup_only_splitloss/checkpoints/epoch011-val_loss1.6952.ckpt"
OUT = Path("/common/users/dm1487/scratch_namo/aquaman/round0/aquaman0_train.h5")
REPORT = OUT.with_suffix(".report.json")
GAMMA, CAP, TOP_M = 0.9, 0.81, 5
GATE_POS, GATE_YAW = 0.10, 0.35  # nominal-vs-actual tolerance; error percentiles reported


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def main():
    device = "cuda"
    src_vals = score_h5(str(CKPT), str(SOURCE), device)  # (431386,60,5) raw, cached

    with h5py.File(SOURCE, "r") as s:
        s_isroot = s["is_root"][:].astype(bool)
        s_amav = s["action_motion_available"][:].astype(bool)
        child_rows = np.where(~s_isroot & s_amav)[0]
        s_xml = s["xml"][:]
        s_obj = s["object_id"][:]
        kids = defaultdict(list)
        for i in child_rows:
            kids[(s_xml[i], s_obj[i])].append(int(i))
        print(f"source colossus children: {len(child_rows)} in {len(kids)} (xml,object) groups", flush=True)
        s_state = s["target_object_state"]
        s_vm, s_rm = s["value_mask"], s["r_mask"]

        shutil.copyfile(DEPLOY, OUT)
        rep = {"children_total": int(len(child_rows)), "groups": len(kids)}
        with h5py.File(OUT, "r+") as d:
            n = d.attrs["n_samples"]
            gm = d.create_dataset("guess_mask", shape=(n, 60, 5), dtype=np.uint8,
                                  compression="lzf", chunks=(1, 60, 5))
            d_isroot = d["is_root"][:].astype(bool)
            d_amav = d["action_motion_available"][:].astype(bool)
            roots = np.where(d_isroot & d_amav)[0]
            print(f"deploy colossus roots: {len(roots)}", flush=True)

            c = dict(no_kids=0, kid_no_untried=0, gate_fail=0, ambiguous_drop=0,
                     matched_exact_cell=0, guessed=0, capped_cells=0)
            pos_errs, targets = [], []
            for nr, ri in enumerate(roots):
                key = (d["xml"][ri], d["object_id"][ri])
                kid_idx = kids.get(key, [])
                lm = (d["value_mask"][ri] > 0.5) & (d["r_mask"][ri] > 0.5)
                capped = lm & (d["ceiling_mask"][ri] > 0.5)
                c["capped_cells"] += int(capped.sum())
                if not kid_idx:
                    c["no_kids"] += 1
                    continue
                st = d["target_object_state"][ri]
                am = d["action_motion"][ri]  # (60,5,3) normalized
                pred = np.empty((60, 5, 3), np.float32)
                pred[..., 0] = st[0] + am[..., 0] * 0.5
                pred[..., 1] = st[1] + am[..., 1] * 0.5
                pred[..., 2] = st[2] + am[..., 2] * np.pi
                cell_flat = np.where(lm.ravel())[0]  # match against all TRIED cells
                pf = pred.reshape(-1, 3)[cell_flat]
                assign = {}  # cell -> (err, kid)
                for ki in kid_idx:
                    kst = s_state[ki]
                    dpos = np.hypot(pf[:, 0] - kst[0], pf[:, 1] - kst[1])
                    dyaw = np.abs(wrap(pf[:, 2] - kst[2]))
                    score = dpos + 0.1 * dyaw
                    j = int(np.argmin(score))
                    if dpos[j] > GATE_POS or dyaw[j] > GATE_YAW:
                        c["gate_fail"] += 1
                        continue
                    cell = int(cell_flat[j])
                    if cell in assign:
                        c["ambiguous_drop"] += 1
                        if score[j] >= assign[cell][0]:
                            continue
                    assign[cell] = (float(score[j]), ki, float(dpos[j]))
                for cell, (sc, ki, dp) in assign.items():
                    e, dd = divmod(cell, 5)
                    if not capped[e, dd]:
                        c["matched_exact_cell"] += 1
                        continue
                    untried = (s_rm[ki] > 0.5) & ~(s_vm[ki] > 0.5)
                    if untried.sum() < 1:
                        c["kid_no_untried"] += 1
                        continue
                    vh = float(np.sort(src_vals[ki][untried])[::-1][:TOP_M].mean())
                    tgt = min(CAP, GAMMA * vh)
                    vt = d["value_target"][ri]
                    cm = d["ceiling_mask"][ri]
                    g = gm[ri]
                    vt[e, dd] = tgt
                    cm[e, dd] = 0.0
                    g[e, dd] = 1
                    d["value_target"][ri] = vt
                    d["ceiling_mask"][ri] = cm
                    gm[ri] = g
                    pos_errs.append(dp)
                    targets.append(tgt)
                    c["guessed"] += 1
                if nr % 2000 == 0:
                    print(f"  root {nr}/{len(roots)} guessed={c['guessed']}", flush=True)

            targets = np.array(targets)
            rep.update(c)
            rep["pos_err_pcts"] = [float(x) for x in np.percentile(pos_errs, [50, 90, 99])] if pos_errs else None
            rep["target_quartiles"] = [float(x) for x in np.percentile(targets, [25, 50, 75])] if len(targets) else None
            rep["target_hist"] = np.histogram(targets, bins=np.linspace(0, 0.81, 28))[0].tolist() if len(targets) else None
            rep["guessed_frac_of_capped"] = c["guessed"] / max(c["capped_cells"], 1)
    REPORT.write_text(json.dumps(rep, indent=1))
    print(json.dumps({k: v for k, v in rep.items() if k != "target_hist"}, indent=1))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
