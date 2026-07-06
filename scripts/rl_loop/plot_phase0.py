#!/usr/bin/env python3
"""Plot Phase-0 GATE (EXP-2026-07-06-rl-only-self-imitation). Two panels:
  (L) grouped bars per difficulty: baseline open@2 / ARM(i)-any / ARM(i)-modelpref / ARM(ii) recoverable, err=std across seeds.
  (R) stacked bars per difficulty: ARM(iii) miss taxonomy as share of ALL episodes (wrong_setup/failed_finish/aliasing_or_control) + solved.
Reads {D}/AGG/results.json. Draws the 85 (proceed) and 65 (stop) gate lines. Saves {D}/AGG/phase0_gate.png."""
import json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

D = os.environ.get("D", "/common/users/dm1487/scratch_namo/eval/phase0_gate")
res = json.load(open(f"{D}/AGG/results.json"))
BINS = ["easy", "medium", "hard", "all"]
x = np.arange(len(BINS))


def col(m):
    return [(res["metrics"][m][b][0] if res["metrics"][m][b] else np.nan) for b in BINS]


def err(m):
    return [(res["metrics"][m][b][1] if res["metrics"][m][b] else 0.0) for b in BINS]


fig, (axL, axR) = plt.subplots(1, 2, figsize=(15, 6))

series = [("base_open", "baseline greedy@2 (fully-learned)", "#888888"),
          ("armi_any", "ARM(i)-any: oracle setup + learned finish", "#2166ac"),
          ("armi_modelpref", "ARM(i)-modelpref", "#67a9cf"),
          ("armii", "ARM(ii): learned setup on GT-valid (recoverable)", "#d6604d")]
w = 0.2
for i, (m, lbl, c) in enumerate(series):
    axL.bar(x + (i - 1.5) * w, col(m), w, yerr=err(m), capsize=3, label=lbl, color=c)
axL.axhline(85, ls="--", lw=1.2, color="green"); axL.text(-0.45, 86.0, "85 proceed", color="green", fontsize=9)
axL.axhline(65, ls="--", lw=1.2, color="red");   axL.text(-0.45, 66.0, "65 stop", color="red", fontsize=9)
axL.set_xticks(x); axL.set_xticklabels(BINS); axL.set_ylabel("open-rate / recoverable (%)"); axL.set_ylim(0, 100)
axL.set_title("Phase-0 arms (CAR, pure2push, greedy, 2push-only)\nmean ± std, 3 NoHz-v3 ckpt-seeds")
axL.legend(fontsize=8, loc="upper right"); axL.grid(axis="y", alpha=0.3)

MISS = ["wrong_setup", "failed_finish", "aliasing_or_control"]
MC = {"wrong_setup": "#b2182b", "failed_finish": "#ef8a62", "aliasing_or_control": "#fddbc7"}
tax = res["tax_all"]
bottom = np.zeros(len(BINS))
solved = np.array(col("base_open"))
for mm in MISS:
    vals = np.array([(tax[b][mm][0] if tax[b][mm] else 0.0) for b in BINS])
    axR.bar(x, vals, 0.55, bottom=bottom, label=mm, color=MC[mm])
    bottom += vals
axR.bar(x, solved, 0.55, bottom=bottom, label="solved (greedy@2)", color="#4d9221")
axR.set_xticks(x); axR.set_xticklabels(BINS); axR.set_ylabel("share of ALL episodes (%)"); axR.set_ylim(0, 100)
axR.set_title("ARM(iii) miss taxonomy (share of all episodes)\n+ solved by fully-learned greedy@2")
axR.legend(fontsize=8, loc="upper right"); axR.grid(axis="y", alpha=0.3)

plt.tight_layout()
out = f"{D}/AGG/phase0_gate.png"
plt.savefig(out, dpi=130, bbox_inches="tight")
print("wrote", out)
