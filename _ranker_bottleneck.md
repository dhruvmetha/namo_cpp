---
type: experiment
status: done
created: 2026-07-04
updated: 2026-07-04
metric: "WHERE the NoHz-v3 q-ranker is stuck: the FIRST push (setup) ranking, not the dive. On the 21 robust misses the true setup sits at MEDIAN rank 38/70 (top-1-is-setup 0%); sim-cost corr w/ first-push rank 0.79 vs dive-rank 0.29; NN scoring is only 3% of wall (not the bottleneck). Single head under-values setups; horizon-conditioning (Hz) does NOT fix it (verified)."
tags:
  - experiment
  - diagnostic
---
# Ranker Bottleneck — where the NoHz-v3 q-ranker gets stuck

Diagnostic (NOT a new run). Reuses the instrumented best-first records from `_full_search` (2push
`fullsearch/nohz_s{1,2,3}`+`rand_s{0..9}`, 1push `fullsearch_1push/*`, interleaved `fullsearch_time/tri_s1`)
+ a small **re-scoring** pass (model loaded on arrakis, CPU) to read where the TRUE setup lands in the model's
ranking on episodes the search never solves. **Objective under test:** the ranker must reach ~100% solve
**before random in wall-time**, with **fast per-candidate suggestions**. Today it tops out at 95.3% (hard 90.2%)
and on easy is *slower* in wall-time than random (7.1 vs 6.3 s).

**Scripts (worktree):** `scripts/sandbox/rankdiag_analyze.py` (record-only Q1-Q4), `rankdiag_rescore.py`
(re-score the true-setup rank + NN/render profile), `rankdiag_plots.py`. Outputs:
`scratch_namo/eval/fullsearch/rankdiag/rankdiag_{analysis,rescore}.json`; plots `assets/rankdiag_*.png`.

## Mechanism (how combine=q ranks the first push vs the dive)
Search = greedy best-first on the labeled object, `combine='q'` → **priority = the raw per-action q**;
`V` (state value) is unused in `combine=q`. Per state the model emits a **(60 edges × 5 depths)** q-map in
**one forward pass** (`LiveScorer.score_state`). The heap holds unsimulated pushes; pop max-q, simulate, on goal
open → stop, else expand its post-push state (a "dive", scored at the same head). `solve_ranks=[r_first, r_dive]`
= the sibling-pool rank of each push in the winning plan; `depth_hist={0:…,1:…}` = sims at push-depth 0 (first
push) vs 1 (dive).

**Two grounding facts, both verified:**
1. **NoHz is a SINGLE, horizon-agnostic head** — trained `+data.budget_h=false` (wandb args), so `score_state`
   never passes `H` to the net; q(first push, h=2) and q(dive, h=1) come from the **same head, same scale**.
   ⇒ the "H1/H2 cross-head scale mismatch" in Q5 **cannot** be the mechanism for NoHz. Refuted at the source.
2. The default **sigmoid squash** compresses q to `[0.503, 0.695]` (std 0.044) vs raw E[bin] `[0.014, 0.825]`
   (std 0.178). But sigmoid is **globally monotone** → it preserves every heap ordering → for `combine=q` the
   squash changes **no** pop order. It is a **red herring** for the search. (It would only matter under
   `combine=blend/product`, which these runs don't use.)

## Result + Verdict
_(Claude, 2026-07-04)_ **The ranker is stuck on the FIRST push (the setup), not the dive.** All splits below are
2push pure2push (n=1018), NoHz pooled over 3 ckpt-seeds; difficulty = per-episode `division`.

### Q1 — Rank of the winning push: H1 (first) vs H2 (dive)
The task's premise (dive = weak link) is **REFUTED**: the dive is the *stronger* ranker.

| push position | #1 = winner | ≤ rank 2 | ≥ rank 5 | median | mean | p99 |
|---|---|---|---|---|---|---|
| **H1 first push** | 50.9% | 69.8% | 22.0% | 0 | **3.28** | 30 |
| **H2 dive** | 70.0% | 82.8% | 12.5% | 0 | **2.05** | 31 |
| random H1 | 14.9% | 39.6% | 43.6% | 4 | 5.95 | 28 |

**By difficulty — H1 collapses on hard, H2 holds up:**

| tier | H1 #1 | H1 median | H1 mean | H1 ≥5 | H2 #1 | H2 mean |
|---|---|---|---|---|---|---|
| easy | 65.2% | 0 | 2.10 | 14.9% | 81.6% | 1.03 |
| medium | 52.3% | 0 | 2.79 | 19.0% | 70.8% | 1.80 |
| **hard** | **39.1%** | **1** | **4.68** | **30.6%** | 61.0% | 3.06 |

Intuition: once you're *at* a real setup's post-push state, the head clearly sees "this second push opens the
goal" (high q) → dive ranks the winner #1 70% of the time. It's the **first** push — a setup that opens
*nothing yet* — the head cannot float to the top on hard scenes. ![[rankdiag_rank_h1_h2.png]]

### Q2 — Why not 100%: the misses are buried setups, all solvable
21/1018 (2.1%) are missed by **all 3** NoHz seeds (⇒ 95.3% headline is the per-seed 4.7% miss). **Every one is
GT-2push-solvable** and **97% are budget-bound** (`n_sim≥900`; only 4 of 144 pooled fails exhausted the tree) —
the solver **is in the tree**, just ranked past 900 sims of exploration. 19/21 hard, 0 easy; median 2 valid
setups (needles). 13/21 are found by ≥1 random seed (solvable-in-budget, model-specific mis-rank); 8/21 by
neither (hardest needles). 2 needles the model solves that **no** random seed ever finds.

**Where the true setup sits in the model's first-push ranking (re-scored):**

| set | median setup rank | mean | top-1 is setup | ≥ rank 5 | median n_setups |
|---|---|---|---|---|---|
| easy (sample) | **0** | 3.0 | 57% | 21% | 13 |
| medium | 3 | 8.2 | 33% | 39% | 4 |
| hard | 4 | 11.3 | 20% | 49% | 1 |
| **fail set (21)** | **38** | 39.5 | **0%** | **90%** | 2 |

On the failures the true setup is **reachable** (0 missing from the pool) but buried at **median rank 38 of
~70** and is **never** the model's #1. The q-gap top-vs-setup is only 0.016 (sigmoid) — the head simply can't
*separate* the lone setup from ~38 useless pushes. ![[rankdiag_unsolved.png]]

### Q3 — The heavy sims tail is dive-sims, but CAUSED by first-push mis-ranking
Sims-to-solve: median **4**, mean 55, p90 164, p99 703 — a long right tail.

| group | n | mean sims | first-push sims (d0) | dive sims (d1) | winning first-push rank |
|---|---|---|---|---|---|
| fast (≤p50) | 1499 | 2.3 | 1.1 | 1.2 | 0.1 |
| tail (>p90) | 291 | **386** | 19.6 | **366.6** | **16.3** |

The tail spends 95% of its sims **diving** — but it dives ~20 *wrong* setups' subtrees first because the
**winning setup is ranked ~16th**. Proof it's the first push, not the dive: **corr(n_sim, first-push rank) =
0.79** vs **corr(n_sim, dive rank) = 0.29**. Tail is 158 hard / 99 med / 34 easy, median 2 setups. This is
**not root-thrash** (depth-0 sims are only 6% of budget) and **not a dive bug** — it's a buried setup feeding
the dive machinery bad branches. ![[rankdiag_simtail.png]]

### Q4 — Suggestion cost: the NN is NOT the bottleneck
Scoring (render+NN, one (60×5) map per call) is **3.0% of wall** overall (easy 4.5%, hard 2.4%); **simulation is
~95%+**. Per-score 66 ms for 300 candidates = **0.22 ms/candidate**. Profiled split: **render 72 ms + NN 20 ms**
(b1); batched NN b16 = 12.3 ms/state (~1.6×). So batching/caching/a lighter scorer buys **~0.5% of wall** —
negligible. "Fast per-candidate suggestion" is **already satisfied**.

**Easy wall-time crossover (model 7.05 s vs random 6.33 s, gap 0.72 s):** = scoring 0.32 s (44%) + model's sims
cost more (178 vs 146 ms/sim → +0.40 s **despite doing fewer sims**, 38 vs 43). A free NN would *not* close it;
easy is at ceiling with no sim-count headroom. The model's per-sim cost being higher (it selects longer/deeper
pushes) is **UNVERIFIED** without push-depth instrumentation, but the per-sim time gap is real in the records.
⇒ the wall-time lever is **fewer sims** (ranking), not a faster net. ![[rankdiag_nncost.png]]

### Q5 — Mechanism: the single head under-values setups (verified), horizon-conditioning doesn't fix it
- **Not** a cross-head mismatch (single head, `budget_h=false`) and **not** the sigmoid squash (monotone →
  ranking-invariant for combine=q). Both refuted above.
- **The real mechanism:** the head predicts near-immediate solvability, so a **setup push (opens nothing yet)
  gets a q barely above a useless push**. Setup rank degrades monotonically **easy 0 → medium 3 → hard 4 →
  fail 38**; on needles (1–2 setups) the lone setup is buried. Direct dive evidence: the winning dive is *not*
  the top child 30% of the time and ≥rank 3 in 17% — real but minor vs the first-push burial.
- **Hz (horizon-conditioned) does NOT help — verified** on the interleaved `tri_s1`: Hz first-push #1 = 49.4%
  (mean 3.40) vs NoHz 51.4% (mean 3.29); Hz even uses **more** sims (59 vs 51 avg). So conditioning on remaining
  budget did not teach setup value. **`dive_bonus` would make it worse** (it dives wrong setups *more*).
  ![[rankdiag_mechanism.png]]

**Verdict [on numbers].** The ranker is stuck at the **first push**. Failure modes ranked by cost toward
"100% before random, fast":
1. **First-push (setup) value under-ranking — THE bottleneck.** Owns the ceiling (misses = setups at median
   rank 38), the sim tail (corr 0.79; buried setup → ~20 wrong dives), and thus the wall-time (sims = 95% of
   wall). Fix this and all three move.
2. Dive (H2) tail — minor (median rank 0; corr with sims only 0.29).
3. NN suggestion cost — negligible (3% of wall); "fast" is already met.
4. Easy wall-time crossover — real but low-value (easy at ceiling; unfixable by a faster net).

**Single highest-leverage fix (direction; HYPOTHESES, not yet tested):** make the first-push priority reflect
**setup value** (value-to-solution), not myopic 1-push solvability. Concretely (a) rank first pushes by a 1-ply
lookahead `V(s1)` — exactly what the predecessor `scorer_beam.py` deliberately does ("P is blind to 2-push
first moves; rank by V(s1)"), which `combine=q` abandoned; costs sims but directly measures setup value; or
(b) retrain the value target to be **solution-path/multi-horizon aware** (the "Policy+Value, not Q" / UVFA
direction) so a setup scores high despite opening nothing. **NOT** horizon-conditioning (Hz, verified no help)
and **NOT** `dive_bonus` (worsens it).

## Next
Test fix (a) cheaply first: re-run best-first with a first-push priority = `V(s1)` (or `combine=blend/product`
variants that inject the state value) on the 371 hard episodes + the 21 misses, and read whether the true
setup's effective rank drops from ~16/38 toward the top and whether hard@900 lifts above 90.2%. If lookahead
wins, it argues the real gain is a setup-aware **value** target for the next model.

## Discussion
_(you ↔ Claude — ask here; newest at bottom.)_

**[Claude 2026-07-04] Does step_penalty share this bottleneck? YES — verified (record-based, same method,
NoHz numbers reproduced exactly ⇒ join correct).** The −1/0/1 reward reshaping did NOT touch the setup-ranking
bottleneck — consistent with its 2push-search wash.

| H1 first-push (setup), rank 0 = winner | #1 % | mean | ≥rank5 % |
|---|---|---|---|
| overall — NoHz / step_pen | 50.9 / 49.3 | 3.28 / 3.41 | 22.0 / 22.6 |
| **hard** — NoHz / step_pen | **39.1 / 34.8** | **4.68 / 5.14** | 30.6 / 32.9 |
| H2 dive, overall — NoHz / step_pen | 70.0 / 69.4 | 2.05 / 2.01 | — |

Robust misses (fail all 3 seeds): NoHz 21, **step_pen 26** (16 shared — a superset-ish failure on hard, not a
disjoint set). So step_pen's setup ranking is ≈NoHz overall and **worse on hard** (the tier that gates 100%);
the dive stays the stronger head for both. **Both models are blind to setups** — the fix must be setup-value-aware
(V(s1) / multi-horizon), not reward-relabeling on the open-now axis.
