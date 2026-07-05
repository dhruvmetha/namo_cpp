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

## Classifier grounding — the q-head scored as a detector vs pure2push GT
_(Claude, 2026-07-04)_ Every observation above is now backed by a measured detector statistic. **Scripts:**
`scripts/sandbox/rankdiag_classifier.py` (re-score + sim-GT, sharded on arrakis CPU, 3 ckpt-seeds),
`rankdiag_cls_agg.py` (metrics + `assets/clsdiag_*.png`). Data: `…/eval/fullsearch/rankdiag/cls/`
(68,345 first-push rows over **all 1018** pure2push episodes; 49,540 second-push rows over a 180-episode
stratified sample). q reported **raw** (E[bin]); sigmoid = 1/(1+e⁻ʳᵃʷ) is monotone → identical AUC/ranks.

**Ground truth — exactly how (no silent proxies).**
- **Setup GT (first push):** pure2push `valid_first_push` (pure2push is all-pure-2push → `valid_1push`=∅). This is
  the exhaustive-1push ∪ sampled-2push key `build_2push_validset.py` writes and that the card/`eval_m3`/fullsearch
  already grade on. The 2-push chain is **same labeled object twice** (trial log keyed by `chosen_object_id`).
- **Opener GT (second push):** DERIVED BY SIMULATION — from a real setup's post-push state s1, push the **labeled
  object** again (matches the same-object chain) and label OPENER iff `goal_open_pts` (≥20/100 s0-sampled goal
  points reachable, frac 0.2 — the collection's own `_validate_opening` criterion). Seed-independent physics.
- **Caveat, MEASURED:** `valid_first_push`'s 2nd-push expansion was **subsampled**, so it is a **lower bound** on
  setups: simulating the labeled-object dive from **label-"wrong" first pushes finds an opener 23% of the time**
  (easy 36% / med 22% / hard 11%). ⇒ setup-detector **precision/FPR vs the label are bounds**; **recall is clean**
  (labeled positives are true). A deconfounded **sim-GT** setup view (720 dove first pushes, true labels) is given.

### Evidence ledger — each claim → its grounding number/plot

| Claim (from Q1–Q5 / verdict) | Grounding statistic (this pass) | Verdict |
|---|---|---|
| Setup under-ranking is THE bottleneck | setup **p@1=0.32** overall / **0.19 hard** (top pick is a valid setup only 19–32%); r@5=0.34; ROC-AUC 0.80→**0.75 hard**; `clsdiag_summary` | ✔ but it's low **top-k**, not zero recall |
| The dive recognizes winners | opener **ROC-AUC 0.87**, recall@op 0.78, **p@1=0.62** (vs setup 0.32) — the dive head is the stronger ranker | ✔ (Q1 confirmed on GT) |
| Head can't reject dead ends | opener **precision@op 0.37** (0.29 hard); dead-end **FP=8.6 above op / 15.7 above the setup's q per wrong subtree**; `clsdiag_deadend` | ✔ measured |
| "No dynamic range" | AUC 0.80/0.87 = real signal; but setup q-gap sig **0.014** (raw 0.077 vs 0.021), overlap 0.13; `clsdiag_qhist` | ↺ signal exists, **top-k ordering fragile** (not absent) |
| "~18 plausible-but-dead dives / wrong subtree" | **8.6 (med 4, max 89)** dead 2nd-pushes score above the opener operating point; **15.7 (med 10, max 109)** above the winning setup's q | ✔ real number ≈ 9–16 |
| Dive-vs-breadth 94/6, NN cost negligible | unchanged (Q3 d0=6%, Q4 NN=3% of wall) | ✔ (prior) |
| step_penalty shares it | unchanged (Discussion below) | ✔ (prior) |

### C1 — Setup-detector (first push): q vs "valid setup", by difficulty (label-GT; 3-seed mean±std)

| tier | n | base | ROC-AUC | PR-AUC | prec@op¹ | recall@op | FPR@op | **p@1** | r@1 | r@3 | r@5 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| overall | 68345 | 0.090 | **0.805** | 0.229 | 0.204 | 0.789 | 0.304 | **0.324** | 0.090 | 0.229 | 0.336 |
| easy | 18840 | 0.191 | 0.799 | 0.391 | 0.374 | 0.799 | 0.316 | 0.483 | 0.036 | 0.114 | 0.191 |
| medium | 26210 | 0.076 | 0.784 | 0.180 | 0.163 | 0.777 | 0.326 | 0.357 | 0.080 | 0.230 | 0.353 |
| **hard** | 23295 | **0.023** | **0.752** | **0.052** | **0.046** | 0.783 | 0.392 | **0.187** | 0.135 | 0.301 | 0.409 |

¹ vs the incomplete label → **lower bound**. Deconfounded **sim-GT** (clean per-first-push labels, 720 dove pushes):
overall AUC 0.771, **precision 0.648**, recall 0.675; **hard AUC 0.750, precision 0.527, recall 0.755**. So true
setup precision is ~0.53–0.65 (not 0.05–0.20) — but **AUC agrees (~0.75–0.80)**: moderate, not sharp, separability.

**Read:** the hypothesis "buries good setups" holds **at the top of the ranking, not at a permissive threshold** —
recall@op is 0.79, but the top pick is a valid setup only **19% of the time on hard** (p@1). On hard the base rate
is **2.3%** (≈1 needle in 43); AUC 0.75 is simply not sharp enough to float that needle to #1 → the search dives
wrong subtrees first (ties to Q2/Q3). ![[clsdiag_roc_pr.png]] ![[clsdiag_qhist.png]]

### C2 — Opener-detector (second push / dive): q vs "opens goal" (sim-GT same-object; 3-seed mean±std)

| tier | n | base | ROC-AUC | PR-AUC | **prec@op** | recall@op | **FPR@op** | p@1 | p@5 | r@5 |
|---|---|---|---|---|---|---|---|---|---|---|
| overall | 12870 | 0.125 | **0.867** | 0.591 | **0.373** | 0.780 | **0.190** | 0.619 | 0.502 | 0.428 |
| easy | 4260 | 0.165 | 0.861 | 0.651 | 0.529 | 0.708 | 0.130 | 0.701 | 0.576 | 0.368 |
| medium | 3810 | 0.124 | 0.885 | 0.603 | 0.360 | 0.840 | 0.213 | 0.620 | 0.510 | 0.471 |
| **hard** | 4800 | 0.089 | **0.866** | 0.553 | **0.290** | 0.786 | 0.192 | 0.534 | 0.418 | 0.448 |

**Read:** hypothesis "decent recall, weak precision" **confirmed** — recall@op 0.78 and AUC **0.87 > setup 0.80**
(the dive is the stronger head, GT-confirming Q1), but **precision@op 0.37 (0.29 hard)**: at its 0.78-recall
operating point ~⅔ of flagged openers are false. Precision/FPR IS the dive head's weak axis.

### C3 — Dead-end false-positives (the "wrong subtree" cost, sim-GT)
Truly-dead wrong subtrees = a label-"wrong" first push whose exhaustive labeled-object dive opens the goal **0**
times (416 subtrees, mean **67** candidate 2nd-pushes each). How many the head scores highly:

| tier | n_dead | subtree size | FP above **opener op-point** mean(med,max) | FP above **winning setup's q** mean(med,max) |
|---|---|---|---|---|
| overall | 416 | 67 | **8.6** (4, 89) | **15.7** (10, 109) |
| easy | 115 | 71 | 8.9 (7, 57) | 13.7 (7, 86) |
| medium | 140 | 65 | 7.2 (4, 72) | 16.7 (10, 83) |
| **hard** | 161 | 67 | 6.9 (2, 94) | 16.2 (11, 109) |

So per wrong subtree the head deems **~9 second pushes as "opener-grade"** and **~16 as better than the correct
setup** — all dead. This is the measured version of the "~18 plausible-but-dead dives"; the head **cannot reject a
dead subtree**. ![[clsdiag_deadend.png]]

### C4 — Separability / dynamic range
One-number separability: setup **ROC-AUC 0.80** (hard 0.75), opener **0.87** (hard 0.87) — **real signal, dive >
setup**. But the setup positive/negative q-overlap is heavy: raw-q medians 0.077 (setup) vs 0.021 (non), **sigmoid
0.519 vs 0.505 → gap 0.014** (reproduces the card's 0.016 top-vs-setup gap; std 0.044) with **13%** of non-setups
scoring ≥ the median setup. So it is **not "no dynamic range"** — there is 0.75–0.87 AUC of signal; the sigmoid
squash + tiny raw gap make the **top-k ordering fragile on low-prevalence (hard) pools**, which is exactly where
best-first lives. `clsdiag_summary` is the one-glance panel (setup vs opener × overall/hard). ![[clsdiag_summary.png]]

### What to optimize — verdict [on numbers]
The three candidate diagnoses resolve cleanly:
- **Low AUC / "no signal" — REJECTED.** Setup AUC 0.80, opener AUC 0.87. The head is not signal-starved.
- **Low setup top-k recall — CONFIRMED (primary).** p@1=0.32 (0.19 hard), r@5=0.34, AUC 0.75 on hard where the
  setup base rate is 2.3%. This gates the **ceiling** (misses = buried setups) and the **sim tail** (buried setup →
  many wrong dives; card corr(n_sim, first-push rank)=**0.79** ≫ dive 0.29). Highest leverage.
- **Low dive precision / dead-end rejection — CONFIRMED (secondary).** opener precision@op 0.37 (0.29 hard);
  **~9–16 dead 2nd-pushes per wrong subtree outscore the operating point / the correct setup.** This sets the
  **cost per wrong dive** (~9–16 wasted sims each), compounding the setup burial into the heavy tail.

⇒ **Optimize a setup-value target FIRST** (multi-horizon / value-to-solution, so a setup that opens nothing yet
still scores high → lift setup **p@1 / hard-AUC**), and **add a dead-end/precision signal on the dive head SECOND**
(so a wrong subtree is abandoned in ~2 dives, not ~9–16 → lift opener **precision@op**). The single number that
picks setup-value as primary: **corr(n_sim, first-push rank)=0.79**. **NOT** horizon-conditioning (Hz, verified no
help) and **NOT** `dive_bonus` (it raises the dead-end FP). **Verified** here: all C1–C4 statistics.
**Residual (fix hypotheses, untested):** that a V(s1) / multi-horizon target actually raises setup p@1/AUC, and
that a precision target lifts opener precision without costing its recall — the card's "Next" experiment.

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
