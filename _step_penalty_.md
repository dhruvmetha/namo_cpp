---
type: experiment
status: live
created: 2026-07-03
updated: 2026-07-04
metric: "−1/0/1 vs 0/0.9/1: 2push SEARCH tied (solve@900 95.4 vs 95.3, sims 96 vs 94) → reject; reactive 2push −2.5 (39.6 vs 42.1), 1push +1.0 (83.3 vs 82.3). 1push search + fair time PENDING"
commit:
tags:
  - experiment
---
# Step penalty

We want to train new model no-horizon model (with "v3" data). However, we change the target "q(s, a)" to 1 for opening (for when it immediately opens), q(s, a) = 0 if it does not open now but will open in the future and q(s, a) = -1 never opens now or in the future. 

At the moment we have trained with successful finish 1, successful setup (future finish exists) 0.9 and no success ever is 0. Change this to what we have described.

Retrain model with this new scheme and compare report the results (1push and 2push), on both search and reactive modes.  We will later compare this with random and no-horizon v3. after we finish. (After the results of random  and no-horizon v3 search are also computed (which they are being done simultaneously in parallel))

## Hypothesis
This "reward" scheme is better suited for ranking (search) than the older scheme (our other parallel experiment thread)

## Plan
_(Claude, 2026-07-04)_ **CAR only, testset `namo_testset_v1`, same v3 data as NoHz-v3.** Retrain the
no-horizon q-scorer with the SIGNED target — **+1** immediate-open · **0** valid-setup (opens later) · **−1**
never-opens — vs the incumbent NoHz-v3 **0 / 0.9 / 1**. Then a **3-way** comparison (random floor / NoHz-v3 /
step_penalty) in **both regimes × both horizons**, split easy/med/hard:
- **Reactive** (`eval_reactive_argmax.py`, region criterion): 2push = open@2, 1push = open@1.
- **Best-first SEARCH** (`time_bestfirst.py`, combine=`q`, budget 900): solve@sims + avg sims-to-solve.

**Reuse (do NOT recompute):** random floor + NoHz-v3 come straight from `_reactive_search.md` (reactive) and
`_full_search.md` (best-first). step_penalty = 3 ckpt-seeds `qfull_nohz_steppen_v3_s{1,2,3}`. Aggregation =
re-binning the existing per-episode leaves with the **canonical** binning (2push = `division` in
`pure2push_divisions.json`; 1push = `solve_rate` tertiles in `onepush_episodes.json`), identical to
`agg_react_search.py` / `agg_fullsearch_bydiff.py`. Best-first uses the verified **positional** episode join
(sorted-xml order; **0/1018 mismatch on all 3 steppen seeds**, basename-normalized).

## Run
_(Claude, 2026-07-04)_ step_penalty ckpts `qfull_nohz_steppen_v3_s{1,2,3}` (reactive shard log shows
`epoch014-val_loss0.6859`). Eval landed on the shared FS; aggregated on arrakis (read-only re-binning, no new
sim). **Eval state:**

| regime | horizon | seeds | episodes/seed | status | location |
|---|---|---|---|---|---|
| reactive | 2push | s1,s2,s3 | 1018 | **DONE** | `eval/react_search_v3/steppen2push_s{1,2,3}` |
| reactive | 1push | s1,s2,s3 | 1323 | **DONE** | `eval/react_search_v3/steppen1push_s{1,2,3}` |
| best-first | 2push | s1,s2,s3 | 1018 | **DONE** | `eval/fullsearch/steppen_s{1,2,3}` |
| best-first | 1push | s1,s2,s3 | 1323 | **⚠ MISSING — never launched** | — |

- **Genuine gap:** step_penalty **1push best-first SEARCH** does not exist anywhere (checked
  `eval/fullsearch_1push/` — only `nohz_s{1,2,3}` + `rand_s{0..9}` baselines present; also empty on Amarel
  `/scratch/dm1487/eval/`). The 2push search verdict below is decisive on the primary axis; 1push search would
  add a secondary datapoint (1push reactive shows a small step_pen edge on hard). Ready to launch on iLab
  (same path the NoHz-v3 1push best-first used on rlab7).
- **Fair wall-TIME PENDING.** The best-first SIMS reported here are machine-independent (valid). A same-hardware
  TIME comparison needs step_penalty timed on the **emeraldrapids-exclusive** setup the NoHz-v3 2push times
  used — those Amarel timing jobs are stalled in queue: 2push `57845891` completed only 5/16 shards (320/1018);
  backfill `57846177` + 1push `57846712` both PD/Priority, empty. **step_penalty t_wall is therefore NOT put on
  the same axis as NoHz's emerald times** (cross-box rule); TIME row deferred.

## Result + Verdict
_(Claude, 2026-07-04)_ **CAR, mean ± std across seeds (random 10 · NoHz-v3 3 · step_penalty 3).** Δ columns =
step_penalty − NoHz-v3 (the hypothesis test). Aggregation `/tmp/steppen_agg.py`; summary
`/tmp/steppen_results.json`.

### Primary — best-first SEARCH ranking (2push, combine=q, budget 900, n=1018)

Solve-rate at increasing sim budgets + avg sims (all episodes) + sims-to-solve (solved only):

| difficulty | ranker | @2 | @10 | @30 | @100 | @300 | @900 | avg sims | to-solve |
|---|---|---|---|---|---|---|---|---|---|
| easy (238) | random | 6.6 ± 1.5 | 34.1 ± 2.4 | 66.3 ± 2.8 | 89.3 ± 1.3 | 98.3 ± 0.6 | 99.8 ± 0.2 | 43 | 42 |
| easy | NoHz-v3 | 55.0 ± 2.1 | 77.6 ± 0.9 | 85.7 ± 0.9 | 92.9 ± 0.9 | 96.1 ± 0.5 | **98.9 ± 0.4** | 38 | 28 |
| easy | **step_pen** | 54.9 ± 3.5 | 76.5 ± 1.8 | 87.0 ± 1.7 | 94.4 ± 0.4 | 96.6 ± 0.3 | **99.2 ± 0.0** | 33 | 26 |
| medium (409) | random | 4.1 ± 0.9 | 21.0 ± 1.4 | 43.6 ± 2.4 | 69.9 ± 2.1 | 88.4 ± 1.2 | 97.0 ± 0.7 | 123 | 98 |
| medium | NoHz-v3 | 40.7 ± 2.4 | 62.8 ± 3.7 | 75.4 ± 2.2 | 86.6 ± 0.5 | 94.1 ± 0.3 | **97.8 ± 0.8** | 62 | 43 |
| medium | **step_pen** | 39.4 ± 2.6 | 63.2 ± 2.2 | 74.5 ± 1.6 | 86.1 ± 1.3 | 93.2 ± 1.2 | **97.4 ± 0.6** | 67 | 44 |
| hard (371) | random | 1.6 ± 0.5 | 10.1 ± 1.0 | 22.0 ± 1.1 | 40.2 ± 1.9 | 58.9 ± 2.2 | 78.7 ± 2.1 | 344 | 197 |
| hard | NoHz-v3 | 26.1 ± 1.5 | 44.0 ± 0.9 | 56.1 ± 0.4 | 69.5 ± 0.8 | 82.1 ± 1.1 | **90.2 ± 0.8** | 165 | 88 |
| hard | **step_pen** | 22.6 ± 2.5 | 40.8 ± 0.9 | 54.1 ± 2.0 | 67.6 ± 2.0 | 81.1 ± 2.3 | **90.7 ± 0.8** | 169 | 97 |
| **all (1018)** | random | 3.7 ± 0.8 | 20.1 ± 0.5 | 41.1 ± 1.0 | 63.6 ± 1.2 | 80.0 ± 0.6 | 91.0 ± 0.8 | 185 | 115 |
| **all** | NoHz-v3 | 38.7 ± 1.3 | 59.4 ± 1.9 | 70.8 ± 1.0 | 81.8 ± 0.4 | 90.2 ± 0.6 | **95.3 ± 0.6** | 94 | 55 |
| **all** | **step_pen** | 36.9 ± 2.8 | 58.1 ± 1.6 | 70.0 ± 1.8 | 81.3 ± 1.3 | 89.6 ± 1.3 | **95.4 ± 0.5** | 96 | 58 |

![[steppen_bestfirst_sims_2push.png]]
_(NoHz-v3 blue and step_penalty green sit on top of each other in every panel; random red well below. Source
`assets/steppen_bestfirst_sims_2push.png`.)_

**Read of the primary axis.** step_penalty and NoHz-v3 are **statistically tied** on 2push search ranking:
- **Ceiling (@900):** all 95.4 vs 95.3 (**+0.1**, within ±0.5–0.6 std). Per tier easy +0.3 / medium −0.4 /
  hard +0.5 — every cell inside the seed band. A wash.
- **Front-loaded / low-budget regime** (where ranking quality shows most — the #1 pick must be the winner):
  step_penalty is **marginally WORSE**. all @2 36.9 vs 38.7 (−1.8), @10 58.1 vs 59.4 (−1.3); on **hard @2**
  22.6 vs 26.1 (**−3.5**). If the signed target helped ranking, this is exactly where it should show — it does
  not.
- **Efficiency:** avg sims 96 vs 94, sims-to-solve 58 vs 55 — step_penalty uses **marginally more** sims
  (easy it's cheaper: 33 vs 38 avg / 26 vs 28 to-solve; hard slightly more: to-solve 97 vs 88).

### Secondary — reactive open-rate (both horizons)

**2push** (open@2)

| difficulty | random | NoHz-v3 | step_penalty | Δ(step−NoHz) |
|---|---|---|---|---|
| easy   | 9.7 ± 1.6 | 61.2 ± 2.4 | 59.8 ± 3.0 | **−1.4** |
| medium | 4.4 ± 0.9 | 44.3 ± 2.9 | 42.1 ± 2.1 | **−2.3** |
| hard   | 1.8 ± 0.6 | 27.5 ± 2.0 | 24.0 ± 2.2 | **−3.5** |
| all    | 4.7 ± 0.6 | 42.1 ± 1.7 | 39.6 ± 2.4 | **−2.5** |

**1push** (open@1)

| difficulty | random | NoHz-v3 | step_penalty | Δ(step−NoHz) |
|---|---|---|---|---|
| easy   | 71.7 ± 2.1 | 98.7 ± 0.4 | 98.5 ± 0.5 | −0.2 |
| medium | 32.6 ± 3.1 | 93.9 ± 0.5 | 94.7 ± 0.7 | **+0.8** |
| hard   |  6.2 ± 1.6 | 54.3 ± 0.4 | 56.8 ± 1.6 | **+2.5** |
| all    | 37.0 ± 1.1 | 82.3 ± 0.2 | 83.3 ± 0.6 | **+1.0** |

![[steppen_reactive.png]]
_(3-way grouped bars by difficulty, both horizons; error bars = std across seeds. Source
`assets/steppen_reactive.png`.)_

**Read of reactive.** Mirror image across horizons: on **2push** step_penalty is consistently **behind** NoHz-v3
(−1.4 → −3.5, worse as difficulty rises, −2.5 overall); on **1push** it is **ahead** on the harder tiers (hard
**+2.5**, medium +0.8, all +1.0), tied on easy. Both crush random everywhere (2push ~8–9×, 1push +45 all).

### Validation (join/binning reproduces the two source cards)

Recomputed the reused `all` rows from the same leaves with my binning — exact match:

| quantity | mine | source card |
|---|---|---|
| reactive 2push all — random | 4.7 | 4.7 (`_reactive_search`) |
| reactive 2push all — NoHz-v3 | 42.1 | 42.1 |
| reactive 1push all — random | 37.0 | 37.0 |
| reactive 1push all — NoHz-v3 | 82.3 | 82.3 |
| best-first 2push all @900 — random | 91.0 | 91.0 (`_full_search`) |
| best-first 2push all @900 — NoHz-v3 | 95.3 | 95.3 |
| best-first 2push all avg sims — random / NoHz | 185 / 94 | 185 / 94 |

NoHz-v3 best-first per-difficulty (easy 98.9 / med 97.8 / hard 90.2 @900; avg sims 38/62/165) also reproduces
`_full_search`'s stratified table cell-for-cell. Positional join verified **0 mismatch / 1018** on all 3
step_penalty seeds (basename-normalized; the divisions file stores full-path xml, records store basename — same
order). Random band is tight (each seed already averages ~1000 episodes → std ≈ binomial SE).

### Verdict [on numbers]

**REJECT the hypothesis on the primary (search-ranking) axis.** The signed **−1/0/1** target does **not** beat
the incumbent **0/0.9/1** at ranking pushes for best-first search on 2push (the richer, exhaustive test):
- @900 solve-rate **tied** (95.4 vs 95.3, +0.1 within noise), across every difficulty tier;
- in the **low-budget / front-loaded** regime — precisely where ranking quality decides — step_penalty is
  **marginally worse** (@2 all −1.8, hard −3.5) and uses **marginally more** sims-to-solve (58 vs 55);
- reactive 2push (a second ranking-flavored read) also favors 0/0.9/1 (−2.5 overall, −3.5 on hard).

The **only** place the signed target edges ahead is **reactive 1push** on the harder tiers (hard +2.5, all
+1.0) — a secondary regime. Net: no evidence that −1/0/1 improves search ranking; it is a wash-to-slightly-worse
on 2push search and mixed on 1push. **1push best-first SEARCH is not yet run**, so the 1push search leg of the
hypothesis is formally open — but the decisive 2push search test does not support the hypothesis.

## Next
- **Close the 1push best-first gap** (3 ckpt-seeds + reuse the `nohz_s*`/`rand_s*` 1push baselines already in
  `eval/fullsearch_1push/`) to settle the 1push search leg — the +2.5 hard reactive edge makes it the one place
  step_penalty *might* win a search cell. Ready to launch on iLab (rlab7 path the NoHz 1push run used).
- **Fair TIME 3-way** once the stalled emeraldrapids-exclusive timing jobs land (or resubmit them) — sims
  already say the story; time will echo it.
- Given the wash, **0/0.9/1 stays the incumbent**; the signed target is not worth adopting for search on
  current evidence. If pursued, the interesting thread is *why* it helps 1push-hard reactive but not 2push
  search — likely a calibration/scale effect, not a ranking win.

## Discussion
_(you ↔ Claude — ask here; I answer inline, dated `**[who YYYY-MM-DD]**`. Newest at the bottom.)_
