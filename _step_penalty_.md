---
type: experiment
status: done
created: 2026-07-03
updated: 2026-07-04
metric: "−1/0/1 vs 0/0.9/1 — SOFTENED reject: 2push SEARCH tied (solve@900 95.4 vs 95.3) → reject; 1push SEARCH small ranking edge (solve@1 +1.0 all, +2.5 hard; ceiling tied ~99.7). Reactive 2push −2.5, 1push +1.0. Fair 3-way wall-time (sapphirerapids-excl, interleaved) CONFIRMS sims: 2push step_pen≈NoHz (15.6 vs 16.0s avg, both ≪ random 26.7s, gap on hard); 1push step_pen a hair faster (0.63 vs 0.70s), both ~½ random"
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
| best-first | 1push | s1,s2,s3 | 1323 | **DONE** (launched here) | `eval/fullsearch_1push/steppen_s{1,2,3}` |

- **1push best-first — the last gap, now CLOSED.** It did not exist anywhere at harvest (checked
  `eval/fullsearch_1push/` — only `nohz_s{1,2,3}` + `rand_s{0..9}` baselines; empty on Amarel too). Launched on
  iLab **rlab7** (`unlimited`, CPU inference) via `scripts/ilab/fullsearch_bestfirst_ilab.slurm`, jobs
  `171268/171269/171270`, combine=q, hmax=1, budget 900, `MODELS=NoHz` with the step_penalty ckpts. Same
  lowest-val_loss ckpt per seed as reactive/2push (s1 `awlkn20p/ep014`, s2 `ud5rquhc/ep012`, s3
  `5smrk5sb/ep012`) so seed identity is consistent across horizons. All 3 COMPLETED (~20–22 min), 1323/1323
  each. Baselines `nohz_s*`/`rand_s*` reused untouched.
- **Fair wall-TIME 3-way — now COMPLETE (see the wall-TIME subsection in Result).** Interleaved
  (NoHz-v3 / step_penalty / random per episode, one `--exclusive` node), pinned **sapphirerapids**
  (`--constraint=sapphirerapids`; emeraldrapids/icelake were ~0-idle, so the strict emerald pin used by
  `_full_search` was infeasible — sapphirerapids is same-microarch across all shards, and its NoHz-v3 anchor
  times match `_full_search`'s emerald within ~3–5 %, sims bit-identical). Jobs: **2push** `57845891` (shards
  0–4, SHARD=64 → eps [0,320)) + `57846177` (shards 10–31, SHARD=32 → eps [320,1018)) = **1018/1018 complete,
  no gaps** — the earlier "missing shards 5–9" was a false alarm (shards 0–4 at double width already cover
  [160,320)); **1push** `57846712` = **1323/1323 complete**. Dirs `eval/fullsearch_time/tri_s1` +
  `eval/fullsearch_time_1push/tri1_s1`. (A speculative backfill array 57852575 for shards 5–9 was submitted then
  **cancelled** on realizing the data was already complete — it would have duplicated eps [160,320).)

## Result + Verdict
_(Claude, 2026-07-04)_ **CAR, mean ± std across seeds (random 10 · NoHz-v3 3 · step_penalty 3).** Δ columns =
step_penalty − NoHz-v3 (the hypothesis test). Aggregation `/tmp/steppen_agg.py` (reactive + 2push best-first)
and `/tmp/steppen_1push_bf.py` (1push best-first); summaries `/tmp/steppen_results.json`,
`/tmp/steppen_1push_bf.json`.

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

### Primary — best-first SEARCH ranking (1push, combine=q, budget 900, n=1323)

Solve-rate at increasing sim budgets + avg sims + sims-to-solve. Tertiles: hard < 0.169 ≤ med < 0.533 ≤ easy.

| difficulty | ranker | @1 | @2 | @5 | @10 | @20 | @900 | avg sims | to-solve |
|---|---|---|---|---|---|---|---|---|---|
| easy (447) | random | 72.4 ± 1.5 | 90.6 ± 0.4 | 99.0 ± 0.2 | 99.7 ± 0.1 | 99.8 ± 0.0 | 99.8 ± 0.0 | 2 | 1 |
| easy | NoHz-v3 | 98.7 ± 0.4 | 99.2 ± 0.1 | 99.6 ± 0.1 | 99.8 ± 0.0 | 99.8 ± 0.0 | **99.8 ± 0.0** | 1 | 1 |
| easy | **step_pen** | 98.5 ± 0.5 | 99.3 ± 0.2 | 99.7 ± 0.1 | 99.8 ± 0.0 | 99.8 ± 0.0 | **99.8 ± 0.0** | 1 | 1 |
| medium (435) | random | 33.1 ± 2.3 | 52.7 ± 2.5 | 82.9 ± 1.2 | 96.4 ± 0.6 | 99.8 ± 0.1 | 100.0 ± 0.0 | 3 | 3 |
| medium | NoHz-v3 | 94.0 ± 0.6 | 96.2 ± 0.6 | 98.3 ± 0.4 | 99.6 ± 0.1 | 99.9 ± 0.1 | **100.0 ± 0.0** | 1 | 1 |
| medium | **step_pen** | 94.7 ± 0.7 | 96.6 ± 1.0 | 99.1 ± 0.2 | 99.6 ± 0.1 | 99.9 ± 0.1 | **100.0 ± 0.0** | 1 | 1 |
| hard (441) | random | 5.9 ± 1.5 | 12.0 ± 1.7 | 27.1 ± 1.8 | 46.3 ± 2.9 | 67.6 ± 1.7 | 99.3 ± 0.1 | 20 | 19 |
| hard | NoHz-v3 | 54.2 ± 0.3 | 62.0 ± 0.8 | 73.9 ± 1.0 | 82.7 ± 0.5 | 90.3 ± 0.2 | **99.2 ± 0.1** | 7 | 7 |
| hard | **step_pen** | 56.7 ± 1.6 | 66.1 ± 1.1 | 77.0 ± 1.2 | 85.3 ± 0.3 | 91.8 ± 0.5 | **99.1 ± 0.0** | 6 | 6 |
| **all (1323)** | random | 37.3 ± 0.9 | 51.9 ± 1.1 | 69.8 ± 0.8 | 80.8 ± 1.0 | 89.1 ± 0.6 | 99.7 ± 0.0 | 8 | 8 |
| **all** | NoHz-v3 | 82.3 ± 0.2 | 85.8 ± 0.2 | 90.6 ± 0.4 | 94.0 ± 0.2 | 96.7 ± 0.1 | **99.7 ± 0.0** | 3 | 3 |
| **all** | **step_pen** | 83.3 ± 0.6 | 87.4 ± 0.7 | 91.9 ± 0.5 | 94.9 ± 0.1 | 97.2 ± 0.2 | **99.6 ± 0.0** | 3 | 3 |

![[steppen_bestfirst_sims_1push.png]]
_(On hard, green step_penalty sits above blue NoHz-v3 for B≈1–30, then both converge to 100 %. Source
`assets/steppen_bestfirst_sims_1push.png`.)_

**Read of 1push search — the one place the signed target wins (modestly).** Opposite of 2push:
- **solve@1** (the pure ranking test — does the #1-ranked push solve in one sim?): step_penalty **83.3 vs 82.3
  (+1.0)**; by tier easy −0.2, medium +0.7, **hard +2.5** (56.7 vs 54.2). This is a *real* small edge in exactly
  the ranking metric the hypothesis targets, and it **exactly mirrors the reactive-1push +2.5-on-hard** signal.
- **Whole low-budget curve leads:** all @2 +1.6, @5 +1.3, @10 +0.9, @20 +0.5; on hard @2 +4.1, @5 +3.1. The gap
  is largest early and closes as budget grows.
- **Ceiling (@900) ties:** all 99.6 vs 99.7 (step_pen a hair lower, within noise); every tier ~99–100 %. The
  1-push pool is tiny (≤~35 pushes) so both models eventually solve everything — the win is purely *front-loaded
  ranking*, not reach.
- **sims-to-solve tied** (both 3 overall; hard 6 vs 7 — step_pen marginally cheaper).
- **Cross-check (independent validation):** best-first solve@1 **exactly** reproduces reactive open@1 — NoHz
  82.3=82.3, step_pen 83.3=83.3, random 37.3≈37.0 (both measure "does rank-1 push open"). Positional join 0
  mismatch/1323 on all 3 steppen seeds.

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

### Fair 3-way wall-TIME (both horizons, sapphirerapids-exclusive interleaved) — the last deferred cell, now CLOSED

_(Claude, 2026-07-04)_ **Interleaved same-hardware timing**: `time_bestfirst.py` runs all 3 rankers
back-to-back **per episode** on one **`--exclusive`** node — the "hz" slot = **NoHz-v3** (`qfull_nohz_v3_v4hq_s1`),
"nohz" slot = **step_penalty** (`qfull_nohz_steppen_v3_s1`), plus **random** (rng=0). Every shard pinned to
**sapphirerapids** (`--constraint=sapphirerapids`; emeraldrapids/icelake were ~0-idle/drained). **1 timing seed**
(s1 ckpts + rng 0) → point estimates, no seed band; the solve-rate seed variance lives in the 3-seed sims tables
above. Data: `eval/fullsearch_time/tri_s1` (2push, 1018/1018) + `eval/fullsearch_time_1push/tri1_s1` (1push,
1323/1323) — both complete, all shards sapphirerapids/hdr (`halk`) nodes. Aggregation `/tmp/steppen_time_agg.py`
(canonical binning, positional join; `/tmp/steppen_time.json`).

**2push — solve@wall-time (t_wall ≤ T) + avg t_wall (s), CAR, sapphirerapids-exclusive**

| difficulty | ranker | 1 s | 5 s | 10 s | 30 s | 60 s | 240 s | avg t_wall | to-solve |
|---|---|---|---|---|---|---|---|---|---|
| easy (238) | NoHz-v3 | 61.8 | 86.1 | 90.8 | 94.5 | 97.1 | 99.2 | 6.39 | 5.51 |
| easy | **step_pen** | 62.6 | 88.2 | 92.4 | 95.0 | 96.2 | 99.2 | **5.80** | 4.93 |
| easy | random | 18.9 | 69.3 | 83.2 | 95.8 | 98.7 | 100.0 | 6.80 | 6.80 |
| medium (409) | NoHz-v3 | 44.0 | 73.1 | 80.4 | 90.5 | 94.6 | 97.8 | 12.14 | 9.15 |
| medium | **step_pen** | 50.1 | 75.6 | 83.9 | 92.2 | 95.4 | 98.0 | **10.48** | 8.00 |
| medium | random | 11.0 | 45.0 | 61.6 | 82.2 | 90.5 | 97.3 | 18.86 | 16.18 |
| hard (371) | NoHz-v3 | 31.3 | 55.0 | 63.1 | 76.3 | 83.0 | 91.4 | 26.28 | 16.42 |
| hard | **step_pen** | 29.9 | 55.3 | 64.7 | 79.5 | 84.6 | 91.1 | **27.40** | 14.70 |
| hard | random | 6.5 | 26.1 | 35.8 | 55.0 | 66.6 | 79.5 | 48.09 | 27.59 |
| **all (1018)** | NoHz-v3 | 43.5 | 69.5 | 76.5 | 86.2 | 91.0 | 95.8 | **15.95** | 10.80 |
| **all** | **step_pen** | 45.7 | 71.1 | 78.9 | 88.2 | 91.7 | 95.8 | **15.55** | 9.58 |
| **all** | random | 11.2 | 43.8 | 57.3 | 75.4 | 83.7 | 91.5 | **26.69** | 17.41 |

**1push — solve@wall-time (t_wall ≤ T) + avg t_wall (s), CAR, sapphirerapids-exclusive**

| difficulty | ranker | 0.5 s | 1 s | 2 s | 5 s | 10 s | 30 s | avg t_wall | to-solve |
|---|---|---|---|---|---|---|---|---|---|
| easy (447) | NoHz-v3 | 97.8 | 99.8 | 99.8 | 99.8 | 99.8 | 99.8 | 0.28 | 0.28 |
| easy | **step_pen** | 99.1 | 99.8 | 99.8 | 99.8 | 99.8 | 99.8 | **0.27** | 0.27 |
| easy | random | 93.7 | 99.6 | 99.8 | 99.8 | 99.8 | 99.8 | 0.24 | 0.24 |
| medium (435) | NoHz-v3 | 89.4 | 97.9 | 99.5 | 100.0 | 100.0 | 100.0 | 0.38 | 0.38 |
| medium | **step_pen** | 92.2 | 99.1 | 99.8 | 100.0 | 100.0 | 100.0 | **0.36** | 0.36 |
| medium | random | 63.2 | 87.6 | 97.5 | 100.0 | 100.0 | 100.0 | 0.58 | 0.58 |
| hard (441) | NoHz-v3 | 53.7 | 71.2 | 81.9 | 94.1 | 98.4 | 99.3 | 1.43 | 1.34 |
| hard | **step_pen** | 57.4 | 76.2 | 84.8 | 94.6 | 98.4 | 99.3 | **1.25** | 1.17 |
| hard | random | 17.2 | 33.6 | 54.9 | 77.8 | 93.7 | 99.3 | 3.25 | 3.18 |
| **all (1323)** | NoHz-v3 | 80.3 | 89.6 | 93.7 | 98.0 | 99.4 | 99.7 | **0.70** | 0.67 |
| **all** | **step_pen** | 82.9 | 91.7 | 94.8 | 98.1 | 99.4 | 99.7 | **0.63** | 0.60 |
| **all** | random | 58.2 | 73.6 | 84.1 | 92.5 | 97.8 | 99.7 | **1.35** | 1.33 |

![[steppen_time_bydiff.png]]
_(3-way success-vs-wall-time by difficulty × horizon; green step_pen sits on top of blue NoHz-v3 in every panel,
red random trails — biggest gap on hard. Source `assets/steppen_time_bydiff.png`; per-horizon
`assets/steppen_time_2push.png`, `assets/steppen_time_1push.png`.)_

**Read — wall-time confirms the sims verdict.** step_penalty and NoHz-v3 are **tied on wall-time**, both far
below random, exactly as the machine-independent sims said:
- **2push:** avg t_wall step_pen **15.55 s** ≈ NoHz-v3 **15.95 s** (−0.4 s, a wash within single-seed noise;
  easy/medium step_pen a hair cheaper, hard +1.1 s but to-solve faster 14.7 vs 16.4 s). Both **≪ random 26.69 s**,
  and — like the solve-rate edge — the gap **lives on hard**: models ~27 s vs random **48 s** (@30 s ≈78 % vs 55 %).
  On easy the three converge (~6–7 s) since the sim count is already tiny.
- **1push:** step_pen **0.63 s** vs NoHz-v3 **0.70 s** (a hair faster, mirroring its front-loaded 1push ranking
  edge; hard **1.25 vs 1.43 s**), both **~½ of random's 1.35 s**. Everything solves sub-second on easy/medium.

**Pooling / validity (this run pools with prior timing).**
- **Exact-sims match (rigorous, machine-independent):** the tri **NoHz-v3** (deterministic) reproduces
  `_full_search`'s `nohz_s1` search **bit-for-bit — 1018/1018 (2push) and 1323/1323 (1push), 0 mismatch** → the
  timed run is the *same search*, just clocked. Random pools **distributionally** (single rng-0 realization
  differs episode-wise, but tri random 2push all @900 sims 91.6 ≈ 91.0, avg sims 184 ≈ 185).
- **Anchor time cross-check vs `_full_search` (emeraldrapids):** on the load-bearing cells the sapphirerapids
  anchors match emeraldrapids within ~3–5 % — **hard NoHz-v3 26.28 ≈ 26.3 s**, hard random 48.1 vs 46.5, all
  NoHz-v3 15.95 vs 15.5, all random 26.69 vs 25.4; easy ~6–7 s both. Since NoHz-v3 **sims are bit-identical**, the
  ≤5 % time delta is pure hardware/jitter → **sapphirerapids ≈ emeraldrapids for this OMP-1 CPU workload**. Per
  the cross-box rule these seconds are **not** placed on `_full_search`'s emeraldrapids axis, but the exact-sims
  match + tight anchor agreement confirm the 3-way pools. The 3-way itself is fully **self-contained** (all three
  rankers interleaved on the same sapphirerapids node per episode), so its fairness needs no external anchor.

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

**SOFTENED REJECT — horizon-split.** With both search horizons now in, the hypothesis ("−1/0/1 is better for
search ranking than 0/0.9/1") is **rejected on 2push** but shows a **small, real edge on 1push** — not the clean
reject the 2push data alone suggested, nor a win.

- **2push search (the harder, exhaustive test) — REJECT.** @900 solve-rate **tied** (95.4 vs 95.3, +0.1 within
  noise, every tier inside the band); in the low-budget/front-loaded regime — where ranking decides —
  step_penalty is **marginally worse** (@2 all −1.8, hard −3.5) and uses **marginally more** sims-to-solve (58
  vs 55). Reactive 2push echoes it (−2.5 all, −3.5 hard).
- **1push search — SMALL EDGE to step_penalty.** In the ranking metric itself — **solve@1** — step_penalty
  **beats** NoHz-v3 **+1.0 overall, +2.5 on hard** (56.7 vs 54.2), and leads the whole low-budget curve
  (@2 +1.6, @5 +1.3). The @900 **ceiling ties** (~99.7 %, tiny 1-push pool → both solve everything), so the win
  is *front-loaded ranking*, not reach. This **confirms** the independent reactive-1push signal (hard +2.5,
  identical).

**Net.** The signed target does **not** improve ranking on the horizon that stresses it (2push, where the model
must rank a *first* push whose payoff is a downstream open) — there it is a wash-to-slightly-worse. It **does**
give a small, consistent ranking bump on 1push-hard (the immediate-open decision), across both search and
reactive. Because the primary/harder 2push test fails and the 1push win is small and ceiling-tied, **0/0.9/1
stays the incumbent** — but the hypothesis is **not cleanly false**: the signed target genuinely sharpens the
*immediate-open* ranking, it just doesn't help the *setup* ranking that 2push needs.

## Next
- **Everything is now closed** — both search horizons, both regimes, **and** the fair 3-way wall-TIME
  (sapphirerapids-exclusive interleaved, both horizons; see Result). Wall-time **confirms** the sims verdict:
  2push step_pen ≈ NoHz-v3 (~16 s, both ≪ random ~27 s, gap on hard), 1push step_pen a hair faster (0.63 vs
  0.70 s), both ~½ random. No loose ends remain.
- The interesting thread the split exposes: the signed target helps the **immediate-open** decision (1push-hard,
  solve@1 +2.5) but not the **setup** decision (2push first-push ranking). That points at the H1 (open-now) vs
  H2 (open-later) head/calibration — a targeted follow-up could apply the signed target **only to the H1 head**
  and keep 0/0.9/1 for setup, to bank the 1push gain without the 2push wash.
- On current evidence **0/0.9/1 remains the deployed ranker**; −1/0/1 is not worth a wholesale switch.

## Discussion
_(you ↔ Claude — ask here; I answer inline, dated `**[who YYYY-MM-DD]**`. Newest at the bottom.)_
