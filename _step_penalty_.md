---
type: experiment
status: done
created: 2026-07-03
updated: 2026-07-04
metric: "−1/0/1 vs 0/0.9/1 — SOFTENED reject: 2push SEARCH tied (solve@900 95.4 vs 95.3) → reject; 1push SEARCH small ranking edge (solve@1 +1.0 all, +2.5 hard; ceiling tied ~99.7). Reactive 2push −2.5, 1push +1.0. Fair time deferred (sims-only, verdict-sufficient)"
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
- **Fair wall-TIME deferred (sims-only shipped).** Best-first SIMS are machine-independent (valid on any box);
  a same-hardware TIME 3-way needs step_penalty timed on the **emeraldrapids-exclusive** setup NoHz-v3's 2push
  times used, and those Amarel timing jobs are stalled (2push `57845891` 5/16 shards; backfill `57846177` +
  1push `57846712` PD/empty). Per orchestrator: **ship sims-only** — the verdict is decisive on sims, and
  step_penalty uses ≥ NoHz sims, so a fair-time run would only reinforce it. step_penalty `t_wall` is NOT put on
  the same axis as NoHz's emerald times (cross-box rule).

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
- **Both search horizons + both regimes are now closed.** Remaining loose end: the fair-TIME 3-way (deferred,
  sims-only shipped) — resurrect the stalled emeraldrapids-exclusive timing jobs only if a wall-clock number is
  wanted; sims already decide it and step_penalty uses ≥ NoHz sims.
- The interesting thread the split exposes: the signed target helps the **immediate-open** decision (1push-hard,
  solve@1 +2.5) but not the **setup** decision (2push first-push ranking). That points at the H1 (open-now) vs
  H2 (open-later) head/calibration — a targeted follow-up could apply the signed target **only to the H1 head**
  and keep 0/0.9/1 for setup, to bank the 1push gain without the 2push wash.
- On current evidence **0/0.9/1 remains the deployed ranker**; −1/0/1 is not worth a wholesale switch.

## Discussion
_(you ↔ Claude — ask here; I answer inline, dated `**[who YYYY-MM-DD]**`. Newest at the bottom.)_
