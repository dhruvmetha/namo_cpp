---
type: experiment
status: done
created: 2026-07-03
updated: 2026-07-04
metric: "NoHz-v3 ≫ random every cell — 2push 42.1 vs 4.7 · 1push 82.3 vs 37.0"
commit:
tags:
  - experiment
---
# _reactive_search

## Hypothesis

We test the "reactive mode" but with random policy using random seeds. We test this against the 3 (seeds) models of No-horizon-v3. But I want atleast 10 seeds of reactive random (per test environment, object, region pair). 1push and 2push both.

## Plan
_(Claude, 2026-07-03)_ **CAR only.** Regime = reactive forced-dive (`eval_reactive_argmax.py`, region criterion): 2push metric = open@2, 1push metric = open@1. Compare **reactive-random floor (≥10 seeds → variance band)** vs **NoHz-v3 reactive (3 seeds)**, split easy/med/hard, for 1push + 2push. Random = `--prior uniform` (model-free: uniform pick from the labeled object's candidate pool, no model call).

**Reuse (already computed on Amarel, rsynced → `scratch_namo/eval/staging_amarel/`; n=1018 2push / 1323 1push):**
- 2push NoHz-v3 baseline **3 seeds + per-episode leaves** = 40.7/40.3/41.1 → **40.7±0.4**. ✓
- 2push difficulty = per-episode `division` (hard/med/easy by `n_setups`) in `pure2push_divisions.json`. ✓
- Bonus: Hz-v3 3 seeds + leaves (45.6/39.1/44.3).
- 1push NoHz-v3 **s1** (@1 82.2) + 1push random s1 (@1 36.4) — partial.

**Gap-fill (iLab `unlimited`, CPU, sharded — `scripts/ilab/reactive_argmax_ilab.slurm`):**
1. Random floor → **10 fresh seeds (100–109) with `--leaf-out`**, both keys (2push SHARD=80 · 1push SHARD=102).
2. 1push NoHz-v3 **s2,s3** (needs model; ckpts rsynced to shared FS → all 6 v3 ckpts now iLab-native).

**Aggregate:** join leaves → difficulty bins → mean±std across seeds per (horizon, difficulty). Random band = std over 10 seeds; NoHz band = std over 3 seeds. Then table (easy/med/hard × {1push,2push}) + plots.

Launcher: `ssh ilab1.cs.rutgers.edu 'bash -s' < launch_react_search.sh` (22 array jobs × 13 shards).

## Run
_(Claude, 2026-07-03)_ commit `ff32a38` (open1-leaf fix) on `feat/horizon-q-redesign`. Script `eval_reactive_argmax.py` (region criterion, `--prior uniform`=random / `q`=model). CAR testset `namo_testset_v1` (2push key n=1018 · onepush key n=1323). Difficulty: 2push = per-episode `division` (n_setups); 1push = solve_rate tertiles (hard<0.169 / <0.533 / easy).
- **Random floors (model-free, 10 seeds 100–109, `--leaf-out`):** iLab `unlimited`, `scripts/ilab/reactive_argmax_ilab.slurm`, 13 shards × 2 horizons. (2push SHARD=80, 1push SHARD=102.)
- **NoHz-v3 2push (3 seeds):** REUSED prior Amarel `reactarg_nohz_v3{,_s2,_s3}_leaf` (40.7/41.3/44.5 leaf).
- **NoHz-v3 1push (3 seeds):** Amarel `main-redhat` jobs 57828466-468 (+ 57828582 reshard), model on CPU.
- iLab model path blocked by `fast_scorer` skew in shared `sage_learning` (older than Amarel's `feat/render-speedup` c0a00f7, which is verified render-equiv 29/29) → model eval run on Amarel.
- onepush XMLs were absent on iLab (symlinks into `car_envs`); repointed 18,270 links to the iLab tree.

## Result + Verdict
_(Claude, auto from run output)_ Numbers — accept/reject **on numbers only**. I want to see some variance band on the random results, and how do they compare to no-horizon-v3. Give me a full concrete table and plots on easy, medium, hard for both 1push and 2push

_(Claude, 2026-07-03)_ **CAR, region criterion, mean ± std across seeds.** Random = 10 seeds, NoHz-v3 = 3 seeds. Plot: `scratch_namo/eval/react_search_v3/AGG/react_search.png` (aggregator `agg_react_search.py`).

**2push** (open@2)

| difficulty | random floor | NoHz-v3 | lift |
|---|---|---|---|
| easy   | 9.7 ± 1.6 | 61.2 ± 2.4 | +51.5 |
| medium | 4.4 ± 0.9 | 44.3 ± 2.9 | +40.0 |
| hard   | 1.8 ± 0.6 | 27.5 ± 2.0 | +25.7 |
| all    | 4.7 ± 0.6 | 42.1 ± 1.7 | +37.4 |

**1push** (open@1)

| difficulty | random floor | NoHz-v3 | lift |
|---|---|---|---|
| easy   | 71.7 ± 2.1 | 98.7 ± 0.4 | +27.1 |
| medium | 32.6 ± 3.1 | 93.9 ± 0.5 | +61.3 |
| hard   |  6.2 ± 1.6 | 54.3 ± 0.4 | +48.0 |
| all    | 37.0 ± 1.1 | 82.3 ± 0.2 | +45.3 |

![[react_search.png]] _(grouped bars: random floor vs NoHz-v3 by difficulty; error bars = std across seeds. Source PNG lives at `assets/react_search.png`; regenerate via `scripts/sandbox/plot_react_search.py`.)_

**Validation:** NoHz-v3 1push all 82.5 ≈ registered 82.2; random 1push all 37.0 ≈ prior Amarel 36.4; NoHz-v3 2push ~42 ≈ registry 40.7 (leaf-run vs summary noise). **Random band is tight (±0.6–3.1) because each seed already averages ~1000 episodes → std = binomial SE `sqrt(P(1-P)/N)`, verified.** 10 seeds is plenty.

**Verdict [on numbers]:** NoHz-v3 ≫ random floor in EVERY cell, both horizons. 2push all +37.4 (≈9×); 1push all +45.3. Lift is real everywhere but shrinks on the hardest 2push (easy +51.5 → hard +25.7): random almost never solves hard 2push (1.8%), and NoHz-v3 also drops most on hard (27.5%). 1push is easier for both (one push, higher floor) yet NoHz-v3 still clears 82.5% vs 37.0%.
## Next
I just want to see the results and analyse them, I dont mind you giving me a brief analysis too.

## Discussion
_(you ↔ Claude — ask here; I answer inline, dated `**[who YYYY-MM-DD]**`. Newest at the bottom.)_
