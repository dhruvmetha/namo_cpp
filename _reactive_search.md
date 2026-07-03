---
type: experiment
status: idea
created: 2026-07-03
metric:
commit:
tags:
  - experiment
---
# _reactive_search

## Hypothesis

We test the "reactive mode" but with random policy using random seeds. We test this against the 3 (seeds) models of No-horizon-v3. But I want atleast 10 seeds of reactive random (per test environment, object, region pair). 1push and 2push both.

## Plan
_(Claude, 2026-07-03)_ **CAR only.** Regime = reactive forced-dive (`eval_reactive_argmax.py`, region
criterion): 2push metric = open@2, 1push metric = open@1. Compare **reactive-random floor (≥10 seeds → variance
band)** vs **NoHz-v3 reactive (3 seeds)**, split easy/med/hard, for 1push + 2push. Random = `--prior uniform`
(model-free: uniform pick from the labeled object's candidate pool, no model call).

**Reuse (already computed on Amarel, rsynced → `scratch_namo/eval/staging_amarel/`; n=1018 2push / 1323 1push):**
- 2push NoHz-v3 baseline **3 seeds + per-episode leaves** = 40.7/40.3/41.1 → **40.7±0.4**. ✓
- 2push difficulty = per-episode `division` (hard/med/easy by `n_setups`) in `pure2push_divisions.json`. ✓
- Bonus: Hz-v3 3 seeds + leaves (45.6/39.1/44.3).
- 1push NoHz-v3 **s1** (@1 82.2) + 1push random s1 (@1 36.4) — partial.

**Gap-fill (iLab `unlimited`, CPU, sharded — `scripts/ilab/reactive_argmax_ilab.slurm`):**
1. Random floor → **10 fresh seeds (100–109) with `--leaf-out`**, both keys (2push SHARD=80 · 1push SHARD=102).
2. 1push NoHz-v3 **s2,s3** (needs model; ckpts rsynced to shared FS → all 6 v3 ckpts now iLab-native).

**Aggregate:** join leaves → difficulty bins → mean±std across seeds per (horizon, difficulty). Random band =
std over 10 seeds; NoHz band = std over 3 seeds. Then table (easy/med/hard × {1push,2push}) + plots.

Launcher: `ssh ilab1.cs.rutgers.edu 'bash -s' < launch_react_search.sh` (22 array jobs × 13 shards).

## Run
_(Claude, auto)_ job id · commit · config · date.

## Result + Verdict
_(Claude, auto from run output)_ Numbers — accept/reject **on numbers only**.
I want to see some variance band on the random results, and how do they compare to no-horizon-v3. Give me a full concrete table and plots on easy, medium, hard for both 1push and 2push
## Next
I just want to see the results and analyse them, I dont mind you giving me a brief analysis too.

## Discussion
_(you ↔ Claude — ask here; I answer inline, dated `**[who YYYY-MM-DD]**`. Newest at the bottom.)_
