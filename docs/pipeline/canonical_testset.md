# Canonical NAMO test set (`namo_testset_v1`)

> ✅ **2026-06-11 REUSED AS-IS for the 0.034 car (horizon-Q build).** The car geometry changed (wheels
> 0.0375→0.034, inside the chassis). The test set was collected at `car=0.0375 / control_steps=550`, but the
> **pure car effect is NEGLIGIBLE: ±0.5% push reach, sub-3 mm, ≤0.02°/depth** (controlled regen diff — regenerate
> primitives at 0.0375 AND 0.034 both at 550, compare). The ~14% "significant" primitive shift seen during the
> car change was the unrelated `482→550` push-duration config change, not the wheels. ⇒ this test set ≈ the new
> 0.034/550 world within eval noise: **no re-collection, no re-labeling — valid for grading 0.034/550 models.**
> See [horizon_q_datasets.md](horizon_q_datasets.md) §1 and [../experiments/horizon_q_build_journal.md](../experiments/horizon_q_build_journal.md) §3.

> ⚠ **2026-06-10 SUCCESS-BAR REGENERATION [USER].** The set was re-collected under a stricter "opened" bar.
> The original `--region-min-reachable-fraction 0.2` flag was **inert** — the real success criterion was the
> default `region_success_min_reachable=1` ("≥1 sampled goal point reachable"). Now **wired** into
> `region_opening.py:_validate_opening` (opt-in; fraction × #goal-points) and the whole set re-collected with
> **goals-per-region=100 + fraction=0.2** = *"≥20% of the goal region reachable"*. Unified depth-2 collection
> over all 2173 scenes (`pkls_2push_v2`) → both tiers, one bar. **Composition shift: genuine-2-push 808 → 1018
> episodes (+26%).** Live labels: `onepush_episodes.json` (1323 eps), `pure2push.json` (1018 eps), `twopush.json`.
> Stale-pyc gotcha: one node (halk0014) ran a pre-tagging `.pyc` → 32 untagged pkls; fixed by clearing
> `__pycache__` + `PYTHONDONTWRITEBYTECODE=1` and re-collecting that shard.


The single source of truth for evaluating Region-Opening scorers / policies / value functions on the **car**. Lives at `/scratch/dm1487/datasets/namo_testset_v1/` (full datasheet + stats in its `README.md`). This doc is the repo-side record so the build is reproducible and we stop confusing test artifacts.

## What it is
- **2173** geometry-clean scenes from the held-out `car_envs/v3/test/{feb,aug9}_car` pool, **0 leaks** into the `v3_scorer_e4` training corpus (proven by geometry, not file names — see below).
- **1-push tier** (20% bar): **1323 episodes / 991 scenes**, key `labels/onepush_episodes.json` (drop-in for `eval_scorer.py --episodes`).
- **2-push tier** (20% bar): **1018 genuine-2-push episodes / 983 scenes** (F1′ = first-pushes that enable a solving second push), key `labels/pure2push.json`. Both tiers from ONE unified depth-2 collection (`pkls_2push_v2`).

## Why the old setup was confusing (and the fix)
The 1-push key was keyed by `outputs/test_*_phase1/...` **symlinks**; the 2-push manifest by `car_envs/v3/test/...` **real paths**. A name-based "0 overlap with train" check is therefore meaningless. **Fix:** re-key everything by `realpath` and prove disjointness by **room geometry** = `md5(sorted walls pos/size/euler + sorted obstacle pos/size/euler)`, goal + robot excluded (two episodes of one room differ only in the goal). 0 / 1128 test scenes share full geometry with any of 66 135 train scenes.

**Deprecated, do not use:** `v3_test_validsets.json` (no `object_center`, not the eval key), `test_2push_solvable_combined.txt` (1-push-contaminated), `test_pure2push_hardness.csv` (capped, builder-less). **Canonical 1-push eval key = `namo_testset_v1/labels/onepush_episodes.json`** (20% bar; the old `v3_test_episodes.json` was the looser "≥1 point" bar and is superseded).

## Pipeline (all committed, reuse-don't-fork)
| step | script | output |
|---|---|---|
| geometry disjointness gate | `scripts/pipeline/verify_geom_disjoint.py` | `stats/geom_disjoint.json` |
| canonical scenes + 1-push key | `scripts/pipeline/build_canonical_testset.py` | `manifests/canonical_scenes.txt`, `labels/onepush.json`, `stats/canonical_stats.json` |
| exhaustive depth-2 collection | `scripts/amarel/testset_2push_collect.slurm` (+ `configs/exhaustive_depth2.yaml`) | `pkls_2push/shard_*/pkls/**/*.pkl` |
| 2-push answer key (F1′) | `scripts/pipeline/build_2push_validset.py` | `labels/twopush.json` |

Build commands and full per-tier statistics: see the datasheet `README.md` in the dataset home.

## Pinned sim settings (match training; do not change without re-collecting)
Collisions ALLOWED (only robot↔non-target aborts), target-region-goal, car primitives `1x_car_d5_`, `config/namo_config_complete_skill15_car_1x.yaml`.
