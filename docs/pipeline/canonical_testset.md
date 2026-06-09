# Canonical NAMO test set (`namo_testset_v1`)

The single source of truth for evaluating Region-Opening scorers / policies / value functions on the **car**.
Lives at `/scratch/dm1487/datasets/namo_testset_v1/` (full datasheet + stats in its `README.md`). This doc is the
repo-side record so the build is reproducible and we stop confusing test artifacts.

## What it is
- **2173** geometry-clean scenes from the held-out `car_envs/v3/test/{feb,aug9}_car` pool, **0 leaks** into the
  `v3_scorer_e4` training corpus (proven by geometry, not file names — see below).
- **1-push tier**: 1228 scenes / 1671 episodes, EXHAUSTIVE 1-push answer key (`valid`/`tried`/`solve_rate`),
  difficulty easy 759 / med 496 / hard 416. This IS the champion test set (`v3_test_episodes.json`).
- **2-push tier**: exhaustive depth-2 ground truth (F1′ = first-pushes that enable a solving second push), collected
  over all 2173 scenes; 985 of them (`twopush_scenes.txt`) are the genuine-2-push analysis subset.

## Why the old setup was confusing (and the fix)
The 1-push key was keyed by `outputs/test_*_phase1/...` **symlinks**; the 2-push manifest by `car_envs/v3/test/...`
**real paths**. A name-based "0 overlap with train" check is therefore meaningless. **Fix:** re-key everything by
`realpath` and prove disjointness by **room geometry** = `md5(sorted walls pos/size/euler + sorted obstacle
pos/size/euler)`, goal + robot excluded (two episodes of one room differ only in the goal). 0 / 1128 test scenes
share full geometry with any of 66 135 train scenes.

**Deprecated, do not use:** `v3_test_validsets.json` (no `object_center`, not the eval key),
`test_2push_solvable_combined.txt` (1-push-contaminated), `test_pure2push_hardness.csv` (capped, builder-less).
**Canonical 1-push eval key = `v3_test_episodes.json`.**

## Pipeline (all committed, reuse-don't-fork)
| step | script | output |
|---|---|---|
| geometry disjointness gate | `scripts/pipeline/verify_geom_disjoint.py` | `stats/geom_disjoint.json` |
| canonical scenes + 1-push key | `scripts/pipeline/build_canonical_testset.py` | `manifests/canonical_scenes.txt`, `labels/onepush.json`, `stats/canonical_stats.json` |
| exhaustive depth-2 collection | `scripts/amarel/testset_2push_collect.slurm` (+ `configs/exhaustive_depth2.yaml`) | `pkls_2push/shard_*/pkls/**/*.pkl` |
| 2-push answer key (F1′) | `scripts/pipeline/build_2push_validset.py` | `labels/twopush.json` |

Build commands and full per-tier statistics: see the datasheet `README.md` in the dataset home.

## Pinned sim settings (match training; do not change without re-collecting)
Collisions ALLOWED (only robot↔non-target aborts), target-region-goal, car primitives `1x_car_d5_`,
`config/namo_config_complete_skill15_car_1x.yaml`.
