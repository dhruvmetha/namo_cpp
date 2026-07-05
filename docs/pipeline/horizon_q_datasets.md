# Horizon-Q datasets — datasheet

Datasets for the budget-conditioned horizon-Q build. Companion to the build journal ([../experiments/horizon_q_build_journal.md](../experiments/horizon_q_build_journal.md)) and the per-episode invariants ([multi_episode_rooms.md](multi_episode_rooms.md)). **All collection at the locked `car=0.034 / control_steps_per_push=550` action space, 20% reachable bar.**

---

## 1. Test set — `namo_testset_v1` — **REUSED AS-IS** (no re-collection)
- **Location:** `/scratch/dm1487/datasets/namo_testset_v1/` (its `README.md` is the full datasheet).
- **Eval keys:** `labels/onepush_episodes.json` (20% bar, 1-push), `labels/pure2push.json` (2-push F1′), `labels/twopush.json`. Scenes: `manifests/canonical_scenes.txt` (2173, geometry-disjoint, 0 leaks).
- **Why reusable at the new action space:** it was collected at `car=0.0375 / 550`. The **pure car effect** (0.0375→0.034, both at 550) was measured **NEGLIGIBLE: ±0.5% reach, sub-3 mm, ≤0.02°/depth** (controlled regen diff, 2026-06-11). The alarming ~14% primitive shift seen earlier was the unrelated `482→550` push-duration config change, not the wheels. ⇒ the 0.0375+550 test set ≈ the 0.034+550 world within eval noise — **valid for grading 0.034/550 models.** No re-collection / no re-labeling.
- **Eval:** `scripts/eval_scorer.py` (referee) → `resolve_robust.sh` (verdict, 3 seeds). Default episodes = `onepush_episodes.json`. Budget-Q adds slices: both deploy regimes · post-push · dead-end · 2-push solve@k/sims.

## 2. H=1 training data — `v4_hq_h1` (collecting)
- **Output:** `/scratch/dm1487/outputs/v4_hq_h1/` (SLURM job 55944720, array 0-59, launched 2026-06-11).
- **Scenes:** `/scratch/dm1487/manifests/v3_feb_top250k.txt` (250 000 v3 feb_car envs). *(aug9_car can be added for composition parity with v3_scorer_e4; feb-only for the first pass.)*
- **Driver:** `scripts/amarel/v3_phase1_collect.slurm` with `GOALS_PER_REGION=100 OUTPUT_DIR=…/v4_hq_h1 PYTHONDONTWRITEBYTECODE=1`. Algorithm `region_opening`, `--region-max-chain-depth 1`, `--region-min-reachable-fraction 0.2`, `--primitive-prefix 1x_car_d5_` (the 0.034/550 primitives), `--config-file config/namo_config_complete_skill15_car_1x.yaml` (control_steps 550, stuck_threshold 5), `--goal-strategy random_rollout` (Pattern B), `--region-allow-collisions`, `--target-goal-region`.
- **Per scene:** **sampled ~30** (edge,depth) cells (random_rollout), NOT exhaustive 300 — this IS the H5 recipe (~30 sampled + masked ≈ exhaustive; the other ~270 are UNKNOWN, masked). Bar = 20% of 100 sampled goal points.
- **Labels live in:** `episode_results[i]['algorithm_stats']['primitive_trial_log']` = list of `{edge_idx, depth, success, wall_collision, movable_collisions, stuck, …}`. Each episode = one `(neighbour, goal)` = one **(pushed object, goal region)** unit. `chain_depth` tag present.
- **Dead-ends: RECORDED** (validated). A scene with no 1-push opener at the 20% bar produces an episode with `success=False`, `failure_reason=all_pushes_failed`, and an all-fail trial_log — it is NOT dropped at collection. **The H0b "no hopeless scenes" bug was in the dataset BUILDER**, which dropped all-zero f_grids.
- **⚠ Build step still needs the H0b fix:** `build_scorer_dataset.py` must **KEEP** dead-end scenes (all-zero f_grid retained), so the budget-Q value can learn "low"/unsolvable. (Task #22.)

## 3. H=2 training data — `v4_hq_h2` (Phase 3, not yet collected)
- Search-distilled on the **informative subset** (scenes where no 1-push opens — derivable from `v4_hq_h1`).
- Per first push: execute→s′; gamma label `1.0` (opens) / `γ≈0.9` (a 2nd push opens) / `0`; **record success-fraction** (robustness, re-recorded each ExIt round). Round-1 verify-heavy.
- Exploration-controlled: uniform/uncertainty first-push selection (NOT confidence), floor ≥25%, setup-state archive (Go-Exploit), disagreement acquisition. Harvest post-push H=1 labels for free. Tag negative types.
- Reuse the tagged depth-2 machinery (`region_opening` chain_depth/parent + `build_2push_validset`).

## 4. Build → H5 → train (the chain)
`pkls (v4_hq_h1) → build_scorer_dataset.py [+keep-dead-ends fix] (join DiT masks + f_grid + r_mask + contact_px) → packed H5 → train_classifier (budget-Q: budget_cond + value_bins HL-Gauss + gamma targets)`. Hold out **by room (xml)**; match samples to episodes by `object_center (~0 mm)`; difficulty per-episode (skill invariants).
