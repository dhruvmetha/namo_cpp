---
type: experiment
status: running
created: 2026-09-14
updated: 2026-09-14
commit: pending
metric: "Tri-An's frozen one-keyhole 600 at 1 mm: solve rate and simulator calls until solved (3000 cap), split by 1push/2push and easy/medium/hard; HY5U and 8 ablation arms, 3 seeds each, best-first search."
tags:
  - experiment
  - ablation
  - one-keyhole
---
# HY5U ablations on the frozen one-keyhole 600

## Why

The registered HY5U ablations were measured on testset v3 at 5 mm, and the 5 mm results are no longer used. [USER 2026-09-14] The ablations are the one set of results to redo at 1 mm, on Tri-An's 600 one-keyhole problems, search only, and with rank-only's no-floor version added.

## Plan

Population: `one_keyhole_frozen600_20260912_v1` (manifest sha256 `bb01f360`), `canonical_1mm`. 300 one-push and 300 two-push problems, each 100 easy, 100 medium and 100 hard. Tiers come from an exhaustive per-problem certificate, expected trials E = (N+1)/(S+1): easy E <= 3, medium E <= 15, hard above.

Arms, 3 seeds each, all shuffle seed 7000: HY5U; HY5 (no unreachable-cell rule); no-family; regression-only; independent contacts; global readout; no-local; no edge identity; rank-only no-floor. The first 24 checkpoints are the exact files in `eval/policy_group_ablations_20260907/plan.json`, SHA256-checked; rank-only no-floor is `aquaman/round0/rankonly_20260912/models/HY5U_rank_only_nofloor_s{1,2,3}/checkpoints/epoch011-*.ckpt`. That is 27 arms x 600 problems = 16,200 runs. Random stays Tri-An's five uniform seeds on this set (`references-v1/random_s*`, frozen code `25c921b`).

Protocol copies Tri-An's one-keyhole campaign helper: best-first search (`eval_bestfirst._evaluate_pooled_task`), up to 2 pushes, 3000 simulator calls, mean5, raw q, discount off, no-op dedupe and jam pruning on, each problem's frozen target points and door objects, solved when 20% of the target points are reachable (`eval_m3.goal_open_pts`), Tri-An's `namo_config.yaml` (`a58e885f`) with 1 mm inflation (`f8ab35e1`), also used as the scorer's render config. Untimed, statistics on.

Runner: `scripts/pipeline/run_one_keyhole_frozen.py`, a port of Tri-An's `evaluate_frozen_one` including the certificate-v1 restore for 11 problems, launched by `scripts/slurm/one_keyhole_frozen.slurm`. Gate before the full launch: rerun HY5U_s2 on a sample and compare solved and call counts with Tri-An's `references-v1/HY5U_s2_search`, and run every arm once.

## Run

Amarel root `/scratch/dm1487/keyhole600_ablations_20260914`. Tracking board: Notion "Full NAMO rerun", cards titled "Keyhole ablations".

## Result

Pending.
