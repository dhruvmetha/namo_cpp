---
type: experiment
status: live
created: 2026-08-31
updated: 2026-08-31
commit: 0312528
metric: "Canonical fixed-physics-v3 success-vs-simulator-calls, split by easy/medium/hard and 1push/2push; three seeds per new arm."
tags:
  - experiment
  - ablation
  - icra27
---
# HY5U ICRA ablations

**Read [docs/problem_and_approach.md](../../problem_and_approach.md) first.** HY5U is a search heuristic that ranks pushes; the simulator remains the verifier. These ablations test which parts of the ranker are responsible for reducing simulator calls.

## Hypotheses

The user selected these controls in discussion on 2026-08-31: no unreachable supervision, no family ranking, regression only, and independent contacts. The no-unreachable arm already exists as HY5 and will be reused.

- Removing family ranking will hurt two-push search most because the best-first queue compares candidate pushes from root and child boards in one episode.
- Removing every ranking term and retaining regression will hurt both horizons because calibrated value regression does not directly train the ordering that search consumes.
- Removing inter-contact self-attention will hurt within-board ordering because each contact can no longer compare itself with other contacts before scoring.

## Plan

All new arms use `/common/users/dm1487/scratch_namo/aquaman/round0/hybrid_train_v1.h5`, the HY5U room-grouped split, setup target 0.5, unreachable-cell regression weight 0.1, 51-bin HL-Gauss head, 12 epochs, batch 256, learning rate 3e-4, and seeds 1/2/3. Grouped batches stay on in every arm so the no-family and regression-only arms change the loss rather than the batch composition.

| arm | one change from HY5U | implementation |
|---|---|---|
| HY5 | remove unreachable-cell supervision | reuse registered HY5 seeds; no retraining |
| HY5U-no-family | remove the episode-family margin loss | `EGMM_LAMBDA=0`; keep per-board ranking |
| HY5U-regression | remove family and per-board ranking | `EGMM_LAMBDA=0`, `RANK_LAMBDA=0`, `LOWER_RANK_LAMBDA=0` |
| HY5U-independent | remove inter-contact self-attention | `NAMO_EDGE_SELF_ATTN=0`; keep the complete HY5U loss |

Before the full fleet, run one full training epoch for each new arm on the exact target CS box, with the real H5 and output path. Require a finite train/validation loss, a checkpoint, a strict reload, and the `eval_scorer-load check` marker. Use the measured epoch time to size the background run. The full fleet starts only for arms that pass.

After training, evaluate all three seeds on both registered fixed-physics-v3 horizons with the canonical HY5U best-first protocol, and report success versus simulator calls for every difficulty tier. Evaluation is a later stage and is not part of the training launch.

## Run

The CS `unlimited` queue projected starts between September 14 and September 21 for every accessible GPU type, so the run uses free direct GPUs on westeros over the same shared filesystem. Three one-epoch target-box smokes started at 2026-08-31 02:58 EDT on GPUs 1/2/3 from commit `32feea2`; each staged the real H5 to `/dev/shm`, loaded 1,302,659 rows with the same 1,172,394/130,265 room-grouped train/validation split, built 234,307 episode families, and entered GPU training.

Smoke outputs are under `$NAMO_SCRATCH/aquaman/round0/ablations_20260831/smoke/`. The background supervisor is `scripts/rl_loop/run_hy5u_ablations_cs.sh`, westeros PID 994972, with log `$NAMO_SCRATCH/aquaman/round0/ablations_20260831/supervisor.log`. It requires `TRAIN DONE` from all three smokes before launching the nine full jobs, and aborts on a traceback, an early trainer exit, or a two-hour smoke timeout. Full outputs will land under `$NAMO_SCRATCH/aquaman/round0/ablations_20260831/models/`.

## Result

Pending.

## Verdict

Pending.
