---
type: experiment
status: live
created: 2026-08-31
updated: 2026-09-02
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

An initial direct-westeros smoke was stopped after seven minutes, before an epoch or checkpoint completed, when the user clarified that long training must be scheduler-owned through SLURM and `srun`. Its partial logs remain under the CS `$NAMO_SCRATCH/aquaman/round0/ablations_20260831/smoke/`; no result from that attempt is usable.

The exact 4,227,636,488-byte H5 was transferred to Amarel and smoke array `61089794_[0-2]` was queued correctly through SLURM plus `srun`, but it was cancelled before starting once a real iLab allocation probe disproved the long `sbatch --test-only` estimate. The iLab account has low fair-share priority after recent usage (`priority=43`, fair-share component 37/3000), yet five-minute probe `255919` backfilled onto a clean ilab3 RTX 4500 Ada after about 20 seconds and completed successfully under `srun`; the mid-September test-only date was not an operational ETA.

The actual target-box smokes were iLab SLURM jobs `255921`, `255922`, and `255923`, each allocated one RTX 4500 Ada on ilab3, 48 GB RAM, and a two-hour limit. No-family and regression-only passed a complete epoch, checkpoint, strict reload, and scorer-load check in 22:38 and 22:39. Independent contacts hit `CUDA error: uncorrectable ECC error encountered` while moving the model to its allocated GPU and produced no checkpoint. SLURM incorrectly recorded that task as `COMPLETED` because the shared wrapper accepted a zero runner status without requiring the scorer-load marker.

The retry hardens `scripts/slurm/train.slurm` so every successful job must emit the `eval_scorer-load check` marker and at least one checkpoint, then reruns all three one-epoch smokes through `scripts/ilab/hy5u_ablations_train.slurm` on a healthy pinned node. Only a clean three-arm smoke releases the nine full trainings; all work remains scheduler-owned and runs the training command through `srun`.

The first hardened retry, array `263859` on rlab4, failed closed before accepting any checkpoint and exposed two independent orchestration bugs. The shared training template treated every SLURM array index as an implicit gamma sweep, so task zero changed the requested `NAMO_GAMMA=0.5` to `0.3`; gamma sweeping is now explicit. The batch shell also staged the H5 before entering `srun`, while rlab4's task step could not see that `/dev/shm` path; the ablation launchers now enter `srun` first and perform staging and training inside the same task step. The invalid outputs under `smoke_ilab_v2` are retained for diagnosis and excluded from every result.

Retry `263862` confirmed the requested gamma but showed that rlab4 removes the staged `/dev/shm` file even within one task step; its `/tmp` is the local RAID root and has 286 GB free, so the next pinned-node smoke stages there and calibrates the resulting epoch time rather than assuming RAM-disk speed. This retry also showed that the site's `srun` can report a failed task while the enclosing array task is recorded `COMPLETED`, so both launchers now verify the scorer-load marker and checkpoint again after `srun` returns. Invalid outputs under `smoke_ilab_v3` are retained for diagnosis and excluded.

The valid pinned-rlab4 smoke writes to `smoke_ilab_v4`: no-family runs as task `263865_0`, while the still-pending 48 GB regression and independent tasks were replaced before execution by 24 GB tasks `263869_[1-2]` so both can fit after task zero releases its allocation. Full fleet `263871_[0-8]` is queued with the scheduler dependency `afterok:263866:263869`; it cannot start unless all three smoke jobs pass the outer checkpoint and scorer-reload gates.

The existing HY5 seeds are being reused as the no-unreachable arm. Their Amarel checkpoint hashes match the CS originals exactly, one current-code canonical eval smoke passed as job `61192621`, and the six full fixed-physics-v3 horizon arrays are `61192630` through `61192635`. These are simulation-count evaluations only; their unpinned wall times will not be compared with another method.

## Result

Pending.

## Verdict

Pending.
