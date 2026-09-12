---
type: experiment
status: running
created: 2026-09-12
updated: 2026-09-12
commit: a97c1754
metric: "Canonical fixed-physics-v3 success-vs-simulator-calls, split by easy/medium/hard and 1push/2push; three seeds."
tags:
  - experiment
  - ablation
  - icra27
---
# HY5U rank-only

**Read [docs/problem_and_approach.md](../../problem_and_approach.md) first.** HY5U is a search heuristic that ranks pushes; the simulator remains the verifier. This arm asks whether the value regression contributes anything the ordering terms do not.

## Hypotheses

[USER 2026-09-12] The September grid in [EXP-2026-08-31](EXP-2026-08-31-hy5u-icra-ablations.md) ran `regression only` but never its mirror. Without the mirror the ablation table only shows that ordering terms matter, not whether regression is doing any work at all.

- [CLAUDE] Rank-only will land within seed noise of HY5U on both horizons. The August rank-pure arm on the older AJ2 lineage landed within about 2 points of the full loss everywhere and took best-in-round two-push solve@2, which is where the "regression is not signal" reading came from. That prior does not transfer for free: HY5U since added the family corpus, the unreachable floor, and inter-contact attention, any of which could have made the value head load-bearing again.
- [CLAUDE] A neutral result is the useful finding, not a null. Paired with regression-only's 18.7-point loss at two-push solve@5, it would say the ordering terms carry the ranker and the value head is an anchor rather than a source of signal.
- [CLAUDE] Watch the score scale. Round 1 measured rank-only inflating the whole range (spread 0.45 to 0.67, dead-board maxima 0.68) once regression stopped holding it tight. Deploy consumes order only under `combine=q`, so a stretch is not automatically a problem, but a pile-up at the bounded head's endpoints would be.

## Plan

One arm, `HY5U_rank_only`, three seeds, one change from HY5U.

The unreachable floor is NOT a separate term in HY5U's loss. `GroupedQ2Dataset` zeroes `f_labels` on unreachable cells and folds them into `loss_mask` at `UNREACH_W`, so HY5U regresses them inside the same HL-Gauss call as the exact cells. Keeping the floor therefore means splitting that call, which is what `scripts/rl_loop/train_q2_rankonly.py` does.

The denominator is the part that is easy to get wrong. HL-Gauss reduces by group mean, so the floor's share of HY5U's regression is `sum(ce * floor) / sum(exact_mask)`, and that denominator counts the exact cells too (~68 tried cells at weight 1.0 against ~230 unreachable at 0.1). Dropping the exact cells from both numerator and denominator would quadruple the surviving term and confound the ablation with a weight change, so the trainer restores HY5U's denominator.

| arm | one change from HY5U | implementation |
|---|---|---|
| HY5U_rank_only | remove exact-cell and censored regression | `train_q2_rankonly.py`; keeps `RANK_LAMBDA=0.1`, `LOWER_RANK_LAMBDA=0.05`, `EGMM_LAMBDA=0.1`, `NAMO_UNREACH_W=0.1` |

Verified before launch, on synthetic batches through the real module: the rank-only loss equals the HY5U loss minus the reachable-exact regression share minus the censored term, |delta| 2.4e-07, with the floor term confirmed non-zero.

`val_loss` is redefined as floor plus per-board rank on the val split, because the stock monitor measures the regression this arm does not train and selecting on it would select on noise. Same consequence as `train_q2_rankpure.py`: this `val_loss` is not comparable to any other registry row, and train_q2's post-training reload line recomputes the regression formula, so a large delta there is expected.

Everything else matches the September grid: `hybrid_train_v1.h5`, room-grouped split, setup target 0.5, 51-bin HL-Gauss head, 12 epochs, batch 256, learning rate 3e-4, grouped batches, edge self-attention, seeds 1/2/3.

Evaluation reuses the canonical protocol so the row sits beside the existing table: fixed-physics v3 at the **5 mm** margin, single object, 1328 one-push and 992 two-push per seed, `hmax=2`, budget 900, `prior=model`, `agg=mean5`, raw `q`, discount off, no-op dedupe and jam-depth pruning on. Registered HY5U, regression-only, and Random artifacts are reused rather than recomputed. Not a pinned-hardware timing run, so no wall-time comparison.

## Run

Placement followed the compute-resources order. arrakis had all five GPUs held by two other users. On iLab, ilab4's Blackwell nodes and ilab3 were full; ilab2 had five free A4500 cards. Amarel showed 200 pending GPU jobs, past the one-hour rule.

Two launches died on iLab infrastructure before any epoch completed, neither in the trainer, which printed its loss configuration correctly both times.

Job `297444` staged the 4.2 GB H5 to `/dev/shm` and lost it mid-epoch. systemd RemoveIPC clears `/dev/shm` when the user's last login session on the node ends, and an ssh to ilab2 to run `squeue` was enough to trigger it.

Job `297445` staged to `/tmp` instead. ilab2's `/tmp` is a dedicated 98 GB NVMe partition but carries a **2 GiB per-user quota**, so the copy died at exactly 2,147,352,576 bytes with "Disk quota exceeded" and fell back to NFS silently. That run then died in a storm of `SemLock._rebuild` FileNotFoundError, because spawned DataLoader workers rebuild POSIX semaphores out of `/dev/shm` and the same wipe destroys them.

So iLab has nowhere node-local to stage this file, and any worker process depends on storage other people's logins can clear. Job `297453` runs the configuration that completed every arm of EXP-2026-08-31: scheduler-owned `srun`, direct NFS reads, zero DataLoader workers, about 45 minutes per epoch. Both findings are written into `scripts/ilab/hy5u_rankonly_train.slurm` with the job numbers.

Smoke `297453` COMPLETED in 49:39. Epoch 0: train_loss 0.8598, val_loss 0.8309, checkpoint written, two-reload delta 0.000e+00, scorer-load check OK at `value_bins=51`. The reload line reported delta 2.0499 against the monitor, which is the expected consequence of redefining `val_loss` and not a fault. Fleet `297459_[0-2]` released on `afterok` and runs three seeds on ilab2, 16-hour limit, all co-located so they share one page cache over the same H5.

### Watch item H3 fired: the score scale collapsed toward ZERO, not up

The scorer-load probe printed `value range=[0.010,0.010]`, so I measured the epoch-0 checkpoint on 32 real val boards instead of trusting one sample.

| quantity, epoch 0 | value |
|---|---|
| global score range over 32 boards | 0.009805 to 0.009806 |
| per-board spread over tried cells, p10 / median / p90 | 0.000001 / 0.000001 / 0.000001 |
| score-vs-label correlation within board, median | 0.089 |

The head is a constant near 0.0098 and the whole ordering signal is 1e-6 wide, which at that magnitude is only a few representable float32 steps.

**UNVERIFIED HYPOTHESIS, one epoch only.** Removing the exact-cell regression removed the only term that pushed any cell UP (openers to 1.0, setups to 0.5). The unreachable floor stayed and pushes ~230 of ~300 cells per board toward 0. The ranking terms are scale-free and indifferent to where the scale sits. Down-force with no counterweight. This is the mirror image of round 1's `RP`, which INFLATED its scale (spread 0.45 to 0.67). But `RP` predates `UNREACH_W` entirely, so it had no absolute down-force at all. The asymmetry is new to this arm, and it is the same mechanism RPEA's autopsy described: an absolute anchor on dead cells "pushes live near-twins down equally".

If it holds, the fix is the variant not selected: drop the floor's REGRESSION too, while keeping unreachable cells in the rank lists, where they still serve as known-worse opponents. `loss_mask` carries them at `UNREACH_W` > 0, so `_rank_list_mask` already includes them and their ordering contribution survives without the absolute zero target.

Not called on one epoch. The gradient to separate scores exists even from a collapsed start. Re-measure at epoch 2 or 3. Widening means a slow start and the fleet stands. Still 1e-6 means the arm is degenerate, stop it rather than hold three shared GPUs for nine hours, and launch the corrected variant.

## Result

Pending.
