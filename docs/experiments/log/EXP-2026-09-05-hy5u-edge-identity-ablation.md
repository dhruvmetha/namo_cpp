---
type: experiment
status: live
created: 2026-09-05
updated: 2026-09-05
commit: 7f1f67d2
metric: "Canonical fixed-physics-v3 success versus simulator calls, split by easy/medium/hard and 1push/2push; three seeds."
tags:
  - experiment
  - ablation
  - architecture
  - icra27
---
# HY5U learned edge-identity ablation

**Read [docs/problem_and_approach.md](../../problem_and_approach.md) first.** HY5U is a search heuristic that ranks candidate pushes; the simulator remains the verifier.

## Hypothesis

The learned identity assigned to each of the 60 contact slots contributes useful ordering beyond the contact's Fourier coordinates. It should matter especially at the four object corners, where two distinct face contacts have exactly the same contact pixel but execute different push directions.

## Plan

Train `HY5U-no-edge-ID` on the exact HY5U `hybrid_train_v1.h5` with the room-grouped split, setup target 0.5, unreachable-cell regression weight 0.1, complete ranking loss, 51-bin HL-Gauss head, 12 epochs, batch 256, learning rate 3e-4, and seeds 1/2/3. Remove only the learned `Embedding(60)`; retain Fourier contact coordinates, the sampled local scene feature, scene cross-attention, inter-contact self-attention, and the complete HY5U loss.

Run one full epoch on the target training box before releasing the three seeds. Require finite training and validation losses, a checkpoint, strict reload, and the `eval_scorer-load check` marker.

After training, evaluate all three seeds on the canonical fixed-physics-v3 1-push and 2-push populations using `hmax=2`, simulator budget 900, `prior=model`, `agg=mean5`, raw `q`, discount off, no-op deduplication on, and jam-depth pruning on. Reuse the registered HY5U and Random results; do not rerun either baseline. Report solve@1 for 1-push and solve@5 for 2-push by easy/medium/hard/overall, plus complete success-versus-simulator-call curves. Do not compare wall time because this is not a pinned-hardware timing run.

## Run

Pending.

## Result

Pending.

## Verdict

Pending.
