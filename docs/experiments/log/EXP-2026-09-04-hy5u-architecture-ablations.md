---
type: experiment
status: live
created: 2026-09-04
updated: 2026-09-04
commit: ffe34dda
metric: "Canonical fixed-physics-v3 success versus simulator calls, split by easy/medium/hard and 1push/2push; three seeds per architecture control."
tags:
  - experiment
  - ablation
  - architecture
  - icra27
---
# HY5U architecture ablations

**Read [docs/problem_and_approach.md](../../problem_and_approach.md) first.** HY5U is a search heuristic that ranks pushes; the simulator remains the verifier. This experiment asks which candidate-specific parts of the ranker architecture produce its low-simulation ordering.

## Hypotheses

The user requested an architecture ablation and explicitly excluded a smaller-model control. The completed independent-contacts arm already removes inter-contact self-attention and will be reused rather than retrained.

- Replacing the 60 contact-token critic with one global scene token and a global 300-action readout will damage low-budget ordering because every action loses candidate-specific local and cross-attention reasoning.
- Removing only the local feature sampled at each contact will produce a smaller loss because Fourier contact position, learned edge identity, scene cross-attention, and inter-contact attention remain intact.

Historical pilots motivate both tests but do not replace them: on the older one-push stack, per-contact cross-attention improved hard solve@1 from 14.3 to 25.6 over global readout, while removing the local gather cost 3.2 points. The current run repeats those comparisons under HY5U's data, loss, seeds, and canonical best-first deployment.

## Plan

Train two new arms on the exact HY5U `hybrid_train_v1.h5` with the room-grouped split, setup target 0.5, unreachable-cell regression weight 0.1, complete ranking loss, 51-bin HL-Gauss head, 12 epochs, batch 256, learning rate 3e-4, and seeds 1/2/3.

| arm | one architecture change from HY5U |
|---|---|
| HY5U-global | replace all contact tokens and contact/scene attention with a CLS-style global scene readout over all 60×5 actions |
| HY5U-no-local | remove the scene feature sampled at each contact; retain Fourier contact coordinates, learned edge identity, scene cross-attention, and inter-contact self-attention |

Use the already completed HY5U-independent arm as the third architecture control. Do not train a width/depth control, no-edge-ID control, patch-resolution control, dual crop, or depth-token arm: the first two are less direct for the paper question and the latter three were already historically flat or negative.

Run a complete one-epoch smoke for both new architectures on the exact target box before releasing the six full trainings. Require finite train and validation losses, a checkpoint, strict reload, and the `eval_scorer-load check` marker. The older architecture experiments supply the exploratory pilot signal; this smoke gates current-code mechanics and calibrates target-box runtime.

After training, evaluate all three seeds of both arms on the canonical fixed-physics-v3 1-push and 2-push populations using `hmax=2`, simulator budget 900, `prior=model`, `agg=mean5`, raw `q`, discount off, no-op deduplication on, and jam-depth pruning on. Report solve@1 for 1-push and solve@5 for 2-push by easy/medium/hard/overall, plus the complete success-versus-simulator-call curves. Do not compare wall time because this campaign is not a pinned-hardware timing run.

## Run

Live placement on 2026-09-04 found iLab's two-hour GPU request projected for 2026-09-09 and Amarel's for 2026-09-05 01:51, beyond the one-hour fallback limit. Arrakis GPUs 1 and 2 were idle RTX 6000 Ada cards, so the target-box smoke and full background fleet use those two GPUs with six data workers per process, staying within the direct-box CPU courtesy limit.

The run uses NAMO commit `ffe34dda` and Sage commit `ceff4bf`.

Background supervisor PID `1655534` started on Arrakis at 14:50 EDT. It writes `architecture_ablations_20260904/supervisor.log` under the round-0 scratch directory, runs the global and no-local smokes concurrently on GPUs 1 and 2, and releases three paired seed waves only after both completion markers pass.

The post-training handoff is precommitted and runs as a separate background supervisor. It waits for all six 12-epoch completion markers, selects the minimum-validation-loss checkpoint from each seed, verifies hashes after transfer to a dedicated Amarel checkout, and submits one target-box smoke per architecture. Each successful smoke releases three 36-task canonical evaluation arrays, followed by strict per-seed aggregation jobs. The evaluation reuses the registered HY5U, Random, and independent-contacts results rather than recomputing them.

Both training smokes passed at 15:08 EDT after 1,116 seconds, and the paired seed-1 full runs started immediately. Post-training handoff supervisor PID `1722492` started on Arrakis at 15:16 EDT using NAMO evaluation commit `61b95a17` and Sage commit `ceff4bf`; it polls at five-minute intervals and will stage the Amarel target-box smokes and dependent full arrays without an interactive session.

## Result

Pending.

## Verdict

Pending.
