---
type: experiment
status: idea
created: 2026-09-15
updated: 2026-09-15
metric: "One-keyhole frozen600 at 1 mm, best-first search, 3000-call cap: simulator calls to success (untimed) and seconds to success on whole Platinum 8358 nodes (timed), split by 1-push/2-push and easy/medium/hard; HY5U with both the contact-index embedding and local feature sampling removed, seeds 1-3."
tags:
  - experiment
  - ablation
  - one-keyhole
  - timing
---
# HY5U without the contact-index embedding and without local feature sampling

## Why

[USER 2026-09-15] Train a new model with no contact-index embedding and no local feature sampling. Test it on the 1 mm one-keyhole set with 300 1-push and 300 2-push problems, once for simulator calls and once for time. Keep a Notion kanban board. Train 3 seeds on the best GPU we can get, and run inference on Amarel.

On that set, each change alone matched or slightly beat HY5U ([EXP-2026-09-14-keyhole600-ablations](EXP-2026-09-14-keyhole600-ablations.md)). This run removes both at once. With both off, a contact's token is only the Fourier encoding of its position, which then attends to the scene and to the other contacts. `python/tests/test_action_motion.py` checks that reordering the contacts only reorders the scores, so no per-index identity is left.

## Plan

Training copies HY5U and the two single ablations (`hy5u-architecture-ablations-hmax2-v3`, `hy5u-edge-identity-ablation-hmax2-v3`): `scripts/rl_loop/run_hy5u_arch_ablations_cs.sh` arm `HY5U_no_local_no_edge`, which sets `NAMO_USE_LOCAL=0 NAMO_USE_EDGE_EMBED=0 NAMO_GLOBAL_READOUT=0`. Everything else is shared: `aquaman/round0/hybrid_train_v1.h5`, `train_q2_round2.py`, 12 epochs, batch 256, `NAMO_GAMMA=0.5`, `NAMO_UNREACH_W=0.1`, grouped episodes, `EGMM_LAMBDA=0.1`, `RANK_LAMBDA=0.1`, `LOWER_RANK_LAMBDA=0.05`, edge self-attention on, action motion off, seeds 1-3, best validation checkpoint. One 1-epoch smoke on seed 1 runs first.

GPUs: arrakis, 3 RTX 6000 Ada, one seed each. The only faster GPUs we can reach, ilab4's 8 RTX 5000 Blackwell, are all in use, and Amarel's GPUs are Ada or Ampere and enter maintenance at 08:00 today. Each seed gets 4 data-loader workers instead of the 6 the single ablations used, because another user's jobs hold about 18 of arrakis's 32 cores. The worker count changes only the random data order, the same kind of difference a new seed makes.

Sim-count test: `scripts/pipeline/run_one_keyhole_frozen.py` exactly as in the keyhole600 ablations (search, 3000 calls, statistics on), 595 problems on Amarel and the 5 problems Amarel's build refuses (40, 277, 373, 463, 504) on arrakis. Compare against the registered `hy5u-ablations-keyhole600-1mm-v1` rows: HY5U, no-local, no edge identity and Random.

Timed test: whole exclusive Platinum 8358 nodes on Amarel, one thread, timing on, statistics off, each row paired with its untimed row. `eval_full_namo_walltime.py` has no one-keyhole path yet; Tri-An's timed campaign added one (`population: one_keyhole`), so it gets ported. Tri-An's timed keyhole rows cannot serve as controls: his Random seeds miss 152 of 3,000 rows, and his build ran 13% slower than ours on Full NAMO. Which control arms to time with the new model is open.

Amarel maintenance runs from 2026-09-15 08:00 to 2026-09-16 23:59, so Amarel inference starts 2026-09-17.

## Run

Stopped [USER 2026-09-15 05:47] before any model trained. The first launch at 05:44 put the smoke on arrakis GPU 0, which another user had just started using, so I stopped it and relaunched on GPUs 1-3 at 05:45; that smoke ran about 2 minutes before the stop. Both partial smoke folders are kept under `$NAMO_SCRATCH/aquaman/round0/architecture_no_local_no_edge_20260915/`, the first in `aborted_gpu0_shared/`. The pinned worktree `ktamp/namo-train-noloc-noedge` at `e8f2ac4d` stays for a relaunch; the second smoke folder must be moved aside first, since the launcher refuses to overwrite a run.

## Result

Pending.
