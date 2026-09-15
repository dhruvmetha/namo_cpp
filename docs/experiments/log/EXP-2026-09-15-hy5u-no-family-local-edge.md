---
type: experiment
status: running
created: 2026-09-15
updated: 2026-09-15
metric: "One-keyhole frozen600 at 1 mm, best-first search, 3000-call cap: simulator calls to success (untimed) and seconds to success on whole Platinum 8358 nodes (timed), split by 1-push/2-push and easy/medium/hard; HY5U with the family margin loss, local feature sampling and the contact-index embedding all removed, seeds 1-3."
tags:
  - experiment
  - ablation
  - one-keyhole
  - timing
---
# HY5U without the family loss, local feature sampling or contact-index embedding

## Why

[USER 2026-09-15] Train a model with no family loss, no local feature sampling and no edge identity, all three at once, 3 seeds on the best GPU we can get. Run inference for this model alone on Amarel, on the 1 mm one-keyhole set with 300 1-push and 300 2-push problems at 3000 simulator calls: once for simulator calls, once timed. No HY5U or Random controls get timed. Keep a Notion kanban board. "No family" means the registered no-family ablation (family margin loss off, same data), chosen by the user over dropping family boards from the training data.

On that set each change alone matched or slightly beat HY5U on two-push problems ([EXP-2026-09-14-keyhole600-ablations](EXP-2026-09-14-keyhole600-ablations.md)). This run removes all three.

## Plan

Training copies HY5U and the single ablations: `scripts/rl_loop/run_hy5u_arch_ablations_cs.sh` arm `HY5U_no_family_local_edge` sets `EGMM_LAMBDA=0 NAMO_USE_LOCAL=0 NAMO_USE_EDGE_EMBED=0 NAMO_GLOBAL_READOUT=0`. Everything else is shared: `aquaman/round0/hybrid_train_v1.h5`, `train_q2_round2.py`, 12 epochs, batch 256, `NAMO_GAMMA=0.5`, `NAMO_UNREACH_W=0.1`, grouped episode batches kept on (as in the registered no-family arm, so only the loss changes), `RANK_LAMBDA=0.1`, `LOWER_RANK_LAMBDA=0.05`, edge self-attention on, action motion off, seeds 1-3, best validation checkpoint. One 1-epoch smoke on seed 1 runs first. With local sampling and the index embedding both off, a contact's token is only the Fourier encoding of its position; `python/tests/test_action_motion.py` checks that reordering the contacts only reorders the scores.

GPUs: arrakis RTX 6000 Ada, one seed each. The only faster GPUs we can reach, ilab4's 8 RTX 5000 Blackwell, are all in use, and Amarel's GPUs are Ada or Ampere and in maintenance. Each seed gets 4 data-loader workers instead of 6, because another user's jobs hold about 18 of arrakis's 32 cores; that changes only the random data order.

Sim-count test: `scripts/pipeline/run_one_keyhole_frozen.py` exactly as in the keyhole600 ablations (search, 3000 calls, statistics on), 595 problems on Amarel. The 5 problems Amarel's build refuses (40, 277, 373, 463, 504) run on arrakis, as before. Results slot into the `hy5u-ablations-keyhole600-1mm-v1` table next to HY5U, no-family, no-local, no edge identity and Random.

Timed test: the best-first search already records total, simulator and scoring time when `record_timing` is on; the keyhole runner had it off. The runner gets a timed switch (timing on, statistics off), and each job holds a whole exclusive Platinum 8358 node on Amarel, runs one problem at a time with every thread pool at 1 and CPU scoring. Each timed row must match its untimed row's result and call count. The 5 refused problems cannot run on Amarel, so the timed set is 595 problems.

Amarel maintenance runs from 2026-09-15 08:00 to 2026-09-16 23:59, so Amarel inference starts 2026-09-17.

## Run

A first version of this card planned only no-local plus no edge identity (commit `e8f2ac4d`, arm `HY5U_no_local_no_edge`). Its smoke started at 05:44 on arrakis GPU 0, which another user had just started using, so I stopped it and relaunched on GPUs 1-3 at 05:45. [USER 05:47] stopped that run about 2 minutes in, before any model trained, and asked for the three-change model instead. Both partial smoke folders stay under `$NAMO_SCRATCH/aquaman/round0/architecture_no_local_no_edge_20260915/`.

## Result

Pending.
