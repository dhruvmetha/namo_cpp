---
type: experiment
status: live
created: 2026-09-06
commit: 9dbe39bd
metric: pending
tags: [experiment, full-namo, multihop, hy5u, same-template, interaction, medium, hard]
---
# HY5U on the controlled two-keyhole cohort

**Read [docs/problem_and_approach.md](../../problem_and_approach.md) first.** HY5U remains one local push ranker inside simulator-verified search. Full NAMO invokes it again after a committed K1 opening and carries the physical scene state into the K2 search.

## Hypothesis

_(user, from chat)_ Try the 40 current-physics two-keyhole scenes with HY5U and measure which complete scenes it solves.

## Plan

Freeze the exact 40-scene approval population from `EXP-2026-09-05`: ten MM, ten MH, ten HM, and ten HH scenes, with seven clean and three K2-contact scenes in each cell. The XML-list SHA-256 is `55b0d2d13918bf349ff13b598a78ef3cfc7b3caba0566d72b83532bbf7920220`; the metadata-manifest SHA-256 is `d1f3550a84446b66ce9fe8ae129afd5f377068ec21e7723aa8c3d0041f43153e`.

Use registered HY5U seed 2 checkpoint `$NAMO_SCRATCH/aquaman/round0/models/HY5U_s2/checkpoints/epoch011-val_loss0.3256.ckpt`, SHA-256 `3cf348cf7ba247f2cb143376371fc06771665793783d12e3b37bf596e0e5a854`. Run ordinary Full NAMO best-first search with model ordering, `hmax=2` per local keyhole, `1x_car_d5_` primitives, raw `q`, discount off, no-op deduplication, jam pruning, and 900 simulator calls reset per keyhole. Enable the read-only next-keyhole audit, but do not enable the rejected strict all-contact preservation gate.

First run one complete scene on rlab7 with one worker. If checkpoint loading, topology selection, Full-NAMO replanning, and artifact writing pass, use the measured runtime to size the 40-scene rlab7 run. Report complete-scene solve count and total simulator calls by ordered donor pair and by clean/contact scene type. Every scene remains in the denominator whether HY5U solves it or not.

## Run

Pending smoke.

## Result

Pending.

## Verdict

Pending.
