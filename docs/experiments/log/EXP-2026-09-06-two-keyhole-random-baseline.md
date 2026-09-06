---
type: experiment
status: live
created: 2026-09-06
commit: 8d2ab031
metric: pending
tags: [experiment, full-namo, multihop, random-baseline, same-template, interaction, medium, hard]
---
# Three-seed random baseline on the controlled two-keyhole cohort

**Read [docs/problem_and_approach.md](../../problem_and_approach.md) first.** Random is the baseline ranker inside the same simulator-verified local search used for HY5U. Full NAMO still solves one local boundary at a time and carries the physical state into the next local search.

## Hypothesis

_(user, from chat)_ Compare Random against HY5U on the same 40 controlled two-keyhole scenes.

## Plan

Reuse the exact 40-scene population evaluated by HY5U: ten MM, ten MH, ten HM, and ten HH scenes, with seven clean and three K2-contact scenes in each cell. The XML-list SHA-256 is `55b0d2d13918bf349ff13b598a78ef3cfc7b3caba0566d72b83532bbf7920220`; the metadata-manifest SHA-256 is `d1f3550a84446b66ce9fe8ae129afd5f377068ec21e7723aa8c3d0041f43153e`.

Run uniform random ordering with the project's standard random seeds 7000, 8000, and 9000. Keep the HY5U protocol fixed: ordinary Full NAMO best-first search, `hmax=2` per local keyhole, `1x_car_d5_` primitives, raw `q`, discount off, no-op deduplication, jam pruning, and 900 simulator calls reset per keyhole. Keep the read-only next-keyhole audit enabled and the rejected strict preservation gate disabled. The random arm skips model scoring; the launcher receives the HY5U path only because its shared shell interface requires `CKPT`.

Smoke one complete scene with random seed 7000 on rlab7 using the production command. If selection, uniform ordering, Full-NAMO replanning, and artifact writing pass, run all 40 scenes for all three seeds on the same node. Report each random seed, the three-seed mean and range, paired solve outcomes against the existing HY5U seed-2 arm, simulator-call curves, and MM/MH/HM/HH by clean/contact. Join evaluation rows to the metadata manifest by `realpath`, never basename.

## Run

Pending.

## Result

Pending.

## Verdict

Pending.
