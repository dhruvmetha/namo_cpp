---
type: experiment
status: live
created: 2026-07-29
thread: rl_loop
robot: car
commit: 2d8b040
tags: [experiment, search, random-baseline, no-op-dedupe, jam-pruning, hmax2, canonical-eval]
---

# Post-pruning canonical search — learned ranker vs seeded random at hmax=2

**Read [docs/problem_and_approach.md](../../problem_and_approach.md) first.** The model is a ranker that orders pushes for a simulator-verified search; this experiment measures whether it finds a verified opening with fewer simulator calls than random after the adopted search pruning.

## Hypotheses

With no-op child deduplication and jam-depth pruning enabled, the setup-only Colossus ranker will beat a three-seed random ranker in simulator calls on every easy/medium/hard tier of both canonical eval sets; allowing `hmax=2` on the 1-push set will also measure whether cheap two-push detours rescue ranker misses without changing the one-opening problem.

## Plan

Run the registered `namo_testset_v1` one-push key (1,323 episodes) and pure-two-push key (1,018 episodes) on Amarel with the same search: `hmax=2`, budget 900, `combine=q`, failure discount `conf` with `tau=0.15`, no-op dedupe on, and jam-depth pruning on. Evaluate the deterministic setup-only checkpoint once and random ordering at seeds 7000/8000/9000. Aggregate solve@{1,2,5,10,30,100,300,900} and simulator calls by easy/medium/hard/all; 1-push uses the fixed per-episode solve-rate bins (hard <0.05, medium <0.30, easy otherwise) and 2-push uses the registered fixed divisions file.

## Run

Pending target-box smoke and calibrated pilot.

## Result

Pending.

## Verdict

Pending.
