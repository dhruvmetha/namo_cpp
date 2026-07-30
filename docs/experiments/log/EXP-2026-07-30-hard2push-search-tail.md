---
type: experiment
status: live
created: 2026-07-30
thread: rl_loop
robot: car
commit: d24434f
tags: [experiment, search, hard-2push, tail, hmax2]
---

# Hard-2push search tail beyond 900 simulator calls

**Read [docs/problem_and_approach.md](../../problem_and_approach.md) first.** This is still one region-opening and the model is still only ordering simulator-verified pushes.

## Hypotheses

The learned ranker's hard-2push curve can rise above its 96.5% solve@900 value when only the budget-truncated tail is searched longer, but it cannot reach 100% under the unchanged search because one of 371 hard episodes already exhausted the queue after 28 simulator calls.

## Plan

Keep the adopted search unchanged (`hmax=2`, `combine=q`, confidence discount τ=0.15, no-op dedupe on, jam-depth pruning on). Build a per-episode tail key from the 13 unsolved hard-2push model rows: retain the 12 rows stopped exactly at budget 900 and exclude the one queue-exhausted row. Smoke one tail episode on Amarel, calibrate calls per wall-second, then choose a longer cap and run only the tail. Splice a tail result into the original curve only if every rerun remains unsolved through call 900; otherwise report simulator-jitter instability rather than an optimistic conditional curve. Stop at the natural queue-exhaustion plateau or the calibrated practical cap; do not claim that more budget can cross the proven 99.73% ceiling.

## Run

Pending.

## Result

Pending.

## Verdict

Pending.
