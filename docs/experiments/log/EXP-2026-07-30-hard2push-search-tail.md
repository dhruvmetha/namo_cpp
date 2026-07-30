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

Smoke job `59590943` ran one budget-truncated tail episode on Amarel `main-redhat` at commit `7c3339d`, with `hmax=2`, budget 1,200, `combine=q`, confidence discount τ=0.15, no-op dedupe on, and jam-depth pruning on. It reproduced the expected unsolved result at exactly 1,200 simulations and completed in 6m26s.

The calibrated 10,000-call worst-case estimate is 53m37s per episode. Amarel job `59592040` piloted two of the 12 tail episodes at the full 10,000-call cap with a 2h wall limit. Both reproduced unsolved-through-900 and then exhausted their queues naturally at 1,461 and 7,099 calls in 4m38s and 40m19s.

The user caught that the existing 2push divisions were based on incomplete-manifest setup counts rather than the available exhaustive-GT setup percentage. Commit `3b4cc1b` makes the fixed GT cuts canonical: hard <5% (140), medium 5–30% (471), easy ≥30% (370), with 37 unmatched roots explicitly unknown; the old 371/409/238 bins remain registered only for historical reproduction. All 12 selected budget-900 tails are in the corrected GT-hard tier, so the tail run remains correctly scoped. The first pilot exhaustion proves a corrected-tier ceiling of at most 139/140 = 99.29% under unchanged search.

## Result

Pending.

## Verdict

Pending.
