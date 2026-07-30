---
type: experiment
status: live
created: 2026-07-30
thread: rl_loop
robot: car
tags: [experiment, search, random, hard-2push, tail, hmax2]
---

# Random hard-2push tails on the finalized GT tiers

**Read [docs/problem_and_approach.md](../../problem_and_approach.md) first.** Random and learned are alternative orderings of the same simulator-verified search.

## Hypotheses

Extending only the budget-truncated hard-2push rows for random seeds 7000, 8000, and 9000 will complete the equal-footing tail plot without rerunning solved or queue-exhausted episodes.

## Plan

Use the finalized 142-episode hard tier from the 35-root GT fill. For each random seed, select only unsolved rows stopped exactly at 900 calls, rerun them with unchanged `hmax=2`, no-op dedupe, jam-depth pruning, and a 10,000-call cap, then splice them back into the full hard tier. Extend any newly-hard learned row under the same rule before plotting all four tails.

## Run

Amarel smoke job `59738428` ran one random hard episode end-to-end on `main-redhat` with the production evaluator and shared-scratch outputs. It completed successfully in 4m31s; the actual 100-call search took 32s, produced one valid leaf row, and verified `hmax=2`, uniform prior, no-op dedupe on, jam-depth pruning on, `combine=q`, and confidence discount τ=0.15. A preceding 1,200-call timing probe (`59738062`) took 11m11s but wrote into Amarel's unpublished home cache, so it is used only for the pessimistic runtime estimate and not as a result artifact. The 10,000-call worst-case estimate is about 80 minutes, with a two-hour production cap.

## Result

Pending.

## Verdict

Pending.
