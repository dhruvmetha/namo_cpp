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

The first production arrays (`59738500`–`59738503`) hit Amarel scratch's expired soft quota after 23 valid results: seed 7000 completed 10/34, seed 8000 completed 13/39, seed 9000 completed 0/34, and the one learned delta completed 0/1. The valid rows were preserved and excluded by exact episode key from retry manifests of 24/26/34/1 missing episodes. Filesystem probe `59745583` verified persistent writes through canonical `/home`; retry smoke `59745687` then produced a valid evaluator leaf artifact there with the unchanged production settings.

The filtered random retry arrays (`59746011`–`59746013`) completed but failed the splice invariant: 37 reruns solved at or before 900 despite being unsolved in the base run. Code inspection found that random seeds depend on each record's position within its original 26-XML shard, so filtering/re-sharding the key changes the baseline. Those rows are rejected. The evaluator now supports `--only-key`: it iterates the original full key and original shard boundaries, skips non-target episodes without changing their indices, and therefore preserves the exact base RNG stream. The learned delta from `59746014` is deterministic and passed the splice invariant, giving 137/142 = 96.5% final hard success with five naturally exhausted searches.

## Result

Pending.

## Verdict

Pending.
