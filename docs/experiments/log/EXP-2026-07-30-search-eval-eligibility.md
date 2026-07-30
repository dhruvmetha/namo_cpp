---
type: experiment
status: done
created: 2026-07-30
thread: rl_loop
robot: car
tags: [experiment, eval-set, search, eligibility]
---

# Exclude search-ineligible easy episodes

**Read [docs/problem_and_approach.md](../../problem_and_approach.md) first.** The comparison measures rank ordering only when the common search candidate set contains a verified solution.

## Hypotheses

The one easy-1push and one easy-2push episode where learned plus all three random seeds exhaust the same tiny queue are candidate-generation failures rather than ranker failures, so excluding them from the registered search eval makes the ranking comparison well-defined and leaves the exhaustive source labels intact.

## Plan

Register the two exact `(xml, object, goal region)` exclusions, derive filtered search manifests and both 2push tier files from the untouched source artifacts, update registry counts, and regenerate all aggregates and plots from the existing raw rows.

## Run

Commit `b7dde0c` registers the two exact per-episode exclusions, derives filtered manifests from the untouched source labels, and makes aggregation filter archival raw rows to the registered manifest. The resulting search eval contains 1,322 1push episodes and 1,017 2push episodes.

## Result

Easy 1push now contains 697 episodes and both learned and all three random seeds reach 100%; learned reaches it at 6 simulator calls versus random at 19. Easy 2push contains 385 episodes after the 35-root GT fill and both reach 100%; learned reaches it at 229 calls versus random at 708.

## Verdict

**Adopt.** These two exclusions remove candidate-generation failures shared by every ordering policy; the exhaustive source labels remain unchanged and auditable.
