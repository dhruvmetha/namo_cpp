---
type: experiment
status: live
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

Pending.

## Result

Pending.

## Verdict

Pending.
