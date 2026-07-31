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

## 2026-07-31 exhaustive-GT cleanup

The extended hard tail exposed four additional episodes whose sampled manifest calls them 2-push-solvable but whose completed exhaustive-GT root contains zero genuine setup pushes: aug9 `set2/benchmark_5/run_0207/pair_000` obstacle 1, aug9 `set1/benchmark_4/run_0245/pair_000` obstacle 3, aug9 `set1/benchmark_4/run_0282/pair_001` obstacle 4, and feb straight100 seed01036 `run_0006/pair_001` obstacle 3. User decision: exclude these exact `(xml, object, region)` records from the canonical search view, preserve the untouched 1,018-episode source manifest, regenerate the derived manifests, and recompute all saved aggregates and plots from existing rows without new simulation.

The first cleanup rebuilt the registry to 1,013 2push episodes: easy 385, medium 488, hard 138, unknown 2. All four budget-900 aggregates and both saved plots were recomputed by filtering the original leaf rows through the registered episode keys; no simulator rerun was needed. The manifest guard passed 8/8 checks.

The final hard episode, feb straight50 seed02037 run 0005 pair 001 obstacle 0, was also removed by user decision after its full GT tree proved a unique successful chain `(edge 26, depth 4) → (edge 28, depth 4)` while learned and all three random orderings exhausted their pruned queues without realizing it. The final registered view is 1,012 episodes: easy 385, medium 488, hard 137, unknown 2; learned now reaches 137/137 = 100% on the natural-exhaustion hard tail, and the source manifest plus exhaustive H5 remain unchanged.
