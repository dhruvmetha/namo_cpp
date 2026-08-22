---
type: experiment
status: running
created: 2026-08-22
thread: ranker-as-policy
robot: car
metric: open@k by difficulty x horizon, policy mode (zero search), k up to 10, against the registered v3 search rows
tags: [experiment, policy, reactive, hy5u, testset-v3, amarel]
---

# HY5U as a policy: what the ranker is worth with the search switched off

**Read [docs/problem_and_approach.md](../../problem_and_approach.md) first.** HY5U is the ranker that orders which pushes the search tries. This card removes the search entirely and asks what the ranker's own first choice is worth, executed for real.

## Question

Run the ranker as a policy: score the live state, take its top push, simulate it for real, check whether the region opened, repeat up to K times. No simulate-and-undo, no queue, no backtracking, so the ranker never gets a second guess at the same state. How far does that get on the canonical v3 one-push and two-push populations at K=5 and K=10, split by difficulty?

## Definition

Policy mode is `scripts/sandbox/eval_reactive_argmax.py`: push *i* is `argmax Q(s_{i-1}, ., H)` over the reachable pushes of the labeled blocking object at the live state, executed with `env.step`, graded by the canonical region criterion (`goal_open_pts`, at least 20 of the 100 s0-sampled goal-region points reachable). Early stop on open, or when the candidate pool empties. Cost is exactly the push index, one simulator call per push.

Two properties of the harness that shape the run:

- **One rollout at `--max-pushes 10` answers K=5 and K=10 both.** Each leaf records `opened_at`, the push index the region opened at (0 = never), so cumulative open@1 through open@10 come from the same trajectory. A separate K=5 job would be a duplicate.
- **`--h` is inert here.** `live_scorer.score_ctx` only passes a budget token when `network.budget_cond` is set; HY5U is a plain ranker, so the first-push query budget changes nothing. Confirmed in the smoke rather than assumed.

## Plan
_(Claude, 2026-08-22)_

**Population.** Fixed-physics v3, identical to the registered search rows: 1,328 one-push episodes (easy 681 / medium 442 / hard 205) and 992 genuine two-push (387 / 487 / 118), resolved through `namo.eval_sets`. Tiers by the same rule `agg_search_eval.py` uses: `bin_of(solve_rate)` for one-push, the `division` field for two-push.

**Arms.** HY5U seeds 1/2/3 (`/cache/home/dm1487/aquaman0/ckpts_bfix/HY5U_s{1,2,3}.ckpt`) and uniform random seeds 7000/8000/9000, both horizons, K=10. Random makes no model call and costs almost nothing, and beating random is the success bar.

**Box.** Amarel `main`, one CPU per shard, `scripts/slurm/policy_argmax_amarel.slurm`. Amarel's `.so` carries `src_tree=9ca4a6e` / `cpp_tree=7e6e802`, byte-identical to this checkout's C++ trees and to the build that produced the v3 search numbers, so the physics is the same one. Nothing between `d32ec06` (the v3 eval commit) and this run's commit touches `src/`, `include/`, `best_first_search.py`, or `live_scorer.py`.

**Validation gates, checked before any result is reported.**

1. One-push policy open@1 must reproduce the registered search solve@1 of 82.5±0.4. These are the same quantity: with `combine=q` the search's first pop is the argmax of the same pool at the same state. A gap wider than sim jitter means the reactive path diverged from the search path, and the run is void.
2. Two-push policy open@1 must be near zero. Those episodes have no one-push solution by construction.
3. Episode counts must land on 1,328 and 992 minus a small skip count, and the aggregator's unmatched-row count must be zero.

## Run
_(Claude, auto from run output)_

Smoke: jobs 60754561 / 60754562 on Amarel `main`, one middle shard per leg per arm (`--array=50,150`, `N1SH=N2SH=100`), commit `2fdab03`.

## Result + Verdict
_(pending)_

## Next

## Discussion
_(you <-> Claude — ask here; I answer inline, dated `**[who YYYY-MM-DD]**`. Newest at the bottom.)_
