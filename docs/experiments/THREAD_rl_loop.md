---
status: hub
thread: rl-loop
tags: [hub]
updated: 2026-07-08
---
# Thread — rl-loop

North-star: can pure forward-rollout self-imitation (no search at train or deploy) learn to rank pushes as well as the supervised NoHz baseline, purely from its own solved trajectories?

**Status:** live — gen-1 forecast falsified (kill-signal-2), but the loop demonstrably learns; diagnosing why gains don't transfer to the canonical testset.

**Done card:** [[EXP-2026-07-06-rl-only-self-imitation]] — gen0→gen1, two walls: diminishing in-distribution returns + episode-composition mismatch (pool is 1-push-dominated, testset pure2push is F=∅ only).

**Live card:** [[EXP-2026-07-08-rl-growth-arms]] — successor, arms N (novelty) vs C (composition: genuine-2push episodes) disentangle whether the flatline was data starvation or a method wall.

**Related hypothesis:** [[policy_value_search_hypothesis]] (scorer-search thread; same π+V split, ExIt search as the data engine instead of rollout-RL).

**Reading:** [[reading_list_self_imitation]] (Tier 1 = the loop we're running).

**Results:** [RESULTS.md](RESULTS.md) §6.

**Open items:** (1) **V-head hl_gauss hang** — V training deadlocks epoch ~2-3 on both arms/gens, evidence + repro in the predecessor card; disabled by default until fixed, blocks the secondary π+V best-first eval. (2) **Growth experiment** ([[EXP-2026-07-08-rl-growth-arms]], `status: idea`) — tests whether growing the pool (novelty vs genuine-2push composition) restores the climb the predecessor lost.
