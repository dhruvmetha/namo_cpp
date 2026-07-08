---
status: hub
thread: scorer-search
tags: [hub]
updated: 2026-07-08
---
# Thread — scorer-search

North-star: learn a single supervised value/ranker (NoHz) that orders pushes to minimize search sims, replacing budget/horizon-conditioning.

**Status:** concluded — **NoHz-v3 is the reigning baseline** (reactive 40.7 / best-first 37.8 @2); this thread now serves as the baseline every other thread is measured against.

**Key cards:** [[EXP-2026-07-06-reactive-mpc-depth5]] (budget plateaus, search stays load-bearing) · [[scorer_1push_results]] · [[model_comparison_report]] · [[informative_1push_training_study]].

**Queued (not run):** [[EXP-2026-06-25-qboot-bootstrapped-q]] (parked) · [[EXP-2026-07-01-pure-v-vs-combineq]].

**Journals/registry:** [[horizon_q_model_registry]] (ACTIVE — ckpt paths, never glob) · [[horizon_q_HANDOFF]] · [[horizon_q_build_journal]] · [[horizon_q_search_redesign_journal]] · [[horizon_q_redesign_execution]] · [[multipush_horizonQ_journal]] · [[policy_framework_journal]] · [[scorer_hacman_journal]].

**Hypothesis on deck:** [[policy_value_search_hypothesis]] (π+V split; the rl-loop hub also links it, since the RL loop trains the same two heads).

**Results:** [RESULTS.md](RESULTS.md) §1-5.

**Open items:** none — concluded. NoHz-v3 stays the baseline to beat until a successor thread clears it.
