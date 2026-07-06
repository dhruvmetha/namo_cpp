"""RL-only off-policy self-imitation loop (EXP-2026-07-06-rl-only-self-imitation).

Forward-only rollouts (no branching/resets) are the data engine; solved trajectories feed a
filtered-BC pi head and Monte-Carlo returns feed a V head, both on the reused sage EdgeCrossAttn
scorer. See docs/experiments/log/EXP-2026-07-06-rl-only-self-imitation.md (## Plan).

Modules: config, episodes, splits, policy, rollout, collector, buffer, build_train_h5,
train_gen, eval_gen, report, run_generation (+ sage_ext for the weighted trainer subclasses).
"""
