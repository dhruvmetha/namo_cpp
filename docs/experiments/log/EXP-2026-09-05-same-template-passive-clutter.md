---
type: experiment
status: live
created: 2026-09-05
commit: b5510d96
metric: verified passive-clutter two-keyhole yield by ordered K1+K2 difficulty tuple
tags: [experiment, full-namo, multihop, composition, same-template, clutter, medium, hard]
---
# Same-template two-keyhole passive-clutter pilot

**Read [docs/problem_and_approach.md](../../problem_and_approach.md) first.** The model remains one local ranker invoked at each boundary by Full NAMO; this experiment changes the complete-scene context, not the ranker target or the definition of a local region-opening episode.

## Hypothesis

_(user, from chat)_ After establishing clean same-template two-keyhole scenes, include environments with interactions with other objects. Start with a simple controlled case before allowing coupled object motion.

## Plan

Keep the accepted 65 blocker-only scenes as the clean baseline and build a separate passive-clutter population in the productive `set2/benchmark_5` room.

For every donor pair, preserve the first donor's wall layout and robot start, transplant K1 and K2 exactly as in the clean composer, and retain one additional movable body from the first donor's host XML as `obstacle_2_movable`. Enumerate each eligible host object as a separate candidate instead of choosing one arbitrarily.

Require the original two-boundary topology and exact blocker order `[[K1], [K2]]`, the solved reachability progression `[false, false, true]`, K1/K2 mechanical independence, and no more than 2 mm or 1 degree of motion by the retained object over the witnessed chain.

Require a measured interaction rather than decoration: replay the accepted K1 action in the corresponding clutter-free scene and keep the candidate only when the retained object changes the reachable contact-edge set for K1 at the initial decision or K2 after K1. Record the clean and clutter edge sets in the manifest.

Target five accepted scenes for each ordered source-tier tuple: medium+medium, medium+hard, hard+medium, and hard+hard. These are donor provenance tiers; do not relabel the altered scene difficulty. Report natural shortfalls without relaxing gates, audit every accepted XML and replay, and render the accepted scenes for visual inspection.

## Run

Pending.

## Result

Pending.

## Verdict

Pending.
