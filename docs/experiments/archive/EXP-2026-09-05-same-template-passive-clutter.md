---
type: experiment
status: done
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

Commit `b5510d96` adds the explicit `same_template_clutter` mode and revalidation support. The focused composer suite passes 25/25. Every candidate keeps one original wall body, K1 and K2, and one host movable renamed to `obstacle_2_movable`; the compositor enumerates every eligible non-K1 host object rather than selecting one by luck.

The first medium+medium calibration attempted 40 candidate variants and accepted none: 37 failed static topology, two preserved the topology but did not change either ranker decision's reachable-edge set, and one exhausted the ten-simulation replay cap. Because the complete four-cell census contains only 998 finite variants, run all cells to exhaustion on ilab3 without changing the 30 cm K1/K2 separation or any acceptance gate. The four single-threaded processes ran concurrently; wall time is not used as a scientific metric.

The complete artifacts live under `$NAMO_SCRATCH/eval/same_template_keyhole_20260905/passive_clutter_v1/`, with one directory per ordered tuple and rendered accepted scenes under `renders/`.

## Result

| K1 source tier | K2 source tier | variants attempted | exact static two-hop | accepted interaction scenes | accepted / attempted |
|---|---|---:|---:|---:|---:|
| medium | medium | 363 | 30 | 1 | 0.28% |
| medium | hard | 277 | 17 | 2 | 0.72% |
| hard | medium | 233 | 22 | 0 | 0.00% |
| hard | hard | 125 | 7 | 0 | 0.00% |
| **all** | **all** | **998** | **76** | **3** | **0.30%** |

The dominant rejection is structural: 745/998 candidates have the wrong hop count, 98 have the wrong blocker order, and 79 put the goal outside free space. Of the 76 exact static two-hop candidates, 24 exhaust the ten-action replay cap, 32 preserve both decisions' edge sets and are therefore decorative, four mechanically move the clutter, 13 fail the exact K1-to-K2-to-goal progression, and three pass every gate.

All three accepted XMLs are distinct geometries with one wall body and exactly three movable bodies. They use four unique donor episodes: one shared medium K1 donor and three K2 donors. Every replay has blocker order `[[K1], [K2]]` and goal reachability `[false, false, true]`. The passive body moves by at most `5.8e-17 m` and `0 degrees`; K1 moves K2 by at most numerical noise, and K2 moves the opened K1 by at most `0.016 mm`.

All three interactions occur at the initial K1 decision. Relative to the clutter-free counterpart, the passive body reduces K1's reachable contact-edge set from 25 edges to 20 while leaving the witnessed K1 action valid. No accepted candidate changes the K2 decision after K1. The three accepted scenes all retain `obstacle_0_movable` from the same host episode, `run_0161/env_0161_pair_000.xml`; the two medium+hard acceptances differ only in K2.

A fresh manifest revalidation accepted all 3/3 scenes and reproduced the topology, exact replay, mechanical-independence, passive-motion, and clean-counterfactual edge-effect gates.

## Verdict

**Feasible but reject this recipe as a balanced interaction generator.** The three survivors are valid controlled passive-interaction scenes and should be kept as diagnostics beside the clean 65-scene baseline, but retaining one object only from K1's original host XML cannot supply hard-K1 interactions or five scenes per ordered tuple. Do not call this a medium/hard interaction benchmark and do not mix its 0.30% yield into the clean population.

The measured bottleneck is not object motion: only four of 998 variants fail the passive-motion gate. It is topology. A native host object usually creates or destroys a region boundary instead of merely occluding some contact approaches. The next interaction generator must select placement by the intended local effect—preserve the two-boundary graph first, then reduce a chosen K1 or K2 decision's reachable-edge set—rather than inherit an arbitrary original host object.
