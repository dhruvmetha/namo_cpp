---
type: experiment
status: live
created: 2026-09-05
commit: pending
metric: pending
tags: [experiment, full-namo, multihop, composition, same-template, interaction, medium, hard]
---
# Two-keyhole mixed-context approval pilot

**Read [docs/problem_and_approach.md](../../problem_and_approach.md) first.** The learned model remains one local ranker invoked successively at two boundaries by Full NAMO. Medium and hard remain properties of the two source local episodes, not labels inferred from the complete composed scene.

## Hypothesis

_(user, from chat)_ Build at least 50 accepted scenes for each ordered medium+medium, medium+hard, hard+medium, and hard+hard pair. A mix of non-interacting scenes and scenes with other-object interactions is acceptable. Interaction includes a designated keyhole blocker contacting or displacing another movable object while opening that single keyhole. First show ten accepted scenes from each ordered pair and continue only after visual approval.

## Plan

Reuse the verified same-template `set2/benchmark_5` blocker-only scenes as the non-interacting source. Extend `scripts/pipeline/compose_keyhole_modules.py` in place with targeted secondary-object placement along a witnessed K1 or K2 blocker motion, rather than retaining an arbitrary host object.

For an interacting candidate, first obtain a clean exact K1-to-K2-to-goal replay, place one sampled canonical-scale movable box ahead of K1 or K2 with positive initial clearance, and rerun the complete chain. Require the static boundary order `[[K1], [K2]]`, goal reachability `[false, false, true]`, explicit `movable_collisions` evidence naming the secondary object on the intended hop, secondary-object motion above 2 mm or 1 degree on that hop, no K1/K2 cross-motion above the existing tolerance, and a final reachable goal.

Build a forty-scene approval cohort with ten scenes per ordered source-tier pair. Prefer a mixture of interaction locations and non-interacting scenes, but do not impose an interaction quota or weaken validation to fill it. Deduplicate by full geometry, revalidate from a fresh process, render all forty scenes in four pair-specific montages, and stop for user approval before scaling to fifty per pair.

## Run

Pending.

## Result

Pending.

## Verdict

Pending.
