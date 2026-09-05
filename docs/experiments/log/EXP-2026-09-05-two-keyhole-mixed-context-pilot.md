---
type: experiment
status: live
created: 2026-09-05
commit: c42d8f1e
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

Commit `c42d8f1e` adds sampled same-template contact augmentation, records per-push movable-collision evidence in the replay trace, and labels the third movable as context in the renderer. The focused composer suite passes 27/27.

The exact production script passed a one-scene smoke on rlab7 as SLURM job `270775_0`. The sampler accepted its first medium+medium candidate, completed the generation command in 6 seconds and the SLURM task in 11 seconds, and wrote the XML, manifest, XML list, and summary under `$NAMO_SCRATCH/eval/keyhole_mixed_context_20260905/smoke_mm/`.

The accepted smoke interaction occurs during K1. The simulator reports `obstacle_2_movable` in `movable_collisions`; the context object translates 64.4 mm while K2 stays fixed to numerical precision; the two donor actions solve with complete-goal reachability `[false, false, true]`.

The required render audit rejected that smoke and the first four-cell contact run before cohort selection: the sampled context box crossed the outer wall even though static topology and replay succeeded. None of those contact outputs count. Commit `c81d3d4b` adds an oriented-box geometry gate that requires every padded context corner inside the room and rejects overlap with any wall or either blocker; the focused suite remains 28/28, and the rejected smoke now deterministically reports `contact_outside_room`.

The corrected production smoke ran on rlab7 as job `270790_0` and completed in 10 seconds. It accepted one medium+medium K2 interaction after five placements, rejecting two out-of-room placements and two incorrect blocker-order scenes. The accepted object is inside the outer walls with physical clearance, the simulator names it in K2's movable-collision trace, it moves only during K2, and the complete-goal trace is `[false, false, true]`; the room, region-map, and graph render passed visual inspection.

The first corrected four-cell run exposed an unbounded rejection tail in MM: the new augmentation path did not forward the existing per-pair replay cap, so one rejected placement spent minutes enumerating donor-action combinations. Task `270793_0` was cancelled after 5:07 with two provisional XMLs and no completed manifest; none count. Commit `7c6c3e18` forwards the existing ten-simulation cap. This does not approximate an acceptance because every accepted row still contains a complete exact replay; it only stops exhaustive work on sampled rejects.

The bounded four-cell contact search ran on rlab7 as SLURM array `270810` under tag `approval40_v3` and completed all cells in 11 seconds to 1:46. MM accepted 3 of 167 sampled placements, MH 3 of 23, HM 3 of 169, and HH 3 of 57. Every accepted contact scene passed the geometry gate, exact blocker-order check, complete two-opening replay, intended-hop simulator collision check, context-motion threshold, K1/K2 independence checks, and final goal reachability.

All twelve accepted interactions occur within K2; the current targeted placements produced no surviving K1 contacts after the full geometry and topology gates. This is acceptable for the approval cohort because the requested mix does not impose a K1/K2 interaction quota, but it remains an explicit coverage limitation rather than a hidden balance claim.

The deterministic cohort selector produced 40 unique full geometries under `$NAMO_SCRATCH/eval/keyhole_mixed_context_20260905/approval40_v3/selected/`: exactly ten scenes in each of MM, MH, HM, and HH, with seven clean scenes and three K2-interaction scenes per cell. Fresh replay validation and rendering remain pending before presentation.

## Result

Pending.

## Verdict

Pending.
