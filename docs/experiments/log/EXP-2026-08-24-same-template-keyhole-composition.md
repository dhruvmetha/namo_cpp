---
type: experiment
status: live
created: 2026-08-24
commit: 4bd9275
metric: verified complete-scene yield by ordered K1+K2 difficulty tuple under one-host same-template composition
tags: [experiment, full-namo, multihop, composition, same-template, medium, hard]
---
# Same-template two-keyhole composition

**Read [docs/problem_and_approach.md](../../problem_and_approach.md) first.** The learned model remains one local raw-Q ranker invoked successively by Full NAMO. This experiment composes complete two-keyhole scenes; it does not redefine a local 2-push episode as a multi-region problem.

## Hypothesis

_(user, from chat)_ Fixed environment templates should let us combine two existing keyhole episodes inside one natural room when their blocker placements do not interfere. The output should retain one template's original walls, not transplant or weld complete donor rooms.

## Correction to the previous pilot

The completed room-stitch pilot copied and joined two complete donor wall layouts. It verified the sequential replay machinery but did not implement the intended same-template transplant. Keep those artifacts only as engineering stress tests. Do not freeze them as the two-keyhole benchmark.

## Plan

Add an explicit `same_template` composition mode. Require both donors to have the same named template and exact wall signature. Keep the first donor's one unchanged wall layout and robot start, remove unrelated movable objects, insert only the two selected blocker bodies at their original template coordinates, and use the second donor's goal.

Treat the unit as `(realpath XML, object_id, goal region)`. Carry difficulty per donor episode. Reject same-XML donor pairs, close blocker placements, incorrect static blocker order, K2 reachability before K1, failure of K1 to expose K2, failure of K2 to reach the final goal, duplicate full geometries, and any replay in which one push mechanically moves the other blocker.

Run tiny one-scene pilots for medium+medium, medium+hard, hard+medium, and hard+hard on productive templates. Render every accepted scene before increasing any target. Scale only after the user accepts the room geometry.

## Run

Commit `4bd9275` adds the explicit `same_template` mode and focused structural and non-interference tests. The focused suite passes 22/22. A census of the canonical one-push donors confirms that every named `set{1,2}/benchmark_{1..5}` template with donors has exactly one wall signature across all of its episodes, so donor blockers share one coordinate system without any wall transformation.

The first local calibration used only `set2/benchmark_5`, targeted one scene per ordered tuple, capped replay sampling at ten actions per pair, and attempted at most 100 pairs. Medium+medium accepted 1/4, medium+hard 1/9, hard+medium 1/7, and hard+hard 1/4. The four acceptances have four distinct full geometries and one shared wall signature. Every XML contains exactly one original `walls` body and no connector, transformed module wall, or added global boundary.

All four accepted replays are solved with blocker order `[[K1], [K2]]` and goal reachability `[false, false, true]`. K1 moved K2 by at most numerical noise, `5.6e-17 m`, and K2 moved the already-opened K1 by at most `0.168 mm`, below the `2 mm` independence gate. The rendered physical rooms, wavefront regions, and region graphs agree. Stop here for user visual approval before any scale build.

The user approved the one-host pilot. A one-scene census then tested every template containing both medium and hard donors, with a 100-pair ceiling per ordered tuple and the same ten-replay cap. Only `set2/benchmark_5` produced accepted scenes for all four tuples. The other eight templates produced zero exact chains in their bounded cells; their failures were dominated by `wrong_hop_count`, with smaller replay and progression tails. Do not force template diversity by weakening topology or adding back unrelated movable objects.

The approved target-box smoke and pilot calibrate the productive room at one acceptance per 4–9 pair attempts, with all four one-scene jobs completing together in seven seconds on ilab3. Build 25 candidates per ordered tuple in four parallel single-threaded processes, cap each pair at ten replay simulations, allow up to 1,000 pair attempts, and retain the existing donor-diversity-first ordering. Audit cross-tuple geometry duplicates, provenance, exact replay traces, and mechanical independence before rendering representatives.

## Result

Pending.

## Verdict

Pending.
