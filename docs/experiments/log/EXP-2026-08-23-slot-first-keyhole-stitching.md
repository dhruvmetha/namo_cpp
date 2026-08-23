---
type: experiment
status: live
created: 2026-08-23
commit: ec0ad58
metric: verified complete-scene yield by ordered K1+K2 difficulty tuple under room-interface stitching
tags: [experiment, full-namo, multihop, composition, room-stitch, medium, hard]
---
# Slot-first two-keyhole stitching

**Read [docs/problem_and_approach.md](../../problem_and_approach.md) first.** The learned model remains one local raw-Q ranker invoked successively by Full NAMO. This experiment builds complete two-keyhole scenes; it does not introduce a horizon-conditioned model or a state-value aggregate.

## Hypothesis

_(user, from chat)_ Existing canonical one-push episodes can be treated as directed room modules and joined through non-interfering interfaces. If each module keeps its local walls, blocker geometry, robot-side region, and goal-side region, then opening K1 should expose K2 while retaining K2's original local solution, allowing productive medium+medium, medium+hard, hard+medium, and hard+hard two-hop populations.

## Scope

Use exactly two canonical one-push donor episodes and two intended movable blockers. Strip unrelated movable objects, preserve the complete fixed-wall context of each donor, cut one controlled portal from K1's goal-side region and one from K2's robot-side region, rigidly align those portals, and join them with a short corridor. Use K1's robot and K2's XML goal. Other-object interactions, individually two-push keyholes, backward search, alternate routes, and training changes remain out of scope.

Difficulty is the ordered per-episode tuple `(K1 tier, K2 tier)`. Source templates may differ. Donor identity remains `(realpath XML, object_id, goal region)`; room XML alone is not an episode.

## Plan

### 1. Structural implementation gate

Add `room_stitch` as a separate mode in `scripts/pipeline/compose_keyhole_modules.py` so the previous `fixed_template` behavior stays reproducible. Infer each directed module interface from its stripped one-donor reachability raster, cut an exact boundary gap, rotate and translate each complete module rigidly, add the corridor walls, and record source templates, interfaces, and transforms in the manifest.

Require focused tests for exact portal geometry, preservation of two wall modules, one robot, two intended blockers, K2's transformed final goal, and cross-template provenance. Run the existing goal-centric progression tests unchanged.

### 2. Easy+easy calibration only

On the current CS checkout, attempt a tiny easy+easy sample across all registered templates. This is a construction calibration, not a benchmark population. Require the static component path to be exactly `[K1, K2]`, the XML goal to be unreachable initially, K2 to be unreachable initially, a known donor K1 action to expose a pushable K2 without reaching the goal, and a known donor K2 action to reach the final goal.

Render every accepted calibration scene before scaling. Stop and diagnose the first dominant exact rejection class if the stitcher cannot produce verified examples.

### 3. Ordered medium/hard pilots

Run bounded pilots in the priority order medium+medium, medium+hard, hard+medium, and hard+hard. Use the same portal geometry and complete goal-centric validator for every tuple. Do not aggregate tuples together and do not loosen gates merely to raise yield.

For each tuple report attempted pairs, interface-construction rejections, static topology rejections, progression-stage rejections, accepted scenes, unique donor episodes, source-template pairs, and geometry identities. Render a representative set from every productive tuple.

### 4. Scale decision

Only after the four bounded pilots establish productive, visually sound tuples, apply the scaled-run procedure and use Amarel's CPU throughput for a larger build. Freeze no evaluation manifest until geometry deduplication, donor provenance, goal-centric oracle replay, and visual inspection pass.

No learned-versus-random Full NAMO evaluation is part of this construction pilot. If a later frozen population is large enough, evaluate the unchanged raw-Q ranker and uniform random under the same Full NAMO search policy and report complete-scene simulator calls separately for medium+medium, medium+hard, hard+medium, and hard+hard.

## Run

Implementation commit `a8be9c1` added the room-interface stitcher and two structural tests. The complete focused suite passed 17/17 on ilab3.

The first easy+easy calibration at `b35b5bd` attempted ten donor pairs and accepted zero because every assembled scene failed during `RLEnvironment` initialization with `vector::_M_fill_insert`. Code inspection traced this to the backend's exact workspace convention: it interprets geoms named `wall_1`, `wall_2`, `wall_3`, and `wall_4` as unrotated left, right, bottom, and top boundaries. Rotating donor rooms without renaming those geoms inverted the inferred wavefront dimensions.

The same diagnostic exposed donors whose robot and XML-goal regions merge after unrelated movables are stripped; those are not standalone one-blocker modules and cannot supply a directed stitch under this experiment's no-other-movable scope. Commit `2300283` prefixes both modules' wall names, adds one unrotated global `wall_1..4` enclosure around the assembly, and rejects any isolated donor whose robot and goal labels coincide. The focused suite remains 17/17. A corrected physics calibration has not run yet.

The corrected ten-pair calibration initialized cleanly and rejected 10/10 as `module_not_keyhole_after_stripping`. A complete donor-role census took 51.2 seconds on ilab3. Easy has 26/237 portable K1-exit donors and 25/237 portable K2-entry donors; medium has 50/160 and 51/160; hard has 27/76 and 27/76. The productive portable pools span `set1/benchmark_3`, `set2/benchmark_3`, and `set2/benchmark_5`, with a few easy donors from `set1/benchmark_1` and one medium-entry donor from `set1/benchmark_5`.

Commit `ec0ad58` changes room-stitch sampling to classify each donor once for its directed slot, record exact pool rejection counts and template composition, and form pairs only from eligible pools. Original source-coordinate separation remains disabled only for room stitching because complete modules are relocated rigidly; same-XML donor pairs remain forbidden.

The eligible-pool easy+easy calibration accepted its target 3 scenes in 8 pair attempts. The three accepted geometries use six distinct donor episodes with zero slot reuse. The five pair-level rejections were three `k1_did_not_expose_k2`, one `final_goal_unreachable`, and one `no_component_path`. Every accepted oracle trace has complete-goal reachability `[false, false, true]`; K2 is unreachable initially, exposes 3–11 reachable edges after K1, and its recorded donor action reaches the final XML goal. Two accepted scenes stitch same-template modules and one stitches `set2/benchmark_5` to `set1/benchmark_3`.

The first render pass verified the wavefront regions and C++ region graphs but exposed that the environment panel's legacy parser ignored prefixed module-wall bodies. The renderer is updated to discover every wall body, apply wall yaw, and compute rotated bounds before the required visual audit.

## Result

Pending.

## Verdict

Pending the easy+easy calibration and the four ordered medium/hard pilots.
