---
type: experiment
status: live
created: 2026-08-23
commit: a8be9c1
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

Implementation commit `a8be9c1` adds the room-interface stitcher and two structural tests. The complete focused suite passes 17/17 on ilab3. No physics calibration or cluster-scale job has run yet.

## Result

Pending.

## Verdict

Pending the easy+easy calibration and the four ordered medium/hard pilots.
