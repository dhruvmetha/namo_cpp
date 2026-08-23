---
type: experiment
status: done
created: 2026-08-23
commit: c1df3b1
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

The ordered medium/hard pilot ran four commands concurrently on ilab3, each targeting five acceptances with a thirty-pair cap. Medium+medium accepted 5/10, medium+hard 5/13, hard+medium 5/8, and hard+hard 5/15. Every tuple used ten unique donor episodes across ten accepted slots with zero reuse. Pair-level failures were dominated by `k1_did_not_expose_k2`, `final_goal_unreachable`, and `no_component_path`; hard+hard also had one `k1_push_failed`, while medium+medium and medium+hard each had one `k2_push_failed`. Two rendered representatives per tuple passed visual inspection: the intact donor rooms, connector, blocker placement, wavefront regions, and C++ graph agree.

Amarel scale preflight uses the dedicated clone `/cache/home/dm1487/projects/namo/namo_slot_stitch` at commit `12c2959`, with the existing Amarel-native binding linked read-only. Smoke job `60768327` failed before simulation because its relative `source env.amarel.sh` did not resolve inside the SLURM wrapper. The corrected absolute-path smoke job `60768328` completed on `main` in 36 seconds, used 88 MB RSS, and accepted 1/1 medium+medium scene with the expected 50-exit/51-entry eligible donor pools.

The scaled candidate build targets 50 verified scenes separately for medium+medium, medium+hard, hard+medium, and hard+hard, with a maximum of 300 pair attempts per tuple. Run four independent one-CPU `main` jobs, pin all numerical thread pools to one, request 4 GB and 30 minutes per job, and monitor accepted manifest counts. This 200-scene output is a candidate population; donor reuse, template balance, geometry uniqueness, oracle traces, and renders must be audited before anything is frozen for Full NAMO evaluation.

Amarel jobs `60768329..60768332` completed without scheduler or simulator failure. Medium+medium reached 50/176 in 10m46s with 43 unique donor episodes across 100 slots; hard+medium reached 50/266 in 3m06s with 51 unique donors; hard+hard reached 50/179 in 2m14s with 22 unique donors. Their reused donor-slot counts are 57, 49, and 78 respectively. Medium+hard exhausted its 300-pair cap at 16 accepted in 17m21s, using 32 distinct donor episodes with zero reuse; its rejection counts were 93 final-goal failures, 79 K1 exposure failures, 79 K2 push failures, 24 no-path scenes, and 9 K1 push failures.

The medium+hard runtime tail came from exhaustively proving incompatible pairs by enumerating recorded donor actions. This proof is unnecessary for sampled generation: accepted scenes need an exact witnessed chain, while rejected pairs do not need exhaustive ground truth. Across all 166 scaled accepted scenes, replay attempts ranged from 2 to 16; medium+hard successes ranged from 2 to 4. Commit `5016b8f` adds an optional per-pair replay cap, leaves revalidation unlimited by default, records `replay_attempt_cap` as a sampling rejection, and adds a focused test. A 50-attempt cap retains more than three times the largest successful trace observed so far while bounding pathological rejects. Run a fresh target-box smoke before a capped medium+hard supplement.

The capped target-box smoke `60768972` completed on Amarel `main` in 33 seconds, used 139 MB RSS, and accepted 1/1 medium+hard scene with one replay attempt and zero rejection. Launch one supplemental medium+hard build with a 50-action per-pair cap, a 50-scene target, and a 1,400-pair ceiling so it may traverse the complete eligible pair pool if necessary. Keep it separate from the uncapped 16-scene output; deduplicate and audit across both only after completion.

The cap-50 supplement `60769078` reached 16 accepted scenes but then spent several minutes inside one capped rejection, showing that fifty physical simulations remains too large for this sampler's tail. Every medium+hard success observed so far used at most four replay attempts, including the cap-50 target-box smoke. Launch a separate cap-10 arm with the same 50-scene target and 1,400-pair ceiling; ten retains a 2.5× margin over the observed successful maximum. Keep the cap-50 job running independently so no accepted artifact is discarded. Compare only verified accepted rows after both jobs stop.

Both supplemental arms reached the same 17-scene frontier before entering another expensive action. The common cause is candidate ordering rather than an acceptance gate: after each success, the generator preferentially selects pairs in which both donor episodes have never appeared in an accepted slot, so the medium+hard tail is forced through the remaining unproductive donor class before it may reuse productive donors in new geometries. Commit `056df88` adds an explicit `--allow-donor-reuse-early` sampling option. It preserves same-XML exclusion, geometry deduplication, exact static topology, and exact `[false, false, true]` replay validation; only pair order changes. Smoke this option on Amarel with the ten-replay cap, then run a separate supplemental arm and report donor reuse rather than concealing it.

Amarel smoke job `60769338` passed at commit `e70b2c6`: it accepted 1/1 medium+hard pair with the ten-replay cap and early donor reuse enabled, with no construction, topology, or replay rejection. Launch a separate 50-scene candidate build with seed 1402, a 1,400-pair ceiling, and the same ten-replay cap. Treat it as a supplement to the original sixteen exact scenes, deduplicate by full geometry identity, and audit provenance and exact traces before selecting a 50-scene medium+hard candidate population.

All three supplemental medium+hard jobs completed normally. Cap-50 job `60769078` reached 50/449 in 24m46s, cap-10 job `60769332` reached 50/479 in 13m34s, and early-reuse cap-10 job `60769340` reached 50/116 in 2m46s. The early-reuse arm is the selected medium+hard population because it changes no validity condition, reaches the same target with 4.1× fewer pair attempts than the non-early-reuse cap-10 arm, and still spans 41 distinct donor episodes and eight ordered source-template pairs. Its 50 accepted scenes contain 50 distinct full geometries, zero same-XML pairs, zero unsolved replays, zero incorrect blocker orders, and zero reachability traces other than `[false, false, true]`. Four representative renders spanning different source-template pairs passed visual inspection.

## Result

The selected construction population contains 200 verified scenes: 50 medium+medium, 50 medium+hard, 50 hard+medium, and 50 hard+hard. The four tuples contain respectively 43, 41, 51, and 22 unique donor episodes and span 8, 8, 9, and 8 ordered source-template pairs. Across the complete population there are 200 distinct full geometry identities, 65 distinct donor episodes, zero cross-tuple geometry duplicates, zero same-XML donor pairs, zero unsolved oracle replays, zero incorrect `K1` then `K2` blocker orders, and zero reachability traces other than `[false, false, true]`.

The medium+medium, hard+medium, and hard+hard manifests live under `$NAMO_SCRATCH/eval/slot_first_keyhole_20260823/scale50_v1/`; the selected medium+hard manifest lives under `$NAMO_SCRATCH/eval/slot_first_keyhole_20260823/medium_hard_reuse_cap10_v4/`. These remain candidate construction artifacts rather than a frozen evaluation registry entry.

## Verdict

The slot-first stitching plan is productive for all four priority ordered tuples. Early donor reuse should be the default sampling order for future medium+hard expansion, with donor reuse reported explicitly and full-geometry deduplication plus exact two-hop replay retained as hard gates. The next separate experiment is a Full NAMO ranker-versus-random evaluation on a frozen selection; no training or ranker change is implied by this construction result.
