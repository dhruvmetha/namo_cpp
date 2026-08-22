---
type: experiment
status: live
created: 2026-08-22
commit: fd1743f
metric: complete-scene solve rate and simulator calls for learned versus random ranking on verified two-keyhole scenes
tags: [experiment, full-namo, multihop, composition, easy, random-baseline]
---
# Verified two-keyhole progression

**Read [docs/problem_and_approach.md](../../problem_and_approach.md) first.** The model remains one local region-opening ranker. This experiment invokes that same ranker successively inside Full NAMO; it does not train or define a multi-hop model.

## Hypothesis

_(user, from chat)_ Two ordinary easy one-push keyholes can be composed inside one fixed wall template so that opening K1 exposes a pushable K2 without reaching the XML goal, and opening K2 reaches the goal. On a frozen set of these scenes, the learned ranker should reach complete solutions in fewer simulator calls than uniform random ordering.

## Scope

Version 0 uses the car robot, exactly two distinct movable blockers, two canonical easy one-push donor episodes, and one fixed wall template per scene. Cross-template transforms, individually two-push keyholes, three-hop chains, semantic room decomposition, alternate-route handling, backward search, and new training data are deferred.

The complete-scene unit is `(realpath XML, XML goal)`. Each donor retains its local episode identity `(realpath donor XML, object_id, goal region)`, its per-episode difficulty, and its known valid action. Difficulty is recorded as the ordered tuple `(K1 difficulty, K2 difficulty)` and is never inherited from the donor room or collapsed into one scene label.

## Plan

### Phase 1: goal-centric scene validator

Extend `scripts/pipeline/compose_keyhole_modules.py` in place. Keep the current static component-path filter, then validate the known donor action chain against the complete-scene goal.

At the initial state require the XML goal to be in free space and unreachable, K1 to be reachable with at least one push edge, K2 to be unreachable, and the deterministic shortest component path to contain exactly the singleton boundaries `[K1]` then `[K2]`.

After the known K1 push require the push to succeed, the XML goal to remain unreachable, K2 to become reachable, and K2 to expose at least one reachable push edge. After the known K2 push require the push to succeed and the XML goal to become reachable.

Keep the pinned 100-point component counts and thresholds as diagnostics only. They must not reject a complete scene. This scene-composition rule does not change the canonical local region-opening success rule of 20 reachable target points.

Record exact donor episode keys, template, actions, component path, boundary objects, reachable objects and edges at every stage, goal reachability at every stage, diagnostic point counts, final blocker poses, geometry identity, and a specific rejection reason.

Add focused tests for a valid progression, initially reachable K2, K1 reaching the goal early, K1 failing to expose K2, K2 exposing no push edge, K2 push failure, final goal failure, and diagnostic point counts below threshold on an otherwise valid scene.

### Phase 2: revalidate and render the existing five

Run the new validator on the five accepted `set2/benchmark_5` easy-easy scenes without overwriting the old artifacts. Require all five to pass the goal-centric progression. Render all five as environment, component-map, and component-graph panels with robot, goal, K1, and K2 labels. Stop and diagnose before scaling if any scene fails.

### Phase 3: bounded 50-scene pilot

Collect 50 accepted easy-easy scenes from `set2/benchmark_5` with a maximum of 500 attempted donor pairs. Do not reuse one donor XML twice within a scene. Deduplicate complete geometry, prefer not to reuse an exact donor episode across frozen evaluation scenes, and join every artifact by realpath rather than basename.

Summarize attempted candidates, every static rejection, initially reachable K2, K1 failure, early goal reachability, failure to expose K2, missing K2 push edges, K2 failure, final goal failure, and accepted scenes. Render a thumbnail montage of every accepted scene plus full deterministic panels for the first ten. If 50 scenes are not found within 500 candidates, stop at the shortfall and diagnose instead of increasing the random search.

Before launching this or any larger run, apply the `scaled-run` and `compute-resources` procedures, smoke-test on the target machine, calibrate runtime, and commit the exact code and card.

### Phase 4: Full-NAMO smoke

Freeze the 50-scene manifest, then use `python/namo/solvability_runner.py`. First replay the recorded K1 and K2 donor actions as an oracle and require every final goal to be reachable.

Run ten scenes with HY5U ordinary Full NAMO, HY5U with `preserve_next_keyhole_access`, and one uniform-random seed under the same search protocol. Use best-first local search, `hmax=2`, the canonical local 20-of-100 goal test, budget 300 simulator calls per keyhole, raw-`q` ordering, discount off, no-op deduplication, and jam pruning. Do not use a state-value aggregate.

Treat next-keyhole preservation as a planner ablation, not as scene validity. The strong preservation gate requires all original K2 contact edges to remain available, whereas the composition contract requires at least one pushable K2 edge.

### Phase 5: 50-scene evaluation

Run HY5U control, HY5U with next-keyhole preservation, and uniform random with three seeds. Learned and random must use the same winning planner policy. Report complete-scene solve rate, solve rate at total simulator-call cutoffs `2, 5, 10, 30, 100, 300`, median calls on jointly solved scenes, learned-only and random-only solves, faster/tied/slower counts, K1 success, conditional K2 success, simulator calls by keyhole, and the iteration-trace failure taxonomy.

Simulator calls are the primary comparison. Do not compare wall-time unless every arm is rerun interleaved on identical hardware. Report random as one three-seed mean with sample standard deviation.

### Phase 6: geometric expansion

Proceed only if the oracle solves every pilot scene, the rendered layouts pass inspection, Full NAMO has no protocol failure, and learned ranking either beats random in simulator calls or exposes one specific fixable failure class.

Try `set2/benchmark_3`, `set2/benchmark_2`, and `set1/benchmark_5` in that order. Give each template a bounded 100-candidate compatibility pilot. Collect 25 accepted scenes only from productive templates and do not brute-force an unproductive template. Freeze a multi-template set only after geometry deduplication, exact donor provenance, training-geometry disjointness verification, and a deterministic visual audit.

The first multi-template evaluation set must contain at least 100 scenes across at least two templates. A target of 150 to 200 is allowed only if the bounded pilots produce them naturally.

### Phase 7: easy-easy verdict

Answer whether one learned local ranker, invoked at successive keyholes by Full NAMO, reaches verified easy-easy complete solutions in fewer simulator calls than uniform random ordering. Report overall and per-template results, control versus preservation-gated planning, K1 and K2 progression, the complete failure taxonomy, and representative renders.

Put detailed results in this card, the main table and figure in `docs/experiments/RESULTS.md`, and the frozen scene and evaluation artifacts in the relevant registry. Commit before every run.

### Phase 8: ordered difficulty progression

Scale one ordered tuple at a time: `easy+medium`, `medium+easy`, `medium+medium`, `medium+hard`, `hard+medium`, then `hard+hard`. Every tuple repeats the bounded composition pilot, goal-centric validation, visual inspection, frozen manifest, oracle replay, learned-versus-random evaluation, and per-keyhole reporting. Do not combine the tuples into an aggregate-only result.

Individually two-push keyholes, longer chains, cross-template stitching, alternate maze routes, backward search, new training data, horizon-conditioned models, semantic room decomposition, and full side-branch preservation remain deferred until this matrix is complete.

## Run

At commit `c31939e`, the goal-centric validator was replayed on ilab3 against the preserved five-scene control manifest and wrote fresh artifacts to `$NAMO_SCRATCH/eval/two_keyhole_progression_20260822/revalidated_five/`; it accepted 5/5 scenes with zero rejection and left the source artifacts untouched.

Every accepted trace had complete-scene goal reachability `[false, false, true]`, K2 had no reachable edge initially and 11–15 reachable edges after K1, and the five geometry identities were unique. The pinned component diagnostics were `[97–100, 0–1]` after K1 and `[97–100, 74–94]` after K2, confirming that the K2 component count is useful trace evidence but must not stand in for complete-scene goal reachability.

The five scenes use eight unique donor episodes across ten donor slots. This is retained as provenance for the old control set; the bounded 50-scene collector will prefer donor-disjoint accepted scenes before freezing its evaluation manifest.

The Phase 3 census found 14 easy donor episodes and only 62 ordered candidate pairs after excluding same-XML pairs and blocker centers closer than 0.30 m, so this fixed-template pool cannot plausibly supply 50 accepted scenes. The required bounded run will therefore exhaust those 62 pairs, report the shortfall, and stop without loosening the geometry or progression gates.

The exact production script passed a one-scene target-box smoke on rlab7 at commit `7404e85`: one scene was accepted after five candidates in three seconds and complete XML, manifest, and summary artifacts landed under `$NAMO_SCRATCH/eval/two_keyhole_progression_20260822/pilot_smoke_direct/`. Scaling the measured candidate rate to all 62 pairs gives roughly 40 seconds under similar rejection mix; the one-hour SLURM limit is deliberately pessimistic, and the full bounded census remains small enough to run directly as one pinned process on rlab7.

The bounded rlab7 census at commit `0adc0b7` exhausted all 62 eligible pairs in 22 seconds and accepted 10, so it triggered the planned shortfall gate rather than reaching 50. All 52 rejections were `wrong_hop_count`; all ten exact-two-hop candidates passed the complete goal-centric replay. The accepted set has ten unique geometry identities, nine unique donor episodes across twenty slots, six three-component graphs, four graphs with one side component, goal-reachability traces `[false, false, true]` on 10/10, and 11–15 reachable K2 edges after K1. Full panels and an all-scene montage are under `$NAMO_SCRATCH/eval/two_keyhole_progression_20260822/easy_easy_bounded50/renders/`.

This ten-scene artifact is a protocol-smoke cohort, not the planned 50-scene evaluation set. Sampling on `set2/benchmark_5` is stopped and its gates remain unchanged; the next Full NAMO step can validate the protocol on these ten, but a benchmark verdict requires additional productive templates to fill the geometry shortfall.

## Result

Pending.

## Verdict

Pending.
