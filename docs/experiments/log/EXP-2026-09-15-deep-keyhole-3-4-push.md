---
status: done
type: experiment
---
# Deep keyhole problems: 3-push and 4-push region openings

User request 2026-09-15: 50-100 three-push and 50-100 four-push keyhole problems, multi-object blocking sets allowed, as environments to run HY5U on. Deadline same night.

## Plan

No generation. The August two-movable exhaustive depth-2 sweeps already answer the expensive half of the question. `exhaustive_hmax2.py` enumerates every reachable (object, edge, depth) at the root and, for each first push that neither opens nor jams, every second push on either movable. A scene whose cells are all `dead` or `blocked` therefore has no 1-push opener and no 2-push setup-finish pair, proved by enumeration rather than sampled. Those scenes need 3 or more pushes, or are impossible.

Counting the eight sweep directories under `$NAMO_SCRATCH/real_buildable_2mov` and deduping on `xml` gives 2,556 unique labelled scenes: 1,034 with an opener, 781 setup-only, and **741 dead at depth 2**. All 741 XMLs resolve on the CS filesystem. Median 120 depth-1 cells, 690 of them at 40 cells or more, spread over 16 generator families. Every one carries two movables, so a chain may switch objects mid-plan, which is the multi-object blocking set the request asks for.

The missing half is a plan and its length. Certify with `eval_policy.py --mode search`, which accepts `--hmax` 3 and 4 unclamped and writes `plan_len` and `sims` per episode. Four arms over all 741 scenes at a 3,000-sim budget: `u3`/`u4` uniform (random) ranker at hmax 3 and 4, `q3`/`q4` HY5U_s1 at the same depths.

Margin is **5 mm**, via `config/margin_5mm/`, because the sweeps predate the margin stamp and ran before the 2026-09-05 drop to 1 mm. Dhruv approved 5 mm for this set. Reading a 1 mm label off a 5 mm sweep is exactly what [[reference_inflation_margin_invalidates_labels]] warns about.

Commensurability check before trusting the prefilter: the sweep takes `region_goals["goal"]` off `get_region_snapshot(100, -1.0, False, 42, True)`, while the harness takes `region_goals[snap["goal_label"]]` through `eval_m3.sample_goal_points`. Measured on 12 scenes, the two produce **identical goal point sets, 12/12**. The sweep's dead-at-2 verdict and the search's success test are the same test.

Depth labels come from the hmax=3 arm failing, never from the hmax=4 arm succeeding. `solve_scene` at hmax=4 dives to the depth limit rather than preferring short plans: over the first 104 scenes it returned plan_len 4 on 46 units against plan_len 3 on 5, while the hmax=3 arm solved everything it touched. A 4-plan found at hmax=4 is therefore no evidence that no 3-plan exists.

Scenes the search opens in 2 pushes or fewer are dropped as `LEAK_shallow`, not relabelled. They ran at 3 of the first 104, which matches the roughly 2-in-478 `set_full_state` warmstart flip the sweep docstring already documents.

## Run

Fanned across 12 CS boxes. Two operational notes worth keeping. Workers spawn about 8 threads each against a `ulimit -u` of 2,000 on every box except westeros, so the first launch at 220 workers on rlab7 killed its parent; and each worker loads the HY5U checkpoint at ~873 MB RSS even on the uniform arms, because `eval_policy.py` builds `BeamPlanner(ckpt=...)` regardless of `--prior`. Slice width, not box capacity, is what caps concurrency: at 8 slices each box holds only ~93 of the 742 uniform units.

Outputs under `$NAMO_SCRATCH/deep_keyhole_20260915/`: `leaf/*.jsonl` per unit, `verdict.json` per scene, `deep3_key.json` and `deep4_key.json` as `eval_policy.py --key` manifests, and `compare.py` for the HY5U against random table split by certified depth.

## Reframing, and it is the main outcome

Certifying "this problem needs exactly 4 pushes" requires proving no 3-push plan exists, a ~2M-chain enumeration per scene that we cannot run. So the depth label would always be an artifact of the search budget. Dhruv's call, 2026-09-15: define the population by the property that IS provable, "no solution within 2 pushes", and report solve rate against simulator calls at hmax=4. The 3-push / 4-push split no longer DEFINES the population. It survives as a secondary cut of the results, labelled by the shortest plan any of the eight arms found, and the depth labels carry their evidence grade so nobody reads them as exact push counts.

That reframing turns the result into an out-of-distribution claim, and both of its premises were checked rather than assumed.

**Training tops out at 2 pushes.** `build_hybrid_h5.py:36` admits only `chain_depth == 2` family rows and `:70` writes old-corpus rows as 1 or 2, so the column only ever holds {1, 2}. The collection config pins `region_max_chain_depth: 2` (`docs/pipeline/full_namo_collection.md:105`). Value targets are precomputed supervised labels with positives in {0.9, 1.0} and no bootstrap, no `max` over a next state, no horizon-Q recurrence (`horizon_q_build_journal.md:416`). Nothing deeper than 2 can reach the targets, even indirectly. HY5U also takes no horizon input, so it cannot know it is being asked a deeper question.

**The rooms are disjoint.** 197,464 unique training rooms hashed by geometry against the 741; zero shared scenes AND zero shared floorplans. Training draws on 432 distinct wall layouts, the evaluation set has 741, and the two sets do not intersect at all, so this is not the same rooms with obstacles moved. Report `geom_disjoint_cspools.json`. Coverage caveat: the H5 names 198,267 unique room paths and the CS pools yielded 197,464 hashes, within 0.4%; set equality is unprovable here because mapping H5 paths to CS files would need name matching, which failure mode #4 forbids.

⛔ The first run of that check was a VACUOUS PASS and nearly shipped. Training paths are Amarel absolutes that do not resolve on CS, so all 198,267 rooms landed in `n_unparseable_train`, the comparison ran against an empty set, and it printed `clean: true`. Read `n_train_unique_scenes` before believing any disjointness verdict. See [[feedback_verify_the_guard_fires]].

## Result

Complete 2026-09-16 01:15. Five random seeds (7000-11000) and three HY5U seeds (s1-s3), best-first at hmax=4, 900-call cap, paired on the 724 scenes every arm ran. Budget moved 3000 -> 900 mid-run on time pressure [USER]; `solve@k` for k <= 900 is identical either way, so the arms that ran at 3000 stay comparable over the whole plotted range.

Whole population, n=724: random **35.2%** [31.9-38.0] solved at 900 calls against HY5U **50.4%** [49.6-51.0], median calls **145-282** against **25-27**. The seed bands never overlap, at any budget.

Split by the shortest plan any of the eight arms found:

| | n | @3 | @5 | @10 | @30 | @100 | @300 | @900 | median calls |
|---|---|---|---|---|---|---|---|---|---|
| 3-push, random | 244 | 0.4 | 1.2 | 5.8 | 14.0 | 30.0 | 50.6 | 70.0 | 112-164 |
| 3-push, HY5U | 244 | 12.3 | 21.8 | 37.0 | 56.4 | 72.8 | 81.1 | 88.9 | 13-15 |
| 4-push, random | 239 | 0.0 | 0.0 | 0.4 | 3.0 | 8.1 | 18.3 | 34.0 | 279-662 |
| 4-push, HY5U | 239 | 0.0 | 4.3 | 11.9 | 25.1 | 41.3 | 50.6 | 63.4 | 47-62 |

**The headline: the advantage WIDENS with depth.** The 900-call ratio is 1.27x on 3-push and **1.86x** on 4-push. Random's median cost roughly triples as the required chain lengthens, 140 calls to 400; HY5U's goes 14 to 55. A ranker trained only on 1-push and 2-push data degrades more gracefully than random ordering as solutions get deeper, in rooms it never saw. That is the generalization claim, measured.

Three numbers come from the geometry rather than the model and serve as internal checks: both arms at 0.0% @1 everywhere, HY5U at 0.0% @3 on the 4-push set (four pushes cannot be found in three calls), picking up at @5, one call off the floor.

Depth labels over the 725 non-leak scenes: **244** with a verified 3-push plan, **2** depth-4 with a depth-3 exhausted certificate, **237** depth-4 on eight-arm evidence without proof, **242** unsolved by every arm within 900 calls. The thirds are near-even, which the early thin-coverage data hid completely.

The 242 unsolved stay IN every number, counted as failures for both arms. Dropping them rescales both curves by nearly the same factor: the ratio moves 1.38x to 1.41x while the headline jumps to 79.5% against 56.2% on n=438. It flatters us and teaches nothing, so it belongs in the text as an aside at most [USER agreed 2026-09-16]. They are budget-limited, not unsolvable.

16 scenes excluded as `set_full_state` warmstart flips, about 2% of 741, matching the 2-in-478 the sweep documents. Detection must use EVERY arm; one arm found only 12, since each extra arm is another chance to trip the flip.

Artifacts: `$NAMO_SCRATCH/deep_keyhole_20260915/` and the shared bundle `/common/users/shared/robot_learning/dm1487/namo/ranking/npush` (tdn39 read, 741 scenes, six manifests, the per-scene depth-2 proof, both figures).

## Harness notes worth keeping

Workers spawn ~8 threads each against a per-user `ulimit -u` of 2000 on every CS box but westeros, and every worker loads the checkpoint at ~873 MB even on `--prior uniform`, because `eval_policy.py` builds `BeamPlanner` before it branches on the prior.

A shared work queue needs an atomic claim AND a reaper. `mkdir` is the claim; the release trap is not enough, because units died mid-run without releasing (logs show partial progress) and no cause was found: every box had hundreds of GB free and zero OOM kills. 405 claims were orphaned at one point. A reaper that asks each box what is genuinely running and releases everything else converges regardless of cause.

A single `xargs` pass exits at the end of the list, so anything freed afterwards has nobody to run it. Loop until a whole pass achieves nothing.

`pkill -f <pattern>` over ssh self-matches, since the remote command line contains the pattern and kills its own shell first. Use `[p]attern`.
