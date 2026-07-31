---
type: experiment
status: done
created: 2026-07-30
thread: rl_loop
robot: car
tags: [experiment, eval-set, exhaustive-gt, 2push, amarel]
---

# Complete exhaustive 2push ground truth

**Read [docs/problem_and_approach.md](../../problem_and_approach.md) first.** Each record is one region-opening episode `(xml, object, goal region)`; the simulator supplies exhaustive truth for ranking analysis.

## Hypotheses

The 37 unmatched pure-2push episodes can be exhaustively swept on Amarel with the existing reference collector by focusing each scene on its missing target object. Completing those roots will remove the `unknown` tier without changing the fixed hard <5% / medium 5–30% / easy ≥30% definition.

## Plan

Build a 37-line manifest from the recorded alignment gap, with every non-target movable object skipped for the goal region. Run the unchanged `ref_fullexhaust.yaml` collector on Amarel `main-redhat`: smoke one target, calibrate it, pilot a few targets, then launch the remaining episodes. Build the fill with `build_rung2_h5.py`, append only completed requested `(xml, object)` trees to the existing H5, and rebuild the GT JSON, canonical divisions, aggregates, and plots.

## Run

Commit `54b42ea` adds the focused-manifest builder, exact target-tree H5 merge, and Amarel collection/build launchers. The generated manifest contains 37 unique `(xml, object, goal)` targets and skips every other movable object in each target's goal region. Amarel `main-redhat` smoke job `59683573` completed one target in 11m03s: exactly `obstacle_3_movable` in `goal`, one root plus 50 depth-2 boards, 2,370 trials, and zero censored finish sweeps. Pilot job `59685510` completed targets 1–3. Production job `59687984` completed 28 targets; targets 9, 10, 12, 27, and 28 reached the 45-minute wall limit. Retry `59703609` landed on `halk` nodes; targets 9, 10, and 27 completed but their files appeared only after delayed filesystem visibility, while 12 and 28 timed out. Skylake retry `59719139` confirmed 35/37 artifacts and advanced targets 12 and 28 to 11,597 and 10,352 primitive trials before their 60-minute limits. The user stopped the final retry `59731403`; targets 12 and 28 remain explicitly unknown.

## Result

The completed 35 trees add 1,937 rows to the 66,456-row base H5, producing `testset_gt_plus35.h5` with 68,393 rows. The initial search view had 1,017 episodes with GT coverage 1,015/1,017 and fixed tiers easy 385, medium 488, hard 142, unknown 2. On 2026-07-31, four hard records whose exhaustive roots contain zero genuine setups and one shared pruned-search queue failure with a verified GT chain were removed from the canonical eval by user decision; the final registered view is 1,012 episodes with coverage 1,010/1,012 and tiers easy 385, medium 488, hard 137, unknown 2. The full 1,018-episode source and H5 are unchanged.

## Verdict

**Adopt the 35-root fill and stop collection.** The two unfinished episodes remain explicit unknowns and are excluded from fixed-tier charts.
