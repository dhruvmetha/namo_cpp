---
type: experiment
status: live
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

Build a 37-line manifest from the recorded alignment gap, with every non-target movable object skipped for the goal region. Run the unchanged `ref_fullexhaust.yaml` collector on Amarel `main-redhat`: smoke one target, calibrate it, pilot a few targets, then launch the remaining episodes. Build the fill with `build_rung2_h5.py`, append only the 37 requested `(xml, object)` trees to the existing H5, and require exactly one new root per target. Rebuild the GT JSON, canonical divisions, aggregates, and plots; the final registry must report 1,018/1,018 GT coverage and zero unknown episodes.

## Run

Commit `54b42ea` adds the focused-manifest builder, exact target-tree H5 merge, and Amarel collection/build launchers. The generated manifest contains 37 unique `(xml, object, goal)` targets and skips every other movable object in each target's goal region. Amarel `main-redhat` smoke job `59683573` completed one target in 11m03s: exactly `obstacle_3_movable` in `goal`, one root plus 50 depth-2 boards, 2,370 trials, and zero censored finish sweeps. Pilot job `59685510` completed targets 1–3. Production job `59687984` completed 28 targets; targets 9, 10, 12, 27, and 28 reached the 45-minute wall limit, so job `59703609` reruns only those five with the script's 60-minute limit.

## Result

Pending.

## Verdict

Pending.
