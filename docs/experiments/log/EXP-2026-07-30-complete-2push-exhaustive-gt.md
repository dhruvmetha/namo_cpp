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

Pending.

## Result

Pending.

## Verdict

Pending.
