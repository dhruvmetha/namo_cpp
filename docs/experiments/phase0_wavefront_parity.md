---
status: frozen
tags: [phase0, wavefront, parity, mujoco-removal]
updated: 2026-08-28
---

# Phase 0: can the policy loop lose MuJoCo, and do the two wavefronts agree?

Frozen at the floor of what the existing bindings can answer. One decision is needed before any further work, and it is Dhruv's.

The commission was to remove MuJoCo from the greedy-policy decision loop, shadow-verified before anything flips. Phase 0 was the offline comparison, with the acceptance criterion set later in the week: the MuJoCo stack and any pure stack must do the same thing, exactly, with the C++ wavefront as the reference because every trial, label and shipped result came from it.

## Established

**Training and deployment share one renderer, so there is no train-deploy skew in the ranker's inputs.** `build_scorer_dataset.py` does no rendering; it joins masks from the diffusion H5 with the per-episode label key. Those masks came from `batch_collection.py`, which calls `generate_all_masks_highres`, the same function the deploy path calls. That function's only unified branch is `WavefrontSnapshotExporter.from_geometry`, pure geometry. The scenario that would have outranked the whole commission does not exist.

**The ranker's input path never needed MuJoCo.** `_episode_data` reads exactly three things from the environment: `get_observation()` for poses, `get_object_info()` for static geometry, and `get_reachable_objects()` for reachability. The first two are plain data. `sage_learning/visualizer.py` imports numpy, cv2, matplotlib and xml.etree, with zero mujoco and zero namo_rl. `image_utils.py` is cv2 and numpy.

**A live silent-degradation path existed in the scorer and is now fixed.** `live_scorer.render_ctx` wrapped the render in `redirect_stdout`, tested the captured text for "falling back to legacy BFS", and set `self.last_fell_back`. The visualizer's own message says that fallback "may use wrong robot size". The print went into a buffer that was discarded, and `grep -rn last_fell_back` across all three repositories found exactly one other hit, an unrelated assignment. Nobody read the flag. Fixed in namo_cpp `1628d1f`, which now raises instead. Measured rate before the fix: 0 fallbacks across 14 renderable captured scenes.

**Both wavefronts read the same constants from the same file.** The runtime hardcodes `namo_config_complete_skill15_car_1x.yaml`, which declares `robot_size: [0.035, 0.035]`, and both sides add the 0.005 tier1 margin, so both inflate 4.0 cm. The 0.052 in `namo_config_car.yaml` belongs to a file the runtime never loads. `robot_inflation: 0.0083` is annotated in the config itself as "Dead config, read by no code path".

**The gate fails at exactly one robot pose on one scene.** The stacks were first reported to disagree 2-versus-3 on `1push/2hop/env1`. That reading is real but incomplete: the answer depends on where the robot is, and neither report stated the pose.

```
        robot pose                    C++ regions   Python
         XML spawn   3 [goal, region_3, robot]        3
       (0.25, 0.10)   2 [goal, robot]                  3   <- the only mismatch
       (0.10, 0.10)   2 [goal, robot]                  2
       (0.40, 0.10)   2 [goal, robot]                  2
       (0.25, 0.45)   2 [region_3, robot_goal]         2
       (0.40, 0.70)   2 [region_3, robot_goal]         2
```

Both partitions move with the robot pose. Five of six poses agree. The one that does not is (0.25, 0.10), which is the pose the repo's own integration tests standardise on, chosen because the XML spawn leaves the car wedged with nothing reachable.

At that pose `obstacle_1_movable` sits at (0.25, 0.202) with half-extent 0.075, putting its lower edge at y = 0.127, so the robot at y = 0.10 is 2.7 cm below it and inside the 4.0 cm inflation. Both stacks then report a 9-cell robot region, which is a 3x3 clearing, and `real_robot` measured the two 3x3 patches overlapping in a single cell. Two clearings of the same size landing one cell apart points at a world-to-cell conversion differing by one. That is the live hypothesis, not a measurement.

`get_environment_bounds` is dynamic with the robot: at spawn it returned xmin = robot_x - 0.0375, and moving the robot moved it. So the lattice origin follows the pose, which composes with everything above, and any harness fetching bounds at a different moment than the snapshot compares shifted grids.

In the Python grid, `robot` and `region_3` have zero contacts at 4-connected and diagonal, and the exporter already uses all eight offsets, so this is not a connectivity-choice artifact. The cause is open.

## What each accessor actually means

The single most reusable thing here. Every one of these looked like the thing the comparison needed, and each answers a different question. All source-read.

`count_reachable_points(points)` is **not** occupancy and **not** robot feasibility. `push_primitive_executor.cpp:229-253` calls `is_goal_reachable(point, goal_tolerance)` with the tolerance set to the 4.0 cm inflation radius, so a point counts as reachable when ANY robot-reachable cell lies within 4 cm of it. It is a tolerance-dilated query about the reachable set. That single fact explains the wall test below and voided analyses on both sides.

`get_region_snapshot` returns `region_labels` as `{region_id: role}`, two or three entries like `{1: 'robot', 2: 'goal'}`. It is not a cell map, and it carries no grid at all.

`sample_region_goals` / `region_goals` returns true interior cell centres, verified at `wavefront_grid.cpp:840-880`: it takes the region's own cell set, shuffles, and emits `grid_to_world + 0.5 * resolution`. No tolerance padding, so a correspondence probe built on it is sound. **But it only samples regions the adjacency BFS reaches from the robot.** On `1push/2hop/env1` at the XML spawn, 3 regions exist and exactly one bundle comes back, because the robot's region is isolated there. Regions disconnected from the robot are invisible to it, which are the ones a partition comparison most wants.

`get_environment_bounds` moves with the robot, as above.

`region_map`, on the Python side only, is indexed `[x, y]` and is the one true per-cell partition available anywhere.

## Dissolved, and the instrument failure that produced each

Five findings were reported and then withdrawn, three mine and two `real_robot`'s. Each was caught by a check rather than by a test passing, and each is written out because the next person will reach for the same instrument.

**`region_labels` is not a cell map.** The first comparator grouped cells by `region_labels` and compared the resulting point sets. That field is `{region_id: role}`, two entries, like `{1: 'robot', 2: 'goal'}`. So the "point sets" were singletons of region ids, and comparing them was comparing labels, the one thing the brief said never to compare. The mandatory sabotage run caught it: with an object moved 5 cm the comparator still called 15 scenes identical. Without the sabotage the report would have read "the two wavefronts agree on 15 of 16 scenes" and meant nothing.

**The lattice was transposed.** `region_map` is indexed `[x, y]`, not `[row, col]`. The grid is 106x163 and the x span is exactly 106 cells. Every cell-to-world conversion was transposed, so every point handed to the simulator was a different place than the cell read from the map. It produced a table showing C++ unable to reach any cell of the region the robot stands in. Caught because the robot at (0.25, 0.10) mapped to cell (24, 54) while the Python robot region sat at (54, 24), an exact transpose. The corrected table is total containment: C++ reaches 9 of 9 robot cells, 1985 of 1985 region_3 cells, and 0 of 4402 goal cells.

**`count_reachable_points` answers a dilated question, and this is the one to remember.** Two lines prove it. `wall_3`'s surface is at y = 0.0, and a robot with a 3.5 cm half-extent cannot centre closer than 3.5 cm to it:

```
0.5 cm above the wall surface: reachable=True
1.0 cm                       : reachable=True
2.0 cm                       : reachable=True
3.0 cm                       : reachable=True
```

At the time this was read as "the accessor does not inflate". The true semantics are worse and are given above: it dilates by 4 cm, so it reports reachability of a neighbourhood rather than of a point. `real_robot` retracted a 27x27 window comparison to the same cause, a third question compared against two others. That analysis invalidated a full one of mine: 2369 cells that Python blocked and C++ "reached", with a thickness histogram running to 12 cells and an attribution spread across six objects. All of it was the inflation shell. Median distance from those cells to the nearest obstacle surface was 2.0 cm, and only 35 of 2369 lay beyond the shared 4.0 cm inflation. Python was blocking them correctly and an uninflated accessor was reporting them reachable. Two different questions, compared as though they were one.

**A comparison reported without its input.** The 2-versus-3 gate failure went into this document with no robot pose recorded. Two sessions then spent a round trip suspecting stale worktrees, stale bindings and differing scene copies, when the answer was that we had run different poses and neither had said so. Checked and ruled out first: 122 commits since the binding was built, none touching `src/`, `include/` or `cpp_bindings/`; scene file md5 identical across both checkouts; and both call shapes giving the same answer on the same build. The pose table above is now the record.

**A clean story from two data points.** After two poses the reading was "Python's partition is pose-independent, C++'s is not". Four more poses killed it: both move. It was one report away from being filed as a semantic difference between the stacks.

## Blocked

The cell-by-cell standard in the acceptance criterion cannot be met with the bindings as they are. It needs an accessor answering "can the robot's centre occupy this cell". `count_reachable_points` answers a different question, as above. `get_region_snapshot` has the right notion of occupancy but returns labels, adjacency and edge objects with no grid. `get_region_connectivity`, `get_reachability_summary` and `sample_region_goals` return no cells either. The Python side has `region_map` on its dataclass, but `get_region_snapshot` flattens it away.

Worth recording how this conclusion was reached, because the sequence matters. It was stated two reports before it was accepted, then abandoned when the seed-teleport idea arrived, because teleporting the robot and flooding looked like a way to extract the C++ partition through an accessor that already existed. It was not, for exactly the reason above. The first answer was right and an elegant workaround displaced it for several hours.

Until an accessor exists, the comparison is limited to what `get_region_snapshot` returns on both sides: region count, adjacency, edge objects, the labels and the booleans. That standard is demonstrably able to miss a partition difference, which is the gap the acceptance criterion was written to close.

## The decision

One question, three answers, and it is Dhruv's.

Add a small read-only debug accessor to the binding, region grid out and nothing else, after the matrix, with this comparison as its first consumer. Or accept the coarse `get_region_snapshot`-level standard, knowing it misses partition differences of exactly the kind now sitting on the books. Or park the commission until after the paper.

`real_robot`'s recommendation, which I share, is the accessor after the matrix, on the grounds that the acceptance gate as set is unmeetable without it.

## Artifacts

`scripts/pipeline/wavefront_parity.py` is the harness, sabotage-first, with the coarse comparison and the limitation documented in its docstring rather than quietly narrowed.

`scripts/pipeline/build_scorer_dataset.py` carries a note recording what the training masks cannot tell us: the mask-generation run captured no fallback flag, so any row rendered under the legacy BFS has wrong-size region channels and there is no field to find those rows by. The 0-of-14 measurement is labelled as being about today's scenes rather than about those rows.

The gate stays failing, on the books, and honestly labelled.
