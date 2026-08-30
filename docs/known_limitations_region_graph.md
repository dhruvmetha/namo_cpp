# Known limitations: region adjacency graph

Live list of things we know are wrong in the region graph and chose not to fix yet, with the reason and what it would take. Opened 2026-08-29 alongside the multi-object edge change. Delete an entry only when the code is fixed, never because it stopped being convenient.

## 1. The single-object pass can miss a connection when a wall splits an object's footprint

`WavefrontGrid::build_region_connectivity_graph` frees one object's inflated footprint, but only the cells where that object is the sole occupant. A wall crossing the footprint leaves solid cells behind and splits the freed cells into separate patches. The flood then starts at `removed_cells[0]` and explores only that patch, recording free neighbours without ever expanding into them. A patch that would join two regions is never visited if the flood started in a different one.

**Why it stays.** Fixing it changes the answer on single-movable scenes, which destroys the byte-parity check that proves the multi-object pass broke nothing. We wanted that check more than we wanted this fix.

**Cost to fix.** Flood every patch instead of the first, then re-baseline every difficulty label in the project. Do it as its own change with its own gate, never bundled.

**How to detect it in the wild.** A scene where deleting an object visibly opens a route but the graph records no edge for it, with the object's footprint straddling a wall.

## 2. Multi-object edges can bridge doorways no push sequence can open

A connected clump of touching movables becomes one blob, and a blob is credited with joining every region its boundary touches. A chain of blocks spanning two separate doorways therefore writes an edge between regions that no achievable push actually connects.

**Why it stays.** The per-object test has the same weakness in milder form, since freeing a footprint is not proof any push can move the object there. The simulator is the verifier in both cases; the graph only proposes candidates.

**Cost to fix.** Per-blob minimum cut, or a reachability check per candidate. Neither is cheap and neither is needed while the simulator checks every candidate anyway.

**How to detect it in the wild.** An edge whose object set spans blocks at opposite ends of the arena, or a blob touching three or more regions.

## 3. The Python exporter still uses single-object semantics

`python/namo/visualization/wavefront_snapshot.py::_build_connectivity` reimplements the old per-object rule. It runs only when `region_use_cpp_unified_wavefront` is off, which is not the live path, and `python/tests/test_wavefront_snapshot_semantics.py` pins its current behaviour.

**Why it stays.** Debug and visualisation path only. Porting it doubles the change and the test would need rewriting.

**Cost to fix.** Port the blob pass to Python and re-record the pinned test.

⛔ **Do not treat the Python answer as ground truth in any future parity check.** It will disagree with C++ on exactly the scenes this work fixed, and the disagreement is the Python side being stale. See [[reference_agreement_is_shared_convention]].

## 4. The pose sampler's fallback guards the wrong condition

`WavefrontGrid::sample_region_goals` falls back to sampling every region when `reachable_labels` is empty. It is never empty, because the adjacency map is pre-seeded with an entry per label and the walk always reaches the robot's own label. So a scene with no edges at all slips past the guard and only the robot region gets sampled.

**Why it stays.** The multi-object pass gives these scenes real edges, so the walk reaches the goal region and the guard stops mattering for the case we hit. The guard is still wrong for any future case with a genuinely isolated goal region.

**Cost to fix.** One line, widen to fire when the walk reached nothing past the robot. Left out of this change to keep the diff to one idea.

## 5. Everything above the graph still assumes one object per doorway

The graph can now say two blocks plug a door. Nothing above it can use that. The episode key is a single pushed object, `restrict_obj` takes a single object at every region-opening call site, the training corpus is single-movable rooms, and the value head scores one object at a time.

**Why it stays.** Separate work, and the graph had to come first because everything reads it.

**Cost to fix.** Its own project. Nothing in this change moves a number the model produces.

## 6. The generator's gate and the runtime wavefront disagree on some scenes

The generator certifies a scene by rasterizing it in numpy at 5 mm with 4.0 cm inflation and 8-connectivity, then requiring the goal to be unreachable from the robot. `v1/solo0`'s log records `open_frac: min=0.000 med=0.000 max=0.000`, so every emitted scene passed. Yet the C++ wavefront labels 7 of those 300 as a single `robot_goal` region, meaning robot and goal are already together and there is no door to open. Blocks do not move on load (measured, 0.00 mm), so both implementations are looking at identical geometry.

Investigated 2026-08-29 and only partly resolved:

- 2 of the 7 (`solo1/rb_00028`, `solo1/rb_00095`) are genuinely open. An independent numpy check puts the goal in the robot's own component, so the C++ label is right and the generator's gate was wrong on those.
- 1 (`solo1/rb_00056`) has the robot inside a blocked cell, which triggers the force-clear of a 3x3 patch around the robot in `find_connected_components`. Different mechanism, plausible, unconfirmed.
- 4 (`solo0/rb_00009`, `rb_00030`, `rb_00078`, `rb_00096`) are unexplained. An independent raster says robot and goal are cleanly separate; the C++ merges them.

⛔ **Do not settle this with another numpy reimplementation.** The one used above disagrees with the C++ grid on 279 to 401 cells per scene, all near inflated boundaries, so it cannot arbitrate. Two implementations disagreeing says the conventions differ, not which is correct. See [[reference_agreement_is_shared_convention]]. Settling it needs the C++ to dump its own blocked grid and flood.

Also worth knowing while reading either side: the C++ treats the goal as a **4.0 cm disc** of cells, not a point (`build_goal_cells`, `rl_env.cpp:171`, radius from `compute_goal_tolerance_m` which equals the inflation radius). A region is labelled `robot_goal` when the robot's component contains ANY cell of that disc. The generator's gate tests one point. On a scene where the goal sits within 4 cm of a doorway the two can legitimately differ, and that alone accounts for the first 2 above.

**Scale of the problem.** 7 of 300 in this pool, and all 7 were already being skipped by the labeller, so nothing downstream consumed them. The reason to care is that it means the gate's certificate is not exactly what the runtime enforces, which could bite a pool where it matters more.
