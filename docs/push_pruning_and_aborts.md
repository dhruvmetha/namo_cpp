# Push pruning and abort conditions

How a push attempt can fail or be skipped, top to bottom. Written for the diff-drive car, but layer 1 is robot-agnostic.

---

## The three layers

A single "push" decision flows through three independent checkpoints:

| Layer | Where | What it decides |
|---|---|---|
| **1. Search pruning** | `region_opening.py` (Python planner) | *Which* (object, edge, depth) primitives are worth trying at all |
| **2. Skill-level abort** | `namo_push_skill.cpp` + `namo_push_controller.cpp` (C++) | Whether the *current* push attempt should bail out mid-execution |
| **3. Feedback to search** | results returned to Python | Updates the planner's blacklist so it stops trying primitives that mechanically can't work |

Layer 1 is logical pruning (skipping dead branches). Layer 2 is physical pruning (the simulation says "no"). Layer 3 closes the loop.

---

## Layer 1: Search pruning (`region_opening.py`)

Tried in order, every step of the inner loop over (edge_idx, depth) candidates ([region_opening.py:2167-2260](../python/namo/planners/opening/region_opening.py#L2167)).

### Always on

- **Already-accessible pre-check.** If the goal region is already reachable from the robot's region, no push is needed — record a 0-push success and stop.
- **`remaining_budget`.** Skip any candidate whose depth would exceed the chain budget (`max_chain_depth`).
- **`region_object_skip` blacklist.** Drop (region, object) pairs the manifest told us to ignore.
- **`region_target_goal_only`** (default on). Only try to open the neighbour that *contains the goal*. Other neighbours are not explored.
- **`stop_after_first_success`** (default off). Stop iterating neighbours once any opens successfully.
- **Score-based ordering.** When using ML strategies, ML-scored primitives are tried first; ties break on depth.

### Disabled in exhaustive mode (`region_exhaustive_mode=True`)

| Lever | What it prunes |
|---|---|
| **`max_solutions_per_neighbor`** (default 10) | Stop after N successes per neighbour |
| **Depth prune** (`min_depth_found`) | Once any success at depth D, skip candidates with depth > D |
| **`solved_edges_this_skill`** | Skip edges that already produced a success |

Exhaustive mode keeps Layer 2 signals (stuck/collision) and budget caps — those are physical, not search heuristics.

### Always-on physical signal (kept even in exhaustive mode)

- **`edge_min_stuck_depth` blacklist.** If edge X got stuck or collided at depth d, skip X at depth ≥ d. Reasoning: a deeper push on the same edge will also wedge — the geometry hasn't changed.

---

## Layer 2: Skill-level abort (C++)

A push attempt has four sub-phases. Each can abort.

### 2a. Navigation to the push start
[`diff_drive_navigation.cpp:check_robot_collision_any`](../src/navigation/diff_drive_navigation.cpp#L62)

The car drives from its current pose to the push start point (rotate → drive → ... → final rotate). Aborts on:

- Robot collision with **any wall** during any nav phase.
- Robot collision with **any non-target movable** during any nav phase.
- Stuck in a nav phase (path can't finish in budget).
- No navigable path to the push edge point.

"Robot" here is **chassis + both wheels** (see [Diff-drive specifics](#diff-drive-specifics) below).

### 2b. Post-positioning placement check
[`namo_push_controller.cpp:353-381`](../src/planning/namo_push_controller.cpp#L353)

After navigation lands the car at the push start, before any wheel command for the push itself, check that the placement is feasible:

- Robot ↔ wall: abort.
- Robot ↔ non-target movable: abort.
- Object touching a wall here is *allowed* — being pre-positioned against the object you're about to push is normal.

### 2c. During the push (periodic, every `stuck_check_stride` ticks)
[`namo_push_controller.cpp:421-475`](../src/planning/namo_push_controller.cpp#L421)

The car drives both wheels at a fixed velocity for `push_steps × 250` sim ticks. Every few ticks it samples physics state and runs four collision queries plus a stuck check.

**Always abort:**
- Robot ↔ wall.
- Robot ↔ non-target movable.
- **Controller-level stuck:** pushed object moved < 1 mm *and* rotated < ~3° for `stuck_ctrl_iterations_threshold` consecutive samples. Abort with `"Controller-level stuck"`.

**Conditional abort** — only when `terminate_on_collision` is on (off by default in region_opening data collection):
- Pushed object ↔ wall. (Recorded as `wall_collision` either way.)
- Pushed object ↔ non-target movable. (Recorded as `movable_collisions` either way.)

So in standard data collection, scraping a wall with the *object* is fine and informative. Scraping a wall with the *robot* is fatal.

### 2d. Between MPC iterations
[`namo_push_skill.cpp:313-349, 677-690`](../src/skills/namo_push_skill.cpp#L313)

The skill executes one push primitive per MPC iteration; multiple iterations may run within a single skill call. Between iterations:

- **Skill-level stuck:** pre→post displacement < 1 mm *and* yaw change < 0.05 rad. After `max_stuck_iterations` (default 2) such iterations in a row, abort the whole skill, set `outputs["stuck"] = "true"`, and add the edge to `stuck_edges`.

There are two stuck layers because they catch different failure modes:
- **Controller-level stuck** (2c) — the simulator is glued *during* a single push step (object jammed against geometry).
- **Skill-level stuck** (2d) — the push step finishes "successfully" but the object barely moves *across* steps (wrong push direction; deeper pushes on this edge won't help either).

### 2e. Misc
- Lost object state (`get_object_state` returns null mid-push) — abort silently.

---

## Layer 3: What the planner learns from each attempt

Each push attempt returns a `step_result.info` dict to Python. The planner uses it to update three pieces of state:

| Field | Meaning | Used by |
|---|---|---|
| `wall_collision` | Any wall contact during the push (robot or object) | F-characterization log |
| `movable_collisions` | Comma-separated list of other movables touched | F-characterization log |
| `collision_object` | Body that triggered an abort, if any | Reporting |
| `stuck` | `"true"` if either stuck layer fired | Updates `edge_min_stuck_depth[edge_idx]` |

`edge_min_stuck_depth` is the bridge from physics back to search. Once an edge is recorded as stuck at depth d, the search skips that edge at depths ≥ d for the rest of this skill call. This is a Layer-2 → Layer-1 feedback loop: the simulation says "this is mechanically infeasible," and the planner doesn't waste time re-asking.

---

## Diff-drive specifics

### What counts as "the robot" in collision checks

Defined by [`DiffDriveAdapter::get_collision_body_names()`](../src/robot/diff_drive_adapter.cpp#L53):

```
{"car", "left_wheel", "right_wheel"}
```

**Casters are excluded** — they're 2.5 mm spheres at z = 2.5 mm under the chassis, geometrically unable to reach a wall before the chassis does.

**Wheels are included.** Geometry of `little_car.xml`:
- Chassis half-width (y-axis): **0.035 m**
- Wheel outer face (y-axis): **0.038 m**

Wheels protrude 3 mm past the chassis sideways. Without including them, a wheel scraping a wall would be silently ignored — the contact is on `left_wheel` body, not `car`, and `bodies_in_collision("car", "wall_X")` is exact-pair (no parent/child traversal).

### When the wheel inclusion matters

| Motion | Outermost contact point | Was wheel-only contact possible before? |
|---|---|---|
| Forward / reverse drive (wall ahead/behind) | Chassis face at x = ±0.035 (wheel x-extent is only ±0.015) | No — chassis hits first |
| Rotate in place (corner-sweep) | Chassis corner (radius 4.95 cm) vs wheel (radius 3.8 cm) | No — chassis corner hits first |
| **Drive alongside a wall (lateral)** | **Wheel face at y = ±0.038, chassis at y = ±0.035** | **Yes — wheel can clip with 0–3 mm of nominal lateral chassis clearance** |

The lateral case is the one this change closes. It only matters when the planner's path leaves 0–3 mm of nominal chassis clearance, which depends on wavefront resolution and obstacle inflation.

### Effect on data collection

Robot-collision aborts get strictly more aggressive — **paths that previously finished with a silent wheel scrape now abort.** Those aborts are real failures: the planner now sees them and blacklists the edge. Expect a small bump in aborted attempts and a corresponding drop in spurious "successful" pushes that physically had a wheel grinding on a wall.

Holonomic (point) robot is unchanged — its `get_collision_body_names()` defaults to `{"robot"}`.
