# Region Opening + Primitive Goal Strategy

## High-Level Goal

Given an environment with a robot, walls, and movable obstacles, **for each neighbor region adjacent to the robot's current region, find a 1- or 2-push sequence that creates connectivity (opens a path) from the robot's region into that neighbor**.

Each (neighbor, object) pair produces one episode — either a success with an action sequence, or a classified failure.

---

## Step 1: Region Decomposition (`snapshot_region_connectivity`)

Uses the `WavefrontSnapshotExporter` to:
1. Rasterize the environment into a 2D grid (obstacle inflation, BFS reachability)
2. Identify **connected components** of free space → these become "regions"
3. Label the region containing the robot as `robot_region_X`
4. Label the region containing the XML goal as `goal`
5. Build a **region adjacency graph** — two regions are neighbors if they'd be connected except for movable objects blocking the boundary
6. Identify **edge objects** — which specific movable objects sit on the boundary between two adjacent regions
7. Sample **region goals** — N random free-space points inside each neighbor region (for validating openings later)

Output: `adjacency`, `edge_objects`, `region_labels`, `region_goals`

---

## Step 2: Neighbor Iteration (`_explore_from_state`)

For each neighbor region of the robot:
1. Reset env to baseline state
2. Pre-check: is this neighbor **already accessible**? (any of its sampled goals reachable via wavefront) — if yes, skip
3. Get candidate blocking objects from `edge_objects[robot_region][neighbor]`
4. Filter to only **reachable** objects (robot can reach them via wavefront BFS)
5. Apply any skip-list from the manifest

---

## Step 3: Per-Object BFS Search (`_search_with_chaining_bfs`)

For each reachable blocking object, run a **layered BFS over push chains**:

### The Primitive Goal Space

The `PrimitiveGoalStrategy` generates a fixed set of push targets for the object:
- **60 edge points** (4 edges × 15 contact points on the object's surface)
- **10 push depths** per edge (push steps 1–10, increasingly far)
- Total: **600 candidate goals** per object per state

Each goal is an absolute SE(2) pose `(x, y, θ)` computed by transforming a precomputed delta `(Δx, Δy, Δθ)` through the object's current orientation. The `.dat` file is shape-specific (square/tall/wide).

### Depth 1 (Single Push)

For each of the 600 candidates (filtered to reachable edges):
1. Restore baseline state
2. Check `_validate_opening` BEFORE push (should return false)
3. Execute push via `env.step(action)`
4. Check `_validate_opening` AFTER push — **success** means at least 1 sampled region goal is now reachable by the robot
5. If collision or stuck → blacklist that edge at that depth (shallower depths still allowed)
6. If success → record goal chain, state observations, collision info

Candidates are sorted by `(-score, depth, edge_idx)`. In pure primitive mode (no ML), score is always 0, so the effective ordering is **(depth ascending, edge_idx ascending)** — shallowest pushes first.

**Pruning:**
- Once a solution is found at depth D, skip all candidates with depth > D
- Once an edge succeeds, skip deeper depths on that edge
- Blacklist: if edge gets stuck at depth K, skip depths ≥ K on that edge

### Depth 2+ (Multi-Push Chains)

If `max_chain_depth ≥ 2` and depth-1 didn't find a solution (or more solutions are needed):

1. All **non-stuck, non-collision** states from depth-1 become **frontier nodes**
2. Frontier is sorted by `(chain_cost, step_cost, -score)` in `cost_first` mode
3. Apply beam width pruning (cap frontier to 10,000 nodes)
4. For each frontier node: restore that state, regenerate 600 primitive goals for the object's new position, execute the same BFS inner loop
5. If the 2nd push creates an opening → success! Reconstruct the full 2-push chain by walking parent pointers

**Cost model:**
- `total_cost = sum(step_costs) + chain_link_cost` (if multi-push)
- `step_cost` = primitive depth index (1-based) at which the push was found
- `chain_link_cost = 11` (config) — penalty for using 2 pushes vs 1
- Only minimum-cost solutions are kept

---

## Step 4: Validation (`_validate_opening`)

After each push:
1. For each sampled goal in the target neighbor region (typically 10 samples):
   - Call `env.set_robot_goal(x, y, θ)` then `env.is_robot_goal_reachable()`
   - This uses the **cached wavefront** from the last skill execution (zero-cost check)
2. If **≥1 goal is reachable** → the opening is valid

---

## Step 5: Observation Collection

For each successful chain:
- **state_observations**: SE(2) poses of all objects BEFORE each push
- **post_action_state_observations**: SE(2) poses AFTER each push
- **reachable_objects_before/after**: which objects the robot can reach before/after
- **Collision tracking**: wall collisions, which other movable objects got bumped

---

## Step 6: Result Assembly

Per neighbor, the planner returns `AttemptResult` objects (one per object tried):
- **Successes**: action sequence, state observations, which region goal validated it, total cost, chain depth
- **Failures**: classified reason (`no_blocking_objects`, `no_reachable_objects`, `all_pushes_failed`, etc.)

These get converted into `ModularEpisodeResult` objects by the collection script and saved as `.pkl` files.

---

## Key Config Parameters (Typical Settings)

| Parameter | Value | Meaning |
|---|---|---|
| `region_max_chain_depth` | 2 | Try 1-push, then 2-push chains |
| `region_frontier_beam_width` | 10000 | Cap 2nd-push frontier to 10K states |
| `region_max_solutions_per_neighbor` | 1 | Find 1 solution per (neighbor, object) |
| `region_max_recorded_solutions_per_neighbor` | 1 | Save 1 solution per (neighbor, object) |
| `region_chain_link_cost` | 11 | 2-push costs +11 over a single push |
| `region_selection_strategy` | cost_first | Prioritize minimum-cost solutions |
| `points_per_face` | 15 | 60 total edge points (4 faces × 15) |
| `goals_per_region` | 10 | 10 validation samples per region |

---

## Summary Flow

```
For each XML environment:
  1. Decompose into regions (wavefront BFS on 2D grid)
  2. For each neighbor region of robot:
     a. For each reachable blocking object on boundary:
        i.  Generate 600 primitive push targets (60 edges × 10 depths)
        ii. Try shallowest-first: push object, check if neighbor opens
        iii. If no 1-push works and depth≤2: expand frontier, try 2-push chains
        iv. Keep minimum-cost solution(s)
     b. Record success/failure per (neighbor, object) triplet
  3. Save all episodes to .pkl
```
