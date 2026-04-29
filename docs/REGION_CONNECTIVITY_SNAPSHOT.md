# Unified Region Snapshot (C++-First)

This note documents the planner-facing region/connectivity snapshot API.

Cell-value conventions are documented in [`WAVEFRONT_CELL_SEMANTICS.md`](WAVEFRONT_CELL_SEMANTICS.md).

## Overview
- Primary API (C++ binding): `env.get_region_snapshot(...)`
- Python helper: `namo.planners.get_region_snapshot(...)`
- Planner wrappers:
  - `namo.planners.get_region_connectivity(...)`
  - `namo.planners.get_region_goal_samples(...)`

The Python helper prefers the C++ unified API and only falls back to Python raster snapshotting when explicitly requested (`use_cpp_unified=False`) or when the binding is unavailable.

## C++ API Shape
`env.get_region_snapshot(goals_per_region=0, goal_radius=-1.0, local_info_only=False, seed=42, use_xml_goal=True)` returns:
- `adjacency`: `Dict[str, Set[str]]`
- `edge_objects`: `Dict[str, Dict[str, Set[str]]]`
- `region_labels`: `Dict[int, str]`
- `region_goals`: `Dict[str, {"points": List[(x, y, theta)], "blocking_objects": List[str]}]`
- `robot_label`: `str`
- `goal_label`: `str`
- `goal_reachable`: `bool`
- `goal_in_free_space`: `bool`

## Usage
```python
from namo.planners import get_region_snapshot

snapshot = get_region_snapshot(
    env,
    goals_per_region=10,
    use_cpp_unified=True,
    goal_radius=None,  # auto: sqrt(hx^2 + hy^2) + tier1_margin
    seed=42,
)

adjacency = snapshot["adjacency"]
edge_objects = snapshot["edge_objects"]
region_labels = snapshot["region_labels"]
robot_label = snapshot["robot_label"]
goal_reachable = snapshot["goal_reachable"]
```

## Determinism
- Region-goal sampling is seeded (`seed` argument).
- Region traversal and representative point selection are deterministic.

## Fallback / Debug Path
Legacy Python snapshotting remains available through:
- `namo.planners.connectivity_snapshot.snapshot_region_connectivity(...)`

Use this for parity checks or debugging only; execution planners should consume the unified C++ snapshot path.
