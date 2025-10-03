# Fast Region Connectivity Helper

This note documents the snapshot-based helper that exposes wavefront region connectivity to Python planners without touching the C++ skill code.

## Overview
- Module: `namo.planners.connectivity_snapshot`
- Entry point: `snapshot_region_connectivity(env, xml_path, config_path, *, goal_radius=0.15, include_snapshot=False)`
- Planner helper: `namo.planners.get_region_connectivity(env)` now prefers the snapshot route automatically when the environment exposes `get_xml_path()` and `get_config_path()`.

## Returned Data
The helper returns a tuple `(adjacency, edge_objects, region_labels, snapshot)` where:
- `adjacency` maps region labels to neighbouring region labels (`Dict[str, Set[str]]`).
- `edge_objects` maps `(region -> neighbour)` pairs to movable object names that unblock that edge (`Dict[str, Dict[str, Set[str]]]`).
- `region_labels` maps numeric IDs to human readable names (`Dict[int, str]`).
- `snapshot` is a `WavefrontSnapshot` when `include_snapshot=True`, otherwise `None`.

These structures mirror the outputs previously produced by the C++ `WavefrontGrid` implementation, so existing planners can consume them without modification.

## Usage Example
```python
from namo.planners import get_region_connectivity

# "env" is an active namo_rl.RLEnvironment
adjacency, edge_objects, labels = get_region_connectivity(env)
```

The helper automatically:
1. Calls the snapshot exporter (`WavefrontSnapshotExporter`) when the environment was constructed through the Python bindings (fast path).
2. Falls back to the legacy `env.get_region_connectivity()` C++ binding when XML/config paths are unavailable (e.g., custom environments).

## Console Helper
For quick inspection without writing a script, use the module runner:

```bash
python -m scripts.snapshot_connectivity_cli \
    --xml /common/users/shared/robot_learning/dm1487/namo/mj_env_configs/aug9/easy/set1/benchmark_1/env_config_100a.xml \
    --config python/config/namo_config.yaml \
    --max-regions 5 --verbose
```

Flags:
- `--max-regions` controls how many regions to print (`-1` prints all).
- `--verbose` includes blocking object details per edge.
- `--resolution` and `--goal-radius` mirror the helper kwargs when tuning the snapshot.

## Performance Notes
- Snapshot export reuses NumPy-based rasterisation and avoids round-tripping through pybind.
- In local tests the fast path completes in milliseconds, even on large benchmark scenes.
- The exporter temporarily resets MuJoCo to gather geometry. Cache planner state before calling it if you depend on transient simulation changes.

## Prerequisites
Activate the MuJoCo virtual environment before importing `namo` modules:
```bash
workon mujoco
export PYTHONPATH=/path/to/namo_cpp/python:$PYTHONPATH
```
Replace the `PYTHONPATH` snippet with the workspace-specific path if different.

## Validation
A simple inline script can validate the helper:
```python
from namo.planners import snapshot_region_connectivity
from namo_rl import RLEnvironment

env = RLEnvironment("templates/benchmark_empty.xml", "config/simple_test.yaml", seed=0)
adj, edge_objects, labels, _ = snapshot_region_connectivity(
    env,
    env.get_xml_path(),
    env.get_config_path(),
)
print(f"regions: {len(adj)}")
print(f"robot neighbours: {sorted(adj['robot'])}")
```
The output should list a small number of regions and neighbours, confirming the helper is wired correctly.
