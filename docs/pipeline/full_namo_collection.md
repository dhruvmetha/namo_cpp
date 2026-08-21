# Full NAMO Collection Configuration

This document is a human-readable version of `python/namo/data_collection/full_namo_collection.yaml`.

It describes the **Full NAMO** data collection setup (algorithm: `full_namo`) and the key parameters passed through to the internal **Region Opening** sub-solver.

## How to run

`full_namo_collection.yaml` is used by:

- `scripts/run_full_namo_collection.sh` (`--algorithm full_namo`)

Example:

```bash
./scripts/run_full_namo_collection.sh --start-idx 0 --end-idx 100 --workers 8
```

## What `full_namo` does (planner summary)

The planner is `python/namo/planners/full_namo/full_namo_planner.py`.

At a high level (`FullNAMOPlanner`):

1. Set the robot goal from the environment.
2. If the goal is already reachable, stop.
3. Compute a **full** region connectivity snapshot (not local-only).
4. Find the robot region label and the goal region label.
5. Run BFS in the region adjacency graph to find a region path.
6. Call `RegionOpeningPlanner.search(..., target_neighbor=next_region)` to open the next region.
7. Apply the resulting post-push state and repeat until the goal is reachable or the iteration limit is hit.

## Field-by-field reference

### Core execution

- `output_dir`: where `*_results.pkl` worker outputs are saved.
- `start_idx`, `end_idx`: subset range in the manifest / discovered XML list.
- `algorithm`: must be `full_namo` for this pipeline.
- `workers`: number of parallel workers (each runs environments independently).
- `episodes_per_env`: number of episodes per environment XML.

### Full NAMO parameters

- `full_namo_max_iterations`: maximum number of region-opening iterations before declaring failure. Each iteration attempts to open exactly one next region along the current region-graph shortest path.

### Region opening sub-solver parameters

These are forwarded via `PlannerConfig.algorithm_params` to the internal `RegionOpeningPlanner`:

- `region_max_chain_depth`: maximum number of pushes allowed per region opening attempt (e.g., 2 allows 2-push chains).
- `region_max_solutions_per_neighbor`: per-object cap on number of solutions to collect.
- `region_frontier_beam_width`: beam cap on frontier states per chain depth (`null`/`0` means unbounded).
- `region_max_recorded_solutions_per_neighbor`: number of successful solutions per object that are kept/logged.
- `region_chain_link_cost`: additional flat cost added for multi-push chains.
- `region_ml_ignore_blacklist`: allow ML-scored candidates to bypass edge blacklists (if using ML strategies).
- `region_selection_strategy`: either `cost_first` or `ml_first`.

### Goal strategy (region opening)

- `goal_strategy`:
  - `primitive`: purely primitive enumeration.
  - `ml`: ML-aligned primitive slots only.
  - `scorer`: learned ranker orders the primitive candidates.
  - `geometric`, `random_rollout`: transport-heuristic and random orderings.

If `goal_strategy: primitive`:

- `shuffle_edges`: shuffle primitive edge ordering.
- `shuffle_seed`: deterministic shuffle seed.

### Generic planner limits

These are shared “compatibility” fields used across planners:

- `max_depth`, `max_goals_per_object`, `max_terminal_checks`.
- `search_timeout`: overall search timeout in seconds.
- `goals_per_region`: number of region-goal samples used for reachability validation.

### Environment + path settings

- `points_per_face`: must match the chosen `config_file` (skill3 vs skill15).
- `xml_dir`: directory containing environment XML files.
- `config_file`: NAMO config file (skill configuration).
- `manifest`: optional manifest file path.

## Current YAML (verbatim)

```yaml
# YAML defaults for modular_parallel_collection.py (Full NAMO)
# Use with: python modular_parallel_collection.py --config-yaml python/namo/data_collection/full_namo_collection.yaml
# CLI flags can override any value here.

# Core execution
output_dir: /common/users/dm1487/namo_data/full_namo_test
end_idx: 100
algorithm: full_namo
workers: 8
episodes_per_env: 1

# Full NAMO specific parameters
full_namo_max_iterations: 20  # Max region openings before giving up

# Region opening sub-solver parameters (inherited by internal RegionOpeningPlanner)
region_max_chain_depth: 2     # 1=single push, 2=two pushes, 3=three pushes per region opening
region_max_solutions_per_neighbor: 1
region_frontier_beam_width: 10000
region_max_recorded_solutions_per_neighbor: 1
region_chain_link_cost: 11
region_ml_ignore_blacklist: false
region_selection_strategy: cost_first

# Goal strategy for region opening sub-problems
# Options: primitive | geometric | ml | scorer | random_rollout
goal_strategy: primitive

# Primitive strategy options (only used when goal_strategy=primitive)
shuffle_edges: false
shuffle_seed: null

# Generic planner limits
max_depth: 5
max_goals_per_object: 5
max_terminal_checks: 5000
search_timeout: 600.0  # 10 minutes for full NAMO (may need multiple region openings)
goals_per_region: 10   # Number of robot goal samples per region for validation

# Points per face configuration (must match config_file)
points_per_face: 15

# Paths - CUSTOMIZE THESE FOR YOUR ENVIRONMENT
xml_dir: /common/users/shared/robot_learning/dm1487/namo/mj_env_configs/nov28/11_11_07deletion/maze_11x11_del70p_seed25188
config_file: config/namo_config_complete_skill15.yaml
# manifest: path/to/manifest.txt  # Optional: pre-generated file list

# Optional pipeline features
verbose: false
filter_minimum_length: false
smooth_solutions: false
max_smooth_actions: 20
refine_actions: false
validate_refinement: false
```
