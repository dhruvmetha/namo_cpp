# Planner Contract Drifts

This note records confirmed config-contract drift in the legacy data-collection path.
These are informational issues and are not part of the exact-`n` solvability runner.

## `region_min_reachable_fraction` is ignored by `RegionOpeningPlanner`

Confirmed behavior:

- `python/namo/data_collection/modular_parallel_collection.py` forwards
  `region_min_reachable_fraction` into `algorithm_params`.
- `python/namo/planners/opening/region_opening.py` does not read that key.
- The planner only reads `region_success_min_reachable`, an absolute integer count.
- Validation compares `reachable_count >= region_success_min_reachable`.

Practical effect:

- A config such as `goals_per_region: 50` and
  `region_min_reachable_fraction: 0.5` does **not** require 25 reachable samples.
- Unless some caller separately sets `region_success_min_reachable`,
  the planner still uses the default threshold of `1`.

## Generic `search_timeout` is ignored by `region_opening` and `full_namo`

Confirmed behavior:

- The legacy collection runner stores `search_timeout` into
  `PlannerConfig.max_search_time_seconds`.
- `RegionOpeningPlanner` does not read `PlannerConfig.max_search_time_seconds`.
- `FullNAMOPlanner` does not read `PlannerConfig.max_search_time_seconds`.
- The planner-side timeout key used by `RegionOpeningPlanner` is
  `region_timeout_per_neighbour_sec`, which is a different contract.

Practical effect:

- Passing `search_timeout` through `modular_parallel_collection.py`
  does not bound `region_opening` or `full_namo`.

## `modular_parallel_collection.py` does not assemble `full_namo` algorithm params

Confirmed behavior:

- The collection runner builds `algorithm_params` for `region_opening`
  and `uniform_rollout_sampler`.
- It does not build the same explicit `algorithm_params` bundle for `full_namo`.
- `FullNAMOPlanner` then falls back to defaults for settings that should have
  been forwarded to the nested `RegionOpeningPlanner`.

Practical effect:

- `full_namo` runs launched through the legacy collection runner can silently
  ignore intended sub-planner settings.

## Status

The exact-`n` solvability runner avoids these stale contracts by:

- building `PlannerConfig` directly
- forwarding only the supported planner keys explicitly
- using a runner-managed per-environment push budget instead of legacy timeout keys
