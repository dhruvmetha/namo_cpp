# Wavefront Cell Semantics

This repository uses a single canonical encoding for wavefront-style grids.

## Canonical Encoding

- `-1`: occupied (and therefore unreachable)
- `0`: free but unreachable
- `1`: free and reachable

## Where It Applies

- `WavefrontPlanner` reachability grid (`get_grid()` / `get_distance_grid()`):
  - Uses full `-1/0/1`.
- `WavefrontGrid` occupancy grids (`get_dynamic_grid()` / `get_static_grid()`):
  - Uses occupancy subset: `-1` occupied, `0` free.
  - `1` is not used in occupancy-only grids.
- Python snapshot/viewer occupancy arrays:
  - Same occupancy subset: `-1` occupied, `0` free.

## Contract Rules

- Do not write ad-hoc debug markers (for example `-3`, `-4`) into live wavefront grids.
- Reachability checks should treat `1` as reachable explicitly.
- Obstacle checks should treat `-1` as occupied explicitly.

## Related Components

- `full_pipeline_fixed/namo/robot_control` wavefront occupancy grid now uses:
  - `OBSTACLE = -1`
  - `FREE = 0`
- `sage_learning` consumes masks/regions and distance fields, not raw C++ wavefront integer grids.
