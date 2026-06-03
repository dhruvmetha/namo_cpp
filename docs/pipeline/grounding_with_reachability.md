# Grounding with Reachability

This document defines how ML-predicted goals are grounded with physical reachability constraints in the NAMO planning pipeline.

## Overview

The core challenge in learned goal prediction for manipulation is **grounding**: ensuring that ML predictions correspond to actions the robot can actually execute. In NAMO, this means checking that predicted push goals align with primitive slots the robot can physically reach.

The grounding pipeline has three stages:
1. **Primitive Generation**: Pre-computed push primitives define a discrete action space
2. **ML Goal Alignment**: Diffusion model predictions are matched to primitive slots
3. **Reachability Filtering**: Only primitives on reachable edges are considered for execution

## Primitives and Edge Points

### Edge Points (60 total)

Each movable object has **60 discrete edge points** around its perimeter:
- 4 sides × 15 contact points per side = 60 edge points
- Indexed as `edge_idx ∈ [0, 59]`
- These represent possible robot contact locations for pushing

### Primitive Slots (600 per object)

Each edge point has multiple **push depths**:
- 10 push depths (steps) per edge point
- Indexed as `depth_idx ∈ [1, 10]`
- Total: 60 edges × 10 depths = **600 primitive slots per object**

Each primitive stores:
```
{
  edge_idx: int,      # Which edge point (0-59)
  depth_idx: int,     # Push depth (1-10)
  Δx, Δy, Δθ: float   # Relative transform in object frame
}
```

Primitives are pre-computed and loaded from binary files for efficiency.

## Reachable Edges

### Wavefront-Based Reachability

The C++ wavefront planner performs BFS from the robot's current position to determine which edge points are reachable:

```cpp
// src/planning/mpc_executor.cpp:302
std::vector<int> MPCExecutor::get_reachable_edges_with_wavefront(const std::string& object_name) {
    // 1. Update wavefront from current robot position
    planner_.update_wavefront(env_, robot_pos);

    // 2. Generate edge points for the object
    controller_.generate_edge_points(object_name, edge_points, ...);

    // 3. Check each edge point against the wavefront grid
    for (size_t edge_idx = 0; edge_idx < edge_count; edge_idx++) {
        int grid_val = grid[edge_x][edge_y];
        if (grid_val > 0) {  // Reachable position
            reachable_edges.push_back(edge_idx);
        }
    }
    return reachable_edges;
}
```

### What Makes an Edge Unreachable?

An edge point is unreachable if:
- The wavefront cannot propagate to that grid cell (blocked by obstacles)
- The cell is occupied by another movable object
- The cell is outside world bounds

### Python API

```python
# Get reachable edge indices for an object
reachable_edges = env.get_reachable_edges(object_id)  # Returns List[int], e.g., [0, 1, 3, 5, ...]
```

## ML Goal Alignment

### Diffusion Model Predictions

The diffusion model predicts goal positions for the object:
```python
ml_goals = [
    {'x': 1.5, 'y': 2.0, 'theta': 0.1},
    {'x': 1.3, 'y': 2.1, 'theta': -0.05},
    ...
]
```

### Alignment to Primitive Slots

Each ML goal is matched to nearby primitive slots using position and angle tolerance:

```python
# python/namo/strategies/primitive_goal_strategy.py
class MLPrimitiveGoalStrategy:
    def __init__(self,
                 match_position_tolerance=0.1,  # meters
                 match_angle_tolerance=0.1,     # radians
                 k_nearest=1):                  # votes per ML goal
```

The alignment process:
1. For each ML goal, find primitive slots within tolerance
2. Each matched slot receives a "vote"
3. Higher vote counts → higher priority for that primitive

Output structure:
```python
aligned_primitives = [
    {'edge_idx': 12, 'depth_idx': 3, 'votes': 5, 'x': 1.5, 'y': 2.0, 'theta': 0.1},
    {'edge_idx': 8,  'depth_idx': 2, 'votes': 3, 'x': 1.3, 'y': 2.1, 'theta': -0.05},
    ...
]
```

## Grounding Metrics

### ReachableAttachment@K (RA@K)

Measures what fraction of top-K aligned primitives point to reachable edges:

```
RA@K = (# of top-K aligned primitives with reachable edge) / K
```

Computed in `scripts/eval_2push.py`:

```python
def compute_ra_at_k_single(aligned_primitives, reachable_edges, k=None):
    # Sort by votes (descending)
    sorted_prims = sorted(aligned_primitives, key=lambda p: -p['votes'])

    # Take top-K
    if k is not None:
        sorted_prims = sorted_prims[:k]

    # Count reachable
    num_reachable = sum(1 for p in sorted_prims if p['edge_idx'] in reachable_edges)
    return (num_reachable, len(sorted_prims))
```

### Interpretation

| RA@K Value | Interpretation |
|------------|----------------|
| 1.0 | Perfect: All top-K ML predictions point to reachable edges |
| 0.5 | Half of top-K predictions are physically executable |
| 0.0 | None of the top-K predictions are reachable (complete failure) |

### Random Baseline

Expected RA under random primitive selection:
```
Random RA = |reachable_edges| / 60
```

If 30 edges are reachable, random selection achieves ~50% RA.

## Grounding in the Planning Pipeline

### Region Opening Flow

```
1. Get blocking object for target region
2. Get reachable edges: reachable = env.get_reachable_edges(object_id)
3. Get ML-aligned primitives: aligned = ml_strategy.get_goals(...)
4. Filter to reachable: candidates = [p for p in aligned if p['edge_idx'] in reachable]
5. Try candidates in vote-order (highest first)
```

### Failure Reasons

The pipeline tracks why grounding fails:

| Failure Reason | Description |
|----------------|-------------|
| `no_reachable_objects` | Blocking objects exist but robot can't reach any |
| `ml_goals_not_aligned` | ML produced goals but none matched primitive slots |
| `no_reachable_edges` | Goals aligned but none on edges robot can reach |
| `all_pushes_failed` | Reachable primitives tried but none succeeded |

### Data Fields in AttemptResult

```python
@dataclass
class AttemptResult:
    ml_goals_generated: int = 0      # Raw ML goals before alignment
    ml_goals_aligned: int = 0        # ML goals that matched a primitive slot
    reachable_edges_count: int = 0   # Number of reachable edges for the object
    aligned_primitives: List[Dict]   # Detailed aligned primitive info
    reachable_edges: List[int]       # List of reachable edge indices
```

## Hybrid Decomposition

The SAGE model uses a two-phase approach:

1. **ML-Only Phase**: Try only ML-aligned primitives on reachable edges
2. **Fallback Phase**: If ML fails, try remaining reachable primitives

This is tracked via:
```python
phase_push_counts: Dict[str, int]  # {"ML-only": X, "primitives": Y}
solved_in_phase: str               # "ML-only", "primitives", or ""
```

## Evaluation Metrics

### Per-Instance Metrics

From `RegionResult` in eval scripts:
```python
ml_aligned_count: int                    # Total aligned primitives
ml_aligned_reachable_count: int          # Aligned primitives that are reachable
ml_aligned_reachable_ratio: float        # ml_aligned_reachable_count / ml_aligned_count
```

### Aggregate Metrics

- **Macro RA@K**: Mean of per-instance RA@K fractions
- **Micro RA@K**: Total reachable / Total considered across all instances

## Key Files

| File | Purpose |
|------|---------|
| `src/skills/namo_push_skill.cpp:692` | `get_reachable_edges()` implementation |
| `src/planning/mpc_executor.cpp:302` | Wavefront-based edge reachability |
| `python/namo/strategies/primitive_goal_strategy.py` | ML goal alignment logic |
| `scripts/eval_2push.py:592-700` | RA@K computation |
| `python/namo/planners/opening/region_opening.py` | Grounding in planning pipeline |

## Summary

Grounding with reachability ensures ML predictions translate to executable actions:

1. **Discretization**: 600 primitive slots per object (60 edges × 10 depths)
2. **Alignment**: Diffusion predictions vote on primitive slots
3. **Filtering**: Only reachable edges are considered
4. **Metrics**: RA@K measures grounding quality
5. **Fallback**: Unaligned primitives provide completeness guarantee

This architecture separates learned prioritization (which primitives to try first) from physical feasibility (which primitives are reachable), enabling robust planning even when ML predictions are partially grounded.
