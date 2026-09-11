"""Planner-side occupied-goal targets, without changing local keyhole labels."""

from dataclasses import dataclass
from typing import Any, Dict, Tuple

import numpy as np


@dataclass(frozen=True)
class GoalClearanceTarget:
    """Remove one object's inflated footprint from a fixed accepted goal cell."""

    object_id: str
    witness_xy: Tuple[float, float]
    cell: Tuple[int, int]
    goal_cells: Tuple[Tuple[float, float], ...]
    resolution: float

    @property
    def label(self) -> str:
        """Stable task identity, not a label in the free-space region graph."""
        return f"goal_cell:{self.cell[0]}:{self.cell[1]}"

    @property
    def samples(self):
        """Fixed target positions for the existing geometric ordering."""
        return [(x, y, 0.0) for x, y in self.goal_cells]

    def is_open(self, env) -> bool:
        """Accept partial multi-object clearance; only the original goal ends NAMO."""
        return bool(env.is_robot_goal_reachable() or not env.object_occupies_point(
            self.object_id, self.witness_xy))


def clearance_targets(snapshot: Dict[str, Any]):
    """Enumerate movable-owned goal cells, nearest the original goal first."""
    info = snapshot["goal_clearance"]
    cells = [cell for cell in info["cells"] if not cell["static_blocked"]]
    goal_cells = tuple(tuple(cell["xy"]) for cell in cells)
    gx, gy = info["goal_xy"]
    cells.sort(key=lambda cell: ((cell["xy"][0] - gx) ** 2 + (cell["xy"][1] - gy) ** 2,
                                 tuple(cell["grid"])))
    return [GoalClearanceTarget(str(obj), tuple(cell["xy"]), tuple(cell["grid"]),
                                goal_cells, float(info["resolution"]))
            for cell in cells for obj in sorted(cell["objects"])]


def goal_diagnostics(snapshot: Dict[str, Any]) -> Dict[str, Any]:
    """Describe terminal occupancy, separately from why the planner stopped."""
    info = snapshot["goal_clearance"]
    usable = [cell for cell in info["cells"] if not cell["static_blocked"]]
    blockers = sorted({obj for cell in usable for obj in cell["objects"]})
    reachable = set(info["reachable_objects"])
    return {
        "goal_reachable": bool(snapshot.get("goal_reachable", False)),
        "goal_has_static_free_cells": bool(usable),
        "goal_has_free_cells": any(not cell["objects"] for cell in usable),
        "goal_occupied_by_movables": bool(usable) and all(cell["objects"] for cell in usable),
        "movable_ids_covering_goal_cells": blockers,
        "reachable_goal_blockers": [obj for obj in blockers if obj in reachable],
        "goal_blockers_without_free_approach": [obj for obj in blockers
                                                if not info["access_regions"].get(obj)],
    }


def goal_mask(cells, resolution, local_bounds, output_size):
    """Rasterize fixed goal cells into the scorer crop using exact pixel coverage.

    Rows increase with world y, matching the canonical renderer's unflipped
    region-map transpose. Fractional coverage retains sub-pixel goal cells.
    """
    xmin, xmax, ymin, ymax = local_bounds
    dx, dy = (xmax - xmin) / output_size, (ymax - ymin) / output_size
    xs = xmin + np.arange(output_size) * dx
    ys = ymin + np.arange(output_size) * dy
    mask = np.zeros((output_size, output_size), dtype=np.float32)
    radius = resolution / 2
    for x, y in cells:
        ox = np.maximum(0, np.minimum(xs + dx, x + radius) - np.maximum(xs, x - radius)) / dx
        oy = np.maximum(0, np.minimum(ys + dy, y + radius) - np.maximum(ys, y - radius)) / dy
        mask += np.outer(oy, ox).astype(np.float32)
    return np.clip(mask, 0, 1)


class GoalClearanceScorer:
    """Reuse trained weights and live scene channels; explicitly paint the goal area."""

    def __init__(self, scorer, target: GoalClearanceTarget):
        self.scorer = scorer
        self.target = target

    def score_state(self, env, target_object, robot_goal, xml_file,
                    region_samples=None, h=1, raw=False):
        """Score a clearance task without BFS-seeding an occupied target cell."""
        ctx, meta = self.scorer.render_ctx(env, target_object, robot_goal, xml_file, region_samples)
        # Channel order belongs to LiveScorer: static, movable, target, robot region, goal region.
        ctx[-1] = goal_mask(self.target.goal_cells, self.target.resolution,
                            meta["local_bounds"], ctx.shape[-1])
        contacts = self.scorer.contact_px_live(env, target_object)
        return self.scorer.score_ctx(ctx, contacts, h=h, raw=raw)
