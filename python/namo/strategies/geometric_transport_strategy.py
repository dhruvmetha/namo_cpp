"""Geometric transport heuristic goal selection strategy.

This module provides goal generation that prioritizes primitives based on
geometric analysis of whether they create openings for the robot to reach
its goal.

Priority levels (sorted by depth first, then priority within same depth):
OPENINGS (clean -> movable -> static):
- Priority 1: No collision, creates opening
- Priority 2: Movable collision, creates opening
- Priority 3: Static collision, creates opening
NO OPENINGS (clean -> movable -> static):
- Priority 4: No collision, no opening
- Priority 5: Movable collision, no opening
- Priority 6: Static collision, no opening
"""

from typing import Any, Dict, List, Optional, Tuple
from collections import Counter
import time

import namo_rl
from .goal_selection_strategy import GoalSelectionStrategy, Goal
from .primitive_goal_strategy import PrimitiveGoalStrategy


class GeometricTransportStrategy(GoalSelectionStrategy):
    """Goal selection using geometric transport heuristic.

    This strategy wraps PrimitiveGoalStrategy and adds priority scoring
    based on whether each primitive target position:
    1. TBD

    The C++ implementation performs:
    - One BFS to find path from robot to goal (with object removed)
    - Geometric checks for each primitive to determine if it blocks this path
    """

    def __init__(
        self,
        primitive_data_dir: str = "data",
        verbose: bool = False,
        profile: bool = False,
    ):
        """Initialize geometric transport strategy.

        Args:
            primitive_data_dir: Directory containing motion primitive .dat files
            verbose: Enable verbose output showing priority distribution
            profile: Collect per-call timing breakdown for analysis
        """
        self._primitive_strategy = PrimitiveGoalStrategy(
            data_dir=primitive_data_dir, verbose=False)
        self.verbose = verbose
        self.profile = profile
        self._profile_state: Dict[str, Any] = {}
        self.reset_profile()

    def reset_profile(self) -> None:
        """Reset accumulated profiling state for the current (env, object) attempt."""
        self._profile_state = {
            "calls": 0,
            "reachable_edges": set(),
            "target_poses_total": 0,
            "priorities_counts": Counter(),
            "timing_ms": Counter(),
            "last_priority_profile_ms": None,
        }

    def get_profile(self) -> Optional[Dict[str, Any]]:
        """Return the current profiling state (safe for pickling), or None if disabled."""
        if not self.profile:
            return None

        timing_ms = dict(self._profile_state.get("timing_ms", {}))
        priorities_counts = dict(self._profile_state.get("priorities_counts", {}))
        reachable_edges = sorted(list(self._profile_state.get("reachable_edges", set())))
        last_priority_profile_ms = self._profile_state.get("last_priority_profile_ms", None)

        return {
            "calls": int(self._profile_state.get("calls", 0)),
            "reachable_edges": reachable_edges,
            "reachable_edges_count": len(reachable_edges),
            "target_poses_total": int(self._profile_state.get("target_poses_total", 0)),
            "priorities_counts": priorities_counts,
            "timing_ms": timing_ms,
            "evaluate_primitive_priorities_profile_ms": last_priority_profile_ms,
        }

    def generate_goals(self,
                      object_id: str,
                      state: namo_rl.RLState,
                      env: namo_rl.RLEnvironment,
                      max_goals: int,
                      region_goals_sampled: Optional[List[Tuple[float, float, float]]] = None
                      ) -> List[List[Goal]]:
        """Generate goals with geometric transport priority scoring.

        Args:
            object_id: Object to generate goals for
            state: Current environment state
            env: Environment instance
            max_goals: Unused (returns all primitives)
            region_goals_sampled: Unused by this strategy (accepted for API compatibility)

        Returns:
            List of goal lists per edge, with score field set to priority
            (higher score = higher priority, i.e., 7 - priority_level)
        """
        def _tick() -> float:
            return time.perf_counter()

        def _add_ms(key: str, start: float, end: float) -> None:
            if self.profile:
                self._profile_state["timing_ms"][key] += (end - start) * 1000.0

        t_total0 = _tick()
        original_state = env.get_full_state()

        try:
            t0 = _tick()
            env.set_full_state(state)
            _add_ms("set_full_state_ms", t0, _tick())

            # 1. Get all primitives (60 edges x 10 depths = 600 goals)
            t0 = _tick()
            goals_per_edge = self._primitive_strategy.generate_goals(
                object_id, state, env, max_goals)
            _add_ms("primitive_generate_goals_ms", t0, _tick())

            if not goals_per_edge:
                _add_ms("total_ms", t_total0, _tick())
                return []

            # 2. Get reachable edges
            t0 = _tick()
            reachable_edges = set(env.get_reachable_edges(object_id))
            _add_ms("get_reachable_edges_ms", t0, _tick())
            if self.profile:
                self._profile_state["reachable_edges"].update(reachable_edges)

            # 3. Collect target poses for batch evaluation
            t0 = _tick()
            target_poses = []
            pose_to_slot = {}

            for edge_idx in reachable_edges:
                if edge_idx >= len(goals_per_edge):
                    continue
                for depth_idx, goal in enumerate(goals_per_edge[edge_idx]):
                    if goal is not None:
                        pose_idx = len(target_poses)
                        target_poses.append([goal.x, goal.y, goal.theta])
                        pose_to_slot[pose_idx] = (edge_idx, depth_idx)
            _add_ms("collect_target_poses_ms", t0, _tick())

            if not target_poses:
                if self.verbose:
                    print(f"No reachable edges for {object_id}")
                _add_ms("total_ms", t_total0, _tick())
                return goals_per_edge

            # 4. Get robot goal
            t0 = _tick()
            robot_goal = env.get_robot_goal()[:2]
            _add_ms("get_robot_goal_ms", t0, _tick())

            # 5. Batch evaluate priorities in C++
            t0 = _tick()
            priorities = env.evaluate_primitive_priorities(
                object_id, target_poses, robot_goal)
            _add_ms("evaluate_primitive_priorities_ms", t0, _tick())
            if self.profile and hasattr(env, "get_last_priority_profile"):
                try:
                    self._profile_state["last_priority_profile_ms"] = dict(env.get_last_priority_profile())
                except Exception:
                    self._profile_state["last_priority_profile_ms"] = None

            # 6. Update goals with priority scores
            # Score: priority 1 -> 6, priority 6 -> 1 (higher = better for sorting)
            t0 = _tick()
            for pose_idx, priority in enumerate(priorities):
                edge_idx, depth_idx = pose_to_slot[pose_idx]
                goal = goals_per_edge[edge_idx][depth_idx]
                goals_per_edge[edge_idx][depth_idx] = Goal(
                    x=goal.x,
                    y=goal.y,
                    theta=goal.theta,
                    score=float(7 - priority),
                    edge_idx=getattr(goal, "edge_idx", edge_idx),
                    depth=getattr(goal, "depth", depth_idx),
                )
            _add_ms("update_goals_ms", t0, _tick())

            if self.profile:
                self._profile_state["calls"] += 1
                self._profile_state["target_poses_total"] += len(target_poses)
                self._profile_state["priorities_counts"].update(priorities)

            if self.verbose:
                counts = Counter(priorities)
                print(f"Geometric transport for {object_id}:")
                print(f"   P1 (clean+opening): {counts.get(1, 0)}")
                print(f"   P2 (movable+opening): {counts.get(2, 0)}")
                print(f"   P3 (static+opening): {counts.get(3, 0)}")
                print(f"   P4 (clean+no opening): {counts.get(4, 0)}")
                print(f"   P5 (movable+no opening): {counts.get(5, 0)}")
                print(f"   P6 (static+no opening): {counts.get(6, 0)}")

            _add_ms("total_ms", t_total0, _tick())
            return goals_per_edge

        finally:
            t0 = _tick()
            env.set_full_state(original_state)
            _add_ms("restore_full_state_ms", t0, _tick())

    @property
    def strategy_name(self) -> str:
        """Return human-readable name of this strategy."""
        return "Geometric Transport Goal Generation"
