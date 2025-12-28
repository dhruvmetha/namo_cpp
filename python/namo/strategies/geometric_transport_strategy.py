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

from typing import List, Optional
from collections import Counter

import namo_rl
from .goal_selection_strategy import GoalSelectionStrategy, Goal
from .primitive_goal_strategy import PrimitiveGoalStrategy


class GeometricTransportStrategy(GoalSelectionStrategy):
    """Goal selection using geometric transport heuristic.

    This strategy wraps PrimitiveGoalStrategy and adds priority scoring
    based on whether each primitive target position:
    1. Collides with static obstacles (Priority 4)
    2. Creates an opening for the robot (Priority 1 or 2)
    3. Blocks the path to the goal (Priority 3)

    The C++ implementation performs:
    - One BFS to find path from robot to goal (with object removed)
    - Geometric checks for each primitive to determine if it blocks this path
    """

    def __init__(self, primitive_data_dir: str = "data", verbose: bool = False):
        """Initialize geometric transport strategy.

        Args:
            primitive_data_dir: Directory containing motion primitive .dat files
            verbose: Enable verbose output showing priority distribution
        """
        self._primitive_strategy = PrimitiveGoalStrategy(
            data_dir=primitive_data_dir, verbose=False)
        self.verbose = verbose

    def generate_goals(self,
                      object_id: str,
                      state: namo_rl.RLState,
                      env: namo_rl.RLEnvironment,
                      max_goals: int) -> List[List[Goal]]:
        """Generate goals with geometric transport priority scoring.

        Args:
            object_id: Object to generate goals for
            state: Current environment state
            env: Environment instance
            max_goals: Unused (returns all primitives)

        Returns:
            List of goal lists per edge, with score field set to priority
            (higher score = higher priority, i.e., 5 - priority_level)
        """
        original_state = env.get_full_state()

        try:
            env.set_full_state(state)

            # 1. Get all primitives (60 edges x 10 depths = 600 goals)
            goals_per_edge = self._primitive_strategy.generate_goals(
                object_id, state, env, max_goals)

            if not goals_per_edge:
                return []

            # 2. Get reachable edges
            reachable_edges = set(env.get_reachable_edges(object_id))

            # 3. Collect target poses for batch evaluation
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

            if not target_poses:
                if self.verbose:
                    print(f"No reachable edges for {object_id}")
                return goals_per_edge

            # 4. Get robot goal
            robot_goal = env.get_robot_goal()[:2]

            # 5. Batch evaluate priorities in C++
            priorities = env.evaluate_primitive_priorities(
                object_id, target_poses, robot_goal)

            # 6. Update goals with priority scores
            # Score: priority 1 -> 6, priority 6 -> 1 (higher = better for sorting)
            for pose_idx, priority in enumerate(priorities):
                edge_idx, depth_idx = pose_to_slot[pose_idx]
                goal = goals_per_edge[edge_idx][depth_idx]
                goals_per_edge[edge_idx][depth_idx] = Goal(
                    x=goal.x,
                    y=goal.y,
                    theta=goal.theta,
                    score=float(7 - priority)
                )

            if self.verbose:
                counts = Counter(priorities)
                print(f"Geometric transport for {object_id}:")
                print(f"   P1 (clean+opening): {counts.get(1, 0)}")
                print(f"   P2 (movable+opening): {counts.get(2, 0)}")
                print(f"   P3 (static+opening): {counts.get(3, 0)}")
                print(f"   P4 (clean+no opening): {counts.get(4, 0)}")
                print(f"   P5 (movable+no opening): {counts.get(5, 0)}")
                print(f"   P6 (static+no opening): {counts.get(6, 0)}")

            return goals_per_edge

        finally:
            env.set_full_state(original_state)

    @property
    def strategy_name(self) -> str:
        """Return human-readable name of this strategy."""
        return "Geometric Transport Goal Generation"
