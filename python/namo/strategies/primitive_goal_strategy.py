"""Primitive-based goal selection strategy for NAMO planning.

This module provides goal generation using precomputed motion primitives
from binary database files. Primitives are shape-specific (square/tall/wide)
and organized by edge points and push steps.
"""

import struct
import os
from dataclasses import dataclass
from typing import List, Dict, Optional
from abc import ABC

import namo_rl
from .goal_selection_strategy import GoalSelectionStrategy, Goal


@dataclass
class Primitive:
    """Motion primitive representation."""
    delta_x: float
    delta_y: float
    delta_theta: float
    edge_idx: int  # 0-59 (4 edges × 15 points)
    push_steps: int  # 1-10


class MotionPrimitiveLoader:
    """Loader for binary motion primitive database files."""

    @staticmethod
    def load_primitives(filepath: str) -> List[Primitive]:
        """Load motion primitives from binary .dat file.

        Binary format:
        - Header: 4 bytes (uint32) = primitive count
        - Each primitive: 14 bytes
          - delta_x: 4 bytes (float)
          - delta_y: 4 bytes (float)
          - delta_theta: 4 bytes (float)
          - edge_idx: 1 byte (uint8)
          - push_steps: 1 byte (uint8)

        Args:
            filepath: Path to .dat file

        Returns:
            List of Primitive objects

        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file format is invalid
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Primitive file not found: {filepath}")

        primitives = []

        with open(filepath, 'rb') as f:
            # Read header (primitive count)
            count_bytes = f.read(4)
            if len(count_bytes) < 4:
                raise ValueError(f"File too short to contain valid header: {filepath}")

            count = struct.unpack('I', count_bytes)[0]  # uint32_t

            # Read primitives
            for i in range(count):
                data = f.read(14)
                if len(data) < 14:
                    raise ValueError(
                        f"Incomplete primitive {i} in {filepath}: "
                        f"expected 14 bytes, got {len(data)}"
                    )

                # Unpack: delta_x, delta_y, delta_theta, edge_idx, push_steps
                delta_x, delta_y, delta_theta, edge_idx, push_steps = struct.unpack('fffBB', data)

                primitives.append(Primitive(
                    delta_x=delta_x,
                    delta_y=delta_y,
                    delta_theta=delta_theta,
                    edge_idx=edge_idx,
                    push_steps=push_steps
                ))

        return primitives


class PrimitiveGoalStrategy(GoalSelectionStrategy):
    """Goal selection strategy using precomputed motion primitives.

    This strategy loads shape-specific motion primitives (square/tall/wide)
    and returns them grouped by edge point, sorted by push steps.

    Returns goals in format: List[List[Goal]] where:
    - Outer list (60 items): one per edge point
    - Inner list (10 items): push steps 1-10 for that edge point
    """

    def __init__(self, data_dir: str = "data", verbose: bool = False):
        """Initialize primitive goal strategy.

        Args:
            data_dir: Directory containing motion_primitives_15_*.dat files
            verbose: Enable verbose output
        """
        self.data_dir = data_dir
        self.verbose = verbose
        self._primitive_cache: Dict[str, List[Primitive]] = {}

    def generate_goals(self,
                      object_id: str,
                      state: namo_rl.RLState,
                      env: namo_rl.RLEnvironment,
                      max_goals: int) -> List[List[Goal]]:
        """Generate primitive-based goals for object.

        Args:
            object_id: Object to generate goals for
            state: Current environment state
            env: Environment instance
            max_goals: Unused (returns all primitives)

        Returns:
            List of 60 goal lists (one per edge point),
            each containing 10 goals (one per push step 1-10)
        """
        # Save and set state to get object pose
        original_state = env.get_full_state()

        try:
            env.set_full_state(state)
            obs = env.get_observation()

            # Get object current pose
            pose_key = f"{object_id}_pose"
            if pose_key not in obs:
                if self.verbose:
                    print(f"Warning: Object {object_id} not found in observation")
                return []

            obj_pose = obs[pose_key]
            obj_x, obj_y, obj_theta = obj_pose[0], obj_pose[1], obj_pose[2]

            # Select primitive file based on object shape
            primitive_file = self._select_primitive_file(object_id, env)

            # Load primitives (use cache)
            if primitive_file not in self._primitive_cache:
                filepath = os.path.join(self.data_dir, primitive_file)
                if self.verbose:
                    print(f"Loading primitives from {filepath}")
                self._primitive_cache[primitive_file] = MotionPrimitiveLoader.load_primitives(filepath)

            primitives = self._primitive_cache[primitive_file]

            # Group primitives by edge_idx
            edge_groups = self._group_by_edge(primitives)

            # Convert to absolute world coordinates
            goals_per_edge = []
            for edge_idx in sorted(edge_groups.keys()):
                edge_primitives = edge_groups[edge_idx]

                # Sort by push_steps
                edge_primitives.sort(key=lambda p: p.push_steps)

                # Convert to absolute goals
                edge_goals = []
                for primitive in edge_primitives:
                    goal = Goal(
                        x=obj_x + primitive.delta_x,
                        y=obj_y + primitive.delta_y,
                        theta=obj_theta + primitive.delta_theta
                    )
                    edge_goals.append(goal)

                goals_per_edge.append(edge_goals)

            if self.verbose:
                print(f"Generated {len(goals_per_edge)} edge groups with "
                      f"{len(goals_per_edge[0]) if goals_per_edge else 0} goals each")

            return goals_per_edge

        finally:
            # Restore original state
            env.set_full_state(original_state)

    def _select_primitive_file(self, object_name: str, env: namo_rl.RLEnvironment) -> str:
        """Select appropriate primitive file based on object shape.

        Uses same logic as C++ NAMOPushSkill:
        - ratio < 1.05: square
        - x > y: wide
        - y > x: tall

        Args:
            object_name: Name of object
            env: Environment instance

        Returns:
            Filename of primitive database
        """
        # Get object dimensions
        object_info = env.get_object_info()

        if object_name not in object_info:
            if self.verbose:
                print(f"Object {object_name} not in object_info, defaulting to square")
            return "motion_primitives_15_square.dat"

        info = object_info[object_name]

        # Get width and height (size is [x, y, z])
        if 'width' in info and 'height' in info:
            x = info['width']
            y = info['height']
        elif 'size' in info:
            x = info['size'][0] if len(info['size']) > 0 else 0.0
            y = info['size'][1] if len(info['size']) > 1 else 0.0
        else:
            if self.verbose:
                print(f"Could not get dimensions for {object_name}, defaulting to square")
            return "motion_primitives_15_square.dat"

        if x <= 0.0 or y <= 0.0:
            if self.verbose:
                print(f"Invalid dimensions for {object_name}: [{x}×{y}], defaulting to square")
            return "motion_primitives_15_square.dat"

        # Calculate aspect ratio
        ratio = max(x, y) / min(x, y)

        if ratio < 1.05:
            # Square: nearly equal dimensions
            if self.verbose:
                print(f"Object {object_name} [{x:.3f}×{y:.3f}] ratio={ratio:.3f} → square")
            return "motion_primitives_15_square.dat"
        elif x > y:
            # Wide: width > height
            if self.verbose:
                print(f"Object {object_name} [{x:.3f}×{y:.3f}] ratio={ratio:.3f} → wide")
            return "motion_primitives_15_wide.dat"
        else:
            # Tall: height > width
            if self.verbose:
                print(f"Object {object_name} [{x:.3f}×{y:.3f}] ratio={ratio:.3f} → tall")
            return "motion_primitives_15_tall.dat"

    def _group_by_edge(self, primitives: List[Primitive]) -> Dict[int, List[Primitive]]:
        """Group primitives by edge index.

        Args:
            primitives: List of all primitives

        Returns:
            Dictionary mapping edge_idx (0-59) to list of primitives
        """
        edge_groups: Dict[int, List[Primitive]] = {}

        for primitive in primitives:
            edge_idx = primitive.edge_idx
            if edge_idx not in edge_groups:
                edge_groups[edge_idx] = []
            edge_groups[edge_idx].append(primitive)

        return edge_groups

    @property
    def strategy_name(self) -> str:
        """Return human-readable name of this strategy."""
        return "Primitive-Based Goal Generation"
