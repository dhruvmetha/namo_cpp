"""Primitive-based goal selection strategy for NAMO planning.

This module provides goal generation using precomputed motion primitives
from binary database files. Primitives are shape-specific (square/tall/wide)
and organized by edge points and push steps.
"""

import struct
import os
import math
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
from abc import ABC

import namo_rl
from .goal_selection_strategy import GoalSelectionStrategy, Goal
from .ml_strategies import MLGoalSelectionStrategy


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
                # if self.verbose:
                #     print(f"Loading primitives from {filepath}")
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
                # Transform primitive deltas through object's current orientation
                # Follows C++ implementation in greedy_planner.cpp:148-164
                edge_goals = []
                cos_theta = math.cos(obj_theta)
                sin_theta = math.sin(obj_theta)

                for primitive in edge_primitives:
                    dx = primitive.delta_x
                    dy = primitive.delta_y

                    goal = Goal(
                        x=obj_x + dx * cos_theta - dy * sin_theta,
                        y=obj_y + dx * sin_theta + dy * cos_theta,
                        theta=obj_theta + primitive.delta_theta
                    )
                    edge_goals.append(goal)

                goals_per_edge.append(edge_goals)

            # if self.verbose:
            #     print(f"Generated {len(goals_per_edge)} edge groups with "
            #           f"{len(goals_per_edge[0]) if goals_per_edge else 0} goals each")

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

        # Get width and height from object_info (uses size_x, size_y, size_z keys)
        if 'size_x' in info and 'size_y' in info:
            x = info['size_x']
            y = info['size_y']
        elif 'width' in info and 'height' in info:
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
            # if self.verbose:
            #     print(f"Object {object_name} [{x:.3f}×{y:.3f}] ratio={ratio:.3f} → square")
            return "motion_primitives_15_square.dat"
        elif x > y:
            # Wide: width > height
            # if self.verbose:
            #     print(f"Object {object_name} [{x:.3f}×{y:.3f}] ratio={ratio:.3f} → wide")
            return "motion_primitives_15_wide.dat"
        else:
            # Tall: height > width
            # if self.verbose:
            #     print(f"Object {object_name} [{x:.3f}×{y:.3f}] ratio={ratio:.3f} → tall")
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


class MLPrimitiveGoalStrategy(GoalSelectionStrategy):
    """Align diffusion goal samples with discrete primitive slots."""

    def __init__(
        self,
        goal_model_path: str,
        primitive_data_dir: str = "data",
        samples: int = 32,
        device: str = "cuda",
        match_position_tolerance: float = 0.05,
        match_angle_tolerance: float = 0.1,
        angle_weight: float = 0.5,
        max_matches: int = 8,
        verbose: bool = False,
        min_goals_threshold: int = 1,
        xml_path: str = None,
        preview_mask_count: int = 0,
        preloaded_model = None,
    ):
        """
        Args:
            goal_model_path: Path to Hydra output directory that contains a trained diffusion model.
            primitive_data_dir: Directory with primitive lookup files.
            samples: Number of diffusion samples to request per inference.
            device: Torch device for the loaded model.
            match_position_tolerance: Maximum positional error (meters) allowed between ML goal and primitive. Default: 0.05m.
            match_angle_tolerance: Maximum angular error (radians) allowed between ML goal and primitive. Default: 0.1 rad (~5.7°).
            angle_weight: Weight used when ranking candidate slots by angular error.
            max_matches: Maximum number of ML goals to align per call.
            verbose: Enable debug output.
            min_goals_threshold: Minimum ML goals required before accepting the inference result.
            preview_mask_count: Number of ML goal masks to preview (0 disables).
            preloaded_model: Optional preloaded GoalInferenceModel to avoid reloading.
        """
        self.verbose = verbose
        self.max_matches = max_matches
        self.match_position_tolerance = match_position_tolerance
        self.match_angle_tolerance = match_angle_tolerance
        self.angle_weight = angle_weight

        self._primitive_strategy = PrimitiveGoalStrategy(
            data_dir=primitive_data_dir,
            verbose=verbose
        )
        self._ml_strategy = MLGoalSelectionStrategy(
            goal_model_path=goal_model_path,
            samples=samples,
            device=device,
            min_goals_threshold=min_goals_threshold,
            verbose=verbose,
            xml_path=xml_path,
            preview_mask_count=preview_mask_count,
            preloaded_model=preloaded_model
        )
        self._default_ml_samples = samples

    def generate_goals(
        self,
        object_id: str,
        state: namo_rl.RLState,
        env: namo_rl.RLEnvironment,
        max_goals: int
    ) -> List[List[Goal]]:
        primitive_goals = self._primitive_strategy.generate_goals(
            object_id,
            state,
            env,
            max_goals
        )

        if not primitive_goals:
            return []

        max_depth = len(primitive_goals[0]) if primitive_goals and primitive_goals[0] else 0
        aligned_goals: List[List[Optional[Goal]]] = [
            [None for _ in range(max_depth)]
            for _ in range(len(primitive_goals))
        ]

        ml_goal_budget = max_goals if max_goals > 0 else self._default_ml_samples
        ml_goals = self._ml_strategy.generate_goals(
            object_id,
            state,
            env,
            ml_goal_budget
        )

        print(f"🎯 ML-Primitive Alignment for {object_id}:")
        print(f"  Primitive slots: {len(primitive_goals)} edges × {max_depth} depths = {len(primitive_goals) * max_depth} total")
        print(f"  ML goals received: {len(ml_goals)}")
        print(f"  Max matches allowed: {self.max_matches}")
        print(f"  Position tolerance: {self.match_position_tolerance}m, Angle tolerance: {self.match_angle_tolerance} rad")

        if not ml_goals:
            print(f"  ⚠️ No ML goals - returning empty aligned structure")
            return aligned_goals

        slot_metadata = self._build_slot_metadata(primitive_goals)
        used_slots = set()
        matches = 0
        skipped_due_to_tolerance = 0

        for ml_goal_idx, ml_goal in enumerate(ml_goals):
            if matches >= self.max_matches:
                print(f"  ⚠️  Stopped at {matches} matches (reached max_matches limit)")
                break

            best_slot = None
            best_score = None
            candidates_checked = 0
            candidates_within_tolerance = 0

            for slot_id, (edge_idx, depth_idx, primitive_goal) in enumerate(slot_metadata):
                if slot_id in used_slots:
                    continue

                candidates_checked += 1
                pos_err, ang_err = self._goal_error(primitive_goal, ml_goal)

                if pos_err > self.match_position_tolerance or ang_err > self.match_angle_tolerance:
                    continue

                candidates_within_tolerance += 1
                score = pos_err + self.angle_weight * ang_err
                if best_score is None or score < best_score:
                    best_score = score
                    best_slot = (slot_id, edge_idx, depth_idx)

            if best_slot is None:
                skipped_due_to_tolerance += 1
                if self.verbose and ml_goal_idx < 5:  # Show first 5 skipped goals
                    print(f"    ⊗ ML goal {ml_goal_idx}: ({ml_goal.x:.3f}, {ml_goal.y:.3f}, {ml_goal.theta:.3f}) - No slot within tolerance (checked {candidates_checked} slots)")
                continue

            slot_id, edge_idx, depth_idx = best_slot
            aligned_goals[edge_idx][depth_idx] = Goal(
                x=ml_goal.x,
                y=ml_goal.y,
                theta=ml_goal.theta
            )
            used_slots.add(slot_id)
            matches += 1

            if self.verbose and matches <= 10:  # Show first 10 matches
                print(f"    ✓ ML goal {ml_goal_idx}: ({ml_goal.x:.3f}, {ml_goal.y:.3f}, {ml_goal.theta:.3f}) → edge {edge_idx}, depth {depth_idx+1} (score: {best_score:.4f}, checked {candidates_within_tolerance} candidates)")

        print(f"  ✅ Aligned {matches}/{len(ml_goals)} ML goals to primitive slots")
        print(f"     Skipped due to tolerance: {skipped_due_to_tolerance}")

        if matches == 0:
            print(f"  ⚠️ WARNING: NO ML goals matched any primitive slots!")
            print(f"     Position tolerance: {self.match_position_tolerance}m, Angle tolerance: {self.match_angle_tolerance} rad")
        else:
            # Show which edges/depths got ML goals
            aligned_edges = set()
            edge_depth_counts = {}
            for edge_idx, edge_goals in enumerate(aligned_goals):
                for depth_idx, goal in enumerate(edge_goals):
                    if goal is not None:
                        aligned_edges.add(edge_idx)
                        if edge_idx not in edge_depth_counts:
                            edge_depth_counts[edge_idx] = []
                        edge_depth_counts[edge_idx].append(depth_idx + 1)

            if aligned_edges:
                sorted_edges = sorted(list(aligned_edges))
                print(f"     Aligned to edges: {sorted_edges}")
                if self.verbose:
                    print(f"     Edge → Depth mapping:")
                    for edge_idx in sorted(edge_depth_counts.keys())[:10]:  # Show first 10
                        depths = sorted(edge_depth_counts[edge_idx])
                        print(f"       Edge {edge_idx}: depths {depths}")

        return aligned_goals

    def _build_slot_metadata(self, primitive_goals: List[List[Goal]]) -> List[Tuple[int, int, Goal]]:
        slots: List[Tuple[int, int, Goal]] = []
        for edge_idx, edge_goals in enumerate(primitive_goals):
            for depth_idx, goal in enumerate(edge_goals):
                slots.append((edge_idx, depth_idx, goal))
        return slots

    @staticmethod
    def _goal_error(primitive_goal: Goal, ml_goal: Goal) -> Tuple[float, float]:
        pos_err = math.hypot(
            primitive_goal.x - ml_goal.x,
            primitive_goal.y - ml_goal.y
        )
        ang_err = abs(MLPrimitiveGoalStrategy._wrap_angle(primitive_goal.theta - ml_goal.theta))
        return pos_err, ang_err

    @staticmethod
    def _wrap_angle(theta: float) -> float:
        while theta > math.pi:
            theta -= 2 * math.pi
        while theta < -math.pi:
            theta += 2 * math.pi
        return theta

    @property
    def strategy_name(self) -> str:
        return "ML Primitive Aligned Goal Generation"
