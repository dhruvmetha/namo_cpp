"""Primitive-based goal selection strategy for NAMO planning.

This module provides goal generation using precomputed motion primitives
from binary database files. Primitives are shape-specific (square/tall/wide)
and organized by edge points and push steps.
"""

import struct
import os
import math
import random
import threading
from concurrent.futures import ThreadPoolExecutor, Future
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Union, Any, TYPE_CHECKING, Sequence
from collections import defaultdict
from abc import ABC

import namo_rl
from .goal_selection_strategy import GoalSelectionStrategy, Goal
from .ml_strategies import MLGoalSelectionStrategy

if TYPE_CHECKING:
    from .primitive_goal_strategy import MLPrimitiveAsyncStrategy


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

    def __init__(self, data_dir: str = "data", verbose: bool = False,
                 shuffle_edges: bool = False, seed: int = None):
        """Initialize primitive goal strategy.

        Args:
            data_dir: Directory containing motion_primitives_15_*.dat files
            verbose: Enable verbose output
            shuffle_edges: If True, randomize edge ordering (useful for averaging difficulty)
            seed: Random seed for reproducible shuffling (None = random each call)
        """
        self.data_dir = data_dir
        self.verbose = verbose
        self.shuffle_edges = shuffle_edges
        self.seed = seed
        self._rng = random.Random(seed) if seed is not None else None
        self._primitive_cache: Dict[str, List[Primitive]] = {}
        self._last_edge_ordering: List[int] = []  # Track ordering for analysis

    def reseed(self, seed: int):
        """Reseed the RNG for a new shuffle. Use for running multiple trials.

        Args:
            seed: New random seed
        """
        self.seed = seed
        self._rng = random.Random(seed)

    def get_last_edge_ordering(self) -> List[int]:
        """Return the edge ordering used in the last generate_goals call.

        Useful for analyzing which ordering led to success/failure.
        """
        return self._last_edge_ordering.copy()

    def generate_goals(self,
                      object_id: str,
                      state: namo_rl.RLState,
                      env: namo_rl.RLEnvironment,
                      max_goals: int,
                      region_goals_sampled: Optional[List[Tuple[float, float, float]]] = None) -> List[List[Goal]]:
        """Generate primitive-based goals for object.

        Args:
            object_id: Object to generate goals for
            state: Current environment state
            env: Environment instance
            max_goals: Unused (returns all primitives)
            region_goals_sampled: Unused by primitive strategy

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

            # Determine edge ordering (sorted or shuffled)
            edge_indices = sorted(edge_groups.keys())
            if self.shuffle_edges:
                if self._rng is not None:
                    # Seeded: reproducible shuffle
                    self._rng.shuffle(edge_indices)
                else:
                    # Unseeded: random each call
                    random.shuffle(edge_indices)

            # Store ordering for analysis
            self._last_edge_ordering = list(edge_indices)

            # Convert to absolute world coordinates
            goals_per_edge = []
            for edge_idx in edge_indices:
                edge_primitives = edge_groups[edge_idx]

                # Sort by push_steps
                edge_primitives.sort(key=lambda p: p.push_steps)

                # Convert to absolute goals
                # Transform primitive deltas through object's current orientation
                # Follows C++ implementation in greedy_planner.cpp:148-164
                edge_goals = []
                cos_theta = math.cos(obj_theta)
                sin_theta = math.sin(obj_theta)

                for depth_idx, primitive in enumerate(edge_primitives):
                    dx = primitive.delta_x
                    dy = primitive.delta_y

                    goal = Goal(
                        x=obj_x + dx * cos_theta - dy * sin_theta,
                        y=obj_y + dx * sin_theta + dy * cos_theta,
                        theta=obj_theta + primitive.delta_theta,
                        edge_idx=edge_idx,      # Actual primitive edge index (from edge_indices, not list position)
                        depth=depth_idx         # 0-indexed depth (depth=0 means push_steps=1)
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
        match_position_tolerance: float = 0.1,
        match_angle_tolerance: float = 0.1,
        angle_weight: float = 0.5,
        max_matches: int = 8,
        verbose: bool = False,
        min_goals_threshold: int = 1,
        xml_path: str = None,
        preview_mask_count: int = 0,
        preloaded_model = None,
        preview_aligned_primitives: bool = False,
        k_nearest: int = 1,
        seed: int = None,
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
            preview_aligned_primitives: If True, save visualization of aligned primitives.
            k_nearest: Number of nearest primitive slots to vote for per ML goal (within tolerance). Default: 1.
            seed: Random seed for diffusion noise (None = random each time).
        """
        self.verbose = verbose
        self.max_matches = max_matches
        self.match_position_tolerance = match_position_tolerance
        self.match_angle_tolerance = match_angle_tolerance
        self.angle_weight = angle_weight
        self.preview_aligned_primitives = preview_aligned_primitives
        self.k_nearest = k_nearest

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
            preloaded_model=preloaded_model,
            seed=seed
        )
        self._default_ml_samples = samples

        # Store last alignment result for visualization
        self._last_alignment_info = None

    def get_last_goal_stats(self) -> dict:
        """Return stats from the last generate_goals call for failure tracking.

        Returns:
            dict with:
                - ml_goals_generated: number of raw ML goals before alignment
                - ml_goals_aligned: number of unique primitive slots that got votes
                - reachable_edges_count: number of edges robot can reach (if available)
                - aligned_primitives: list of dicts with edge_idx, depth_idx, x, y, theta, votes
                - ml_goals_raw: list of dicts with x, y, theta for each ML goal
                - reachable_edges: list of reachable edge indices
        """
        if self._last_alignment_info is None:
            return {
                'ml_goals_generated': 0,
                'ml_goals_aligned': 0,
                'reachable_edges_count': 0,
                'aligned_primitives': [],
                'ml_goals_raw': [],
                'reachable_edges': [],
            }

        # Convert aligned primitives to serializable format
        aligned_primitives = []
        for p in self._last_alignment_info.get('aligned_primitives', []):
            goal = p.get('goal')
            aligned_primitives.append({
                'edge_idx': p.get('edge_idx'),
                'depth_idx': p.get('depth_idx'),
                'x': goal.x if goal else None,
                'y': goal.y if goal else None,
                'theta': goal.theta if goal else None,
                'votes': p.get('votes', 0),
            })

        # Convert ML goals to serializable format
        ml_goals_raw = []
        for g in self._last_alignment_info.get('ml_goals', []):
            ml_goals_raw.append({
                'x': g.x,
                'y': g.y,
                'theta': g.theta,
            })

        return {
            'ml_goals_generated': self._last_alignment_info.get('total_ml_goals', 0),
            'ml_goals_aligned': self._last_alignment_info.get('total_aligned', 0),
            'reachable_edges_count': len(self._last_alignment_info.get('reachable_edges', set())),
            'aligned_primitives': aligned_primitives,
            'ml_goals_raw': ml_goals_raw,
            'reachable_edges': sorted(list(self._last_alignment_info.get('reachable_edges', set()))),
        }

    def generate_goals(
        self,
        object_id: str,
        state: namo_rl.RLState,
        env: namo_rl.RLEnvironment,
        max_goals: int,
        region_goals_sampled: Optional[List[Tuple[float, float, float]]] = None
    ) -> List[List[Goal]]:
        primitive_goals = self._primitive_strategy.generate_goals(
            object_id,
            state,
            env,
            max_goals,
            region_goals_sampled
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
            ml_goal_budget,
            region_goals_sampled
        )

        if self.verbose:
            print(f"🎯 ML-Primitive Alignment for {object_id}:")
            print(f"  Primitive slots: {len(primitive_goals)} edges × {max_depth} depths = {len(primitive_goals) * max_depth} total")
            print(f"  ML goals received: {len(ml_goals)}")
            print(f"  Max matches allowed: {self.max_matches}")
            print(f"  Position tolerance: {self.match_position_tolerance}m, Angle tolerance: {self.match_angle_tolerance} rad")
            print(f"  K-nearest neighbors: {self.k_nearest}")

        if not ml_goals:
            if self.verbose:
                print(f"  ⚠️ No ML goals - returning empty aligned structure")
            return aligned_goals

        slot_metadata = self._build_slot_metadata(primitive_goals)
        slot_accumulators = defaultdict(lambda: {"x": 0.0, "y": 0.0, "sin": 0.0, "cos": 0.0, "count": 0})
        matches = 0
        skipped_due_to_tolerance = 0

        for ml_goal_idx, ml_goal in enumerate(ml_goals):
            # Collect all slots within tolerance with their scores
            candidates_within_tolerance = []

            for slot_id, (edge_idx, depth_idx, primitive_goal) in enumerate(slot_metadata):
                pos_err, ang_err = self._goal_error(primitive_goal, ml_goal)

                # Filter by tolerance - only consider slots within both thresholds
                if pos_err > self.match_position_tolerance or ang_err > self.match_angle_tolerance:
                    continue

                score = pos_err + self.angle_weight * ang_err
                candidates_within_tolerance.append((score, slot_id, edge_idx, depth_idx))

            if not candidates_within_tolerance:
                skipped_due_to_tolerance += 1
                if self.verbose and ml_goal_idx < 5:  # Show first 5 skipped goals
                    print(f"    ⊗ ML goal {ml_goal_idx}: ({ml_goal.x:.3f}, {ml_goal.y:.3f}, {ml_goal.theta:.3f}) - No slots within tolerance")
                continue

            # Sort by score (ascending) and take top-k
            candidates_within_tolerance.sort(key=lambda x: x[0])
            top_k_candidates = candidates_within_tolerance[:self.k_nearest]

            # Vote for each of the k-nearest slots
            for score, slot_id, edge_idx, depth_idx in top_k_candidates:
                acc = slot_accumulators[slot_id]
                acc["count"] += 1

                if "goal" not in acc:
                    # Retrieve the correct primitive goal from metadata using slot_id
                    _, _, correct_primitive_goal = slot_metadata[slot_id]
                    acc["goal"] = correct_primitive_goal

            matches += 1  # Count ML goals that had at least one match

        # Construct aligned goals from accumulators
        for slot_id, data in slot_accumulators.items():
            edge_idx, depth_idx, _ = slot_metadata[slot_id]
            count = data["count"]
            stored_goal = data["goal"]

            aligned_goals[edge_idx][depth_idx] = Goal(
                x=stored_goal.x,
                y=stored_goal.y,
                theta=stored_goal.theta,
                score=count,  # Store vote count as score
                edge_idx=edge_idx,   # Preserve edge index for C++ direct execution
                depth=depth_idx      # Preserve depth for C++ direct execution
            )
            matches += 1
            
            if self.verbose and matches <= 10:
                print(f"    ✓ Slot edge {edge_idx} depth {depth_idx+1}: {count} votes")

        if self.verbose or matches == 0:
            print(f"  ✅ Aligned {matches}/{len(ml_goals)} ML goals to primitive slots for {object_id}")
            if skipped_due_to_tolerance > 0:
                 print(f"     Skipped due to tolerance: {skipped_due_to_tolerance}")

        if matches == 0:
            print(f"  ⚠️ WARNING: NO ML goals matched any primitive slots!")
            print(f"     Position tolerance: {self.match_position_tolerance}m, Angle tolerance: {self.match_angle_tolerance} rad")
        else:
            # Show which edges/depths got ML goals only in verbose
            if self.verbose:
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

        # Store alignment info for visualization
        aligned_primitives_info = []
        for edge_idx, edge_goals in enumerate(aligned_goals):
            for depth_idx, goal in enumerate(edge_goals):
                if goal is not None:
                    aligned_primitives_info.append({
                        'edge_idx': edge_idx,
                        'depth_idx': depth_idx,
                        'goal': goal,
                        'votes': goal.score if hasattr(goal, 'score') else 1
                    })

        # Sort by votes (descending), then edge_idx and depth_idx (ascending) for deterministic ordering
        aligned_primitives_info.sort(key=lambda x: (-x['votes'], x['edge_idx'], x['depth_idx']))

        # Get reachable edges for visualization
        reachable_edges = set()
        try:
            env.set_full_state(state)
            reachable_edges = set(env.get_reachable_edges(object_id))
        except Exception as e:
            if self.verbose:
                print(f"  ⚠️ Could not get reachable edges: {e}")

        self._last_alignment_info = {
            'object_id': object_id,
            'object_pose': self._get_object_pose(state, env, object_id),
            'aligned_primitives': aligned_primitives_info,
            'ml_goals': ml_goals,
            'total_ml_goals': len(ml_goals),
            'total_aligned': len(aligned_primitives_info),
            'reachable_edges': reachable_edges
        }

        # Save visualization if enabled
        if self.preview_aligned_primitives and aligned_primitives_info:
            self._save_alignment_preview()

        return aligned_goals

    def _get_object_pose(self, state, env, object_id: str) -> Tuple[float, float, float]:
        """Get the current pose of an object."""
        original_state = env.get_full_state()
        try:
            env.set_full_state(state)
            obs = env.get_observation()
            pose_key = f"{object_id}_pose"
            if pose_key in obs:
                pose = obs[pose_key]
                return (pose[0], pose[1], pose[2])
            return (0.0, 0.0, 0.0)
        finally:
            env.set_full_state(original_state)

    def _save_alignment_preview(self):
        """Save visualization of aligned primitives."""
        import numpy as np

        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            from matplotlib.patches import FancyArrow, Rectangle
            from matplotlib.transforms import Affine2D
        except ImportError:
            print("   ⚠️ matplotlib not available for alignment preview")
            return

        info = self._last_alignment_info
        if not info or not info['aligned_primitives']:
            return

        object_id = info['object_id']
        obj_x, obj_y, obj_theta = info['object_pose']
        aligned = info['aligned_primitives']
        reachable_edges = info.get('reachable_edges', set())
        ml_goals = info.get('ml_goals', [])

        fig, ax = plt.subplots(1, 1, figsize=(12, 10))

        # Draw object at current position
        obj_size = 0.3  # Approximate object size for visualization
        rect = Rectangle(
            (obj_x - obj_size/2, obj_y - obj_size/2),
            obj_size, obj_size,
            angle=np.degrees(obj_theta),
            rotation_point='center',
            fill=True, facecolor='cyan', edgecolor='black', linewidth=2,
            label='Current Position'
        )
        ax.add_patch(rect)

        # Draw ML predicted goals first (dashed magenta boxes, in background)
        for i, ml_goal in enumerate(ml_goals):
            ml_rect = Rectangle(
                (ml_goal.x - obj_size/2, ml_goal.y - obj_size/2),
                obj_size, obj_size,
                angle=np.degrees(ml_goal.theta),
                rotation_point='center',
                fill=False, edgecolor='magenta', linewidth=1.5, linestyle='--', alpha=0.7
            )
            ax.add_patch(ml_rect)
            # Small label
            ax.text(ml_goal.x, ml_goal.y - obj_size/2 - 0.05, f'ML{i}', fontsize=6,
                   ha='center', va='top', color='magenta', alpha=0.8)

        # Separate aligned primitives by reachability
        reachable_aligned = [p for p in aligned if p['edge_idx'] in reachable_edges]
        unreachable_aligned = [p for p in aligned if p['edge_idx'] not in reachable_edges]

        # Color map for priority (higher votes = more red, lower = more blue)
        max_votes = max(p['votes'] for p in aligned) if aligned else 1

        # Draw UNREACHABLE primitives first (gray, in background)
        for prim_info in unreachable_aligned:
            goal = prim_info['goal']
            edge_idx = prim_info['edge_idx']
            depth_idx = prim_info['depth_idx']
            votes = prim_info['votes']

            # Gray for unreachable
            goal_rect = Rectangle(
                (goal.x - obj_size/2, goal.y - obj_size/2),
                obj_size, obj_size,
                angle=np.degrees(goal.theta),
                rotation_point='center',
                fill=True, facecolor='lightgray', edgecolor='gray', linewidth=1, alpha=0.4
            )
            ax.add_patch(goal_rect)

            # Add edge/depth info (smaller, grayed out)
            ax.text(goal.x, goal.y, f'E{edge_idx}', fontsize=6, ha='center', va='center',
                   color='gray', alpha=0.6)

        # Draw REACHABLE primitives with execution order (colored, in foreground)
        # Re-rank only reachable ones
        reachable_aligned_sorted = sorted(reachable_aligned, key=lambda x: x['votes'], reverse=True)

        for rank, prim_info in enumerate(reachable_aligned_sorted):
            goal = prim_info['goal']
            votes = prim_info['votes']
            edge_idx = prim_info['edge_idx']
            depth_idx = prim_info['depth_idx']

            # Color based on votes (normalized) - green to red
            cmap = plt.cm.RdYlGn_r
            color = cmap(votes / max_votes) if max_votes > 0 else 'blue'

            # Draw goal position as rectangle
            goal_rect = Rectangle(
                (goal.x - obj_size/2, goal.y - obj_size/2),
                obj_size, obj_size,
                angle=np.degrees(goal.theta),
                rotation_point='center',
                fill=True, facecolor=color, edgecolor='black', linewidth=1.5, alpha=0.8
            )
            ax.add_patch(goal_rect)

            # Draw arrow from current position to goal
            ax.annotate('', xy=(goal.x, goal.y), xytext=(obj_x, obj_y),
                       arrowprops=dict(arrowstyle='->', color=color, lw=1.5, alpha=0.6))

            # Add rank number at goal position (rank among REACHABLE only)
            ax.text(goal.x, goal.y, f'{rank+1}', fontsize=10, fontweight='bold',
                   ha='center', va='center', color='white',
                   bbox=dict(boxstyle='circle', facecolor='black', alpha=0.8))

            # Add edge/depth info near the goal
            ax.text(goal.x + obj_size/2 + 0.05, goal.y, f'E{edge_idx}D{depth_idx+1}\n({votes}v)',
                   fontsize=7, ha='left', va='center', alpha=0.9)

        # Set axis limits with padding (include ML goals)
        all_x = [obj_x] + [p['goal'].x for p in aligned] + [g.x for g in ml_goals]
        all_y = [obj_y] + [p['goal'].y for p in aligned] + [g.y for g in ml_goals]
        margin = 1.0
        ax.set_xlim(min(all_x) - margin, max(all_x) + margin)
        ax.set_ylim(min(all_y) - margin, max(all_y) + margin)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('World X (m)')
        ax.set_ylabel('World Y (m)')

        # Title with summary including reachability info
        reachable_count = len(reachable_aligned)
        unreachable_count = len(unreachable_aligned)
        ax.set_title(f'ML-Primitive Alignment: {object_id}\n'
                    f'{info["total_aligned"]} primitives from {info["total_ml_goals"]} ML goals | '
                    f'Reachable edges: {sorted(list(reachable_edges))}\n'
                    f'✓ {reachable_count} reachable (numbered) | ✗ {unreachable_count} unreachable (gray)',
                    fontsize=11, fontweight='bold')

        # Add legend
        legend_elements = [
            plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='cyan',
                      markersize=15, label='Current Position'),
            plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='none',
                      markersize=15, markeredgecolor='magenta', linestyle='--',
                      label=f'ML Predictions ({len(ml_goals)})'),
            plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='green',
                      markersize=15, markeredgecolor='black', label='Reachable (low votes)'),
            plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='red',
                      markersize=15, markeredgecolor='black', label='Reachable (high votes)'),
            plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='lightgray',
                      markersize=15, markeredgecolor='gray', label='Unreachable'),
        ]
        ax.legend(handles=legend_elements, loc='upper right')

        # Save
        save_path = os.path.join(os.getcwd(), f"ml_primitive_alignment_{object_id}.png")
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"   📁 Saved primitive alignment preview: {save_path}")
        plt.close(fig)

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


class MLPrimitiveFallbackStrategy(GoalSelectionStrategy):
    """ML-first goal selection with full primitive fallback.

    This strategy combines ML goal prediction with complete primitive coverage:
    1. Generate ALL primitives (60 edges × 10 depths = 600 goals)
    2. Run ML inference to get goal samples
    3. Align ML goals to primitives, accumulating votes
    4. Return FULL grid where:
       - ML-aligned slots have score = vote_count (tried first)
       - Non-ML slots have score = 0 (fallback, tried after ML goals)

    Execution order: sorted by (-score, depth) so high-confidence ML goals
    are tried first, then fallback primitives ordered by depth.
    """

    def __init__(
        self,
        goal_model_path: str,
        primitive_data_dir: str = "data",
        samples: int = 32,
        device: str = "cuda",
        match_position_tolerance: float = 0.1,
        match_angle_tolerance: float = 0.1,
        angle_weight: float = 0.5,
        max_matches: int = 8,
        verbose: bool = False,
        min_goals_threshold: int = 1,
        xml_path: str = None,
        preview_mask_count: int = 0,
        preloaded_model=None,
        preview_aligned_primitives: bool = False,
        k_nearest: int = 1,
        seed: int = None,
    ):
        """
        Args:
            goal_model_path: Path to Hydra output directory with trained diffusion model.
            primitive_data_dir: Directory with primitive lookup files.
            samples: Number of diffusion samples to request per inference.
            device: Torch device for the loaded model.
            match_position_tolerance: Max positional error (meters) for ML-to-primitive matching.
            match_angle_tolerance: Max angular error (radians) for ML-to-primitive matching.
            angle_weight: Weight for angular error when ranking candidate slots.
            max_matches: Maximum number of ML goals to align per call.
            verbose: Enable debug output.
            min_goals_threshold: Minimum ML goals required before accepting inference result.
            xml_path: XML file path for ML model context.
            preview_mask_count: Number of ML goal masks to preview (0 disables).
            preloaded_model: Optional preloaded GoalInferenceModel to avoid reloading.
            preview_aligned_primitives: If True, save visualization of aligned primitives.
            k_nearest: Number of nearest primitive slots to vote for per ML goal.
            seed: Random seed for diffusion noise (None = random each time).
        """
        self.verbose = verbose
        self.max_matches = max_matches
        self.match_position_tolerance = match_position_tolerance
        self.match_angle_tolerance = match_angle_tolerance
        self.angle_weight = angle_weight
        self.preview_aligned_primitives = preview_aligned_primitives
        self.k_nearest = k_nearest

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
            preloaded_model=preloaded_model,
            seed=seed
        )
        self._default_ml_samples = samples

        # Store last alignment result for stats/visualization
        self._last_alignment_info = None

    def get_last_goal_stats(self) -> dict:
        """Return stats from the last generate_goals call for failure tracking."""
        if self._last_alignment_info is None:
            return {
                'ml_goals_generated': 0,
                'ml_goals_aligned': 0,
                'reachable_edges_count': 0,
                'aligned_primitives': [],
                'ml_goals_raw': [],
                'reachable_edges': [],
                'fallback_primitives_count': 0,
            }

        aligned_primitives = []
        for p in self._last_alignment_info.get('aligned_primitives', []):
            goal = p.get('goal')
            aligned_primitives.append({
                'edge_idx': p.get('edge_idx'),
                'depth_idx': p.get('depth_idx'),
                'x': goal.x if goal else None,
                'y': goal.y if goal else None,
                'theta': goal.theta if goal else None,
                'votes': p.get('votes', 0),
            })

        ml_goals_raw = []
        for g in self._last_alignment_info.get('ml_goals', []):
            ml_goals_raw.append({
                'x': g.x,
                'y': g.y,
                'theta': g.theta,
            })

        return {
            'ml_goals_generated': self._last_alignment_info.get('total_ml_goals', 0),
            'ml_goals_aligned': self._last_alignment_info.get('total_aligned', 0),
            'reachable_edges_count': len(self._last_alignment_info.get('reachable_edges', set())),
            'aligned_primitives': aligned_primitives,
            'ml_goals_raw': ml_goals_raw,
            'reachable_edges': sorted(list(self._last_alignment_info.get('reachable_edges', set()))),
            'fallback_primitives_count': self._last_alignment_info.get('fallback_count', 0),
        }

    def generate_goals(
        self,
        object_id: str,
        state: namo_rl.RLState,
        env: namo_rl.RLEnvironment,
        max_goals: int,
        region_goals_sampled: Optional[List[Tuple[float, float, float]]] = None
    ) -> List[List[Goal]]:
        """Generate goals with ML prioritization and full primitive fallback.

        Returns a full grid (60 edges × 10 depths) where:
        - ML-aligned slots have score = vote_count (positive)
        - Non-ML slots have score = 0 (fallback)
        """
        # Phase 1: Generate ALL primitives
        primitive_goals = self._primitive_strategy.generate_goals(
            object_id, state, env, max_goals, region_goals_sampled
        )

        if not primitive_goals:
            return []

        num_edges = len(primitive_goals)
        max_depth = len(primitive_goals[0]) if primitive_goals[0] else 0

        # Phase 2: Initialize output grid with all primitives (score=0)
        # Copy primitive goals with score=0 as fallback
        output_goals: List[List[Goal]] = []
        for edge_idx, edge_goals in enumerate(primitive_goals):
            edge_output = []
            for depth_idx, goal in enumerate(edge_goals):
                # Create new Goal with score=0 (fallback), preserving edge_idx/depth
                fallback_goal = Goal(
                    x=goal.x,
                    y=goal.y,
                    theta=goal.theta,
                    score=0.0,  # Fallback priority
                    edge_idx=goal.edge_idx if goal.edge_idx >= 0 else edge_idx,
                    depth=goal.depth if goal.depth >= 0 else depth_idx
                )
                edge_output.append(fallback_goal)
            output_goals.append(edge_output)

        # Phase 3: Run ML inference
        ml_goal_budget = max_goals if max_goals > 0 else self._default_ml_samples
        ml_goals = self._ml_strategy.generate_goals(
            object_id, state, env, ml_goal_budget, region_goals_sampled
        )

        if self.verbose:
            print(f"🎯 ML-Primitive Fallback for {object_id}:")
            print(f"   Primitive grid: {num_edges} edges × {max_depth} depths = {num_edges * max_depth} total")
            print(f"   ML goals received: {len(ml_goals)}")

        # Phase 4: Align ML goals to primitives and update scores
        slot_metadata = self._build_slot_metadata(primitive_goals)
        slot_votes: Dict[int, int] = defaultdict(int)  # slot_id -> vote count
        aligned_count = 0
        skipped_tolerance = 0

        for ml_goal in ml_goals:
            # Find slots within tolerance
            candidates_within_tolerance = []

            for slot_id, (edge_idx, depth_idx, primitive_goal) in enumerate(slot_metadata):
                pos_err, ang_err = self._goal_error(primitive_goal, ml_goal)

                if pos_err > self.match_position_tolerance or ang_err > self.match_angle_tolerance:
                    continue

                score = pos_err + self.angle_weight * ang_err
                candidates_within_tolerance.append((score, slot_id, edge_idx, depth_idx))

            if not candidates_within_tolerance:
                skipped_tolerance += 1
                continue

            # Vote for top-k nearest slots
            candidates_within_tolerance.sort(key=lambda x: x[0])
            top_k = candidates_within_tolerance[:self.k_nearest]

            for score, slot_id, edge_idx, depth_idx in top_k:
                slot_votes[slot_id] += 1

            aligned_count += 1

        # Phase 5: Update output grid with ML vote scores
        ml_aligned_slots = 0
        for slot_id, votes in slot_votes.items():
            edge_idx, depth_idx, _ = slot_metadata[slot_id]
            # Update the goal's score to vote count, preserving edge_idx/depth
            old_goal = output_goals[edge_idx][depth_idx]
            output_goals[edge_idx][depth_idx] = Goal(
                x=old_goal.x,
                y=old_goal.y,
                theta=old_goal.theta,
                score=float(votes),  # ML priority based on votes
                edge_idx=old_goal.edge_idx,
                depth=old_goal.depth
            )
            ml_aligned_slots += 1

        fallback_count = num_edges * max_depth - ml_aligned_slots

        if self.verbose:
            print(f"   ML-aligned slots: {ml_aligned_slots} (score > 0)")
            print(f"   Fallback slots: {fallback_count} (score = 0)")
            if skipped_tolerance > 0:
                print(f"   ML goals outside tolerance: {skipped_tolerance}")

        # Store alignment info for stats
        aligned_primitives_info = []
        for slot_id, votes in slot_votes.items():
            edge_idx, depth_idx, _ = slot_metadata[slot_id]
            goal = output_goals[edge_idx][depth_idx]
            aligned_primitives_info.append({
                'edge_idx': edge_idx,
                'depth_idx': depth_idx,
                'goal': goal,
                'votes': votes
            })
        # Sort by votes (descending), then edge_idx and depth_idx (ascending) for deterministic ordering
        aligned_primitives_info.sort(key=lambda x: (-x['votes'], x['edge_idx'], x['depth_idx']))

        # Get reachable edges for stats
        reachable_edges = set()
        try:
            env.set_full_state(state)
            reachable_edges = set(env.get_reachable_edges(object_id))
        except Exception:
            pass

        self._last_alignment_info = {
            'object_id': object_id,
            'aligned_primitives': aligned_primitives_info,
            'ml_goals': ml_goals,
            'total_ml_goals': len(ml_goals),
            'total_aligned': ml_aligned_slots,
            'reachable_edges': reachable_edges,
            'fallback_count': fallback_count,
        }

        return output_goals

    def _build_slot_metadata(self, primitive_goals: List[List[Goal]]) -> List[Tuple[int, int, Goal]]:
        """Build flat list of (edge_idx, depth_idx, goal) for all primitive slots."""
        slots = []
        for edge_idx, edge_goals in enumerate(primitive_goals):
            for depth_idx, goal in enumerate(edge_goals):
                slots.append((edge_idx, depth_idx, goal))
        return slots

    @staticmethod
    def _goal_error(primitive_goal: Goal, ml_goal: Goal) -> Tuple[float, float]:
        """Compute position and angle error between primitive and ML goal."""
        pos_err = math.hypot(
            primitive_goal.x - ml_goal.x,
            primitive_goal.y - ml_goal.y
        )
        ang_err = abs(MLPrimitiveFallbackStrategy._wrap_angle(primitive_goal.theta - ml_goal.theta))
        return pos_err, ang_err

    @staticmethod
    def _wrap_angle(theta: float) -> float:
        """Wrap angle to [-pi, pi]."""
        while theta > math.pi:
            theta -= 2 * math.pi
        while theta < -math.pi:
            theta += 2 * math.pi
        return theta

    @property
    def strategy_name(self) -> str:
        return "ML Primitive Fallback Goal Generation"


@dataclass
class AsyncGoalResult:
    """Result from async goal generation with ML inference running in background.

    This allows primitive execution to start immediately while ML inference
    runs in parallel. Caller polls for ML completion and merges scores dynamically.
    """
    # Immediate: all primitives with score=0 (fallback)
    primitive_goals: List[List[Goal]]

    # Async: ML inference future (None if no ML)
    ml_future: Optional[Future] = None

    # Reference to strategy for alignment callback
    _strategy_ref: Optional['MLPrimitiveAsyncStrategy'] = None

    # Captured data for ML alignment
    _object_id: Optional[str] = None

    # Track merge state
    ml_merged: bool = False
    ml_aligned_slots: Optional[Dict[Tuple[int, int], float]] = None

    # Stats
    ml_goals_count: int = 0
    ml_inference_time_ms: float = 0.0

    def poll_ml_ready(self) -> bool:
        """Check if ML inference is complete (non-blocking).

        Returns:
            True if ML is ready to merge, False otherwise.
        """
        if self.ml_future is None or self.ml_merged:
            return False
        return self.ml_future.done()

    def get_ml_scores(self) -> Dict[Tuple[int, int], float]:
        """Get ML scores mapping (edge_idx, depth_idx) -> vote_count.

        Blocks if ML not ready. Call only after poll_ml_ready() returns True
        for non-blocking behavior.

        Returns:
            Dict mapping (edge_idx, depth_idx) to vote count (float).
        """
        if self.ml_merged:
            return self.ml_aligned_slots or {}

        if self.ml_future is None:
            self.ml_merged = True
            self.ml_aligned_slots = {}
            return {}

        try:
            # Get ML goals (blocks if not ready)
            ml_result = self.ml_future.result()
            ml_goals = ml_result.get('goals', [])
            self.ml_inference_time_ms = ml_result.get('inference_time_ms', 0.0)
            self.ml_goals_count = len(ml_goals)

            # Align to primitives and get scores
            if self._strategy_ref and ml_goals:
                self.ml_aligned_slots = self._strategy_ref._align_ml_to_primitives(
                    ml_goals,
                    self.primitive_goals,
                    self._object_id
                )
            else:
                self.ml_aligned_slots = {}

        except Exception as e:
            # ML failed, continue with primitives only
            print(f"⚠️ ML inference failed: {e}")
            self.ml_aligned_slots = {}

        self.ml_merged = True
        return self.ml_aligned_slots

    def cancel_if_pending(self) -> bool:
        """Attempt to cancel pending ML inference.

        Returns:
            True if successfully cancelled, False if already running/completed.
        """
        if self.ml_future is None or self.ml_merged:
            return False

        return self.ml_future.cancel()


class MLPrimitiveAsyncStrategy(GoalSelectionStrategy):
    """Async ML inference with primitive pre-execution.

    This strategy enables parallel execution:
    1. Generate primitives immediately (sync, ~1ms)
    2. Capture env state for ML (sync, ~5ms)
    3. Submit ML inference to background thread (async, ~1-2s)
    4. Return AsyncGoalResult for immediate primitive execution
    5. Caller polls for ML completion and merges scores dynamically

    Benefits:
    - Primitives start executing immediately (no ML wait)
    - If primitive finds solution before ML ready, we're done faster
    - If ML ready, its high-confidence goals get priority
    - Pruning from primitive phase applies to ML goals too
    """

    # Shared executor for ML inference (reuse across calls)
    _executor: Optional[ThreadPoolExecutor] = None
    _executor_lock = threading.Lock()

    # Cancellation event - set when current inference should be cancelled
    _cancel_event: Optional[threading.Event] = None

    def __init__(
        self,
        goal_model_path: str,
        primitive_data_dir: str = "data",
        samples: int = 32,
        device: str = "cuda",
        match_position_tolerance: float = 0.1,
        match_angle_tolerance: float = 0.1,
        angle_weight: float = 0.5,
        verbose: bool = False,
        min_goals_threshold: int = 1,
        xml_path: str = None,
        preloaded_model=None,
        k_nearest: int = 1,
        max_workers: int = 1,
        seed: int = None,
        **kwargs,  # Accept extra kwargs for compatibility
    ):
        """Initialize async ML primitive strategy.

        Args:
            goal_model_path: Path to trained goal inference model.
            primitive_data_dir: Directory with primitive lookup files.
            samples: Number of diffusion samples for ML inference.
            device: Torch device for ML model.
            match_position_tolerance: Max position error for ML-to-primitive matching.
            match_angle_tolerance: Max angle error for ML-to-primitive matching.
            angle_weight: Weight for angle error in matching score.
            verbose: Enable debug output.
            min_goals_threshold: Minimum ML goals required.
            xml_path: XML file path for ML model context.
            preloaded_model: Optional preloaded GoalInferenceModel.
            k_nearest: Number of nearest slots to vote for per ML goal.
            max_workers: Thread pool size for async ML inference.
            seed: Random seed for diffusion noise (None = random each time).
        """
        self.verbose = verbose
        self.match_position_tolerance = match_position_tolerance
        self.match_angle_tolerance = match_angle_tolerance
        self.angle_weight = angle_weight
        self.k_nearest = k_nearest
        self._default_ml_samples = samples

        # Initialize primitive strategy (sync, fast)
        self._primitive_strategy = PrimitiveGoalStrategy(
            data_dir=primitive_data_dir,
            verbose=verbose
        )

        # Initialize ML strategy (will capture JSON, inference runs async)
        self._ml_strategy = MLGoalSelectionStrategy(
            goal_model_path=goal_model_path,
            samples=samples,
            device=device,
            min_goals_threshold=min_goals_threshold,
            verbose=verbose,
            xml_path=xml_path,
            preloaded_model=preloaded_model,
            seed=seed
        )

        # Initialize thread pool
        self._init_executor(max_workers)

        # Warmup CUDA in the worker thread (first inference is slow due to context init)
        if preloaded_model is not None:
            self._warmup_cuda_in_thread()

        # Stats tracking
        self._last_alignment_info = None

    @classmethod
    def _init_executor(cls, max_workers: int = 1):
        """Initialize shared thread pool executor (singleton).

        Always uses 1 worker since GPU can only run one ML inference at a time.
        """
        with cls._executor_lock:
            if cls._executor is None:
                cls._executor = ThreadPoolExecutor(
                    max_workers=1,  # Always 1 - GPU runs one inference at a time
                    thread_name_prefix="ml_goal_async"
                )
            if cls._cancel_event is None:
                cls._cancel_event = threading.Event()

    @classmethod
    def cancel_all_pending(cls):
        """Signal cancellation to any running inference.

        The running inference will check this event and return early.
        """
        if cls._cancel_event is not None:
            cls._cancel_event.set()

    @classmethod
    def clear_cancellation(cls):
        """Clear the cancellation signal before starting new inference."""
        if cls._cancel_event is not None:
            cls._cancel_event.clear()

    # Class-level flag to track if warmup has been done (singleton pattern)
    _warmup_done = False
    _warmup_lock = threading.Lock()

    def _warmup_cuda_in_thread(self):
        """Run a warmup inference in the worker thread to initialize CUDA/model.

        First model inference is slow (~20s) due to CUDA kernel compilation.
        Running warmup during strategy init makes actual inferences fast (~100ms).

        This is a singleton operation - only runs once across all instances.
        """
        with self._warmup_lock:
            if MLPrimitiveAsyncStrategy._warmup_done:
                return  # Already warmed up
            MLPrimitiveAsyncStrategy._warmup_done = True

        import torch
        import time

        def warmup_task():
            try:
                # Get the actual model and run a dummy forward pass
                model = self._ml_strategy._goal_model
                if model is not None and hasattr(model, 'model'):
                    device = self._ml_strategy.device
                    # Create dummy input matching model's expected shape
                    # The model expects 5 context channels: static, movable, target, reachable, goal_region
                    context_channels = 5
                    dummy_input = torch.randn(1, context_channels, 64, 64, device=device)
                    with torch.no_grad():
                        # Run full inference with actual number of steps
                        # This ensures all CUDA kernels are compiled
                        _ = model.model.sample_from_model(
                            dummy_input,
                            samples=self._default_ml_samples,  # Use actual sample count
                            num_steps=5  # Typical DDIM steps
                        )
                    torch.cuda.synchronize()
                else:
                    # Fallback to simple tensor warmup
                    device = self._ml_strategy.device
                    x = torch.randn(1, 8, 64, 64, device=device)
                    _ = x * 2
                    torch.cuda.synchronize()
            except Exception as e:
                pass  # Warmup failure is non-fatal

        # Submit warmup and wait for completion
        future = self._executor.submit(warmup_task)
        future.result()  # Block until warmup done

    def generate_goals(
        self,
        object_id: str,
        state: namo_rl.RLState,
        env: namo_rl.RLEnvironment,
        max_goals: int,
        region_goals_sampled: Optional[List[Tuple[float, float, float]]] = None
    ) -> AsyncGoalResult:
        """Generate primitives immediately, start ML inference async.

        Returns:
            AsyncGoalResult that can be polled for ML completion.
            primitive_goals are ready immediately (all score=0).
            Call poll_ml_ready() and get_ml_scores() to merge ML results.
        """
        import time
        start_time = time.time()

        # Phase 1: Generate ALL primitives (sync, ~1ms)
        primitive_goals = self._primitive_strategy.generate_goals(
            object_id, state, env, max_goals, region_goals_sampled
        )

        if not primitive_goals:
            return AsyncGoalResult(primitive_goals=[], ml_future=None)

        # Initialize all primitives with score=0 (fallback priority), preserving edge_idx/depth
        output_goals: List[List[Goal]] = []
        for edge_idx, edge_goals in enumerate(primitive_goals):
            edge_output = []
            for depth_idx, goal in enumerate(edge_goals):
                edge_output.append(Goal(
                    x=goal.x,
                    y=goal.y,
                    theta=goal.theta,
                    score=0.0,
                    edge_idx=goal.edge_idx if goal.edge_idx >= 0 else edge_idx,
                    depth=goal.depth if goal.depth >= 0 else depth_idx
                ))
            output_goals.append(edge_output)

        primitive_time_ms = (time.time() - start_time) * 1000

        # Phase 2: Capture env state for ML (sync, main thread, ~5ms)
        # This must happen in main thread since env is not thread-safe
        json_capture_start = time.time()
        try:
            json_message = self._ml_strategy._create_json_message_for_goals(
                object_id, state, env
            )
        except Exception as e:
            if self.verbose:
                print(f"⚠️ Failed to create JSON for ML: {e}")
            json_message = None

        json_time_ms = (time.time() - json_capture_start) * 1000

        if json_message is None:
            # Can't run ML, return primitives only
            if self.verbose:
                print(f"🎯 Async goals for {object_id}: primitives only (JSON creation failed)")
            return AsyncGoalResult(
                primitive_goals=output_goals,
                ml_future=None,
                _strategy_ref=self,
                _object_id=object_id
            )

        # Phase 3: Submit ML inference to background thread (async)
        ml_budget = max_goals if max_goals > 0 else self._default_ml_samples

        ml_future = self._executor.submit(
            self._run_ml_inference_only,
            json_message,
            object_id,
            ml_budget
        )

        total_setup_ms = (time.time() - start_time) * 1000

        if self.verbose:
            num_edges = len(output_goals)
            num_depths = len(output_goals[0]) if output_goals else 0
            print(f"🚀 Async ML started for {object_id}:")
            print(f"   Primitives ready: {num_edges} edges × {num_depths} depths")
            print(f"   Setup time: {total_setup_ms:.1f}ms (primitives: {primitive_time_ms:.1f}ms, JSON: {json_time_ms:.1f}ms)")
            print(f"   ML inference running in background...")

        return AsyncGoalResult(
            primitive_goals=output_goals,
            ml_future=ml_future,
            _strategy_ref=self,
            _object_id=object_id,
            ml_merged=False,
            ml_aligned_slots=None
        )

    def _run_ml_inference_only(
        self,
        json_message: Dict[str, Any],
        object_id: str,
        ml_budget: int
    ) -> Dict[str, Any]:
        """Run ML inference in background thread (no env access).

        This is thread-safe as it only uses the pre-captured JSON data
        and the ML model (which handles its own CUDA context).

        Returns:
            Dict with 'goals' list and 'inference_time_ms'.
        """
        import time
        start_time = time.time()

        # Check if cancelled before starting
        if self._cancel_event is not None and self._cancel_event.is_set():
            return {'goals': [], 'inference_time_ms': 0, 'cancelled': True}

        try:
            # Ensure model is loaded (lazy loading)
            if self._ml_strategy._goal_model is None:
                self._ml_strategy._load_model()

            if self._ml_strategy._goal_model is None:
                return {
                    'goals': [],
                    'inference_time_ms': (time.time() - start_time) * 1000,
                    'error': 'Model failed to load'
                }

            # Run model inference (pure computation, no env access)
            goals = self._ml_strategy._goal_model.infer(
                json_message=json_message,
                xml_path=json_message["xml_path"],
                robot_goal=json_message["robot_goal"],
                selected_object=object_id,
                samples=ml_budget
            )

            inference_time_ms = (time.time() - start_time) * 1000

            # Check if cancelled after inference (results will be discarded)
            if self._cancel_event is not None and self._cancel_event.is_set():
                return {'goals': [], 'inference_time_ms': inference_time_ms, 'cancelled': True}

            # Convert to Goal objects
            goal_objects = []
            for goal_data in (goals or []):
                if 'x' in goal_data and 'y' in goal_data and 'theta' in goal_data:
                    goal_objects.append(Goal(
                        x=float(goal_data['x']),
                        y=float(goal_data['y']),
                        theta=float(goal_data['theta'])
                    ))

            return {
                'goals': goal_objects,
                'inference_time_ms': inference_time_ms
            }

        except Exception as e:
            return {
                'goals': [],
                'inference_time_ms': (time.time() - start_time) * 1000,
                'error': str(e)
            }

    def _align_ml_to_primitives(
        self,
        ml_goals: List[Goal],
        primitive_goals: List[List[Goal]],
        object_id: str
    ) -> Dict[Tuple[int, int], float]:
        """Align ML goals to primitive slots, return score mapping.

        Args:
            ml_goals: List of ML-generated goals.
            primitive_goals: Full primitive grid.
            object_id: Object ID (for logging).

        Returns:
            Dict mapping (edge_idx, depth_idx) -> vote_count (float).
        """
        if not ml_goals:
            return {}

        slot_votes: Dict[Tuple[int, int], int] = defaultdict(int)

        # Build flat slot list for matching
        slot_metadata: List[Tuple[int, int, Goal]] = []
        for edge_idx, edge_goals in enumerate(primitive_goals):
            for depth_idx, goal in enumerate(edge_goals):
                slot_metadata.append((edge_idx, depth_idx, goal))

        # Align each ML goal to nearest primitive slots
        aligned_count = 0
        skipped_tolerance = 0

        for ml_goal in ml_goals:
            candidates = []

            for edge_idx, depth_idx, prim_goal in slot_metadata:
                pos_err = math.hypot(prim_goal.x - ml_goal.x, prim_goal.y - ml_goal.y)
                ang_err = abs(self._wrap_angle(prim_goal.theta - ml_goal.theta))

                if pos_err > self.match_position_tolerance:
                    continue
                if ang_err > self.match_angle_tolerance:
                    continue

                score = pos_err + self.angle_weight * ang_err
                candidates.append((score, edge_idx, depth_idx))

            if not candidates:
                skipped_tolerance += 1
                continue

            # Vote for top-k nearest slots
            candidates.sort(key=lambda x: x[0])
            for _, edge_idx, depth_idx in candidates[:self.k_nearest]:
                slot_votes[(edge_idx, depth_idx)] += 1

            aligned_count += 1

        if self.verbose:
            print(f"   🎯 ML alignment: {aligned_count}/{len(ml_goals)} goals → {len(slot_votes)} slots")
            if skipped_tolerance > 0:
                print(f"      Skipped (tolerance): {skipped_tolerance}")

        # Store stats
        self._last_alignment_info = {
            'object_id': object_id,
            'total_ml_goals': len(ml_goals),
            'aligned_count': aligned_count,
            'slots_with_votes': len(slot_votes),
            'skipped_tolerance': skipped_tolerance,
        }

        return {k: float(v) for k, v in slot_votes.items()}

    @staticmethod
    def _wrap_angle(theta: float) -> float:
        """Wrap angle to [-pi, pi]."""
        while theta > math.pi:
            theta -= 2 * math.pi
        while theta < -math.pi:
            theta += 2 * math.pi
        return theta

    def get_last_goal_stats(self) -> dict:
        """Return stats from the last alignment."""
        if self._last_alignment_info is None:
            return {}
        return self._last_alignment_info.copy()

    @property
    def strategy_name(self) -> str:
        return "ML Primitive Async Goal Generation"
