"""Primitive-based goal selection strategy for NAMO planning.

This module provides goal generation using precomputed motion primitives
from binary database files. Primitives are shape-specific (square/tall/wide)
and organized by edge points and push steps.
"""

import hashlib
import struct
import os
import math
import random
import json
import time
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Union, Any, Sequence
from collections import defaultdict
from abc import ABC

import namo_rl
from .goal_selection_strategy import GoalSelectionStrategy, Goal
from .ml_strategies import MLGoalSelectionStrategy


def _namo_get_ml_artifacts_dir() -> Optional[Path]:
    raw = os.environ.get("NAMO_ML_ARTIFACTS_DIR")
    if not raw:
        return None
    try:
        path = Path(raw)
        path.mkdir(parents=True, exist_ok=True)
        return path
    except Exception:
        return None


def _namo_unique_path(path: Path) -> Path:
    if not path.exists():
        return path
    stem = path.stem
    suffix = path.suffix
    parent = path.parent
    for i in range(1, 10_000):
        candidate = parent / f"{stem}_{i:04d}{suffix}"
        if not candidate.exists():
            return candidate
    return parent / f"{stem}_{time.time_ns()}{suffix}"


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
                 shuffle_edges: bool = False, seed: int = None,
                 primitive_prefix: str = "", max_push_steps: Optional[int] = None):
        """Initialize primitive goal strategy.

        Args:
            data_dir: Directory containing motion_primitives_15_*.dat files
            verbose: Enable verbose output
            shuffle_edges: If True, randomize edge ordering (useful for averaging difficulty)
            seed: Random seed for reproducible shuffling (None = random each call)
            primitive_prefix: Prefix on primitive filename to select per-robot calibration.
                "" → motion_primitives_15_*.dat (30 cm point-robot, legacy)
                "car_" → car_motion_primitives_15_*.dat (7 cm diff-drive car)
            max_push_steps: Optional cap on primitive depth enumeration. When set,
                primitives with push_steps > max_push_steps are discarded before
                edge grouping so all downstream strategies share the same cap.
        """
        self.data_dir = data_dir
        self.verbose = verbose
        self.shuffle_edges = shuffle_edges
        self.seed = seed
        self.primitive_prefix = primitive_prefix
        self.max_push_steps = max_push_steps
        self._rng = random.Random(seed) if seed is not None else None
        self._primitive_cache: Dict[str, List[Primitive]] = {}
        self._primitive_sha256_cache: Dict[str, str] = {}
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
            if self.max_push_steps is not None:
                primitives = [
                    primitive
                    for primitive in primitives
                    if primitive.push_steps <= self.max_push_steps
                ]

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
            return f"{self.primitive_prefix}motion_primitives_15_square.dat"

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
            return f"{self.primitive_prefix}motion_primitives_15_square.dat"

        if x <= 0.0 or y <= 0.0:
            if self.verbose:
                print(f"Invalid dimensions for {object_name}: [{x}×{y}], defaulting to square")
            return f"{self.primitive_prefix}motion_primitives_15_square.dat"

        # Calculate aspect ratio
        ratio = max(x, y) / min(x, y)

        if ratio < 1.05:
            shape = "square"
        elif x > y:
            shape = "wide"
        else:
            shape = "tall"
        return f"{self.primitive_prefix}motion_primitives_15_{shape}.dat"

    def primitive_database_provenance(self, object_name: str, env: namo_rl.RLEnvironment) -> Dict[str, str]:
        """Identify the exact shape-specific primitive database used for this object."""
        primitive_file = self._select_primitive_file(object_name, env)
        filepath = os.path.realpath(os.path.join(self.data_dir, primitive_file))
        if filepath not in self._primitive_sha256_cache:
            digest = hashlib.sha256()
            with open(filepath, "rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
            self._primitive_sha256_cache[filepath] = digest.hexdigest()
        shape_family = primitive_file.rsplit("_", 1)[-1].removesuffix(".dat")
        return {
            "primitive_database_id": primitive_file,
            "primitive_database_sha256": self._primitive_sha256_cache[filepath],
            "shape_family": shape_family,
        }

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


class RandomRolloutGoalStrategy(PrimitiveGoalStrategy):
    """Random-rollout goal strategy.

    Returns the same primitive-aligned goals as PrimitiveGoalStrategy, but:
      1. Assigns each candidate a random score in (0, 1), so the chain-depth
         BFS's `sort by (-score, depth, edge_idx)` produces a uniformly
         random ordering of candidates per state.
      2. Optionally caps the number of non-None candidates returned
         (`samples_per_state`) so the BFS tries only K random primitives per
         state instead of all ~600. Combined with `max_chain_depth`, this
         yields trial-style exploration: each chain through the BFS is a
         short random walk through primitive-aligned pushes.

    Because the BFS caches goals per node (see region_opening:1305), each
    frontier node at chain depth 2 gets its own independent random ordering
    — different trials explore different trajectories.
    """

    def __init__(self, data_dir: str = "data", verbose: bool = False,
                 samples_per_state: Optional[int] = None, seed: Optional[int] = None,
                 primitive_prefix: str = "", max_push_steps: Optional[int] = None):
        super().__init__(
            data_dir=data_dir,
            verbose=verbose,
            shuffle_edges=False,
            seed=seed,
            primitive_prefix=primitive_prefix,
            max_push_steps=max_push_steps,
        )
        self.samples_per_state = samples_per_state
        self._score_rng = random.Random(seed) if seed is not None else random

    def generate_goals(self,
                       object_id: str,
                       state: namo_rl.RLState,
                       env: namo_rl.RLEnvironment,
                       max_goals: int,
                       region_goals_sampled: Optional[List[Tuple[float, float, float]]] = None) -> List[List[Goal]]:
        # Get the 600 primitive-aligned goals (all with score=0).
        goals_per_edge = super().generate_goals(
            object_id, state, env, max_goals, region_goals_sampled
        )

        # Assign each non-None goal a random score in (0, 1). The BFS sort by
        # `-score` then orders them uniformly at random per state.
        for edge_goals in goals_per_edge:
            for goal in edge_goals:
                if goal is not None:
                    goal.score = self._score_rng.random() * 0.9 + 0.05

        # Optional thinning: keep only K random candidates per state,
        # blank the rest. This is the lever that makes each chain "thin".
        if self.samples_per_state is not None and self.samples_per_state > 0:
            all_candidates = [
                (e_idx, d_idx)
                for e_idx, edge_goals in enumerate(goals_per_edge)
                for d_idx, goal in enumerate(edge_goals)
                if goal is not None
            ]
            k = min(self.samples_per_state, len(all_candidates))
            if k < len(all_candidates):
                kept = set(self._score_rng.sample(all_candidates, k))
                for e_idx, edge_goals in enumerate(goals_per_edge):
                    for d_idx in range(len(edge_goals)):
                        if (e_idx, d_idx) not in kept:
                            edge_goals[d_idx] = None

        if self.verbose:
            kept_count = sum(
                1 for edge_goals in goals_per_edge for g in edge_goals if g is not None
            )
            print(f"      🎲 RandomRollout: {kept_count} candidates (samples_per_state={self.samples_per_state})")

        return goals_per_edge

    @property
    def strategy_name(self) -> str:
        return "Random-Rollout Goal Generation"


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
        primitive_prefix: str = "",
        max_push_steps: Optional[int] = None,
        namo_config_path: Optional[str] = None,
        sampler_method: Optional[str] = None,
        num_steps: Optional[int] = None,
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
            primitive_prefix: Filename prefix selecting the per-robot primitive set
                (e.g. "1x_car_"). MUST match the robot the diffusion model was
                trained for, otherwise ML goals (small car-scale moves) cannot
                align to the slot poses (legacy point-robot pushes are 0.5-6 m).
                Forwarded to the inner PrimitiveGoalStrategy. Default "" (legacy).
            max_push_steps: Optional cap on primitive depth enumeration, mirroring
                the executor's motion_primitives.max_push_steps. Forwarded too.
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
            verbose=verbose,
            primitive_prefix=primitive_prefix,
            max_push_steps=max_push_steps,
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
            seed=seed,
            namo_config_path=namo_config_path,
            sampler_method=sampler_method,
            num_steps=num_steps,
        )
        self._default_ml_samples = samples

        # Store last alignment result for visualization
        self._last_alignment_info = None
        self._profile_state = {
            "generate_goals_calls": 0,
            "ml_mask_vote_attach_calls": 0,
            "ml_mask_vote_attach_ms_total": 0.0,
            "ml_goals_seen_total": 0,
        }

    def reset_diffusion_call_counter(self) -> None:
        """Reset diffusion/infer call counter for the underlying ML goal sampler."""
        if hasattr(self._ml_strategy, "reset_diffusion_call_counter"):
            self._ml_strategy.reset_diffusion_call_counter()

    def get_diffusion_call_counter(self) -> int:
        """Get diffusion/infer calls since last reset (best-effort)."""
        if hasattr(self._ml_strategy, "get_diffusion_call_counter"):
            return int(self._ml_strategy.get_diffusion_call_counter())
        return 0

    def reset_profile(self) -> None:
        self._profile_state = {
            "generate_goals_calls": 0,
            "ml_mask_vote_attach_calls": 0,
            "ml_mask_vote_attach_ms_total": 0.0,
            "ml_goals_seen_total": 0,
        }

    def get_profile(self) -> Dict[str, Any]:
        calls = int(self._profile_state.get("ml_mask_vote_attach_calls", 0))
        total_ms = float(self._profile_state.get("ml_mask_vote_attach_ms_total", 0.0))
        return {
            "generate_goals_calls": int(self._profile_state.get("generate_goals_calls", 0)),
            "ml_mask_vote_attach_calls": calls,
            "ml_mask_vote_attach_ms_total": total_ms,
            "ml_mask_vote_attach_ms_avg": (total_ms / calls) if calls > 0 else 0.0,
            "ml_goals_seen_total": int(self._profile_state.get("ml_goals_seen_total", 0)),
        }

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
            prof = self.get_profile()
            return {
                'ml_goals_generated': 0,
                'ml_goals_aligned': 0,
                'reachable_edges_count': 0,
                'aligned_primitives': [],
                'ml_goals_raw': [],
                'reachable_edges': [],
                'ml_diffusion_calls': self.get_diffusion_call_counter(),
                'ml_mask_vote_attach_calls': int(prof.get('ml_mask_vote_attach_calls', 0)),
                'ml_mask_vote_attach_ms_total': float(prof.get('ml_mask_vote_attach_ms_total', 0.0)),
                'ml_mask_vote_attach_ms_avg': float(prof.get('ml_mask_vote_attach_ms_avg', 0.0)),
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

        prof = self.get_profile()
        return {
            'ml_goals_generated': self._last_alignment_info.get('total_ml_goals', 0),
            'ml_goals_aligned': self._last_alignment_info.get('total_aligned', 0),
            'reachable_edges_count': len(self._last_alignment_info.get('reachable_edges', set())),
            'aligned_primitives': aligned_primitives,
            'ml_goals_raw': ml_goals_raw,
            'reachable_edges': sorted(list(self._last_alignment_info.get('reachable_edges', set()))),
            'ml_diffusion_calls': self.get_diffusion_call_counter(),
            'ml_mask_vote_attach_calls': int(prof.get('ml_mask_vote_attach_calls', 0)),
            'ml_mask_vote_attach_ms_total': float(prof.get('ml_mask_vote_attach_ms_total', 0.0)),
            'ml_mask_vote_attach_ms_avg': float(prof.get('ml_mask_vote_attach_ms_avg', 0.0)),
        }

    def generate_goals(
        self,
        object_id: str,
        state: namo_rl.RLState,
        env: namo_rl.RLEnvironment,
        max_goals: int,
        region_goals_sampled: Optional[List[Tuple[float, float, float]]] = None
    ) -> List[List[Goal]]:
        self._profile_state["generate_goals_calls"] += 1

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
        ml_goal_vote_details: List[Dict[str, Any]] = []
        self._profile_state["ml_goals_seen_total"] += int(len(ml_goals))
        try:
            alignment_ml_call_id = int(getattr(ml_goals[0], "ml_call_id", -1)) if ml_goals else -1
        except Exception:
            alignment_ml_call_id = -1

        vote_attach_start = time.perf_counter()
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
                # Still record an entry (empty votes) so downstream analysis can align
                # "no-match" samples with their stored masks.
                try:
                    sample_index = int(getattr(ml_goal, "sample_index", ml_goal_idx))
                except Exception:
                    sample_index = int(ml_goal_idx)
                try:
                    ml_call_id = int(getattr(ml_goal, "ml_call_id", -1))
                except Exception:
                    ml_call_id = -1
                ml_goal_vote_details.append(
                    {
                        "sample_index": sample_index,
                        "ml_call_id": ml_call_id,
                        "mask_path": getattr(ml_goal, "mask_path", None),
                        "x": float(ml_goal.x),
                        "y": float(ml_goal.y),
                        "theta": float(ml_goal.theta),
                        "voted_primitives": [],
                    }
                )
                if self.verbose and ml_goal_idx < 5:  # Show first 5 skipped goals
                    print(f"    ⊗ ML goal {ml_goal_idx}: ({ml_goal.x:.3f}, {ml_goal.y:.3f}, {ml_goal.theta:.3f}) - No slots within tolerance")
                continue

            # Sort by score (ascending) and take top-k
            candidates_within_tolerance.sort(key=lambda x: x[0])
            top_k_candidates = candidates_within_tolerance[:self.k_nearest]

            # Record which primitive slots this ML sample voted for.
            try:
                sample_index = int(getattr(ml_goal, "sample_index", ml_goal_idx))
            except Exception:
                sample_index = int(ml_goal_idx)
            try:
                ml_call_id = int(getattr(ml_goal, "ml_call_id", -1))
            except Exception:
                ml_call_id = -1

            goal_weight = float(getattr(ml_goal, "score", 1.0))
            if not math.isfinite(goal_weight) or goal_weight <= 0.0:
                goal_weight = 1.0

            ml_goal_vote_details.append(
                {
                    "sample_index": sample_index,
                    "ml_call_id": ml_call_id,
                    "mask_path": getattr(ml_goal, "mask_path", None),
                    "x": float(ml_goal.x),
                    "y": float(ml_goal.y),
                    "theta": float(ml_goal.theta),
                    "vote_weight": goal_weight,
                    "voted_primitives": [
                        {
                            "edge_idx": int(edge_idx),
                            "depth_idx": int(depth_idx),
                            "score": float(score),
                            "slot_id": int(slot_id),
                        }
                        for (score, slot_id, edge_idx, depth_idx) in top_k_candidates
                    ],
                }
            )

            # Vote for each of the k-nearest slots
            for score, slot_id, edge_idx, depth_idx in top_k_candidates:
                acc = slot_accumulators[slot_id]
                acc["count"] += goal_weight

                if "goal" not in acc:
                    # Retrieve the correct primitive goal from metadata using slot_id
                    _, _, correct_primitive_goal = slot_metadata[slot_id]
                    acc["goal"] = correct_primitive_goal

            matches += 1  # Count ML goals that had at least one match
        vote_attach_ms = (time.perf_counter() - vote_attach_start) * 1000.0
        self._profile_state["ml_mask_vote_attach_calls"] += 1
        self._profile_state["ml_mask_vote_attach_ms_total"] += max(0.0, float(vote_attach_ms))

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
                depth=depth_idx,     # Preserve depth for C++ direct execution
                # Propagate diffusion call id for joining with saved masks/votes artifacts.
                ml_call_id=int(alignment_ml_call_id),
                sample_index=-1,
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
            'ml_goal_votes': ml_goal_vote_details,
            'total_ml_goals': len(ml_goals),
            'total_aligned': len(aligned_primitives_info),
            'reachable_edges': reachable_edges,
            'ml_mask_vote_attach_ms': max(0.0, float(vote_attach_ms)),
        }

        # Best-effort: persist per-sample vote mapping + the aggregated
        # vote-ranked primitive slots, alongside the saved masks. The ranked
        # list is aligned_primitives_info, already sorted high->low vote, so a
        # reader gets the executed-push candidates in priority order and can
        # join to the local/action masks via ml_call_id.
        artifacts_root = _namo_get_ml_artifacts_dir()
        if artifacts_root is not None and (ml_goal_vote_details or aligned_primitives_info):
            call_id = -1
            for g in ml_goals:
                cid = getattr(g, "ml_call_id", -1)
                if isinstance(cid, int) and cid >= 0:
                    call_id = cid
                    break
            if call_id >= 0:
                out_dir = artifacts_root / "ml_goal_samples" / str(object_id) / f"call_{call_id:06d}"
            else:
                out_dir = artifacts_root / "primitive_votes" / str(object_id)
            try:
                out_dir.mkdir(parents=True, exist_ok=True)
                # JSON-safe, vote-ranked (high->low) aligned slots.
                # push_steps = depth_idx + 1 (executor convention).
                ranked_aligned_slots = []
                for rank, p in enumerate(aligned_primitives_info):
                    g = p.get("goal")
                    ranked_aligned_slots.append({
                        "rank": int(rank),
                        "edge_idx": int(p["edge_idx"]),
                        "depth_idx": int(p["depth_idx"]),
                        "push_steps": int(p["depth_idx"]) + 1,
                        "votes": float(p["votes"]),
                        "x": float(g.x) if g is not None else None,
                        "y": float(g.y) if g is not None else None,
                        "theta": float(g.theta) if g is not None else None,
                    })
                payload = {
                    "object_id": str(object_id),
                    "ml_call_id": int(call_id),
                    "k_nearest": int(self.k_nearest),
                    "match_position_tolerance": float(self.match_position_tolerance),
                    "match_angle_tolerance": float(self.match_angle_tolerance),
                    "angle_weight": float(self.angle_weight),
                    "created_unix_sec": time.time(),
                    "total_ml_goals": int(len(ml_goals)),
                    "total_aligned_slots": int(len(ranked_aligned_slots)),
                    "ranked_aligned_slots": ranked_aligned_slots,
                    "ml_goal_votes": ml_goal_vote_details,
                }
                out_path = _namo_unique_path(out_dir / "primitive_votes.json")
                out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
            except Exception:
                pass

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
