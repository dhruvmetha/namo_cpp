"""Primitive-based goal selection strategy for NAMO planning.

This module provides goal generation using precomputed motion primitives
from binary database files. Primitives are shape-specific (square/tall/wide)
and organized by edge points and push steps.
"""

import struct
import os
import json
import math
import random
import threading
from concurrent.futures import ThreadPoolExecutor, Future
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple, Any
from collections import defaultdict
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

    def __init__(
        self,
        data_dir: str = "data",
        verbose: bool = False,
        shuffle_edges: bool = False,
        seed: int = None,
    ):
        """Initialize primitive goal strategy.

        Args:
            data_dir: Directory containing motion_primitives_15_*.dat files
            verbose: Enable verbose output
            shuffle_edges: If True, randomize edge ordering (primarily for ablations)
            seed: Optional seed for reproducible shuffling
        """
        self.data_dir = data_dir
        # Default configs often set `primitive_data_dir: data`, which is CWD-relative and fragile.
        # If we're running outside `namo_cpp/`, fall back to `<repo>/namo_cpp/data`.
        if self.data_dir == "data" and not os.path.isdir(self.data_dir):
            candidate = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../data"))
            if os.path.isdir(candidate):
                self.data_dir = candidate
        self.verbose = verbose
        self.shuffle_edges = shuffle_edges
        self.seed = seed
        self._rng = random.Random(seed) if seed is not None else None
        self._last_edge_ordering: List[int] = []
        self._primitive_cache: Dict[str, List[Primitive]] = {}

    def reseed(self, seed: int):
        self.seed = seed
        self._rng = random.Random(seed)

    def get_last_edge_ordering(self) -> List[int]:
        return self._last_edge_ordering.copy()

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
            # Determine edge ordering (sorted or shuffled)
            edge_indices = sorted(edge_groups.keys())
            if self.shuffle_edges:
                if self._rng is not None:
                    self._rng.shuffle(edge_indices)
                else:
                    random.shuffle(edge_indices)
            self._last_edge_ordering = list(edge_indices)

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
        goals_per_region: int = None,
        preview_aligned_primitives: bool = False,
        k_nearest: int = 1,
        score_metric: str = "pos+w*ang",
        seed: int = None,
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
            goals_per_region: Number of region goal samples to include (vector models only).
            preview_aligned_primitives: If True, save visualization of aligned primitives.
            k_nearest: Number of nearest primitive slots to vote for per ML goal (within tolerance).
            score_metric: How to rank primitive slots by (pos_err, ang_err). Options:
                - "pos+w*ang": pos_err + w * ang_err  (current default; w has units m/rad)
                - "l2": sqrt(pos_err^2 + (w*ang_err)^2)
                - "normalized_l2": sqrt((pos_err/tol_pos)^2 + (w*ang_err/tol_ang)^2)
        """
        self.verbose = verbose
        self.max_matches = max_matches
        self.match_position_tolerance = match_position_tolerance
        self.match_angle_tolerance = match_angle_tolerance
        self.angle_weight = angle_weight
        self.score_metric = str(score_metric or "pos+w*ang").strip().lower()
        self.preview_aligned_primitives = preview_aligned_primitives
        self.k_nearest = max(1, int(k_nearest)) if k_nearest is not None else 1

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
            goals_per_region=goals_per_region,
            seed=seed,
            sampler_method=sampler_method,
            num_steps=num_steps,
        )
        self._default_ml_samples = samples

        # Store last alignment result for visualization
        self._last_alignment_info = None

    def get_last_goal_stats(self) -> dict:
        """Return stats from the last generate_goals call for failure tracking.

        Mirrors the stats contract used by the 2push evaluation framework.
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

        reachable_edges = self._last_alignment_info.get('reachable_edges', set()) or set()
        return {
            'ml_goals_generated': self._last_alignment_info.get('total_ml_goals', 0),
            'ml_goals_aligned': self._last_alignment_info.get('total_aligned', 0),
            'reachable_edges_count': len(reachable_edges),
            'aligned_primitives': aligned_primitives,
            'ml_goals_raw': ml_goals_raw,
            'reachable_edges': sorted(list(reachable_edges)),
        }

    def _slot_score(self, *, pos_err: float, ang_err: float) -> float:
        metric = self.score_metric
        if metric == "pos+w*ang":
            return pos_err + (self.angle_weight * ang_err)
        if metric == "l2":
            return math.hypot(pos_err, self.angle_weight * ang_err)
        if metric == "normalized_l2":
            pos_scale = max(float(self.match_position_tolerance), 1e-9)
            ang_scale = max(float(self.match_angle_tolerance), 1e-9)
            return math.hypot(pos_err / pos_scale, (self.angle_weight * ang_err) / ang_scale)
        # Fallback to the historic behavior.
        return pos_err + (self.angle_weight * ang_err)

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

        # NOTE: We map ML goals to the closest primitive slots over *all* edges (reachable + unreachable).
        # The region-opening planner filters by reachability later (see RegionOpeningPlanner._search_bfs),
        # but we still compute `reachable_edges` here to (a) color dots in the preview and (b) show which
        # voted primitive the planner would attempt first among reachable edges.
        reachable_edges: set[int] = set()
        try:
            original_state = env.get_full_state()
            env.set_full_state(state)
            reachable_edges = set(env.get_reachable_edges(object_id))
        except Exception as e:
            if self.verbose:
                print(f"  ⚠️ Could not get reachable edges for alignment: {e}")
        finally:
            try:
                env.set_full_state(original_state)
            except Exception:
                pass

        slot_metadata = self._build_slot_metadata(primitive_goals)
        slot_accumulators = defaultdict(lambda: {"x": 0.0, "y": 0.0, "sin": 0.0, "cos": 0.0, "count": 0})
        goals_within_tolerance = 0
        goals_outside_tolerance = 0
        ml_goal_match_debug: List[Dict[str, Any]] = []
        debug_topk = 5 if (self.preview_aligned_primitives or self.verbose) else 0

        for ml_goal_idx, ml_goal in enumerate(ml_goals):
            best_unfiltered = None  # for debugging: (score, slot_id, edge_idx, depth_idx, pos_err, ang_err)
            candidates_reachable = []
            candidates_all = [] if debug_topk else None

            for slot_id, (edge_idx, depth_idx, primitive_goal) in enumerate(slot_metadata):
                pos_err, ang_err = self._goal_error(primitive_goal, ml_goal)
                score = self._slot_score(pos_err=pos_err, ang_err=ang_err)
                if best_unfiltered is None or score < best_unfiltered[0]:
                    best_unfiltered = (score, slot_id, edge_idx, depth_idx, pos_err, ang_err)

                if candidates_all is not None:
                    candidates_all.append((score, pos_err, ang_err, slot_id, edge_idx, depth_idx, (not reachable_edges) or (edge_idx in reachable_edges)))

                # NOTE: We intentionally map over *all* primitive slots, regardless of reachability.
                # The region-opening planner later filters by `reachable_edge_indices` before executing.
                candidates_reachable.append((score, pos_err, ang_err, slot_id, edge_idx, depth_idx))

            if not candidates_reachable:
                continue

            candidate_pool_sorted = sorted(candidates_reachable, key=lambda x: x[0])
            # For debugging only: what the planner could actually try (reachable edges).
            # If reachability is unknown (empty set), treat everything as reachable.
            candidate_pool_reachable_sorted = [
                cand for cand in candidate_pool_sorted
                if (not reachable_edges) or (cand[4] in reachable_edges)
            ]
            top_k = candidate_pool_sorted[: max(1, self.k_nearest)]

            # Count tolerance satisfaction based on the best (closest) candidate only.
            best_score, best_pos, best_ang, _best_slot_id, _best_edge, _best_depth = top_k[0]
            best_within_tolerance = (best_pos <= self.match_position_tolerance) and (best_ang <= self.match_angle_tolerance)
            if best_within_tolerance:
                goals_within_tolerance += 1
            else:
                goals_outside_tolerance += 1

            # Vote weighting: Borda-style so the planner's order matches the closeness ranking.
            # For k=5, votes are [5, 4, 3, 2, 1] for ranks 1..5.
            voted_slots_debug = []
            for rank, (cand_score, cand_pos, cand_ang, slot_id, edge_idx, depth_idx) in enumerate(top_k):
                vote_weight = max(1, int(self.k_nearest) - rank)
                acc = slot_accumulators[slot_id]
                acc["count"] += vote_weight
                if "goal" not in acc:
                    _, _, correct_primitive_goal = slot_metadata[slot_id]
                    acc["goal"] = correct_primitive_goal

                within_tolerance = (cand_pos <= self.match_position_tolerance) and (cand_ang <= self.match_angle_tolerance)
                voted_slots_debug.append({
                    "rank": rank + 1,
                    "vote_weight": vote_weight,
                    "score": cand_score,
                    "pos_err": cand_pos,
                    "ang_err": cand_ang,
                    "slot_id": slot_id,
                    "edge_idx": edge_idx,
                    "depth_idx": depth_idx,
                    "within_tolerance": within_tolerance,
                    "reachable": (not reachable_edges) or (edge_idx in reachable_edges),
                })

            top_reachable = []
            top_all = []
            if debug_topk:
                if candidate_pool_reachable_sorted:
                    top_reachable = candidate_pool_reachable_sorted[:debug_topk]
                if candidates_all:
                    top_all = sorted(candidates_all, key=lambda x: x[0])[:debug_topk]

            if self.verbose and ml_goal_idx < 5:
                reach_note = "reachable" if (not reachable_edges or _best_edge in reachable_edges) else "unreachable"
                tol_note = "✓ within tol" if best_within_tolerance else "⊗ outside tol"
                print(
                    f"    ↪ ML goal {ml_goal_idx}: ({ml_goal.x:.3f}, {ml_goal.y:.3f}, {ml_goal.theta:.3f}) "
                    f"→ E{_best_edge}D{_best_depth+1} ({reach_note}) | pos={best_pos:.3f}, ang={best_ang:.3f} ({tol_note})"
                )
                if top_all:
                    print(
                        f"      Top-{min(debug_topk, len(top_all))} candidates "
                        f"(all edges, score_metric={self.score_metric}, w={self.angle_weight}):"
                    )
                    for rank, cand in enumerate(top_all):
                        c_score, c_pos, c_ang, _slot, c_edge, c_depth, c_reach = cand
                        reach_tag = "reachable" if c_reach else "unreachable"
                        print(f"        {rank+1}) E{c_edge}D{c_depth+1} ({reach_tag}): score={c_score:.3f}, pos={c_pos:.3f}, ang={c_ang:.3f}")
                if reachable_edges and top_reachable:
                    print(
                        f"      Top-{min(debug_topk, len(top_reachable))} candidates "
                        f"(reachable edges only, score_metric={self.score_metric}, w={self.angle_weight}):"
                    )
                    for rank, cand in enumerate(top_reachable):
                        c_score, c_pos, c_ang, _slot, c_edge, c_depth = cand
                        print(f"        {rank+1}) E{c_edge}D{c_depth+1}: score={c_score:.3f}, pos={c_pos:.3f}, ang={c_ang:.3f}")

            ml_goal_match_debug.append({
                "ml_goal_idx": ml_goal_idx,
                "ml_goal": ml_goal,
                "matched": best_within_tolerance,
                "voted_slots": voted_slots_debug,
                "best_overall": best_unfiltered,
                "top_candidates_reachable": [
                    {
                        "rank": rank + 1,
                        "score": float(c_score),
                        "pos_err": float(c_pos),
                        "ang_err": float(c_ang),
                        "slot_id": int(c_slot),
                        "edge_idx": int(c_edge),
                        "depth_idx": int(c_depth),
                    }
                    for rank, (c_score, c_pos, c_ang, c_slot, c_edge, c_depth) in enumerate(top_reachable)
                ] if top_reachable else [],
                "top_candidates_all": [
                    {
                        "rank": rank + 1,
                        "score": float(c_score),
                        "pos_err": float(c_pos),
                        "ang_err": float(c_ang),
                        "slot_id": int(c_slot),
                        "edge_idx": int(c_edge),
                        "depth_idx": int(c_depth),
                        "reachable": bool(c_reach),
                    }
                    for rank, (c_score, c_pos, c_ang, c_slot, c_edge, c_depth, c_reach) in enumerate(top_all)
                ] if top_all else [],
            })

        # Construct aligned goals from accumulators
        slot_prints = 0
        for slot_id, data in slot_accumulators.items():
            edge_idx, depth_idx, _ = slot_metadata[slot_id]
            count = data["count"]
            stored_goal = data["goal"]
            
            aligned_goals[edge_idx][depth_idx] = Goal(
                x=stored_goal.x,
                y=stored_goal.y,
                theta=stored_goal.theta,
                score=count  # Store vote count as score
            )
            slot_prints += 1
            if self.verbose and slot_prints <= 10:
                print(f"    ✓ Slot edge {edge_idx} depth {depth_idx+1}: {count} votes")

        if self.verbose:
            print(f"  ✅ Mapped {len(ml_goals)} ML goals to nearest primitive slots for {object_id}")
            print(f"     Within tolerance: {goals_within_tolerance}/{len(ml_goals)} | Outside tolerance: {goals_outside_tolerance}/{len(ml_goals)}")

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

        # Sort by votes (descending) to get execution order
        aligned_primitives_info.sort(key=lambda x: x['votes'], reverse=True)

        self._last_alignment_info = {
            'object_id': object_id,
            'object_pose': self._get_object_pose(state, env, object_id),
            'aligned_primitives': aligned_primitives_info,
            'ml_goals': ml_goals,
            'total_ml_goals': len(ml_goals),
            'total_aligned': len(aligned_primitives_info),
            'reachable_edges': reachable_edges,
            'slot_metadata': slot_metadata,
            'ml_goal_match_debug': ml_goal_match_debug,
            'xml_path': getattr(self._ml_strategy, "xml_path", None),
            'goals_within_tolerance': goals_within_tolerance,
            'goals_outside_tolerance': goals_outside_tolerance,
            'match_params': {
                'match_position_tolerance': float(self.match_position_tolerance),
                'match_angle_tolerance': float(self.match_angle_tolerance),
                'angle_weight': float(self.angle_weight),
                'k_nearest': int(self.k_nearest),
                'score_metric': str(self.score_metric),
            },
        }

        # Save visualization if enabled (even if nothing matched, to debug why)
        if self.preview_aligned_primitives:
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
        if not info:
            return

        object_id = info['object_id']
        obj_x, obj_y, obj_theta = info['object_pose']
        aligned = info.get('aligned_primitives', [])
        reachable_edges = info.get('reachable_edges', set())
        ml_goals = info.get('ml_goals', [])
        slot_metadata = info.get('slot_metadata', [])
        ml_goal_match_debug = info.get('ml_goal_match_debug', [])

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

        # Draw ALL primitive slots as faint dots for context.
        # This is intentionally lightweight (dots not boxes) to keep the plot readable.
        if slot_metadata:
            reachable_x = []
            reachable_y = []
            unreachable_x = []
            unreachable_y = []

            for edge_idx, _depth_idx, goal in slot_metadata:
                if edge_idx in reachable_edges:
                    reachable_x.append(goal.x)
                    reachable_y.append(goal.y)
                else:
                    unreachable_x.append(goal.x)
                    unreachable_y.append(goal.y)

            if unreachable_x:
                ax.scatter(unreachable_x, unreachable_y, s=6, c='lightgray', alpha=0.15, label='All primitives (unreachable)')
            if reachable_x:
                ax.scatter(reachable_x, reachable_y, s=6, c='gray', alpha=0.25, label='All primitives (reachable)')

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

        # Draw explicit ML->primitive mapping for the top-1 voted slot per ML goal (when available).
        # This makes it obvious which primitive slot the sampler will vote for.
        slot_by_id = {slot_id: entry for slot_id, entry in enumerate(slot_metadata)}
        for entry in ml_goal_match_debug[:10]:  # safety cap to avoid unreadable plots
            ml_goal = entry.get("ml_goal")
            voted_slots = entry.get("voted_slots") or []
            top_candidates_reachable = entry.get("top_candidates_reachable") or []
            top_candidates_all = entry.get("top_candidates_all") or []
            if ml_goal is None:
                continue

            # Draw top candidates (ranked by score) as faint outline boxes for sanity-checking the metric.
            # This is useful for single-sample debugging but gets cluttered for many samples.
            if len(ml_goals) <= 1 and len(voted_slots) <= 1:
                # Prefer reachable candidates (what the planner can actually try); fall back to all candidates.
                candidate_list = top_candidates_reachable if top_candidates_reachable else top_candidates_all
                for cand in candidate_list[:5]:
                    # Skip the top-1 candidate since it will be highlighted by the mapping/planner box anyway.
                    if int(cand.get("rank", 0)) == 1:
                        continue
                    slot_entry = slot_by_id.get(int(cand.get("slot_id", -1)))
                    if slot_entry is None:
                        continue
                    _e, _d, goal = slot_entry
                    outline = Rectangle(
                        (goal.x - obj_size / 2, goal.y - obj_size / 2),
                        obj_size,
                        obj_size,
                        angle=np.degrees(goal.theta),
                        rotation_point='center',
                        fill=False,
                        edgecolor='blue',
                        linewidth=1.0,
                        alpha=0.25,
                        linestyle=':',
                        zorder=7,
                    )
                    ax.add_patch(outline)
                    ax.text(
                        goal.x,
                        goal.y,
                        str(cand.get("rank")),
                        fontsize=7,
                        color='blue',
                        ha='center',
                        va='center',
                        alpha=0.8,
                        zorder=8,
                    )
            if voted_slots:
                # Draw voted primitives (k-nearest) as translucent boxes; thickness/alpha follows vote weight.
                max_vote_weight = max(int(v.get("vote_weight", 1)) for v in voted_slots) if voted_slots else 1
                for vote in voted_slots:
                    slot_entry = slot_by_id.get(int(vote.get("slot_id", -1)))
                    if slot_entry is None:
                        continue
                    _edge_idx, _depth_idx, prim_goal = slot_entry
                    within_tolerance = bool(vote.get("within_tolerance", True))
                    vote_weight = int(vote.get("vote_weight", 1))
                    rank = int(vote.get("rank", 1))

                    # Visual encoding:
                    # - Higher vote_weight => higher alpha, thicker border.
                    # - Outside tolerance => dashed border/arrow.
                    alpha = 0.06 + 0.22 * (vote_weight / max(1, max_vote_weight))
                    linewidth = 1.0 + 1.0 * (vote_weight / max(1, max_vote_weight))
                    style = '-' if within_tolerance else '--'

                    prim_rect = Rectangle(
                        (prim_goal.x - obj_size / 2, prim_goal.y - obj_size / 2),
                        obj_size,
                        obj_size,
                        angle=np.degrees(prim_goal.theta),
                        rotation_point='center',
                        fill=True,
                        facecolor='magenta',
                        edgecolor='magenta',
                        linewidth=linewidth,
                        alpha=alpha,
                        zorder=8,
                        linestyle=style,
                    )
                    ax.add_patch(prim_rect)

                    # Light mapping arrow for each voted primitive (ranked by weight).
                    arrow_alpha = 0.15 + 0.45 * (vote_weight / max(1, max_vote_weight))
                    ax.annotate(
                        '',
                        xy=(prim_goal.x, prim_goal.y),
                        xytext=(ml_goal.x, ml_goal.y),
                        arrowprops=dict(
                            arrowstyle='->',
                            color='magenta',
                            lw=1.5,
                            alpha=arrow_alpha,
                            linestyle=style,
                        ),
                    )

                    # Tiny rank label near the voted primitive (no text boxes).
                    ax.text(
                        prim_goal.x,
                        prim_goal.y,
                        str(rank),
                        fontsize=8,
                        fontweight='bold',
                        color='black',
                        ha='center',
                        va='center',
                        alpha=0.8,
                        zorder=9,
                    )
            else:
                # No mapping recorded (unexpected in normal operation). Keep a minimal, non-intrusive cue.
                best_overall = entry.get("best_overall")
                if best_overall is None:
                    continue
                _score, _slot_id, edge_idx, depth_idx, pos_err, ang_err = best_overall
                prim = slot_metadata[_slot_id][2] if _slot_id is not None and _slot_id < len(slot_metadata) else None
                if prim is None:
                    continue
                ax.annotate(
                    '',
                    xy=(prim.x, prim.y),
                    xytext=(ml_goal.x, ml_goal.y),
                    arrowprops=dict(arrowstyle='->', color='gray', lw=1.5, alpha=0.4, linestyle='--'),
                )

        # Determine which primitive the planner will try first (highest votes, reachable edges only).
        planner_choice = None
        if aligned:
            reachable_aligned = [p for p in aligned if p['edge_idx'] in reachable_edges] if reachable_edges else aligned
            if reachable_aligned:
                planner_choice = sorted(
                    reachable_aligned,
                    key=lambda x: (-float(x.get('votes', 0.0)), int(x.get('depth_idx', 0))),
                )[0]

        if planner_choice is not None:
            goal = planner_choice['goal']
            edge_idx = planner_choice['edge_idx']
            depth_idx = planner_choice['depth_idx']
            planner_rect = Rectangle(
                (goal.x - obj_size / 2, goal.y - obj_size / 2),
                obj_size,
                obj_size,
                angle=np.degrees(goal.theta),
                rotation_point='center',
                fill=True,
                facecolor='magenta',
                edgecolor='black',
                linewidth=3.0,
                alpha=0.45,
                zorder=10,
            )
            ax.add_patch(planner_rect)
            ax.text(
                goal.x,
                goal.y + obj_size / 2 + 0.05,
                f"planner tries\nE{edge_idx}D{depth_idx+1}",
                fontsize=7,
                ha='center',
                va='bottom',
                color='black',
                bbox=dict(boxstyle='round,pad=0.15', facecolor='white', alpha=0.7, edgecolor='black'),
                zorder=11,
            )

        # Set axis limits with padding (include ML goals)
        all_x = [obj_x] + [p['goal'].x for p in aligned] + [g.x for g in ml_goals] + [g.x for _e, _d, g in slot_metadata]
        all_y = [obj_y] + [p['goal'].y for p in aligned] + [g.y for g in ml_goals] + [g.y for _e, _d, g in slot_metadata]
        margin = 1.0
        ax.set_xlim(min(all_x) - margin, max(all_x) + margin)
        ax.set_ylim(min(all_y) - margin, max(all_y) + margin)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('World X (m)')
        ax.set_ylabel('World Y (m)')

        # Title with summary including reachability info
        slot_count = len(slot_metadata)
        match_params = info.get("match_params", {})
        score_metric = str(match_params.get("score_metric", self.score_metric))
        if score_metric == "pos+w*ang":
            score_note = f"score=pos+w*ang (w={match_params.get('angle_weight', self.angle_weight):.3f})"
        elif score_metric == "l2":
            score_note = f"score=sqrt(pos^2+(w*ang)^2) (w={match_params.get('angle_weight', self.angle_weight):.3f})"
        elif score_metric == "normalized_l2":
            score_note = f"score=sqrt((pos/tp)^2+(w*ang/ta)^2) (w={match_params.get('angle_weight', self.angle_weight):.3f})"
        else:
            score_note = f"score_metric={score_metric} (w={match_params.get('angle_weight', self.angle_weight):.3f})"
        match_note = (
            f"tol={match_params.get('match_position_tolerance', self.match_position_tolerance):.3f}m/"
            f"{match_params.get('match_angle_tolerance', self.match_angle_tolerance):.3f}rad, "
            f"k={match_params.get('k_nearest', self.k_nearest)}, "
            f"{score_note}"
        )
        goals_within = int(info.get("goals_within_tolerance", 0))
        goals_outside = int(info.get("goals_outside_tolerance", 0))
        ax.set_title(
            f'ML→Primitive Mapping: {object_id}\n'
            f'{info["total_aligned"]} mapped slots from {info["total_ml_goals"]} ML goals | '
            f'{slot_count} total primitive slots | {match_note}\n'
            f'Within tolerance: {goals_within} | Outside tolerance: {goals_outside}\n'
            f'Reachable edges: {sorted(list(reachable_edges)) if reachable_edges else "(unknown)"}',
            fontsize=11,
            fontweight='bold',
        )

        # Add legend
        legend_elements = [
            plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='cyan',
                      markersize=15, label='Current Position'),
            plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='none',
                      markersize=15, markeredgecolor='magenta', linestyle='--',
                      label=f'ML Predictions ({len(ml_goals)})'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='gray',
                      markersize=6, label='All primitive centers (reachable)'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='lightgray',
                      markersize=6, label='All primitive centers (unreachable)'),
            plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='magenta',
                      markersize=12, markeredgecolor='magenta', label='Voted primitives'),
            plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='magenta',
                      markersize=15, markeredgecolor='black', label='Planner-chosen primitive'),
            plt.Line2D([0], [0], color='magenta', lw=2, linestyle='-',
                       label='Mapping (within tol)'),
            plt.Line2D([0], [0], color='magenta', lw=2, linestyle='--',
                       label='Mapping (outside tol)'),
        ]
        ax.legend(handles=legend_elements, loc='upper right')

        # Save
        xml_path = info.get("xml_path")
        env_tag = None
        if xml_path:
            env_tag = os.path.splitext(os.path.basename(str(xml_path)))[0]
            env_tag = "".join(c if (c.isalnum() or c in {"-", "_"}) else "_" for c in env_tag)
        prefix = f"ml_primitive_alignment_{env_tag}_" if env_tag else "ml_primitive_alignment_"
        save_path = os.path.join(os.getcwd(), f"{prefix}{object_id}.png")
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"   📁 Saved primitive alignment preview: {save_path}")
        plt.close(fig)

        # Also emit a JSON sidecar with the exact votes/scores used, so it's easy to audit
        # cases where a voted primitive looks unintuitive from the plot alone.
        try:
            def _goal_to_dict(g: Goal) -> Dict[str, float]:
                return {"x": float(g.x), "y": float(g.y), "theta": float(g.theta)}

            slot_by_id_for_json = {slot_id: entry for slot_id, entry in enumerate(slot_metadata)}

            def _with_primitive_goal(entry: Dict[str, Any]) -> Dict[str, Any]:
                out = dict(entry)
                slot_entry = slot_by_id_for_json.get(int(out.get("slot_id", -1)))
                if slot_entry is not None:
                    _edge, _depth, prim_goal = slot_entry
                    out["primitive_goal"] = _goal_to_dict(prim_goal)
                return out

            ml_goal_debug_json = []
            for entry in ml_goal_match_debug:
                ml_goal_obj = entry.get("ml_goal")
                if ml_goal_obj is None:
                    continue

                best_overall = entry.get("best_overall")
                best_overall_json = None
                if best_overall is not None and isinstance(best_overall, (tuple, list)) and len(best_overall) >= 6:
                    _score, _slot_id, edge_idx, depth_idx, pos_err, ang_err = best_overall[:6]
                    prim = None
                    slot_entry = slot_by_id_for_json.get(int(_slot_id))
                    if slot_entry is not None:
                        prim = _goal_to_dict(slot_entry[2])
                    best_overall_json = {
                        "score": float(_score),
                        "slot_id": int(_slot_id),
                        "edge_idx": int(edge_idx),
                        "depth_idx": int(depth_idx),
                        "pos_err": float(pos_err),
                        "ang_err": float(ang_err),
                        "primitive_goal": prim,
                    }

                ml_goal_debug_json.append(
                    {
                        "ml_goal_idx": int(entry.get("ml_goal_idx", -1)),
                        "ml_goal": _goal_to_dict(ml_goal_obj),
                        "matched": bool(entry.get("matched", False)),
                        "voted_slots": [_with_primitive_goal(v) for v in (entry.get("voted_slots") or [])],
                        "top_candidates_reachable": entry.get("top_candidates_reachable") or [],
                        "top_candidates_all": entry.get("top_candidates_all") or [],
                        "best_overall": best_overall_json,
                    }
                )

            aligned_json = []
            for p in aligned:
                g = p.get("goal")
                if g is None:
                    continue
                edge_idx = int(p.get("edge_idx", -1))
                aligned_json.append(
                    {
                        "edge_idx": edge_idx,
                        "depth_idx": int(p.get("depth_idx", -1)),
                        "goal": _goal_to_dict(g),
                        "votes": float(p.get("votes", 0.0)),
                        "reachable": (not reachable_edges) or (edge_idx in reachable_edges),
                    }
                )

            planner_choice_json = None
            if planner_choice is not None:
                g = planner_choice.get("goal")
                if g is not None:
                    edge_idx = int(planner_choice.get("edge_idx", -1))
                    planner_choice_json = {
                        "edge_idx": edge_idx,
                        "depth_idx": int(planner_choice.get("depth_idx", -1)),
                        "goal": _goal_to_dict(g),
                        "votes": float(planner_choice.get("votes", 0.0)),
                        "reachable": (not reachable_edges) or (edge_idx in reachable_edges),
                    }

            debug_payload = {
                "xml_path": str(xml_path) if xml_path else None,
                "object_id": object_id,
                "object_pose": {"x": float(obj_x), "y": float(obj_y), "theta": float(obj_theta)},
                "reachable_edges": sorted(list(reachable_edges)) if reachable_edges else [],
                "match_params": info.get("match_params", {}),
                "ml_goals": [_goal_to_dict(g) for g in ml_goals],
                "aligned_primitives": aligned_json,
                "planner_choice": planner_choice_json,
                "ml_goal_match_debug": ml_goal_debug_json,
            }

            json_path = os.path.splitext(save_path)[0] + ".json"
            with open(json_path, "w") as f:
                json.dump(debug_payload, f, indent=2)
            print(f"   🧾 Saved alignment debug JSON: {json_path}")
        except Exception as e:
            print(f"   ⚠️ Failed to save alignment debug JSON: {e}")

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
    """ML-first goal selection with full primitive fallback."""

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
        goals_per_region: int = None,
        sampler_method: Optional[str] = None,
        num_steps: Optional[int] = None,
    ):
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
            seed=seed,
            goals_per_region=goals_per_region,
            sampler_method=sampler_method,
            num_steps=num_steps,
        )
        self._default_ml_samples = samples
        self._last_alignment_info = None

    def get_last_goal_stats(self) -> dict:
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
        max_goals: int
    ) -> List[List[Goal]]:
        # Phase 1: Generate ALL primitives
        primitive_goals = self._primitive_strategy.generate_goals(
            object_id, state, env, max_goals
        )

        if not primitive_goals:
            return []

        num_edges = len(primitive_goals)
        max_depth = len(primitive_goals[0]) if primitive_goals[0] else 0

        # Phase 2: Initialize output grid with all primitives (score=0 fallback)
        output_goals: List[List[Goal]] = []
        for edge_goals in primitive_goals:
            edge_output = []
            for goal in edge_goals:
                edge_output.append(Goal(
                    x=goal.x,
                    y=goal.y,
                    theta=goal.theta,
                    score=0.0
                ))
            output_goals.append(edge_output)

        # Phase 3: Run ML inference
        ml_goal_budget = max_goals if max_goals > 0 else self._default_ml_samples
        ml_goals = self._ml_strategy.generate_goals(
            object_id, state, env, ml_goal_budget
        )

        if self.verbose:
            print(f"🎯 ML-Primitive Fallback for {object_id}:")
            print(f"   Primitive grid: {num_edges} edges × {max_depth} depths = {num_edges * max_depth} total")
            print(f"   ML goals received: {len(ml_goals)}")

        # Phase 4: Align ML goals to primitives and update scores
        slot_metadata = self._build_slot_metadata(primitive_goals)
        slot_votes: Dict[int, int] = defaultdict(int)
        aligned_count = 0
        skipped_tolerance = 0

        for ml_goal in ml_goals:
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

            candidates_within_tolerance.sort(key=lambda x: x[0])
            top_k = candidates_within_tolerance[:self.k_nearest]

            for score, slot_id, edge_idx, depth_idx in top_k:
                slot_votes[slot_id] += 1

            aligned_count += 1

        ml_aligned_slots = 0
        for slot_id, votes in slot_votes.items():
            edge_idx, depth_idx, _ = slot_metadata[slot_id]
            old_goal = output_goals[edge_idx][depth_idx]
            output_goals[edge_idx][depth_idx] = Goal(
                x=old_goal.x,
                y=old_goal.y,
                theta=old_goal.theta,
                score=float(votes)
            )
            ml_aligned_slots += 1

        fallback_count = num_edges * max_depth - ml_aligned_slots

        if self.verbose:
            print(f"   ML-aligned slots: {ml_aligned_slots} (score > 0)")
            print(f"   Fallback slots: {fallback_count} (score = 0)")
            if skipped_tolerance > 0:
                print(f"   ML goals outside tolerance: {skipped_tolerance}")

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
        aligned_primitives_info.sort(key=lambda x: x['votes'], reverse=True)

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
        slots = []
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
        ang_err = abs(MLPrimitiveFallbackStrategy._wrap_angle(primitive_goal.theta - ml_goal.theta))
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
        return "ML Primitive Fallback Goal Generation"


@dataclass
class AsyncGoalResult:
    """Result from async goal generation with ML inference running in background.

    The caller can start executing fallback primitives immediately while ML
    inference runs on a background thread. When ML completes, call
    `get_ml_scores()` to retrieve slot vote scores for reprioritization.
    """

    # Immediate: primitives (score=0) for execution
    primitive_goals: List[List[Goal]]

    # Async: ML inference future (None if ML unavailable)
    ml_future: Optional[Future] = None

    # Strategy reference used to align ML goals when ready
    _strategy_ref: Optional['MLPrimitiveAsyncStrategy'] = None
    _object_id: Optional[str] = None

    # Merge state
    ml_merged: bool = False
    ml_aligned_slots: Optional[Dict[Tuple[int, int], float]] = None

    # Stats
    ml_goals_count: int = 0
    ml_inference_time_ms: float = 0.0

    def poll_ml_ready(self) -> bool:
        if self.ml_future is None or self.ml_merged:
            return False
        return self.ml_future.done()

    def get_ml_scores(self) -> Dict[Tuple[int, int], float]:
        """Get ML slot votes mapping (edge_idx, depth_idx) -> vote_count.

        Blocks if ML future is not done.
        """
        if self.ml_merged:
            return self.ml_aligned_slots or {}

        if self.ml_future is None:
            self.ml_merged = True
            self.ml_aligned_slots = {}
            return {}

        try:
            ml_result = self.ml_future.result()
            ml_goals = ml_result.get('goals', [])
            self.ml_inference_time_ms = ml_result.get('inference_time_ms', 0.0)
            self.ml_goals_count = len(ml_goals)

            if self._strategy_ref and ml_goals:
                self.ml_aligned_slots = self._strategy_ref._align_ml_to_primitives(
                    ml_goals,
                    self.primitive_goals,
                    self._object_id or ""
                )
            else:
                self.ml_aligned_slots = {}
        except Exception as e:
            print(f"⚠️ Async ML inference failed: {e}")
            self.ml_aligned_slots = {}

        self.ml_merged = True
        return self.ml_aligned_slots or {}


class MLPrimitiveAsyncStrategy(GoalSelectionStrategy):
    """Generate primitives immediately; run ML inference asynchronously."""

    _executor: Optional[ThreadPoolExecutor] = None
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
        max_matches: int = 8,
        verbose: bool = False,
        min_goals_threshold: int = 1,
        xml_path: str = None,
        preview_mask_count: int = 0,
        preloaded_model=None,
        k_nearest: int = 1,
        max_workers: int = 1,
        seed: int = None,
        goals_per_region: int = None,
        sampler_method: Optional[str] = None,
        num_steps: Optional[int] = None,
    ):
        self.verbose = verbose
        self.max_matches = max_matches
        self.match_position_tolerance = match_position_tolerance
        self.match_angle_tolerance = match_angle_tolerance
        self.angle_weight = angle_weight
        self.k_nearest = max(1, int(k_nearest)) if k_nearest is not None else 1

        self._primitive_strategy = PrimitiveGoalStrategy(
            data_dir=primitive_data_dir,
            verbose=False,
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
            goals_per_region=goals_per_region,
            sampler_method=sampler_method,
            num_steps=num_steps,
        )
        self._default_ml_samples = samples

        # Store last stats for failure tracking
        self._last_goal_stats: Optional[Dict[str, Any]] = None

        # Initialize singleton executor
        if MLPrimitiveAsyncStrategy._executor is None:
            MLPrimitiveAsyncStrategy._executor = ThreadPoolExecutor(
                max_workers=max(1, int(max_workers)),
                thread_name_prefix="ml_async",
            )
            MLPrimitiveAsyncStrategy._cancel_event = threading.Event()

    def _slot_score(self, pos_err: float, ang_err: float) -> float:
        return pos_err + (self.angle_weight * ang_err)

    @staticmethod
    def _wrap_angle(theta: float) -> float:
        while theta > math.pi:
            theta -= 2 * math.pi
        while theta < -math.pi:
            theta += 2 * math.pi
        return theta

    @staticmethod
    def _goal_error(primitive_goal: Goal, ml_goal: Goal) -> Tuple[float, float]:
        pos_err = math.hypot(
            primitive_goal.x - ml_goal.x,
            primitive_goal.y - ml_goal.y
        )
        ang_err = abs(MLPrimitiveAsyncStrategy._wrap_angle(primitive_goal.theta - ml_goal.theta))
        return pos_err, ang_err

    def _align_ml_to_primitives(
        self,
        ml_goals: List[Goal],
        primitive_goals: List[List[Goal]],
        object_id: str
    ) -> Dict[Tuple[int, int], float]:
        if not ml_goals or not primitive_goals:
            self._last_goal_stats = {
                'ml_goals_generated': len(ml_goals) if ml_goals else 0,
                'ml_goals_aligned': 0,
                'reachable_edges_count': 0,
                'aligned_primitives': [],
                'ml_goals_raw': [],
                'reachable_edges': [],
            }
            return {}

        slot_metadata: List[Tuple[int, int, Goal]] = []
        for edge_idx, edge_goals in enumerate(primitive_goals):
            for depth_idx, goal in enumerate(edge_goals):
                slot_metadata.append((edge_idx, depth_idx, goal))

        slot_votes: Dict[Tuple[int, int], int] = defaultdict(int)

        for ml_goal in ml_goals:
            candidates: List[Tuple[float, int, int]] = []
            for edge_idx, depth_idx, prim_goal in slot_metadata:
                pos_err, ang_err = self._goal_error(prim_goal, ml_goal)
                if pos_err > self.match_position_tolerance:
                    continue
                if ang_err > self.match_angle_tolerance:
                    continue
                score = self._slot_score(pos_err, ang_err)
                candidates.append((score, edge_idx, depth_idx))

            if not candidates:
                continue

            candidates.sort(key=lambda x: x[0])
            for _, edge_idx, depth_idx in candidates[: self.k_nearest]:
                slot_votes[(edge_idx, depth_idx)] += 1

        slot_scores = {k: float(v) for k, v in slot_votes.items()}

        # Store last goal stats in the same schema as other ML strategies
        aligned_primitives = [
            {
                'edge_idx': edge_idx,
                'depth_idx': depth_idx,
                'x': None,
                'y': None,
                'theta': None,
                'votes': votes,
            }
            for (edge_idx, depth_idx), votes in sorted(
                slot_scores.items(), key=lambda kv: (-kv[1], kv[0][0], kv[0][1])
            )
        ]
        self._last_goal_stats = {
            'ml_goals_generated': len(ml_goals),
            'ml_goals_aligned': len(slot_scores),
            'reachable_edges_count': 0,
            'aligned_primitives': aligned_primitives,
            'ml_goals_raw': [{'x': g.x, 'y': g.y, 'theta': g.theta} for g in ml_goals],
            'reachable_edges': [],
        }

        return slot_scores

    def get_last_goal_stats(self) -> dict:
        return self._last_goal_stats or {
            'ml_goals_generated': 0,
            'ml_goals_aligned': 0,
            'reachable_edges_count': 0,
            'aligned_primitives': [],
            'ml_goals_raw': [],
            'reachable_edges': [],
        }

    def generate_goals(
        self,
        object_id: str,
        state: namo_rl.RLState,
        env: namo_rl.RLEnvironment,
        max_goals: int
    ) -> AsyncGoalResult:
        primitive_goals = self._primitive_strategy.generate_goals(object_id, state, env, max_goals)
        if not primitive_goals:
            return AsyncGoalResult(primitive_goals=[], ml_future=None)

        # Initialize primitives with score=0 (fallback)
        output_goals: List[List[Goal]] = []
        for edge_goals in primitive_goals:
            edge_output = []
            for goal in edge_goals:
                edge_output.append(Goal(x=goal.x, y=goal.y, theta=goal.theta, score=0.0))
            output_goals.append(edge_output)

        # Capture JSON in main thread (env is not thread-safe)
        try:
            json_message = self._ml_strategy._create_json_message_for_goals(object_id, state, env)
        except Exception as e:
            if self.verbose:
                print(f"⚠️ Failed to create JSON for async ML: {e}")
            json_message = None

        if json_message is None or MLPrimitiveAsyncStrategy._executor is None:
            return AsyncGoalResult(
                primitive_goals=output_goals,
                ml_future=None,
                _strategy_ref=self,
                _object_id=object_id,
            )

        ml_budget = max_goals if max_goals > 0 else self._default_ml_samples
        ml_future = MLPrimitiveAsyncStrategy._executor.submit(
            self._run_ml_inference_only,
            json_message,
            object_id,
            ml_budget,
        )

        return AsyncGoalResult(
            primitive_goals=output_goals,
            ml_future=ml_future,
            _strategy_ref=self,
            _object_id=object_id,
            ml_merged=False,
            ml_aligned_slots=None,
        )

    def _run_ml_inference_only(
        self,
        json_message: Dict[str, Any],
        object_id: str,
        ml_budget: int,
    ) -> Dict[str, Any]:
        import time
        start_time = time.time()

        cancel = MLPrimitiveAsyncStrategy._cancel_event
        if cancel is not None and cancel.is_set():
            return {'goals': [], 'inference_time_ms': 0.0, 'cancelled': True}

        try:
            if self._ml_strategy._goal_model is None:
                self._ml_strategy._load_model()
            if self._ml_strategy._goal_model is None:
                return {'goals': [], 'inference_time_ms': (time.time() - start_time) * 1000, 'error': 'Model failed to load'}

            # Run model inference (thread-safe: uses captured JSON only)
            goals = self._ml_strategy._goal_model.infer(
                json_message=json_message,
                xml_path=json_message.get("xml_path"),
                robot_goal=json_message.get("robot_goal"),
                selected_object=object_id,
                samples=ml_budget,
            )

            inference_time_ms = (time.time() - start_time) * 1000
            if cancel is not None and cancel.is_set():
                return {'goals': [], 'inference_time_ms': inference_time_ms, 'cancelled': True}

            # Convert to Goal objects
            goal_objects: List[Goal] = []
            for goal_data in (goals or []):
                if isinstance(goal_data, Goal):
                    goal_objects.append(goal_data)
                    continue
                if isinstance(goal_data, dict) and 'x' in goal_data and 'y' in goal_data and 'theta' in goal_data:
                    goal_objects.append(Goal(x=float(goal_data['x']), y=float(goal_data['y']), theta=float(goal_data['theta'])))

            return {'goals': goal_objects, 'inference_time_ms': inference_time_ms}
        except Exception as e:
            return {'goals': [], 'inference_time_ms': (time.time() - start_time) * 1000, 'error': str(e)}

    @property
    def strategy_name(self) -> str:
        return "ML Primitive Async Goal Generation"
