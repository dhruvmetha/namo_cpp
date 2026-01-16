#!/usr/bin/env python3
"""
2-Push Evaluation Script

Compares multiple learned models for 2-push problems.
Categorizes problems by chain depth (1-push vs 2-push) based on reference.

Usage:
    python eval_2push.py --config eval_2push_config.yaml
    python eval_2push.py --config eval_2push_config.yaml --output-dir ./my_plots

Example config (eval_2push_config.yaml):

    reference:
      name: "Search"
      dir: /path/to/search/results

    baselines:
      - name: "No Heuristic"
        dir: /path/to/no_heuristic/results

    learned:
      - name: "Model A"
        dir: /path/to/model_a/results
      - name: "Model B"
        dir: /path/to/model_b/results

    settings:
      output_dir: ./eval_plots
      time_cutoff_max: 100000
      time_step: 500
"""

import pickle
import argparse
from glob import glob
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Generator, List, Optional, Tuple, Set
from collections import defaultdict

import yaml
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl


# =============================================================================
# Shared Utilities
# =============================================================================

# Type aliases for clarity
EnvKeyPair = Tuple[str, str]
ModelData = Dict[str, Dict[str, 'RegionResult']]


def iter_matched_triplets(
    model_data: ModelData,
    reference_data: ModelData,
    require_ref_success: bool = True,
) -> Generator[Tuple[str, str, 'RegionResult', 'RegionResult'], None, None]:
    """
    Iterate over triplets present in both model and reference data.

    Yields (env, key, ref_result, model_result) tuples.
    This eliminates the repeated nested loop pattern throughout the codebase.
    """
    for env in model_data:
        if env not in reference_data:
            continue
        for key in model_data[env]:
            if key not in reference_data[env]:
                continue
            ref_result = reference_data[env][key]
            if require_ref_success and not ref_result.success:
                continue
            yield env, key, ref_result, model_data[env][key]


def compute_stats_list(values: List[float]) -> Dict[str, float]:
    """Compute common statistics (median, mean, IQR) for a list of values."""
    if not values:
        return {'median': 0.0, 'mean': 0.0, 'p25': 0.0, 'p75': 0.0, 'std': 0.0}
    arr = np.array(values)
    return {
        'median': float(np.median(arr)),
        'mean': float(np.mean(arr)),
        'p25': float(np.percentile(arr, 25)),
        'p75': float(np.percentile(arr, 75)),
        'std': float(np.std(arr)),
    }


def get_collision_category(ref_result: 'RegionResult') -> str:
    """Determine collision category from reference result."""
    has_wall = ref_result.wall_collision
    has_movable = ref_result.movable_collisions > 0
    if has_wall and has_movable:
        return 'both'
    elif has_wall:
        return 'wall_only'
    elif has_movable:
        return 'movable_only'
    return 'none'


def compute_percentile_thresholds(
    values: List[float],
    percentiles: Tuple[float, float] = (33.33, 66.67),
) -> Dict[str, float]:
    """Compute percentile thresholds for difficulty categorization."""
    if not values:
        return {'p33': 0.0, 'p66': 0.0, 'min': 0.0, 'max': 0.0}
    return {
        'p33': float(np.percentile(values, percentiles[0])),
        'p66': float(np.percentile(values, percentiles[1])),
        'min': float(min(values)),
        'max': float(max(values)),
    }


def assign_difficulty(value: float, thresholds: Dict[str, float]) -> str:
    """Assign difficulty category based on thresholds."""
    if value <= thresholds['p33']:
        return 'easy'
    elif value <= thresholds['p66']:
        return 'medium'
    return 'hard'

# Set up nicer plot style
plt.style.use('seaborn-v0_8-whitegrid')
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.size'] = 11
mpl.rcParams['axes.titlesize'] = 14
mpl.rcParams['axes.titleweight'] = 'bold'
mpl.rcParams['axes.labelsize'] = 12
mpl.rcParams['xtick.labelsize'] = 10
mpl.rcParams['ytick.labelsize'] = 10
mpl.rcParams['legend.fontsize'] = 10
mpl.rcParams['figure.facecolor'] = 'white'
mpl.rcParams['axes.facecolor'] = 'white'
mpl.rcParams['axes.edgecolor'] = '#333333'
mpl.rcParams['axes.linewidth'] = 0.8
mpl.rcParams['grid.alpha'] = 0.3
mpl.rcParams['axes.spines.top'] = False
mpl.rcParams['axes.spines.right'] = False


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class ModelConfig:
    """Configuration for a single model."""
    name: str
    dir: str
    color: Optional[str] = None
    success_dir: Optional[str] = None  # Custom name for success subdirectory


@dataclass
class EvalConfig:
    """Configuration for 2-push evaluation."""
    # References (multiple oracle runs with different seeds for consistency analysis)
    references: List[ModelConfig] = field(default_factory=list)

    # Legacy: single reference (for backwards compatibility)
    reference: ModelConfig = None

    # Baselines (non-learned)
    baselines: List[ModelConfig] = field(default_factory=list)

    # Learned models to compare
    learned: List[ModelConfig] = field(default_factory=list)

    # Plot settings
    output_dir: str = "./eval_2push_plots"
    time_cutoff_max: int = 10000  # ms
    time_step: int = 100  # ms
    push_cutoff_max: int = 20  # max number of pushes
    push_step: int = 1  # step size for push cutoffs

    # ReachableAttachment@K thresholds (None = all ranked primitives)
    ra_at_k_values: List[Optional[int]] = field(default_factory=lambda: [10, 50, 100, None])

    # Success@B budget values (number of simulation-verified checks)
    success_at_budget_values: List[int] = field(default_factory=lambda: [50, 100, 200])

    # Success@T time budget values (milliseconds)
    success_at_time_values: List[float] = field(default_factory=lambda: [5000, 10000, 30000])

    # Colors for models (colorblind-friendly palette)
    model_colors: List[str] = field(default_factory=lambda: [
        '#4C72B0',  # muted blue
        '#DD8452',  # muted orange
        '#55A868',  # muted green
        '#C44E52',  # muted red
        '#8172B3',  # muted purple
        '#937860',  # muted brown
        '#DA8BC3',  # muted pink
        '#8C8C8C',  # gray
    ])

    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'EvalConfig':
        """Load configuration from YAML file."""
        with open(yaml_path, 'r') as f:
            data = yaml.safe_load(f)

        config = cls()

        # Parse reference(s) - supports both single and multiple
        if 'reference' in data:
            ref_data = data['reference']
            # Check if it's a list (multiple references) or single dict
            if isinstance(ref_data, list):
                for ref in ref_data:
                    config.references.append(ModelConfig(
                        name=ref.get('name', 'Search'),
                        dir=ref['dir'],
                        color=ref.get('color'),
                    ))
                # Set first reference as primary (for backwards compatibility)
                if config.references:
                    config.reference = config.references[0]
            else:
                # Single reference (legacy format)
                config.reference = ModelConfig(
                    name=ref_data.get('name', 'Search'),
                    dir=ref_data['dir'],
                    color=ref_data.get('color'),
                )
                config.references = [config.reference]

        # Parse baselines
        if 'baselines' in data and data['baselines']:
            for b in data['baselines']:
                config.baselines.append(ModelConfig(
                    name=b.get('name', 'Baseline'),
                    dir=b['dir'],
                    color=b.get('color'),
                ))

        # Parse learned models (list)
        if 'learned' in data and data['learned']:
            for m in data['learned']:
                config.learned.append(ModelConfig(
                    name=m.get('name', 'Learned'),
                    dir=m['dir'],
                    color=m.get('color'),
                    success_dir=m.get('success_dir'),
                ))

        if 'settings' in data:
            settings = data['settings']
            config.output_dir = settings.get('output_dir', config.output_dir)
            config.time_cutoff_max = settings.get('time_cutoff_max', config.time_cutoff_max)
            config.time_step = settings.get('time_step', config.time_step)
            config.push_cutoff_max = settings.get('push_cutoff_max', config.push_cutoff_max)
            config.push_step = settings.get('push_step', config.push_step)

        return config


# =============================================================================
# Data Loading
# =============================================================================

@dataclass
class RegionResult:
    """Results for a single env+region+object triplet."""
    success: bool = False
    pushes: int = 0
    solutions: int = 0  # solutions_total_for_neighbour
    solutions_found: int = 0  # solutions_found_for_neighbour
    ratio: float = 0.0
    time_taken: float = 0.0
    failure_reason: str = ""
    xml_file: str = ""
    region: str = ""
    object_id: str = ""
    chain_depth: int = 0
    ml_goals_raw: List[Any] = field(default_factory=list)
    search_solutions: List[Any] = field(default_factory=list)
    # Interaction types
    wall_collision: bool = False
    movable_collisions: int = 0
    # Explicit phase tracking fields
    phase_push_counts: Optional[Dict[str, int]] = None  # {"ML-only": X, "primitives": Y}
    solved_in_phase: str = ""  # "ML-only", "primitives", or ""
    # ML prediction grounding metrics
    ml_aligned_count: int = 0  # Number of aligned primitives from ML
    ml_aligned_reachable_count: int = 0  # Number of those that are reachable
    # Full data for RA@K computation
    aligned_primitives: List[Dict] = field(default_factory=list)  # Each has 'edge_idx', 'votes'
    reachable_edges: set = field(default_factory=set)  # Set of reachable edge indices

    @property
    def solved_by_learned(self) -> bool:
        """Check if solved by learned stage (ML-only phase)."""
        return self.success and self.solved_in_phase == "ML-only"

    @property
    def solved_by_fallback(self) -> bool:
        """Check if solved by fallback stage (primitives phase)."""
        return self.success and self.solved_in_phase == "primitives"

    @property
    def ml_aligned_reachable_ratio(self) -> float:
        """Fraction of aligned ML primitives that are reachable (grounding metric)."""
        if self.ml_aligned_count == 0:
            return 0.0
        return self.ml_aligned_reachable_count / self.ml_aligned_count


def load_pickle_data(
    data_dir: str,
    reference_data: Optional[Dict[str, Dict[str, RegionResult]]] = None,
) -> Tuple[Dict[str, Dict[str, RegionResult]], Dict[str, int]]:
    """
    Load evaluation data from pickle files.

    Args:
        data_dir: Glob pattern for pickle files
        reference_data: If provided, only include triplets that exist in reference

    Returns:
        per_env_per_key: {xml_file_name: {region_label::object_id: RegionResult}}
        failure_reasons: {reason: count}
    """
    per_env_per_key: Dict[str, Dict[str, RegionResult]] = {}
    failure_reasons: Dict[str, int] = defaultdict(int)

    for file in glob(data_dir, recursive=True):
        if 'collection_summary' in file:
            continue

        try:
            with open(file, "rb") as f:
                data = pickle.load(f)

            episode_results = data.get('episode_results', [])
            if not episode_results:
                continue

            keys_done = set()

            for ep in episode_results:
                xml_file = ep.get('xml_file', '')
                xml_file_name = xml_file  # Use full path as env identifier

                alg_stats = ep.get('algorithm_stats', {})
                region_label = alg_stats.get('neighbour_region_label')
                object_id = alg_stats.get('chosen_object_id', '')

                if region_label is None or not object_id:
                    continue

                # Key by (env, region, object) triplet
                key = f"{region_label}::{object_id}"

                # If reference provided, only include matching triplets
                if reference_data is not None:
                    if xml_file_name not in reference_data:
                        continue
                    if key not in reference_data[xml_file_name]:
                        continue
                    if not reference_data[xml_file_name][key].success:
                        continue

                # Only process each key once per file
                if key in keys_done:
                    continue
                keys_done.add(key)

                # Track failure reasons
                failure_reason = alg_stats.get('failure_reason', 'unknown')
                failure_reasons[failure_reason] += 1

                # Initialize env dict if needed
                if xml_file_name not in per_env_per_key:
                    per_env_per_key[xml_file_name] = {}

                # Extract stats
                pushes = alg_stats.get('pushes_total_for_neighbour', 0)
                solutions = alg_stats.get('solutions_total_for_neighbour', 0)
                solutions_found = alg_stats.get('solutions_found_for_neighbour', 0)
                solution_found = ep.get('solution_found', False)
                time_taken = ep.get('search_time_ms', 0)
                chain_depth = alg_stats.get('chain_depth', 0)

                # Extract interaction types
                wall_collision = ep.get('any_wall_collision', False)
                movable_collisions = ep.get('unique_movable_collision_count', 0)

                # Extract explicit phase tracking fields
                phase_push_counts = alg_stats.get('phase_push_counts', None)
                solved_in_phase = alg_stats.get('solved_in_phase', "")

                # Extract ML grounding metrics (aligned primitives vs reachable edges)
                aligned_primitives = alg_stats.get('aligned_primitives') or []
                reachable_edges_raw = alg_stats.get('reachable_edges') or []
                reachable_edges = set(reachable_edges_raw)
                ml_aligned_count = len(aligned_primitives)
                ml_aligned_reachable_count = 0
                if aligned_primitives and reachable_edges:
                    ml_aligned_reachable_count = sum(
                        1 for p in aligned_primitives
                        if p.get('edge_idx') in reachable_edges
                    )

                result = RegionResult(
                    success=solution_found and pushes > 0,
                    pushes=pushes,
                    solutions=solutions,
                    solutions_found=solutions_found,
                    ratio=solutions / pushes if pushes > 0 else 0.0,
                    time_taken=time_taken,
                    failure_reason=failure_reason,
                    xml_file=xml_file,
                    region=region_label,
                    object_id=object_id,
                    chain_depth=chain_depth,
                    ml_goals_raw=alg_stats.get('ml_goals_raw', []),
                    search_solutions=ep.get('search_solutions', []),
                    wall_collision=wall_collision,
                    movable_collisions=movable_collisions,
                    phase_push_counts=phase_push_counts,
                    solved_in_phase=solved_in_phase,
                    ml_aligned_count=ml_aligned_count,
                    ml_aligned_reachable_count=ml_aligned_reachable_count,
                    aligned_primitives=aligned_primitives,
                    reachable_edges=reachable_edges,
                )

                per_env_per_key[xml_file_name][key] = result

        except Exception as e:
            print(f"Error loading {file}: {e}")
            continue

    return per_env_per_key, dict(failure_reasons)


# =============================================================================
# Analysis
# =============================================================================

@dataclass
class DepthStats:
    """Statistics for a specific chain depth."""
    pushes: List[int] = field(default_factory=list)
    times: List[float] = field(default_factory=list)
    solutions: List[int] = field(default_factory=list)
    solutions_found: List[int] = field(default_factory=list)
    successes: int = 0
    total: int = 0
    # Interaction tracking
    wall_collisions: int = 0
    movable_collisions_list: List[int] = field(default_factory=list)
    # ML grounding metrics (how many aligned primitives are reachable)
    ml_aligned_counts: List[int] = field(default_factory=list)
    ml_aligned_reachable_counts: List[int] = field(default_factory=list)

    @property
    def success_rate(self) -> float:
        return self.successes / self.total if self.total > 0 else 0.0

    @property
    def median_pushes(self) -> float:
        return float(np.median(self.pushes)) if self.pushes else 0.0

    @property
    def mean_pushes(self) -> float:
        return float(np.mean(self.pushes)) if self.pushes else 0.0

    @property
    def pushes_iqr(self) -> Tuple[float, float]:
        """Return (25th percentile, 75th percentile) for pushes."""
        if not self.pushes:
            return (0.0, 0.0)
        return (float(np.percentile(self.pushes, 25)), float(np.percentile(self.pushes, 75)))

    @property
    def median_time(self) -> float:
        return float(np.median(self.times)) if self.times else 0.0

    @property
    def mean_time(self) -> float:
        return float(np.mean(self.times)) if self.times else 0.0

    @property
    def time_iqr(self) -> Tuple[float, float]:
        """Return (25th percentile, 75th percentile) for times."""
        if not self.times:
            return (0.0, 0.0)
        return (float(np.percentile(self.times, 25)), float(np.percentile(self.times, 75)))

    @property
    def total_solutions(self) -> int:
        return int(np.sum(self.solutions)) if self.solutions else 0

    @property
    def mean_solutions(self) -> float:
        return float(np.mean(self.solutions)) if self.solutions else 0.0

    @property
    def wall_collision_rate(self) -> float:
        """Rate of successful runs that had wall collisions."""
        return self.wall_collisions / self.successes if self.successes > 0 else 0.0

    @property
    def mean_movable_collisions(self) -> float:
        """Mean number of movable object collisions per successful run."""
        return float(np.mean(self.movable_collisions_list)) if self.movable_collisions_list else 0.0

    @property
    def any_movable_collision_rate(self) -> float:
        """Rate of successful runs that had any movable collisions."""
        if not self.movable_collisions_list:
            return 0.0
        return sum(1 for c in self.movable_collisions_list if c > 0) / len(self.movable_collisions_list)

    @property
    def mean_ml_aligned_reachable_ratio(self) -> float:
        """Mean fraction of aligned ML primitives that are reachable (grounding metric)."""
        if not self.ml_aligned_counts:
            return 0.0
        # Compute ratio per-sample, then average
        ratios = []
        for aligned, reachable in zip(self.ml_aligned_counts, self.ml_aligned_reachable_counts):
            if aligned > 0:
                ratios.append(reachable / aligned)
        return float(np.mean(ratios)) if ratios else 0.0

    @property
    def total_ml_aligned(self) -> int:
        """Total aligned ML primitives across all samples."""
        return sum(self.ml_aligned_counts)

    @property
    def total_ml_aligned_reachable(self) -> int:
        """Total aligned ML primitives that are reachable across all samples."""
        return sum(self.ml_aligned_reachable_counts)

    @property
    def aggregate_ml_aligned_reachable_ratio(self) -> float:
        """Aggregate ratio: total reachable / total aligned (different from mean of ratios)."""
        total_aligned = self.total_ml_aligned
        if total_aligned == 0:
            return 0.0
        return self.total_ml_aligned_reachable / total_aligned


@dataclass
class ModelStats:
    """Statistics for a model across chain depths."""
    name: str
    depth_1: DepthStats = field(default_factory=DepthStats)
    depth_2: DepthStats = field(default_factory=DepthStats)
    failure_reasons: Dict[str, int] = field(default_factory=dict)

    def get_depth(self, name: str) -> DepthStats:
        return getattr(self, name)

    @property
    def total_successes(self) -> int:
        return self.depth_1.successes + self.depth_2.successes

    @property
    def total_trials(self) -> int:
        return self.depth_1.total + self.depth_2.total

    @property
    def overall_success_rate(self) -> float:
        return self.total_successes / self.total_trials if self.total_trials > 0 else 0.0

    @property
    def all_pushes(self) -> List[int]:
        return self.depth_1.pushes + self.depth_2.pushes

    @property
    def all_times(self) -> List[float]:
        return self.depth_1.times + self.depth_2.times


# =============================================================================
# ReachableAttachment@K Computation
# =============================================================================

def compute_ra_at_k_single(
    aligned_primitives: List[Dict],
    reachable_edges: set,
    k: Optional[int] = None,
) -> Tuple[int, int]:
    """
    Compute ReachableAttachment@K for a single instance.

    Args:
        aligned_primitives: List of dicts with 'edge_idx' and 'votes'
        reachable_edges: Set of reachable edge indices
        k: Top-K to consider (None = all ranked)

    Returns:
        (num_reachable, num_considered)
    """
    if not aligned_primitives:
        return (0, 0)

    # Sort by votes descending (higher score = better rank)
    sorted_prims = sorted(aligned_primitives, key=lambda p: -p.get('votes', 1))

    # Take top-K (or all if K is None or larger than available)
    if k is not None:
        sorted_prims = sorted_prims[:k]

    num_considered = len(sorted_prims)
    num_reachable = sum(1 for p in sorted_prims if p.get('edge_idx') in reachable_edges)

    return (num_reachable, num_considered)


@dataclass
class RAatKStats:
    """ReachableAttachment@K statistics."""
    k: Optional[int]  # None means all
    macro: float = 0.0  # Mean of per-instance fractions
    micro: float = 0.0  # Total reachable / total considered
    n_instances: int = 0
    total_reachable: int = 0
    total_considered: int = 0


def compute_ra_at_k(
    model_data: Dict[str, Dict[str, RegionResult]],
    reference_data: Dict[str, Dict[str, RegionResult]],
    k: Optional[int] = None,
    depth_filter: Optional[int] = None,
) -> RAatKStats:
    """
    Compute ReachableAttachment@K across all instances.

    Args:
        model_data: Model results by env/key
        reference_data: Reference results for filtering
        k: Top-K to consider (None = all)
        depth_filter: Only include problems with this chain depth (1 or 2)

    Returns:
        RAatKStats with macro/micro averages
    """
    results = []  # List of (reachable, considered) per instance

    for _, _, ref_result, model_result in iter_matched_triplets(model_data, reference_data):
        if depth_filter is not None and ref_result.chain_depth != depth_filter:
            continue

        reachable, considered = compute_ra_at_k_single(
            model_result.aligned_primitives, model_result.reachable_edges, k=k
        )
        if considered > 0:
            results.append((reachable, considered))

    stats = RAatKStats(k=k)
    if not results:
        return stats

    stats.n_instances = len(results)
    stats.total_reachable = sum(r for r, c in results)
    stats.total_considered = sum(c for r, c in results)
    per_instance_fracs = [r / c for r, c in results]
    stats.macro = float(np.mean(per_instance_fracs))
    stats.micro = stats.total_reachable / stats.total_considered if stats.total_considered > 0 else 0.0

    return stats


def compute_random_baseline(
    model_data: Dict[str, Dict[str, RegionResult]],
    reference_data: Dict[str, Dict[str, RegionResult]],
    depth_filter: Optional[int] = None,
    total_edges: int = 60,
) -> float:
    """
    Compute expected RA under random primitive selection.

    Returns mean of (|reachable_edges| / total_edges) across instances.
    """
    ratios = []

    for _, _, ref_result, model_result in iter_matched_triplets(model_data, reference_data):
        if depth_filter is not None and ref_result.chain_depth != depth_filter:
            continue
        if model_result.aligned_primitives:
            ratios.append(len(model_result.reachable_edges) / total_edges)

    return float(np.mean(ratios)) if ratios else 0.0


def categorize_problem(ref_result: RegionResult) -> str:
    """
    Categorize a problem as 'blocked', '1push', '2push', or 'unsolvable'.

    - blocked: pushes=0 and not successful (robot can't reach any objects)
    - 1push: chain_depth=1 and successful (true 1-push solvable)
    - 2push: chain_depth=2 and successful (true 2-push solvable)
    - unsolvable: search failed (exhausted budget)
    """
    if ref_result.pushes == 0 and not ref_result.success:
        return 'blocked'
    elif ref_result.chain_depth == 1 and ref_result.success:
        return '1push'
    elif ref_result.chain_depth == 2 and ref_result.success:
        return '2push'
    else:
        # Search failed (exhausted budget) - not solvable by search
        return 'unsolvable'


def compute_stats(
    model_data: Dict[str, Dict[str, RegionResult]],
    reference_data: Dict[str, Dict[str, RegionResult]],
    model_name: str,
    failure_reasons: Optional[Dict[str, int]] = None,
) -> ModelStats:
    """
    Compute statistics for a model, categorized by chain depth.

    Chain depth is determined by the reference (search) solution.
    """
    stats = ModelStats(name=model_name)
    if failure_reasons:
        stats.failure_reasons = failure_reasons

    for _, _, ref_result, model_result in iter_matched_triplets(
        model_data, reference_data, require_ref_success=False
    ):
        category = categorize_problem(ref_result)
        if category in ('blocked', 'unsolvable'):
            continue

        depth_stats = stats.depth_1 if category == '1push' else stats.depth_2
        depth_stats.total += 1

        # Track ML grounding metrics for all samples (not just successful)
        if model_result.ml_aligned_count > 0:
            depth_stats.ml_aligned_counts.append(model_result.ml_aligned_count)
            depth_stats.ml_aligned_reachable_counts.append(model_result.ml_aligned_reachable_count)

        if model_result.success:
            depth_stats.successes += 1
            depth_stats.pushes.append(model_result.pushes)
            depth_stats.times.append(model_result.time_taken)
            depth_stats.solutions.append(model_result.solutions)
            depth_stats.solutions_found.append(model_result.solutions_found)
            if model_result.wall_collision:
                depth_stats.wall_collisions += 1
            depth_stats.movable_collisions_list.append(model_result.movable_collisions)

    return stats


def compute_time_based_success(
    model_data: Dict[str, Dict[str, RegionResult]],
    reference_data: Dict[str, Dict[str, RegionResult]],
    config: EvalConfig,
    learned_only: bool = False,
) -> Dict[str, Dict[str, List[float]]]:
    """
    Compute success rate as a function of time cutoff.

    Args:
        learned_only: If True, only count successes from ML-only phase (no fallback).

    Returns:
        {category: {'cutoffs': [...], 'rates': [...]}}
    """
    cutoffs = np.arange(0, config.time_cutoff_max + config.time_step, config.time_step)

    # Collect times by category
    times_by_category: Dict[str, List[float]] = {'1push': [], '2push': []}
    totals_by_category: Dict[str, int] = {'1push': 0, '2push': 0}

    for _, _, ref_result, model_result in iter_matched_triplets(
        model_data, reference_data, require_ref_success=False
    ):
        category = categorize_problem(ref_result)
        if category in ('blocked', 'unsolvable'):
            continue

        cat_key = '1push' if category == '1push' else '2push'
        totals_by_category[cat_key] += 1

        # Check success based on learned_only flag
        if learned_only:
            if model_result.solved_by_learned:
                times_by_category[cat_key].append(model_result.time_taken)
        else:
            if model_result.success:
                times_by_category[cat_key].append(model_result.time_taken)

    # Compute rates at each cutoff
    result = {}
    for cat in ['1push', '2push']:
        times = np.array(times_by_category[cat])
        total = totals_by_category[cat]
        rates = []
        for cutoff in cutoffs:
            if total > 0:
                successes = np.sum(times <= cutoff) if len(times) > 0 else 0
                rates.append(successes / total)
            else:
                rates.append(0.0)
        result[cat] = {'cutoffs': cutoffs.tolist(), 'rates': rates}

    return result


def compute_collision_success_stats(
    model_data: Dict[str, Dict[str, RegionResult]],
    reference_data: Dict[str, Dict[str, RegionResult]],
) -> Dict[str, Dict[str, int]]:
    """
    Compute success rates broken down by collision type.

    Collision categories:
    - none: No wall or movable collisions
    - wall_only: Wall collision but no movable collisions
    - movable_only: Movable collisions but no wall collision
    - both: Both wall and movable collisions

    Returns:
        {collision_type: {'successes': int, 'total': int}}
    """
    stats = {cat: {'successes': 0, 'total': 0}
             for cat in ['none', 'wall_only', 'movable_only', 'both']}

    for _, _, ref_result, model_result in iter_matched_triplets(model_data, reference_data):
        cat = get_collision_category(ref_result)
        stats[cat]['total'] += 1
        if model_result.success:
            stats[cat]['successes'] += 1

    return stats


def compute_chain_depth_confusion_matrix(
    model_data: Dict[str, Dict[str, RegionResult]],
    reference_data: Dict[str, Dict[str, RegionResult]],
) -> Dict[str, Dict[str, int]]:
    """
    Compute confusion matrix between oracle chain depth and model's achieved chain depth.

    For successful model runs, compares what chain depth the oracle needed vs
    what chain depth the model achieved.

    Returns:
        {
            'oracle_1_model_1': count,  # True 1-push, solved with 1 push
            'oracle_1_model_2': count,  # True 1-push, solved with 2 pushes (over-pushed)
            'oracle_2_model_1': count,  # True 2-push, solved with 1 push (impossible normally)
            'oracle_2_model_2': count,  # True 2-push, solved with 2 pushes
            'oracle_1_failed': count,   # True 1-push, model failed
            'oracle_2_failed': count,   # True 2-push, model failed
        }
    """
    matrix = {
        'oracle_1_model_1': 0, 'oracle_1_model_2': 0, 'oracle_1_model_2plus': 0,
        'oracle_2_model_1': 0, 'oracle_2_model_2': 0, 'oracle_2_model_2plus': 0,
        'oracle_1_failed': 0, 'oracle_2_failed': 0,
    }

    for _, _, ref_result, model_result in iter_matched_triplets(model_data, reference_data):
        oracle_depth = ref_result.chain_depth
        if oracle_depth not in (1, 2):
            continue

        prefix = f'oracle_{oracle_depth}_'
        if model_result.success:
            model_depth = model_result.chain_depth
            if model_depth == 1:
                matrix[prefix + 'model_1'] += 1
            elif model_depth == 2:
                matrix[prefix + 'model_2'] += 1
            else:
                matrix[prefix + 'model_2plus'] += 1
        else:
            matrix[prefix + 'failed'] += 1

    return matrix


def compute_collision_bucket_efficiency(
    model_data: Dict[str, Dict[str, RegionResult]],
    reference_data: Dict[str, Dict[str, RegionResult]],
) -> Dict[str, Dict[str, Any]]:
    """
    Compute efficiency stats (checks, time) per collision bucket for solved cases.

    Returns:
        {collision_type: {'pushes': [...], 'times': [...], 'successes': int, 'total': int}}
    """
    stats = {cat: {'pushes': [], 'times': [], 'successes': 0, 'total': 0}
             for cat in ['none', 'wall_only', 'movable_only', 'both']}

    for _, _, ref_result, model_result in iter_matched_triplets(model_data, reference_data):
        cat = get_collision_category(ref_result)
        stats[cat]['total'] += 1
        if model_result.success:
            stats[cat]['successes'] += 1
            stats[cat]['pushes'].append(model_result.pushes)
            stats[cat]['times'].append(model_result.time_taken)

    return stats


def compute_difficulty_stratification(
    model_data: Dict[str, Dict[str, RegionResult]],
    reference_data: Dict[str, Dict[str, RegionResult]],
    depth_filter: Optional[int] = None,  # 1 for 1-push, 2 for 2-push, None for all
    difficulty_mapping: Optional[Dict[Tuple[str, str], str]] = None,  # Pre-computed (env, key) -> difficulty
    thresholds: Optional[Dict[str, float]] = None,  # p33, p66 from multi-reference
) -> Dict[str, Dict[str, Any]]:
    """
    Stratify problems by difficulty based on pre-computed difficulty mapping (from multi-reference
    median pushes) or single-reference oracle pushes as fallback.

    Args:
        depth_filter: If specified, only include problems where oracle solved in exactly this many pushes.
        difficulty_mapping: Pre-computed (env, key) -> difficulty from multi-reference analysis.
        thresholds: Pre-computed {p33, p66, min, max} from multi-reference analysis.

    Returns:
        {difficulty: {'pushes': [...], 'times': [...], 'successes': int, 'total': int,
                      'oracle_push_range': (min, max)}}
    """
    # Initialize stats
    stats = {
        'easy': {'pushes': [], 'times': [], 'successes': 0, 'total': 0, 'oracle_pushes': []},
        'medium': {'pushes': [], 'times': [], 'successes': 0, 'total': 0, 'oracle_pushes': []},
        'hard': {'pushes': [], 'times': [], 'successes': 0, 'total': 0, 'oracle_pushes': []},
    }

    # If no difficulty mapping provided, compute from single reference
    if difficulty_mapping is None:
        # Collect oracle pushes for fallback categorization
        oracle_pushes = []
        problem_keys = []

        for env in reference_data:
            if env not in model_data:
                continue
            for key in reference_data[env]:
                if key not in model_data[env]:
                    continue
                ref_result = reference_data[env][key]
                if ref_result.success:
                    if depth_filter is not None and ref_result.chain_depth != depth_filter:
                        continue
                    oracle_pushes.append(ref_result.pushes)
                    problem_keys.append((env, key))

        if not oracle_pushes:
            return {
                'easy': {'pushes': [], 'times': [], 'successes': 0, 'total': 0, 'oracle_push_range': (0, 0)},
                'medium': {'pushes': [], 'times': [], 'successes': 0, 'total': 0, 'oracle_push_range': (0, 0)},
                'hard': {'pushes': [], 'times': [], 'successes': 0, 'total': 0, 'oracle_push_range': (0, 0)},
            }

        # Compute percentiles from single reference
        p33 = np.percentile(oracle_pushes, 33.33)
        p66 = np.percentile(oracle_pushes, 66.67)

        # Build mapping from single reference
        difficulty_mapping = {}
        for (env, key), oracle_push in zip(problem_keys, oracle_pushes):
            if oracle_push <= p33:
                difficulty_mapping[(env, key)] = 'easy'
            elif oracle_push <= p66:
                difficulty_mapping[(env, key)] = 'medium'
            else:
                difficulty_mapping[(env, key)] = 'hard'

    # Process each triplet using the difficulty mapping
    for env in model_data:
        if env not in reference_data:
            continue
        for key in model_data[env]:
            if key not in reference_data[env]:
                continue

            ref_result = reference_data[env][key]
            if not ref_result.success:
                continue

            # Apply depth filter
            if depth_filter is not None and ref_result.chain_depth != depth_filter:
                continue

            # Get difficulty from mapping
            difficulty = difficulty_mapping.get((env, key))
            if difficulty is None:
                continue  # Triplet not in mapping (e.g., filtered out)

            stats[difficulty]['total'] += 1
            stats[difficulty]['oracle_pushes'].append(ref_result.pushes)

            # Check if model solved it
            model_result = model_data[env][key]
            if model_result.success:
                stats[difficulty]['successes'] += 1
                stats[difficulty]['pushes'].append(model_result.pushes)
                stats[difficulty]['times'].append(model_result.time_taken)

    # Compute oracle push ranges for each difficulty
    for diff in stats:
        if stats[diff]['oracle_pushes']:
            stats[diff]['oracle_push_range'] = (
                min(stats[diff]['oracle_pushes']),
                max(stats[diff]['oracle_pushes'])
            )
        else:
            stats[diff]['oracle_push_range'] = (0, 0)
        # Remove oracle_pushes list (not needed in output)
        del stats[diff]['oracle_pushes']

    return stats


@dataclass
class HybridStats:
    """Statistics for hybrid model decomposition (learned vs fallback)."""
    # Counts
    solved_by_learned: int = 0
    solved_by_fallback: int = 0
    failed: int = 0
    total: int = 0

    # Efficiency for learned solutions
    learned_pushes: List[int] = field(default_factory=list)
    learned_times: List[float] = field(default_factory=list)

    # Efficiency for fallback solutions
    fallback_pushes: List[int] = field(default_factory=list)
    fallback_times: List[float] = field(default_factory=list)

    # Checks before fallback (for fallback cases)
    checks_before_fallback: List[int] = field(default_factory=list)

    @property
    def learned_rate(self) -> float:
        """Fraction solved by learned stage."""
        return self.solved_by_learned / self.total if self.total > 0 else 0.0

    @property
    def fallback_rate(self) -> float:
        """Fraction solved by fallback."""
        return self.solved_by_fallback / self.total if self.total > 0 else 0.0

    @property
    def success_rate(self) -> float:
        """Overall success rate."""
        return (self.solved_by_learned + self.solved_by_fallback) / self.total if self.total > 0 else 0.0

    @property
    def learned_median_pushes(self) -> float:
        return float(np.median(self.learned_pushes)) if self.learned_pushes else 0.0

    @property
    def learned_pushes_iqr(self) -> Tuple[float, float]:
        if not self.learned_pushes:
            return (0.0, 0.0)
        return (float(np.percentile(self.learned_pushes, 25)),
                float(np.percentile(self.learned_pushes, 75)))

    @property
    def learned_median_time(self) -> float:
        return float(np.median(self.learned_times)) if self.learned_times else 0.0

    @property
    def learned_time_iqr(self) -> Tuple[float, float]:
        if not self.learned_times:
            return (0.0, 0.0)
        return (float(np.percentile(self.learned_times, 25)),
                float(np.percentile(self.learned_times, 75)))

    @property
    def fallback_median_pushes(self) -> float:
        return float(np.median(self.fallback_pushes)) if self.fallback_pushes else 0.0

    @property
    def fallback_pushes_iqr(self) -> Tuple[float, float]:
        if not self.fallback_pushes:
            return (0.0, 0.0)
        return (float(np.percentile(self.fallback_pushes, 25)),
                float(np.percentile(self.fallback_pushes, 75)))

    @property
    def fallback_median_time(self) -> float:
        return float(np.median(self.fallback_times)) if self.fallback_times else 0.0

    @property
    def fallback_time_iqr(self) -> Tuple[float, float]:
        if not self.fallback_times:
            return (0.0, 0.0)
        return (float(np.percentile(self.fallback_times, 25)),
                float(np.percentile(self.fallback_times, 75)))

    @property
    def median_checks_before_fallback(self) -> float:
        return float(np.median(self.checks_before_fallback)) if self.checks_before_fallback else 0.0

    @property
    def checks_before_fallback_iqr(self) -> Tuple[float, float]:
        if not self.checks_before_fallback:
            return (0.0, 0.0)
        return (float(np.percentile(self.checks_before_fallback, 25)),
                float(np.percentile(self.checks_before_fallback, 75)))

    @property
    def fallback_phase_only_pushes(self) -> List[int]:
        """Compute pushes in fallback phase only (total - ML phase) for each fallback case."""
        if len(self.fallback_pushes) != len(self.checks_before_fallback):
            return []  # Can't compute if lists don't align
        return [total - ml for total, ml in zip(self.fallback_pushes, self.checks_before_fallback)]

    @property
    def fallback_phase_only_median(self) -> float:
        phase_only = self.fallback_phase_only_pushes
        return float(np.median(phase_only)) if phase_only else 0.0

    @property
    def fallback_phase_only_iqr(self) -> Tuple[float, float]:
        phase_only = self.fallback_phase_only_pushes
        if not phase_only:
            return (0.0, 0.0)
        return (float(np.percentile(phase_only, 25)),
                float(np.percentile(phase_only, 75)))


def compute_hybrid_stats(
    model_data: Dict[str, Dict[str, RegionResult]],
    reference_data: Dict[str, Dict[str, RegionResult]],
    depth_filter: Optional[int] = None,  # 1 for 1-push, 2 for 2-push, None for all
) -> HybridStats:
    """
    Compute hybrid decomposition stats (learned vs fallback).

    Uses explicit phase tracking (solved_in_phase field):
      - solved_in_phase == "ML-only" → LEARNED
      - solved_in_phase == "primitives" → FALLBACK
    """
    stats = HybridStats()

    for _, _, ref_result, model_result in iter_matched_triplets(model_data, reference_data):
        if depth_filter is not None and ref_result.chain_depth != depth_filter:
            continue

        stats.total += 1

        if model_result.solved_by_learned:
            stats.solved_by_learned += 1
            stats.learned_pushes.append(model_result.pushes)
            stats.learned_times.append(model_result.time_taken)
        elif model_result.solved_by_fallback:
            stats.solved_by_fallback += 1
            stats.fallback_pushes.append(model_result.pushes)
            stats.fallback_times.append(model_result.time_taken)
            if model_result.phase_push_counts and "ML-only" in model_result.phase_push_counts:
                stats.checks_before_fallback.append(model_result.phase_push_counts["ML-only"])
        else:
            stats.failed += 1

    return stats


def compute_hybrid_stats_by_difficulty(
    model_data: Dict[str, Dict[str, RegionResult]],
    reference_data: Dict[str, Dict[str, RegionResult]],
    depth_filter: Optional[int] = None,
    difficulty_mapping: Optional[Dict[Tuple[str, str], str]] = None,
) -> Dict[str, Dict[str, int]]:
    """
    Compute hybrid decomposition stats (learned vs fallback) per difficulty bucket.

    Difficulty is based on pre-computed difficulty mapping (from multi-reference median pushes)
    or single-reference oracle pushes as fallback.

    Returns:
        {difficulty: {'total': N, 'learned': N, 'fallback': N, 'failed': N}}
    """
    # Initialize stats
    stats = {
        'easy': {'total': 0, 'learned': 0, 'fallback': 0, 'failed': 0},
        'medium': {'total': 0, 'learned': 0, 'fallback': 0, 'failed': 0},
        'hard': {'total': 0, 'learned': 0, 'fallback': 0, 'failed': 0},
    }

    # If no difficulty mapping provided, compute from single reference
    if difficulty_mapping is None:
        oracle_pushes = []
        problem_keys = []

        for env in reference_data:
            if env not in model_data:
                continue
            for key in reference_data[env]:
                if key not in model_data[env]:
                    continue
                ref_result = reference_data[env][key]
                if ref_result.success:
                    if depth_filter is not None and ref_result.chain_depth != depth_filter:
                        continue
                    oracle_pushes.append(ref_result.pushes)
                    problem_keys.append((env, key))

        if not oracle_pushes:
            return stats

        p33 = np.percentile(oracle_pushes, 33.33)
        p66 = np.percentile(oracle_pushes, 66.67)

        difficulty_mapping = {}
        for (env, key), oracle_push in zip(problem_keys, oracle_pushes):
            if oracle_push <= p33:
                difficulty_mapping[(env, key)] = 'easy'
            elif oracle_push <= p66:
                difficulty_mapping[(env, key)] = 'medium'
            else:
                difficulty_mapping[(env, key)] = 'hard'

    # Process each triplet using the difficulty mapping
    for env in model_data:
        if env not in reference_data:
            continue
        for key in model_data[env]:
            if key not in reference_data[env]:
                continue

            ref_result = reference_data[env][key]
            if not ref_result.success:
                continue

            if depth_filter is not None and ref_result.chain_depth != depth_filter:
                continue

            difficulty = difficulty_mapping.get((env, key))
            if difficulty is None:
                continue

            model_result = model_data[env][key]
            stats[difficulty]['total'] += 1

            if model_result.solved_by_learned:
                stats[difficulty]['learned'] += 1
            elif model_result.solved_by_fallback:
                stats[difficulty]['fallback'] += 1
            else:
                stats[difficulty]['failed'] += 1

    return stats


# =============================================================================
# Multi-Reference Consistency Analysis
# =============================================================================

@dataclass
class TripletConsistency:
    """Consistency metrics for a single triplet across multiple oracle runs."""
    env: str
    key: str
    pushes: List[int]  # Pushes from each oracle
    times: List[float]  # Times from each oracle
    chain_depths: List[int]  # Chain depth from each oracle
    all_successful: bool  # True if all oracles solved it

    @property
    def mean_pushes(self) -> float:
        return float(np.mean(self.pushes)) if self.pushes else 0.0

    @property
    def std_pushes(self) -> float:
        return float(np.std(self.pushes)) if len(self.pushes) > 1 else 0.0

    @property
    def cv_pushes(self) -> float:
        """Coefficient of variation for pushes (std/mean)."""
        if self.mean_pushes == 0:
            return 0.0
        return self.std_pushes / self.mean_pushes

    @property
    def mean_time(self) -> float:
        return float(np.mean(self.times)) if self.times else 0.0

    @property
    def std_time(self) -> float:
        return float(np.std(self.times)) if len(self.times) > 1 else 0.0

    @property
    def cv_time(self) -> float:
        """Coefficient of variation for time (std/mean)."""
        if self.mean_time == 0:
            return 0.0
        return self.std_time / self.mean_time

    @property
    def min_pushes(self) -> int:
        return min(self.pushes) if self.pushes else 0

    @property
    def max_pushes(self) -> int:
        return max(self.pushes) if self.pushes else 0

    @property
    def push_range(self) -> int:
        """Range of pushes (max - min)."""
        return self.max_pushes - self.min_pushes

    @property
    def median_pushes(self) -> float:
        return float(np.median(self.pushes)) if self.pushes else 0.0

    @property
    def p25_pushes(self) -> float:
        """25th percentile of pushes."""
        return float(np.percentile(self.pushes, 25)) if len(self.pushes) >= 2 else self.min_pushes

    @property
    def p75_pushes(self) -> float:
        """75th percentile of pushes."""
        return float(np.percentile(self.pushes, 75)) if len(self.pushes) >= 2 else self.max_pushes


def compute_multi_reference_consistency(
    reference_data_list: List[Dict[str, Dict[str, RegionResult]]],
    reference_names: List[str],
) -> Tuple[Dict[str, TripletConsistency], Set[Tuple[str, str]]]:
    """
    Compute consistency metrics across multiple oracle runs.

    Args:
        reference_data_list: List of reference data dicts (one per oracle run)
        reference_names: Names of each reference run

    Returns:
        Tuple of:
        - consistency_data: {(env, key): TripletConsistency}
        - common_triplets: Set of (env, key) tuples present in ALL references
    """
    if not reference_data_list:
        return {}, set()

    # Find common triplets across all references (intersection)
    common_triplets = None
    for ref_data in reference_data_list:
        ref_keys = set()
        for env in ref_data:
            for key in ref_data[env]:
                if ref_data[env][key].success:  # Only successful triplets
                    ref_keys.add((env, key))
        if common_triplets is None:
            common_triplets = ref_keys
        else:
            common_triplets = common_triplets & ref_keys

    if common_triplets is None:
        common_triplets = set()

    # Compute consistency for each common triplet
    consistency_data = {}
    for env, key in common_triplets:
        pushes = []
        times = []
        chain_depths = []

        for ref_data in reference_data_list:
            if env in ref_data and key in ref_data[env]:
                result = ref_data[env][key]
                if result.success:
                    pushes.append(result.pushes)
                    times.append(result.time_taken)
                    chain_depths.append(result.chain_depth)

        consistency_data[(env, key)] = TripletConsistency(
            env=env,
            key=key,
            pushes=pushes,
            times=times,
            chain_depths=chain_depths,
            all_successful=len(pushes) == len(reference_data_list)
        )

    return consistency_data, common_triplets


def categorize_by_consistency(
    consistency_data: Dict[Tuple[str, str], TripletConsistency],
    depth_filter: Optional[int] = None,
) -> Tuple[Dict[str, List[TripletConsistency]], Dict[str, float]]:
    """
    Categorize triplets by difficulty using median pushes with data-driven percentiles.

    Uses median pushes across oracles (robust to outliers).

    Categories:
    - easy: Low median pushes (bottom 33%)
    - medium: Medium median pushes (middle 33%)
    - hard: High median pushes (top 33%)

    Returns:
        Tuple of:
        - {category: List[TripletConsistency]}
        - {threshold_name: value} - actual threshold values from the data
    """
    # Filter by chain depth if specified
    filtered = []
    for tc in consistency_data.values():
        if not tc.all_successful:
            continue
        if depth_filter is not None:
            # Use mode of chain depths
            if tc.chain_depths and max(set(tc.chain_depths), key=tc.chain_depths.count) != depth_filter:
                continue
        filtered.append(tc)

    if not filtered:
        return {'easy': [], 'medium': [], 'hard': []}, {'p33': 0, 'p66': 0}

    # Compute percentiles using MEDIAN pushes (robust to outliers)
    median_pushes_list = [tc.median_pushes for tc in filtered]
    p33 = np.percentile(median_pushes_list, 33.33)
    p66 = np.percentile(median_pushes_list, 66.67)

    thresholds = {
        'p33': p33,
        'p66': p66,
        'min': min(median_pushes_list),
        'max': max(median_pushes_list),
    }

    categories = {'easy': [], 'medium': [], 'hard': []}

    for tc in filtered:
        if tc.median_pushes <= p33:
            categories['easy'].append(tc)
        elif tc.median_pushes <= p66:
            categories['medium'].append(tc)
        else:
            categories['hard'].append(tc)

    return categories, thresholds


def build_difficulty_mapping(
    categories: Dict[str, List['TripletConsistency']],
) -> Dict[Tuple[str, str], str]:
    """
    Build a mapping from (env, key) -> difficulty from categorized triplets.

    Args:
        categories: {difficulty: List[TripletConsistency]} from categorize_by_consistency

    Returns:
        {(env, key): difficulty} mapping
    """
    mapping = {}
    for difficulty, triplets in categories.items():
        for tc in triplets:
            mapping[(tc.env, tc.key)] = difficulty
    return mapping


@dataclass
class ConsistencyStats:
    """Aggregate consistency statistics."""
    n_triplets: int = 0
    n_references: int = 0

    # Push statistics
    mean_of_means_pushes: float = 0.0
    mean_of_medians_pushes: float = 0.0
    mean_std_pushes: float = 0.0
    mean_cv_pushes: float = 0.0

    # Thresholds from data
    threshold_p33: float = 0.0  # Easy/medium boundary
    threshold_p66: float = 0.0  # Medium/hard boundary

    # Time statistics
    mean_of_means_time: float = 0.0
    mean_std_time: float = 0.0
    mean_cv_time: float = 0.0

    # Category counts
    n_easy: int = 0
    n_medium: int = 0
    n_hard: int = 0

    # Detailed lists for plotting
    all_mean_pushes: List[float] = field(default_factory=list)
    all_median_pushes: List[float] = field(default_factory=list)
    all_std_pushes: List[float] = field(default_factory=list)
    all_cv_pushes: List[float] = field(default_factory=list)


def compute_consistency_stats(
    consistency_data: Dict[Tuple[str, str], TripletConsistency],
    categories: Dict[str, List[TripletConsistency]],
    thresholds: Dict[str, float],
    n_references: int,
) -> ConsistencyStats:
    """Compute aggregate consistency statistics."""
    stats = ConsistencyStats(n_references=n_references)

    all_mean_pushes = []
    all_median_pushes = []
    all_std_pushes = []
    all_cv_pushes = []
    all_mean_times = []
    all_std_times = []
    all_cv_times = []

    for tc in consistency_data.values():
        if not tc.all_successful:
            continue
        all_mean_pushes.append(tc.mean_pushes)
        all_median_pushes.append(tc.median_pushes)
        all_std_pushes.append(tc.std_pushes)
        all_cv_pushes.append(tc.cv_pushes)
        all_mean_times.append(tc.mean_time)
        all_std_times.append(tc.std_time)
        all_cv_times.append(tc.cv_time)

    stats.n_triplets = len(all_mean_pushes)

    if all_mean_pushes:
        stats.mean_of_means_pushes = float(np.mean(all_mean_pushes))
        stats.mean_of_medians_pushes = float(np.mean(all_median_pushes))
        stats.mean_std_pushes = float(np.mean(all_std_pushes))
        stats.mean_cv_pushes = float(np.mean(all_cv_pushes))

    if all_mean_times:
        stats.mean_of_means_time = float(np.mean(all_mean_times))
        stats.mean_std_time = float(np.mean(all_std_times))
        stats.mean_cv_time = float(np.mean(all_cv_times))

    # Store thresholds
    stats.threshold_p33 = thresholds.get('p33', 0)
    stats.threshold_p66 = thresholds.get('p66', 0)

    stats.n_easy = len(categories.get('easy', []))
    stats.n_medium = len(categories.get('medium', []))
    stats.n_hard = len(categories.get('hard', []))

    stats.all_mean_pushes = all_mean_pushes
    stats.all_median_pushes = all_median_pushes
    stats.all_std_pushes = all_std_pushes
    stats.all_cv_pushes = all_cv_pushes

    return stats


def print_consistency_report(
    consistency_stats: ConsistencyStats,
    categories: Dict[str, List[TripletConsistency]],
    depth_name: str = "All",
):
    """Print consistency report to console."""
    stats = consistency_stats
    total = stats.n_triplets

    print(f"\n  [{depth_name}] {stats.n_triplets} triplets, {stats.n_references} refs, CV={stats.mean_cv_pushes:.2f}")

    if total > 0:
        print(f"  Thresholds: Easy ≤{stats.threshold_p33:.0f}, Medium {stats.threshold_p33:.0f}-{stats.threshold_p66:.0f}, Hard >{stats.threshold_p66:.0f} pushes")
        print(f"  Distribution: Easy {stats.n_easy} ({100*stats.n_easy/total:.0f}%), "
              f"Medium {stats.n_medium} ({100*stats.n_medium/total:.0f}%), "
              f"Hard {stats.n_hard} ({100*stats.n_hard/total:.0f}%)")


def save_consistency_report_to_markdown(
    consistency_stats_all: ConsistencyStats,
    consistency_stats_1push: ConsistencyStats,
    consistency_stats_2push: ConsistencyStats,
    categories_all: Dict[str, List[TripletConsistency]],
    categories_1push: Dict[str, List[TripletConsistency]],
    categories_2push: Dict[str, List[TripletConsistency]],
    reference_names: List[str],
    output_path: str,
):
    """Save consistency report to markdown file."""
    lines = []
    lines.append("# Multi-Oracle Consistency Analysis")
    lines.append("")
    lines.append(f"Analyzed **{len(reference_names)}** oracle runs with different shuffle seeds.")
    lines.append("")

    for depth_name, stats, cats in [
        ("All Depths", consistency_stats_all, categories_all),
        ("1-Push Problems", consistency_stats_1push, categories_1push),
        ("2-Push Problems", consistency_stats_2push, categories_2push),
    ]:
        lines.append(f"## {depth_name}")
        lines.append("")
        lines.append(f"**{stats.n_triplets}** common triplets | Mean CV: {stats.mean_cv_pushes:.2f}")
        lines.append("")

        total = stats.n_triplets
        if total > 0:
            lines.append("### Thresholds (data-driven percentiles)")
            lines.append("")
            lines.append(f"| Difficulty | Median Pushes |")
            lines.append(f"|------------|---------------|")
            lines.append(f"| Easy | ≤ {stats.threshold_p33:.0f} |")
            lines.append(f"| Medium | {stats.threshold_p33:.0f} - {stats.threshold_p66:.0f} |")
            lines.append(f"| Hard | > {stats.threshold_p66:.0f} |")
            lines.append("")

            lines.append("### Distribution")
            lines.append("")
            lines.append("| Category | Count | % |")
            lines.append("|----------|------:|--:|")
            lines.append(f"| Easy | {stats.n_easy} | {100*stats.n_easy/total:.0f}% |")
            lines.append(f"| Medium | {stats.n_medium} | {100*stats.n_medium/total:.0f}% |")
            lines.append(f"| Hard | {stats.n_hard} | {100*stats.n_hard/total:.0f}% |")
            lines.append("")

        if cats.get('hard'):
            lines.append("### Hardest Problems")
            lines.append("")
            lines.append("| Triplet | Median | Std |")
            lines.append("|---------|-------:|----:|")
            sorted_hard = sorted(cats['hard'], key=lambda x: -x.median_pushes)[:10]
            for tc in sorted_hard:
                lines.append(f"| {tc.key} | {tc.median_pushes:.0f} | {tc.std_pushes:.1f} |")
            lines.append("")

    with open(output_path, 'w') as f:
        f.write('\n'.join(lines))
    print(f"  Saved: {output_path}")


def plot_consistency_scatter(
    consistency_data: Dict[Tuple[str, str], TripletConsistency],
    categories: Dict[str, List[TripletConsistency]],
    depth_filter: Optional[int] = None,
    output_path: Optional[str] = None,
):
    """Plot mean pushes vs CV (variance) scatter plot."""
    fig, ax = plt.subplots(figsize=(10, 6))

    category_colors = {
        'easy': '#55A868',      # green
        'medium': '#DD8452',    # orange
        'hard': '#C44E52',      # red
    }

    for cat_name, cat_triplets in categories.items():
        if not cat_triplets:
            continue
        x = [tc.median_pushes for tc in cat_triplets]
        y = [tc.cv_pushes for tc in cat_triplets]
        ax.scatter(x, y, c=category_colors.get(cat_name, '#888888'),
                   label=f"{cat_name.capitalize()} ({len(cat_triplets)})",
                   alpha=0.6, s=30)

    ax.set_xlabel('Median Pushes (across oracles)')
    ax.set_ylabel('Coefficient of Variation (std/mean)')
    ax.axhline(y=0.3, color='#888888', linestyle='--', linewidth=0.8, alpha=0.5, label='CV=0.3 threshold')
    ax.legend(loc='upper right')
    ax.grid(True, linestyle='--', alpha=0.3)

    depth_str = f"{depth_filter}-Push" if depth_filter else "All"
    ax.set_title(f"Difficulty Consistency Analysis ({depth_str} Problems)", fontweight='bold')

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white')
        print(f"Saved: {output_path}")
    return fig


def plot_push_variance_histogram(
    consistency_data: Dict[Tuple[str, str], TripletConsistency],
    depth_filter: Optional[int] = None,
    output_path: Optional[str] = None,
):
    """Plot histogram of push standard deviation across oracles."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Filter by depth
    filtered = [tc for tc in consistency_data.values()
                if tc.all_successful and
                (depth_filter is None or
                 (tc.chain_depths and max(set(tc.chain_depths), key=tc.chain_depths.count) == depth_filter))]

    if not filtered:
        return fig

    # Left: Histogram of std dev
    ax1 = axes[0]
    std_values = [tc.std_pushes for tc in filtered]
    ax1.hist(std_values, bins=30, color='#4C72B0', edgecolor='white', alpha=0.8)
    ax1.axvline(x=np.mean(std_values), color='#C44E52', linestyle='--', linewidth=2, label=f'Mean={np.mean(std_values):.1f}')
    ax1.set_xlabel('Standard Deviation of Pushes')
    ax1.set_ylabel('Count')
    ax1.set_title('Distribution of Push Variance')
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.3)

    # Right: Histogram of CV
    ax2 = axes[1]
    cv_values = [tc.cv_pushes for tc in filtered]
    ax2.hist(cv_values, bins=30, color='#55A868', edgecolor='white', alpha=0.8)
    ax2.axvline(x=np.mean(cv_values), color='#C44E52', linestyle='--', linewidth=2, label=f'Mean={np.mean(cv_values):.2f}')
    ax2.axvline(x=0.3, color='#888888', linestyle=':', linewidth=1.5, label='CV=0.3 (variable)')
    ax2.set_xlabel('Coefficient of Variation (std/mean)')
    ax2.set_ylabel('Count')
    ax2.set_title('Distribution of Push CV')
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.3)

    depth_str = f"{depth_filter}-Push" if depth_filter else "All"
    fig.suptitle(f"Push Variance Analysis ({depth_str} Problems, N={len(filtered)})", fontweight='bold')

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white')
        print(f"Saved: {output_path}")
    return fig


# =============================================================================
# Plotting
# =============================================================================

def get_model_color(idx: int, config: EvalConfig) -> str:
    """Get color for a model by index."""
    return config.model_colors[idx % len(config.model_colors)]


def style_boxplot(bp, colors: List[str]):
    """Apply consistent styling to a boxplot."""
    for i, patch in enumerate(bp['boxes']):
        patch.set_facecolor(colors[i % len(colors)])
        patch.set_alpha(0.85)
        patch.set_edgecolor('white')
        patch.set_linewidth(0.5)
    for median in bp['medians']:
        median.set_color('#333333')
        median.set_linewidth(2)
    for whisker in bp['whiskers']:
        whisker.set_color('#666666')
    for cap in bp['caps']:
        cap.set_color('#666666')


def plot_success_rates(
    model_stats: List[ModelStats],
    config: EvalConfig,
    output_path: Optional[str] = None,
):
    """Plot success rate comparison across models (2-push only)."""
    n_models = len(model_stats)

    fig, ax = plt.subplots(figsize=(8, 5))

    names = [s.name for s in model_stats]
    rates = [s.depth_2.success_rate for s in model_stats]
    counts = [(s.depth_2.successes, s.depth_2.total) for s in model_stats]

    bars = ax.bar(range(len(names)), rates,
                  color=[get_model_color(i, config) for i in range(len(names))],
                  edgecolor='white', linewidth=0.5)

    for bar, rate, (succ, total) in zip(bars, rates, counts):
        ax.annotate(f'{rate:.0%}\n({succ}/{total})',
                   xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                   xytext=(0, 4), textcoords="offset points",
                   ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=15, ha='right')
    ax.set_ylim(0, 1.15)
    ax.set_ylabel('Success Rate')
    ax.set_title('2-Push Problems - Success Rate')
    ax.axhline(y=1.0, color='#888888', linestyle='--', linewidth=0.8, alpha=0.5)

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white')
        print(f"Saved: {output_path}")
    return fig


def _plot_boxplot(
    model_stats: List[ModelStats],
    config: EvalConfig,
    data_getter: Callable[[ModelStats], List],
    ylabel: str,
    title: str,
    output_path: Optional[str] = None,
):
    """Generic boxplot for model statistics (2-push only)."""
    names = [s.name for s in model_stats]
    fig, ax = plt.subplots(figsize=(8, 5))

    data = [data_getter(s) or [0] for s in model_stats]
    bp = ax.boxplot(data, tick_labels=names, patch_artist=True, showfliers=False)
    style_boxplot(bp, [get_model_color(i, config) for i in range(len(names))])

    ax.tick_params(axis='x', rotation=15)
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white')
        print(f"Saved: {output_path}")
    return fig


def plot_pushes_boxplot(
    model_stats: List[ModelStats],
    config: EvalConfig,
    output_path: Optional[str] = None,
):
    """Plot pushes comparison as boxplots (2-push only)."""
    return _plot_boxplot(
        model_stats, config,
        lambda s: s.depth_2.pushes,
        'Pushes to Success',
        '2-Push Problems - Pushes Distribution',
        output_path
    )


def plot_time_boxplot(
    model_stats: List[ModelStats],
    config: EvalConfig,
    output_path: Optional[str] = None,
):
    """Plot time comparison as boxplots (2-push only)."""
    return _plot_boxplot(
        model_stats, config,
        lambda s: s.depth_2.times,
        'Time to Success (ms)',
        '2-Push Problems - Time Distribution',
        output_path
    )


def plot_time_vs_success(
    time_data: Dict[str, Dict[str, Dict[str, List[float]]]],  # {model_name: {category: {cutoffs, rates}}}
    config: EvalConfig,
    output_path: Optional[str] = None,
    learned_only_data: Optional[Dict[str, Dict[str, Dict[str, List[float]]]]] = None,
):
    """Plot success rate vs time cutoff (2-push only).

    Args:
        time_data: Total success data {model_name: {category: {cutoffs, rates}}}
        config: Evaluation config
        output_path: Path to save plot
        learned_only_data: ML-only phase success data (same format as time_data)
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # Get N from the first model's 2push data
    n_problems = 0
    for cat_data in time_data.values():
        if '2push' in cat_data:
            n_problems = cat_data['2push'].get('total', 0)
            break

    for model_idx, (model_name, cat_data) in enumerate(time_data.items()):
        if '2push' in cat_data:
            cutoffs_ms = cat_data['2push']['cutoffs']
            cutoffs_s = [c / 1000.0 for c in cutoffs_ms]  # Convert to seconds
            rates = cat_data['2push']['rates']
            color = get_model_color(model_idx, config)
            ax.plot(cutoffs_s, rates, label=model_name,
                   color=color, linewidth=2)

            # Plot learned-only curve if available for this model
            if learned_only_data and model_name in learned_only_data:
                lo_cat_data = learned_only_data[model_name]
                if '2push' in lo_cat_data:
                    lo_rates = lo_cat_data['2push']['rates']
                    ax.plot(cutoffs_s, lo_rates, label=model_name.replace('Hybrid ', ''),
                           color=color, linewidth=2, linestyle='--', alpha=0.7)

    ax.set_xlabel('Time cutoff (s)')
    ax.set_ylabel('Success Rate')
    ax.set_ylim(0, 1.05)
    ax.set_xlim(0, config.time_cutoff_max / 1000.0)
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend(loc='lower right')

    ax.set_title(f"2-Push Problems (N={n_problems}) - Success Rate @ Time Cutoff", fontsize=14, fontweight='bold')

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white')
        print(f"Saved: {output_path}")
    return fig


def compute_push_based_success(
    model_data: Dict[str, Dict[str, RegionResult]],
    reference_data: Dict[str, Dict[str, RegionResult]],
    config: EvalConfig,
    learned_only: bool = False,
) -> Dict[str, Dict[str, List[float]]]:
    """
    Compute success rate as a function of push count cutoff.

    Args:
        learned_only: If True, only count successes from ML-only phase (no fallback).

    Returns:
        {category: {'cutoffs': [...], 'rates': [...], 'total': int}}
    """
    cutoffs = list(range(0, config.push_cutoff_max + 1, config.push_step))

    # Collect pushes by category
    pushes_by_category: Dict[str, List[int]] = {'1push': [], '2push': []}
    totals_by_category: Dict[str, int] = {'1push': 0, '2push': 0}

    for env in model_data:
        if env not in reference_data:
            continue
        for key in model_data[env]:
            if key not in reference_data[env]:
                continue

            ref_result = reference_data[env][key]
            model_result = model_data[env][key]

            category = categorize_problem(ref_result)
            if category in ('blocked', 'unsolvable'):
                continue

            cat_key = '1push' if category == '1push' else '2push'
            totals_by_category[cat_key] += 1

            # Check success based on learned_only flag
            if learned_only:
                if model_result.solved_by_learned:
                    pushes_by_category[cat_key].append(model_result.pushes)
            else:
                if model_result.success:
                    pushes_by_category[cat_key].append(model_result.pushes)

    # Compute rates at each cutoff
    result = {}
    for cat in ['1push', '2push']:
        pushes = np.array(pushes_by_category[cat])
        total = totals_by_category[cat]
        rates = []
        for cutoff in cutoffs:
            if total > 0:
                successes = np.sum(pushes <= cutoff) if len(pushes) > 0 else 0
                rates.append(successes / total)
            else:
                rates.append(0.0)
        result[cat] = {'cutoffs': cutoffs, 'rates': rates, 'total': total}

    return result


def _compute_success_at_threshold(
    model_data: Dict[str, Dict[str, RegionResult]],
    reference_data: Dict[str, Dict[str, RegionResult]],
    thresholds: List,
    metric_getter: Callable[['RegionResult'], float],
    depth_filter: int = 2,
) -> Dict[Any, Dict[str, Any]]:
    """
    Generic helper to compute success rate at specific thresholds.

    Args:
        model_data: Model results by env/key
        reference_data: Reference results for filtering
        thresholds: List of threshold values
        metric_getter: Function to extract metric from RegionResult (e.g., pushes or time_taken)
        depth_filter: Chain depth to filter (1 or 2)

    Returns:
        {threshold: {'successes': int, 'total': int, 'rate': float}}
    """
    metrics_list = []
    total = 0

    for _, _, ref_result, model_result in iter_matched_triplets(
        model_data, reference_data, require_ref_success=True
    ):
        if ref_result.chain_depth != depth_filter:
            continue

        total += 1
        if model_result.success:
            metrics_list.append(metric_getter(model_result))

    metrics = np.array(metrics_list) if metrics_list else np.array([])
    result = {}

    for threshold in thresholds:
        if total > 0:
            successes = int(np.sum(metrics <= threshold)) if len(metrics) > 0 else 0
            rate = successes / total
        else:
            successes = 0
            rate = 0.0
        result[threshold] = {'successes': successes, 'total': total, 'rate': rate}

    return result


def compute_success_at_budget(
    model_data: Dict[str, Dict[str, RegionResult]],
    reference_data: Dict[str, Dict[str, RegionResult]],
    budgets: List[int],
    depth_filter: int = 2,
) -> Dict[int, Dict[str, Any]]:
    """
    Compute success rate at specific verification budgets (Success@B).

    This is a constant-compute comparison: what success rate does each method
    achieve when limited to B simulation-verified push evaluations?
    """
    return _compute_success_at_threshold(
        model_data, reference_data, budgets,
        metric_getter=lambda r: r.pushes,
        depth_filter=depth_filter
    )


def compute_success_at_time_budget(
    model_data: Dict[str, Dict[str, RegionResult]],
    reference_data: Dict[str, Dict[str, RegionResult]],
    time_budgets: List[float],
    depth_filter: int = 2,
) -> Dict[float, Dict[str, Any]]:
    """
    Compute success rate at specific time budgets (Success@T).

    This is a constant-time comparison: what success rate does each method
    achieve when limited to T milliseconds?
    """
    return _compute_success_at_threshold(
        model_data, reference_data, time_budgets,
        metric_getter=lambda r: r.time_taken,
        depth_filter=depth_filter
    )


def _compute_success_at_threshold_by_difficulty(
    model_data: Dict[str, Dict[str, RegionResult]],
    reference_data: Dict[str, Dict[str, RegionResult]],
    thresholds: List,
    metric_getter: Callable[['RegionResult'], float],
    depth_filter: int = 2,
    difficulty_mapping: Optional[Dict[Tuple[str, str], str]] = None,
) -> Dict[str, Dict[Any, Dict[str, Any]]]:
    """
    Generic helper to compute success rate at thresholds, stratified by difficulty.

    Returns:
        {difficulty: {threshold: {'successes': int, 'total': int, 'rate': float}}}
    """
    # Build difficulty mapping if not provided
    if difficulty_mapping is None:
        oracle_pushes = []
        problem_keys = []

        for env in reference_data:
            if env not in model_data:
                continue
            for key in reference_data[env]:
                if key not in model_data[env]:
                    continue
                ref_result = reference_data[env][key]
                if ref_result.success and ref_result.chain_depth == depth_filter:
                    oracle_pushes.append(ref_result.pushes)
                    problem_keys.append((env, key))

        if not oracle_pushes:
            empty_result = {t: {'successes': 0, 'total': 0, 'rate': 0.0} for t in thresholds}
            return {'easy': empty_result.copy(), 'medium': empty_result.copy(), 'hard': empty_result.copy()}

        p33 = np.percentile(oracle_pushes, 33.33)
        p66 = np.percentile(oracle_pushes, 66.67)

        difficulty_mapping = {}
        for (env, key), oracle_push in zip(problem_keys, oracle_pushes):
            if oracle_push <= p33:
                difficulty_mapping[(env, key)] = 'easy'
            elif oracle_push <= p66:
                difficulty_mapping[(env, key)] = 'medium'
            else:
                difficulty_mapping[(env, key)] = 'hard'

    # Collect metrics by difficulty
    metrics_by_difficulty: Dict[str, List[float]] = {'easy': [], 'medium': [], 'hard': []}
    totals_by_difficulty: Dict[str, int] = {'easy': 0, 'medium': 0, 'hard': 0}

    for env in model_data:
        if env not in reference_data:
            continue
        for key in model_data[env]:
            if key not in reference_data[env]:
                continue

            ref_result = reference_data[env][key]
            if not ref_result.success or ref_result.chain_depth != depth_filter:
                continue

            difficulty = difficulty_mapping.get((env, key))
            if difficulty is None:
                continue

            model_result = model_data[env][key]
            totals_by_difficulty[difficulty] += 1
            if model_result.success:
                metrics_by_difficulty[difficulty].append(metric_getter(model_result))

    # Compute rates for each difficulty and threshold
    result = {}
    for diff in ['easy', 'medium', 'hard']:
        metrics = np.array(metrics_by_difficulty[diff]) if metrics_by_difficulty[diff] else np.array([])
        total = totals_by_difficulty[diff]
        result[diff] = {}

        for threshold in thresholds:
            if total > 0:
                successes = int(np.sum(metrics <= threshold)) if len(metrics) > 0 else 0
                rate = successes / total
            else:
                successes = 0
                rate = 0.0
            result[diff][threshold] = {'successes': successes, 'total': total, 'rate': rate}

    return result


def compute_success_at_budget_by_difficulty(
    model_data: Dict[str, Dict[str, RegionResult]],
    reference_data: Dict[str, Dict[str, RegionResult]],
    budgets: List[int],
    depth_filter: int = 2,
    difficulty_mapping: Optional[Dict[Tuple[str, str], str]] = None,
) -> Dict[str, Dict[int, Dict[str, Any]]]:
    """
    Compute success rate at specific verification budgets, stratified by difficulty.

    Returns:
        {difficulty: {budget: {'successes': int, 'total': int, 'rate': float}}}
    """
    return _compute_success_at_threshold_by_difficulty(
        model_data, reference_data, budgets,
        metric_getter=lambda r: r.pushes,
        depth_filter=depth_filter,
        difficulty_mapping=difficulty_mapping
    )


def compute_success_at_time_budget_by_difficulty(
    model_data: Dict[str, Dict[str, RegionResult]],
    reference_data: Dict[str, Dict[str, RegionResult]],
    time_budgets: List[float],
    depth_filter: int = 2,
    difficulty_mapping: Optional[Dict[Tuple[str, str], str]] = None,
) -> Dict[str, Dict[float, Dict[str, Any]]]:
    """
    Compute success rate at specific time budgets, stratified by difficulty.

    Returns:
        {difficulty: {time_budget: {'successes': int, 'total': int, 'rate': float}}}
    """
    return _compute_success_at_threshold_by_difficulty(
        model_data, reference_data, time_budgets,
        metric_getter=lambda r: r.time_taken,
        depth_filter=depth_filter,
        difficulty_mapping=difficulty_mapping
    )


def compute_time_based_success_by_difficulty(
    model_data: Dict[str, Dict[str, RegionResult]],
    reference_data: Dict[str, Dict[str, RegionResult]],
    config: EvalConfig,
    depth_filter: int = 2,
    difficulty_mapping: Optional[Dict[Tuple[str, str], str]] = None,
    learned_only: bool = False,
) -> Dict[str, Dict[str, Any]]:
    """
    Compute success rate as a function of time cutoff, stratified by difficulty.

    Difficulty is based on pre-computed difficulty mapping (from multi-reference median pushes)
    or single-reference oracle pushes as fallback.

    Args:
        learned_only: If True, only count successes from ML-only phase (no fallback).

    Returns:
        {difficulty: {'cutoffs': [...], 'rates': [...], 'total': int}}
    """
    cutoffs = np.arange(0, config.time_cutoff_max + config.time_step, config.time_step)

    # If no difficulty mapping provided, compute from single reference
    if difficulty_mapping is None:
        oracle_pushes = []
        problem_keys = []

        for env in reference_data:
            if env not in model_data:
                continue
            for key in reference_data[env]:
                if key not in model_data[env]:
                    continue
                ref_result = reference_data[env][key]
                if ref_result.success and ref_result.chain_depth == depth_filter:
                    oracle_pushes.append(ref_result.pushes)
                    problem_keys.append((env, key))

        if not oracle_pushes:
            return {
                'easy': {'cutoffs': cutoffs.tolist(), 'rates': [0.0] * len(cutoffs), 'total': 0},
                'medium': {'cutoffs': cutoffs.tolist(), 'rates': [0.0] * len(cutoffs), 'total': 0},
                'hard': {'cutoffs': cutoffs.tolist(), 'rates': [0.0] * len(cutoffs), 'total': 0},
            }

        p33 = np.percentile(oracle_pushes, 33.33)
        p66 = np.percentile(oracle_pushes, 66.67)

        difficulty_mapping = {}
        for (env, key), oracle_push in zip(problem_keys, oracle_pushes):
            if oracle_push <= p33:
                difficulty_mapping[(env, key)] = 'easy'
            elif oracle_push <= p66:
                difficulty_mapping[(env, key)] = 'medium'
            else:
                difficulty_mapping[(env, key)] = 'hard'

    # Collect times by difficulty using the mapping
    times_by_difficulty: Dict[str, List[float]] = {'easy': [], 'medium': [], 'hard': []}
    totals_by_difficulty: Dict[str, int] = {'easy': 0, 'medium': 0, 'hard': 0}

    for env in model_data:
        if env not in reference_data:
            continue
        for key in model_data[env]:
            if key not in reference_data[env]:
                continue

            ref_result = reference_data[env][key]
            if not ref_result.success or ref_result.chain_depth != depth_filter:
                continue

            difficulty = difficulty_mapping.get((env, key))
            if difficulty is None:
                continue

            model_result = model_data[env][key]
            totals_by_difficulty[difficulty] += 1

            # Check success based on learned_only flag
            if learned_only:
                if model_result.solved_by_learned:
                    times_by_difficulty[difficulty].append(model_result.time_taken)
            else:
                if model_result.success:
                    times_by_difficulty[difficulty].append(model_result.time_taken)

    # Compute rates at each cutoff
    result = {}
    for diff in ['easy', 'medium', 'hard']:
        times = np.array(times_by_difficulty[diff])
        total = totals_by_difficulty[diff]
        rates = []
        for cutoff in cutoffs:
            if total > 0:
                successes = np.sum(times <= cutoff) if len(times) > 0 else 0
                rates.append(successes / total)
            else:
                rates.append(0.0)
        result[diff] = {'cutoffs': cutoffs.tolist(), 'rates': rates, 'total': total}

    return result


def compute_push_based_success_by_difficulty(
    model_data: Dict[str, Dict[str, RegionResult]],
    reference_data: Dict[str, Dict[str, RegionResult]],
    config: EvalConfig,
    depth_filter: int = 2,
    difficulty_mapping: Optional[Dict[Tuple[str, str], str]] = None,
    learned_only: bool = False,
) -> Dict[str, Dict[str, Any]]:
    """
    Compute success rate as a function of push count cutoff, stratified by difficulty.

    Difficulty is based on pre-computed difficulty mapping (from multi-reference median pushes)
    or single-reference oracle pushes as fallback.

    Args:
        learned_only: If True, only count successes from ML-only phase (no fallback).

    Returns:
        {difficulty: {'cutoffs': [...], 'rates': [...], 'total': int}}
    """
    cutoffs = list(range(0, config.push_cutoff_max + 1, config.push_step))

    # If no difficulty mapping provided, compute from single reference
    if difficulty_mapping is None:
        oracle_pushes = []
        problem_keys = []

        for env in reference_data:
            if env not in model_data:
                continue
            for key in reference_data[env]:
                if key not in model_data[env]:
                    continue
                ref_result = reference_data[env][key]
                if ref_result.success and ref_result.chain_depth == depth_filter:
                    oracle_pushes.append(ref_result.pushes)
                    problem_keys.append((env, key))

        if not oracle_pushes:
            return {
                'easy': {'cutoffs': cutoffs, 'rates': [0.0] * len(cutoffs), 'total': 0},
                'medium': {'cutoffs': cutoffs, 'rates': [0.0] * len(cutoffs), 'total': 0},
                'hard': {'cutoffs': cutoffs, 'rates': [0.0] * len(cutoffs), 'total': 0},
            }

        p33 = np.percentile(oracle_pushes, 33.33)
        p66 = np.percentile(oracle_pushes, 66.67)

        difficulty_mapping = {}
        for (env, key), oracle_push in zip(problem_keys, oracle_pushes):
            if oracle_push <= p33:
                difficulty_mapping[(env, key)] = 'easy'
            elif oracle_push <= p66:
                difficulty_mapping[(env, key)] = 'medium'
            else:
                difficulty_mapping[(env, key)] = 'hard'

    # Collect pushes by difficulty using the mapping
    pushes_by_difficulty: Dict[str, List[int]] = {'easy': [], 'medium': [], 'hard': []}
    totals_by_difficulty: Dict[str, int] = {'easy': 0, 'medium': 0, 'hard': 0}

    for env in model_data:
        if env not in reference_data:
            continue
        for key in model_data[env]:
            if key not in reference_data[env]:
                continue

            ref_result = reference_data[env][key]
            if not ref_result.success or ref_result.chain_depth != depth_filter:
                continue

            difficulty = difficulty_mapping.get((env, key))
            if difficulty is None:
                continue

            model_result = model_data[env][key]
            totals_by_difficulty[difficulty] += 1

            # Check success based on learned_only flag
            if learned_only:
                if model_result.solved_by_learned:
                    pushes_by_difficulty[difficulty].append(model_result.pushes)
            else:
                if model_result.success:
                    pushes_by_difficulty[difficulty].append(model_result.pushes)

    # Compute rates at each cutoff
    result = {}
    for diff in ['easy', 'medium', 'hard']:
        pushes = np.array(pushes_by_difficulty[diff])
        total = totals_by_difficulty[diff]
        rates = []
        for cutoff in cutoffs:
            if total > 0:
                successes = np.sum(pushes <= cutoff) if len(pushes) > 0 else 0
                rates.append(successes / total)
            else:
                rates.append(0.0)
        result[diff] = {'cutoffs': cutoffs, 'rates': rates, 'total': total}

    return result


def plot_pushes_vs_success(
    push_data: Dict[str, Dict[str, Dict[str, List[float]]]],  # {model_name: {category: {cutoffs, rates, total}}}
    config: EvalConfig,
    output_path: Optional[str] = None,
    learned_only_data: Optional[Dict[str, Dict[str, Dict[str, List[float]]]]] = None,
):
    """Plot success rate vs push count cutoff (2-push only).

    Args:
        push_data: Total success data {model_name: {category: {cutoffs, rates, total}}}
        config: Evaluation config
        output_path: Path to save plot
        learned_only_data: ML-only phase success data (same format as push_data)
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # Get N from the first model's 2push data
    n_problems = 0
    for cat_data in push_data.values():
        if '2push' in cat_data:
            n_problems = cat_data['2push'].get('total', 0)
            break

    for model_idx, (model_name, cat_data) in enumerate(push_data.items()):
        if '2push' in cat_data:
            cutoffs = cat_data['2push']['cutoffs']
            rates = cat_data['2push']['rates']
            color = get_model_color(model_idx, config)
            ax.plot(cutoffs, rates, label=model_name,
                   color=color, linewidth=2)

            # Plot learned-only curve if available for this model
            if learned_only_data and model_name in learned_only_data:
                lo_cat_data = learned_only_data[model_name]
                if '2push' in lo_cat_data:
                    lo_rates = lo_cat_data['2push']['rates']
                    ax.plot(cutoffs, lo_rates, label=model_name.replace('Hybrid ', ''),
                           color=color, linewidth=2, linestyle='--', alpha=0.7)

    ax.set_xlabel('Simulation-verified push evaluations (# checks)')
    ax.set_ylabel('Success Rate')
    ax.set_ylim(0, 1.05)
    ax.set_xlim(0, config.push_cutoff_max)
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend(loc='lower right')

    ax.set_title(f"2-Push Problems (N={n_problems}) - Success Rate @ Push Evaluations", fontsize=14, fontweight='bold')

    # Add caption
    fig.text(0.5, -0.02, "One evaluation = one simulated feasibility check of a candidate push (not an executed push).",
             ha='center', fontsize=9, style='italic', color='#666666')
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white')
        print(f"Saved: {output_path}")
    return fig


def _plot_success_by_difficulty(
    data_by_difficulty: Dict[str, Dict[str, Dict[str, Any]]],
    config: EvalConfig,
    xlabel: str,
    xlim_max: float,
    title: str,
    cutoff_transform: Callable[[List], List] = lambda x: x,
    output_path: Optional[str] = None,
    learned_only_data: Optional[Dict[str, Dict[str, Dict[str, Any]]]] = None,
):
    """Generic success rate by difficulty plot (time or pushes)."""
    difficulty_levels = ['easy', 'medium', 'hard']
    difficulty_labels = {'easy': 'Easy', 'medium': 'Medium', 'hard': 'Hard'}
    difficulty_colors = {'easy': '#55A868', 'medium': '#DD8452', 'hard': '#C44E52'}

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)

    for ax_idx, diff in enumerate(difficulty_levels):
        ax = axes[ax_idx]
        n_problems = next(
            (d[diff].get('total', 0) for d in data_by_difficulty.values() if diff in d), 0
        )

        zero_success_models = []
        for model_idx, (model_name, diff_data) in enumerate(data_by_difficulty.items()):
            if diff not in diff_data:
                continue
            cutoffs = cutoff_transform(diff_data[diff]['cutoffs'])
            rates = diff_data[diff]['rates']
            color = get_model_color(model_idx, config)

            if max(rates) == 0:
                ax.plot(cutoffs, rates, label=model_name, color=color,
                       linewidth=1.5, linestyle='--', alpha=0.7)
                zero_success_models.append(model_name)
            else:
                ax.plot(cutoffs, rates, label=model_name, color=color, linewidth=2)

            # Plot learned-only curve if available for this model
            if learned_only_data and model_name in learned_only_data:
                lo_diff_data = learned_only_data[model_name]
                if diff in lo_diff_data:
                    lo_rates = lo_diff_data[diff]['rates']
                    ax.plot(cutoffs, lo_rates, label=model_name.replace('Hybrid ', ''),
                           color=color, linewidth=2, linestyle='--', alpha=0.7)

        if zero_success_models:
            ax.annotate(f"0%: {', '.join(zero_success_models)}", xy=(0.02, 0.02),
                       xycoords='axes fraction', fontsize=8, color='#666666', style='italic',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor='#cccccc'))

        ax.set_xlabel(xlabel)
        if ax_idx == 0:
            ax.set_ylabel('Success Rate')
        ax.set_ylim(0, 1.05)
        ax.set_xlim(0, xlim_max)
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.set_title(f"{difficulty_labels[diff]} (N={n_problems})", fontsize=12,
                    fontweight='bold', color=difficulty_colors[diff])

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, -0.02),
              ncol=min(len(labels), 4), fontsize=10)
    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white')
        print(f"Saved: {output_path}")
    return fig


def plot_time_vs_success_by_difficulty(
    time_data_by_difficulty: Dict[str, Dict[str, Dict[str, Any]]],
    config: EvalConfig,
    output_path: Optional[str] = None,
    learned_only_data: Optional[Dict[str, Dict[str, Dict[str, Any]]]] = None,
):
    """Plot success rate vs time cutoff, with subplots for each difficulty level."""
    return _plot_success_by_difficulty(
        time_data_by_difficulty, config,
        xlabel='Time cutoff (s)',
        xlim_max=config.time_cutoff_max / 1000.0,
        title="Success Rate @ Time Cutoff by Difficulty",
        cutoff_transform=lambda x: [c / 1000.0 for c in x],  # ms to seconds
        output_path=output_path,
        learned_only_data=learned_only_data,
    )


def plot_pushes_vs_success_by_difficulty(
    push_data_by_difficulty: Dict[str, Dict[str, Dict[str, Any]]],
    config: EvalConfig,
    output_path: Optional[str] = None,
    learned_only_data: Optional[Dict[str, Dict[str, Dict[str, Any]]]] = None,
):
    """Plot success rate vs push count cutoff, with subplots for each difficulty level."""
    return _plot_success_by_difficulty(
        push_data_by_difficulty, config,
        xlabel='# Checks',
        xlim_max=config.push_cutoff_max,
        title="Success Rate @ Push Evaluations by Difficulty",
        output_path=output_path,
        learned_only_data=learned_only_data,
    )


def plot_interactions(
    model_stats: List[ModelStats],
    config: EvalConfig,
    output_path: Optional[str] = None,
):
    """Plot interaction statistics (wall and movable collisions) - 2-push only."""
    n_models = len(model_stats)
    names = [s.name for s in model_stats]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Plot 1: Wall collision rate
    ax1 = axes[0]
    rates = [s.depth_2.wall_collision_rate for s in model_stats]
    bars = ax1.bar(range(len(names)), rates,
                   color=[get_model_color(i, config) for i in range(len(names))],
                   edgecolor='white', linewidth=0.5)

    for bar, rate in zip(bars, rates):
        if rate > 0:
            ax1.annotate(f'{rate:.0%}',
                       xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                       xytext=(0, 3), textcoords="offset points",
                       ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax1.set_xticks(range(len(names)))
    ax1.set_xticklabels(names, rotation=15, ha='right')
    ax1.set_ylim(0, 1.15)
    ax1.set_ylabel('Wall Collision Rate')
    ax1.set_title('2-Push Problems - Wall Collision Rate\n(among successful runs)')
    ax1.grid(True, axis='y', linestyle='--', alpha=0.6)

    # Plot 2: Movable collision rate
    ax2 = axes[1]
    rates = [s.depth_2.any_movable_collision_rate for s in model_stats]
    bars = ax2.bar(range(len(names)), rates,
                   color=[get_model_color(i, config) for i in range(len(names))],
                   edgecolor='white', linewidth=0.5)

    for bar, rate in zip(bars, rates):
        if rate > 0:
            ax2.annotate(f'{rate:.0%}',
                       xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                       xytext=(0, 3), textcoords="offset points",
                       ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax2.set_xticks(range(len(names)))
    ax2.set_xticklabels(names, rotation=15, ha='right')
    ax2.set_ylim(0, 1.15)
    ax2.set_ylabel('Movable Collision Rate')
    ax2.set_title('2-Push Problems - Movable Object Collision Rate\n(among successful runs)')
    ax2.grid(True, axis='y', linestyle='--', alpha=0.6)

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white')
        print(f"Saved: {output_path}")
    return fig


def plot_collision_success_rates(
    collision_stats: Dict[str, Dict[str, Dict[str, int]]],  # {model_name: {collision_type: {successes, total}}}
    config: EvalConfig,
    output_path: Optional[str] = None,
):
    """
    Plot success rates by collision type for each model.

    Shows a grouped bar chart with collision types on x-axis and models as groups.
    """
    collision_types = ['none', 'wall_only', 'movable_only', 'both']
    collision_labels = ['No Collision', 'Wall Only', 'Movable Only', 'Both']
    model_names = list(collision_stats.keys())
    n_models = len(model_names)

    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(collision_types))
    width = 0.35

    for i, model_name in enumerate(model_names):
        rates = []
        counts = []
        for ct in collision_types:
            stats = collision_stats[model_name][ct]
            rate = stats['successes'] / stats['total'] if stats['total'] > 0 else 0.0
            rates.append(rate)
            counts.append((stats['successes'], stats['total']))

        offset = (i - n_models/2 + 0.5) * width
        bars = ax.bar(x + offset, rates, width * 0.9, label=model_name,
                      color=get_model_color(i, config), edgecolor='white', linewidth=0.5)

        # Add percentage labels on bars
        for bar, rate, (succ, total) in zip(bars, rates, counts):
            if total > 0:
                y_pos = bar.get_height()
                ax.annotate(f'{rate:.0%}',
                           xy=(bar.get_x() + bar.get_width()/2, y_pos),
                           xytext=(0, 4), textcoords="offset points",
                           ha='center', va='bottom', fontsize=9, fontweight='bold')
                # Add count below the bar
                ax.annotate(f'n={total}',
                           xy=(bar.get_x() + bar.get_width()/2, 0),
                           xytext=(0, -12), textcoords="offset points",
                           ha='center', va='top', fontsize=8, color='#666666')

    ax.set_xticks(x)
    ax.set_xticklabels(collision_labels)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel('Success Rate')
    ax.set_title('Success Rate by Collision Type Required')
    ax.legend(loc='upper right', frameon=True, fancybox=True, shadow=False)

    # Add horizontal line at 100%
    ax.axhline(y=1.0, color='#888888', linestyle='--', linewidth=0.8, alpha=0.5)

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white')
        print(f"Saved: {output_path}")
    return fig


def plot_confusion_matrix(
    confusion_matrices: Dict[str, Dict[str, int]],  # {model_name: matrix}
    config: EvalConfig,
    output_path: Optional[str] = None,
):
    """
    Plot confusion matrix showing oracle 2-push vs model achieved chain depth.

    Shows how models solve 2-push problems:
    - Model 1-push: Solved with 1 push (shouldn't happen for true 2-push)
    - Model 2-push: Solved with exactly 2 pushes (optimal)
    - Model 2+ push: Solved with more than 2 pushes (over-pushed)
    - Failed: Did not solve
    """
    model_names = list(confusion_matrices.keys())
    n_models = len(model_names)

    fig, ax = plt.subplots(figsize=(10, 6))

    # Build data: each model's breakdown for oracle 2-push problems
    categories = ['Model\n1-Push', 'Model\n2-Push', 'Model\n2+ Push', 'Failed']
    x = np.arange(len(categories))
    width = 0.8 / n_models

    for i, model_name in enumerate(model_names):
        matrix = confusion_matrices[model_name]
        oracle_2_total = (matrix['oracle_2_model_1'] + matrix['oracle_2_model_2'] +
                         matrix['oracle_2_model_2plus'] + matrix['oracle_2_failed'])

        if oracle_2_total > 0:
            counts = [matrix['oracle_2_model_1'], matrix['oracle_2_model_2'],
                     matrix['oracle_2_model_2plus'], matrix['oracle_2_failed']]
            rates = [c / oracle_2_total for c in counts]
        else:
            counts = [0, 0, 0, 0]
            rates = [0, 0, 0, 0]

        offset = (i - n_models/2 + 0.5) * width
        bars = ax.bar(x + offset, rates, width * 0.9, label=f'{model_name} (n={oracle_2_total})',
                      color=get_model_color(i, config), edgecolor='white', linewidth=0.5)

        for bar, rate, count in zip(bars, rates, counts):
            if rate > 0.02:  # Only show label if > 2%
                ax.annotate(f'{rate:.0%}',
                           xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                           xytext=(0, 2), textcoords="offset points",
                           ha='center', va='bottom', fontsize=8, fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel('Percentage of 2-Push Problems')
    ax.set_title('2-Push Problems - Model Result Breakdown', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', frameon=True, fancybox=True)
    ax.axhline(y=1.0, color='#888888', linestyle='--', linewidth=0.8, alpha=0.5)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white')
        print(f"Saved: {output_path}")
    return fig


def plot_hybrid_decomposition(
    hybrid_stats: Dict[str, HybridStats],  # {model_name: HybridStats}
    config: EvalConfig,
    output_path: Optional[str] = None,
):
    """
    Plot hybrid decomposition: stacked bar of outcome fractions + fallback distribution.

    Panel A: Stacked bar showing % solved by learned, % solved by fallback, % failed
    Panel B: Box plot of checks before fallback triggered
    """
    model_names = list(hybrid_stats.keys())
    n_models = len(model_names)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Panel A: Stacked bar of outcome fractions
    ax1 = axes[0]

    learned_rates = [hybrid_stats[m].learned_rate for m in model_names]
    fallback_rates = [hybrid_stats[m].fallback_rate for m in model_names]
    failed_rates = [1 - hybrid_stats[m].success_rate for m in model_names]
    totals = [hybrid_stats[m].total for m in model_names]

    x = np.arange(n_models)
    width = 0.6

    # Stack: learned (bottom), fallback (middle), failed (top)
    bars_learned = ax1.bar(x, learned_rates, width, label='Solved by Learned',
                           color='#55A868', edgecolor='white', linewidth=0.5)
    bars_fallback = ax1.bar(x, fallback_rates, width, bottom=learned_rates,
                            label='Solved by Fallback', color='#DD8452',
                            edgecolor='white', linewidth=0.5)
    bars_failed = ax1.bar(x, failed_rates, width,
                          bottom=[l + f for l, f in zip(learned_rates, fallback_rates)],
                          label='Failed', color='#C44E52', edgecolor='white', linewidth=0.5)

    # Add labels
    for i, (lr, fr, fail, total) in enumerate(zip(learned_rates, fallback_rates, failed_rates, totals)):
        # Learned label
        if lr > 0.05:
            ax1.text(i, lr/2, f'{lr:.0%}', ha='center', va='center',
                    fontsize=10, fontweight='bold', color='white')
        # Fallback label
        if fr > 0.05:
            ax1.text(i, lr + fr/2, f'{fr:.0%}', ha='center', va='center',
                    fontsize=10, fontweight='bold', color='white')
        # Failed label
        if fail > 0.05:
            ax1.text(i, lr + fr + fail/2, f'{fail:.0%}', ha='center', va='center',
                    fontsize=10, fontweight='bold', color='white')
        # Total count on top
        ax1.text(i, 1.02, f'n={total}', ha='center', va='bottom', fontsize=9, color='#666666')

    ax1.set_xticks(x)
    ax1.set_xticklabels(model_names, rotation=15, ha='right')
    ax1.set_ylim(0, 1.15)
    ax1.set_ylabel('Fraction of Problems')
    ax1.set_title('Hybrid Decomposition: Learned vs Fallback')
    ax1.legend(loc='upper right', frameon=True)
    ax1.axhline(y=1.0, color='#888888', linestyle='--', linewidth=0.8, alpha=0.5)

    # Panel B: Checks before fallback (for fallback cases only)
    ax2 = axes[1]

    data = []
    labels = []
    for m in model_names:
        if hybrid_stats[m].checks_before_fallback:
            data.append(hybrid_stats[m].checks_before_fallback)
            labels.append(f"{m}\n(n={len(hybrid_stats[m].checks_before_fallback)})")
        else:
            data.append([0])
            labels.append(f"{m}\n(n=0)")

    bp = ax2.boxplot(data, tick_labels=labels, patch_artist=True, showfliers=False)

    for i, patch in enumerate(bp['boxes']):
        patch.set_facecolor(get_model_color(i, config))
        patch.set_alpha(0.85)
        patch.set_edgecolor('white')

    for median in bp['medians']:
        median.set_color('#333333')
        median.set_linewidth(2)

    ax2.tick_params(axis='x', rotation=15)
    ax2.set_ylabel('Checks Before Fallback (ml_goals_aligned)')
    ax2.set_title('Fallback Trigger Point Distribution\n(for problems solved by fallback)')
    ax2.grid(True, axis='y', linestyle='--', alpha=0.6)

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white')
        print(f"Saved: {output_path}")
    return fig


def plot_collision_bucket_efficiency(
    collision_efficiency: Dict[str, Dict[str, Dict[str, Any]]],  # {model: {bucket: {pushes, times, ...}}}
    config: EvalConfig,
    output_path: Optional[str] = None,
):
    """
    Plot efficiency (checks, time) by collision bucket for each model.

    Two panels: checks boxplot and time boxplot per collision type.
    """
    collision_types = ['none', 'wall_only', 'movable_only', 'both']
    collision_labels = ['No Collision', 'Wall Only', 'Movable Only', 'Both']
    model_names = list(collision_efficiency.keys())
    n_models = len(model_names)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Panel A: Checks by collision type
    ax1 = axes[0]
    positions = []
    data_checks = []
    colors = []
    tick_positions = []
    tick_labels = []

    pos = 0
    for ct_idx, ct in enumerate(collision_types):
        group_start = pos
        for m_idx, model in enumerate(model_names):
            pushes = collision_efficiency[model][ct]['pushes']
            data_checks.append(pushes if pushes else [0])
            positions.append(pos)
            colors.append(get_model_color(m_idx, config))
            pos += 1
        pos += 0.5  # Gap between groups
        tick_positions.append((group_start + pos - 1.5) / 2)
        tick_labels.append(collision_labels[ct_idx])

    bp1 = ax1.boxplot(data_checks, positions=positions, widths=0.7, patch_artist=True, showfliers=False)
    for patch, color in zip(bp1['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.85)

    ax1.set_xticks(tick_positions)
    ax1.set_xticklabels(tick_labels)
    ax1.set_ylabel('Checks to Solution')
    ax1.set_title('Efficiency by Collision Type - Checks')
    ax1.grid(True, axis='y', linestyle='--', alpha=0.6)

    # Add legend
    legend_handles = [plt.Rectangle((0, 0), 1, 1, facecolor=get_model_color(i, config))
                     for i in range(n_models)]
    ax1.legend(legend_handles, model_names, loc='upper right')

    # Panel B: Time by collision type
    ax2 = axes[1]
    positions = []
    data_times = []
    colors = []

    pos = 0
    for ct in collision_types:
        for m_idx, model in enumerate(model_names):
            times = collision_efficiency[model][ct]['times']
            # Convert to seconds for readability
            data_times.append([t/1000 for t in times] if times else [0])
            positions.append(pos)
            colors.append(get_model_color(m_idx, config))
            pos += 1
        pos += 0.5

    bp2 = ax2.boxplot(data_times, positions=positions, widths=0.7, patch_artist=True, showfliers=False)
    for patch, color in zip(bp2['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.85)

    ax2.set_xticks(tick_positions)
    ax2.set_xticklabels(tick_labels)
    ax2.set_ylabel('Time to Solution (seconds)')
    ax2.set_title('Efficiency by Collision Type - Time')
    ax2.grid(True, axis='y', linestyle='--', alpha=0.6)
    ax2.legend(legend_handles, model_names, loc='upper right')

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white')
        print(f"Saved: {output_path}")
    return fig


def print_summary(model_stats: List[ModelStats]):
    """Print summary statistics (2-push only)."""
    print("\n" + "=" * 80)
    print("2-PUSH EVALUATION SUMMARY")
    print("=" * 80)

    for stats in model_stats:
        print(f"\n{'─' * 40}")
        print(f"Model: {stats.name}")
        print(f"{'─' * 40}")

        cat_stats = stats.depth_2
        print(f"  Success: {cat_stats.successes}/{cat_stats.total} = {cat_stats.success_rate:.4f}")
        if cat_stats.pushes:
            print(f"  Pushes:  median={cat_stats.median_pushes:.1f}, mean={cat_stats.mean_pushes:.1f}")
        if cat_stats.times:
            print(f"  Time:    median={cat_stats.median_time:.1f}ms, mean={cat_stats.mean_time:.1f}ms")
        if cat_stats.solutions:
            print(f"  Solutions: total={cat_stats.total_solutions}, mean={cat_stats.mean_solutions:.1f}")
        if cat_stats.successes > 0:
            print(f"  Interactions: wall_col={cat_stats.wall_collision_rate:.1%}, movable_col={cat_stats.any_movable_collision_rate:.1%}")
        if cat_stats.ml_aligned_counts:
            n_samples = len(cat_stats.ml_aligned_counts)
            print(f"  ML Grounding: {cat_stats.mean_ml_aligned_reachable_ratio:.1%} reachable "
                  f"({cat_stats.total_ml_aligned_reachable}/{cat_stats.total_ml_aligned} aligned primitives, n={n_samples})")

        if stats.failure_reasons:
            print(f"\n  Failure Reasons:")
            for reason, count in sorted(stats.failure_reasons.items(), key=lambda x: -x[1]):
                print(f"    {reason}: {count}")


# =============================================================================
# Markdown Report Helpers
# =============================================================================

def _format_time_budget(t_ms: float) -> str:
    """Format time budget in ms for display (seconds or ms)."""
    if t_ms >= 1000:
        return f"@{t_ms/1000:.0f}s"
    return f"@{t_ms:.0f}ms"


def _report_summary_table(lines: List[str], model_stats: List['ModelStats']):
    """Generate summary table section."""
    lines.append("## Summary Table\n")
    lines.append("**Definitions:**")
    lines.append("- **Success**: First valid 2-push plan found satisfying clearance + executability")
    lines.append("- **Checks**: # simulation-verified candidate push primitive evaluations until first solution")
    lines.append("- **Time**: End-to-end wall-clock until first solution (includes inference+decode+scoring+verification)\n")

    lines.append("| Model | Success % | Checks (median [IQR]) | Time (s) (median [IQR]) |")
    lines.append("|-------|-----------|----------------------|-------------------------|")
    for stats in model_stats:
        cat_stats = stats.depth_2
        success_str = f"**{cat_stats.success_rate:.1%}** ({cat_stats.successes}/{cat_stats.total})"
        if cat_stats.pushes:
            p_iqr = cat_stats.pushes_iqr
            checks_str = f"{cat_stats.median_pushes:.0f} [{p_iqr[0]:.0f}, {p_iqr[1]:.0f}]"
        else:
            checks_str = "-"
        if cat_stats.times:
            t_iqr = cat_stats.time_iqr
            time_str = f"{cat_stats.median_time/1000:.1f} [{t_iqr[0]/1000:.1f}, {t_iqr[1]/1000:.1f}]"
        else:
            time_str = "-"
        lines.append(f"| {stats.name} | {success_str} | {checks_str} | {time_str} |")
    lines.append("")


def _report_success_at_budget(
    lines: List[str],
    success_at_budget: Dict[str, Dict[int, Dict[str, Any]]],
    config: 'EvalConfig',
):
    """Generate Success@Budget section."""
    lines.append("## Success@Budget (Constant-Compute Comparison)\n")
    lines.append("**Definition:**")
    lines.append("- **Success@B**: Success rate when limited to B simulation-verified push evaluations")
    lines.append("- Enables fair comparison by fixing verification compute budget\n")

    budget_values = config.success_at_budget_values if config else [50, 100, 200]
    b_headers = [f"@{b}" for b in budget_values]
    header = "| Model |" + " | ".join(b_headers) + " |"
    sep = "|-------|" + "|".join(["--------"] * len(budget_values)) + "|"
    lines.append(header)
    lines.append(sep)

    for name in success_at_budget:
        row = f"| {name} |"
        for b in budget_values:
            stats = success_at_budget[name].get(b, {})
            row += f" **{stats['rate']:.1%}** |" if stats else " - |"
        lines.append(row)
    lines.append("")


def _report_success_at_time(
    lines: List[str],
    success_at_time: Dict[str, Dict[float, Dict[str, Any]]],
    config: 'EvalConfig',
):
    """Generate Success@Time section."""
    lines.append("## Success@Time (Constant-Time Comparison)\n")
    lines.append("**Definition:**")
    lines.append("- **Success@T**: Success rate when limited to T seconds of search time")
    lines.append("- Enables fair comparison by fixing wall-clock time budget\n")

    time_values = config.success_at_time_values if config else [5000, 10000, 30000]
    t_headers = [_format_time_budget(t) for t in time_values]
    header = "| Model |" + " | ".join(t_headers) + " |"
    sep = "|-------|" + "|".join(["--------"] * len(time_values)) + "|"
    lines.append(header)
    lines.append(sep)

    for name in success_at_time:
        row = f"| {name} |"
        for t in time_values:
            stats = success_at_time[name].get(t, {})
            row += f" **{stats['rate']:.1%}** |" if stats else " - |"
        lines.append(row)
    lines.append("")


def _report_ml_grounding(lines: List[str], model_stats: List['ModelStats']):
    """Generate ML Prediction Grounding section."""
    lines.append("## ML Prediction Grounding\n")
    lines.append("**Definition:**")
    lines.append("- **Grounding %**: Fraction of ML-aligned primitive slots where the robot can actually reach the push edge")
    lines.append("- Measures how well ML predictions are grounded in physical reachability\n")

    lines.append("| Model | Grounding % | Reachable / Aligned | N samples |")
    lines.append("|-------|-------------|---------------------|-----------|")
    for stats in model_stats:
        cat_stats = stats.depth_2
        if cat_stats.ml_aligned_counts:
            n_samples = len(cat_stats.ml_aligned_counts)
            grounding_pct = f"**{cat_stats.mean_ml_aligned_reachable_ratio:.1%}**"
            ratio_str = f"{cat_stats.total_ml_aligned_reachable}/{cat_stats.total_ml_aligned}"
            lines.append(f"| {stats.name} | {grounding_pct} | {ratio_str} | {n_samples} |")
        else:
            lines.append(f"| {stats.name} | - | - | 0 |")
    lines.append("")


def _report_ra_at_k(
    lines: List[str],
    ra_at_k_stats: Dict[str, Dict[Optional[int], 'RAatKStats']],
    config: 'EvalConfig',
):
    """Generate ReachableAttachment@K section."""
    lines.append("## ReachableAttachment@K\n")
    lines.append("**Definition:**")
    lines.append("- **RA@K**: Among top-K ML-ranked primitives (by vote count), fraction with reachable push attachments")
    lines.append("- Measures how well ML ranking prioritizes physically feasible pushes\n")

    k_values = config.ra_at_k_values if config else [10, 50, 100, None]
    k_headers = [f"@{k}" if k is not None else "@All" for k in k_values]
    header = "| Model |" + " | ".join(k_headers) + " |"
    sep = "|-------|" + "|".join(["--------"] * len(k_values)) + "|"
    lines.append(header)
    lines.append(sep)

    for name in ra_at_k_stats:
        row = f"| {name} |"
        for k in k_values:
            ra = ra_at_k_stats[name].get(k)
            row += f" **{ra.macro:.1%}** |" if ra and ra.n_instances > 0 else " - |"
        lines.append(row)
    lines.append("")

    # Detailed table
    lines.append("### Detailed RA@K Statistics\n")
    lines.append("| Model | K | Macro | Micro | Reachable/Total | N |")
    lines.append("|-------|---|-------|-------|-----------------|---|")
    for name in ra_at_k_stats:
        for k in k_values:
            ra = ra_at_k_stats[name].get(k)
            k_str = str(k) if k is not None else "All"
            if ra and ra.n_instances > 0:
                lines.append(f"| {name} | {k_str} | {ra.macro:.1%} | {ra.micro:.1%} | "
                           f"{ra.total_reachable}/{ra.total_considered} | {ra.n_instances} |")
            else:
                lines.append(f"| {name} | {k_str} | - | - | - | 0 |")
    lines.append("")


def _report_hybrid_decomposition(
    lines: List[str],
    hybrid_stats: Dict[str, 'HybridStats'],
    hybrid_stats_by_difficulty: Optional[Dict[str, Dict[str, Dict[str, int]]]] = None,
):
    """Generate Hybrid Decomposition section."""
    lines.append("## Hybrid Decomposition\n")
    lines.append("**Definitions:**")
    lines.append("- **LEARNED**: Solved during ML-only phase (ML-scored primitives)")
    lines.append("- **FALLBACK**: ML phase exhausted, solved during primitives phase")
    lines.append("- **FAILED**: Neither phase found a solution\n")

    # Outcome breakdown
    lines.append("### Outcome Breakdown\n")
    lines.append("| Model | N | Learned | Fallback | Failed |")
    lines.append("|-------|---|---------|----------|--------|")
    for name, hs in hybrid_stats.items():
        learned_str = f"{hs.learned_rate:.1%} ({hs.solved_by_learned})"
        fallback_str = f"{hs.fallback_rate:.1%} ({hs.solved_by_fallback})"
        failed_str = f"{(1 - hs.success_rate):.1%} ({hs.failed})"
        lines.append(f"| {name} | {hs.total} | {learned_str} | {fallback_str} | {failed_str} |")
    lines.append("")

    # Learned cases efficiency
    lines.append("### Learned Cases: Efficiency\n")
    lines.append("*Problems solved by ML-only phase.*\n")
    lines.append("| Model | N | Checks (median [IQR]) | Time (s) (median [IQR]) |")
    lines.append("|-------|---|----------------------|-------------------------|")
    for name, hs in hybrid_stats.items():
        n_learned = hs.solved_by_learned
        if hs.learned_pushes:
            l_iqr = hs.learned_pushes_iqr
            learned_checks = f"{hs.learned_median_pushes:.0f} [{l_iqr[0]:.0f}, {l_iqr[1]:.0f}]"
            lt_iqr = hs.learned_time_iqr
            learned_time = f"{hs.learned_median_time/1000:.1f} [{lt_iqr[0]/1000:.1f}, {lt_iqr[1]/1000:.1f}]"
        else:
            learned_checks = "-"
            learned_time = "-"
        lines.append(f"| {name} | {n_learned} | {learned_checks} | {learned_time} |")
    lines.append("")

    # Fallback cases efficiency
    any_fallback = any(hs.solved_by_fallback > 0 for hs in hybrid_stats.values())
    if any_fallback:
        lines.append("### Fallback Cases: Efficiency\n")
        lines.append("*Problems where ML phase exhausted, solved by primitives phase. Totals include both phases.*\n")
        lines.append("| Model | N | Checks (median [IQR]) | Time (s) (median [IQR]) |")
        lines.append("|-------|---|----------------------|-------------------------|")
        for name, hs in hybrid_stats.items():
            n_fallback = hs.solved_by_fallback
            if hs.fallback_pushes:
                f_iqr = hs.fallback_pushes_iqr
                total_checks = f"{hs.fallback_median_pushes:.0f} [{f_iqr[0]:.0f}, {f_iqr[1]:.0f}]"
                ft_iqr = hs.fallback_time_iqr
                total_time = f"{hs.fallback_median_time/1000:.1f} [{ft_iqr[0]/1000:.1f}, {ft_iqr[1]/1000:.1f}]"
            else:
                total_checks = "-"
                total_time = "-"
            lines.append(f"| {name} | {n_fallback} | {total_checks} | {total_time} |")
        lines.append("")

    # Hybrid stats by difficulty
    if hybrid_stats_by_difficulty:
        lines.append("### Outcome by Difficulty\n")
        lines.append("*Learned vs Fallback breakdown per difficulty bucket (based on oracle pushes).*\n")
        lines.append("| Model | Difficulty | N | Learned | Fallback | Failed |")
        lines.append("|-------|------------|---|---------|----------|--------|")

        difficulty_levels = ['easy', 'medium', 'hard']
        difficulty_labels = {'easy': 'Easy', 'medium': 'Medium', 'hard': 'Hard'}

        for name in hybrid_stats_by_difficulty:
            for diff in difficulty_levels:
                stats = hybrid_stats_by_difficulty[name][diff]
                n = stats['total']
                if n > 0:
                    learned_pct = f"{stats['learned']/n:.1%} ({stats['learned']})"
                    fallback_pct = f"{stats['fallback']/n:.1%} ({stats['fallback']})"
                    failed_pct = f"{stats['failed']/n:.1%} ({stats['failed']})"
                else:
                    learned_pct = "-"
                    fallback_pct = "-"
                    failed_pct = "-"
                lines.append(f"| {name} | {difficulty_labels[diff]} | {n} | {learned_pct} | {fallback_pct} | {failed_pct} |")
        lines.append("")


def _report_collision_stratification(
    lines: List[str],
    collision_efficiency: Dict[str, Dict[str, Dict[str, Any]]],
):
    """Generate Collision Stratification section."""
    collision_types = ['none', 'wall_only', 'movable_only', 'both']
    collision_labels = {'none': 'No Collision', 'wall_only': 'Wall Only',
                        'movable_only': 'Movable Only', 'both': 'Both'}

    lines.append("## Success Rate by Collision Type\n")
    lines.append("Stratifies problems by the type of collisions encountered in the oracle solution. ")
    lines.append("**No Collision**: Direct push to goal. **Wall Only**: Push causes wall contact. ")
    lines.append("**Movable Only**: Push causes contact with other movable objects. ")
    lines.append("**Both**: Push causes both wall and movable object contact.\n")

    header = "| Model |" + "".join(f" {collision_labels[ct]} |" for ct in collision_types)
    lines.append(header)
    sep = "|-------|" + "".join("-------------|" for _ in collision_types)
    lines.append(sep)

    for name in collision_efficiency:
        row = f"| {name} |"
        for ct in collision_types:
            stats = collision_efficiency[name][ct]
            rate = stats['successes'] / stats['total'] if stats['total'] > 0 else 0.0
            row += f" {rate:.1%} ({stats['successes']}/{stats['total']}) |"
        lines.append(row)
    lines.append("")

    # Efficiency by bucket
    lines.append("### Efficiency by Collision Type (Solved Cases Only)\n")
    lines.append("*Note: Efficiency numbers are computed over solved cases only; models with lower success rates may appear more efficient due to selection bias (e.g., only succeeding on easier instances).*\n")
    lines.append("| Model | Collision Type | N | Median Checks | Median Time (s) |")
    lines.append("|-------|----------------|---|---------------|-----------------|")

    for name in collision_efficiency:
        for ct in collision_types:
            stats = collision_efficiency[name][ct]
            pushes = stats['pushes']
            times = stats['times']
            n_solved = len(pushes)
            if pushes:
                med_checks = f"{np.median(pushes):.0f}"
                med_time = f"{np.median(times)/1000:.1f}"
            else:
                med_checks = "-"
                med_time = "-"
            lines.append(f"| {name} | {collision_labels[ct]} | {n_solved} | {med_checks} | {med_time} |")
    lines.append("")


def _report_difficulty_stratification(
    lines: List[str],
    difficulty_stratification: Dict[str, Dict[str, Dict[str, Any]]],
):
    """Generate Difficulty Stratification section."""
    difficulty_levels = ['easy', 'medium', 'hard']
    difficulty_labels = {'easy': 'Easy', 'medium': 'Medium', 'hard': 'Hard'}

    lines.append("## Difficulty Stratification (by Oracle Pushes)\n")
    lines.append("Categorizes problems by computational difficulty using the oracle's push count as a proxy. ")
    lines.append("Problems requiring fewer simulation-verified checks are considered easier. ")
    lines.append("Enables analysis of model performance across different difficulty levels.\n")
    lines.append("*Problems split into thirds by oracle pushes: Easy (fewest 33%), Medium (middle 33%), Hard (most 33%).*\n")

    # Oracle push ranges
    first_model = list(difficulty_stratification.keys())[0]
    range_info = []
    for diff in difficulty_levels:
        r = difficulty_stratification[first_model][diff]['oracle_push_range']
        range_info.append(f"**{difficulty_labels[diff]}**: {r[0]}–{r[1]} pushes")
    lines.append(f"Oracle push ranges: {', '.join(range_info)}\n")

    # Success rate table
    lines.append("### Success Rate by Difficulty\n")
    header = "| Model |" + "".join(f" {difficulty_labels[d]} |" for d in difficulty_levels)
    lines.append(header)
    sep = "|-------|" + "".join("------------|" for _ in difficulty_levels)
    lines.append(sep)

    for name in difficulty_stratification:
        row = f"| {name} |"
        for diff in difficulty_levels:
            stats = difficulty_stratification[name][diff]
            rate = stats['successes'] / stats['total'] if stats['total'] > 0 else 0.0
            row += f" {rate:.1%} ({stats['successes']}/{stats['total']}) |"
        lines.append(row)
    lines.append("")

    # Efficiency by difficulty
    lines.append("### Efficiency by Difficulty (Solved Cases Only)\n")
    lines.append("*Note: Efficiency computed over solved cases only; selection bias may apply.*\n")
    lines.append("| Model | Difficulty | N | Median Checks | Median Time (s) |")
    lines.append("|-------|------------|---|---------------|-----------------|")

    for name in difficulty_stratification:
        for diff in difficulty_levels:
            stats = difficulty_stratification[name][diff]
            pushes = stats['pushes']
            times = stats['times']
            n_solved = len(pushes)
            if pushes:
                med_checks = f"{np.median(pushes):.0f}"
                med_time = f"{np.median(times)/1000:.1f}"
            else:
                med_checks = "-"
                med_time = "-"
            lines.append(f"| {name} | {difficulty_labels[diff]} | {n_solved} | {med_checks} | {med_time} |")
    lines.append("")


def _report_legacy_tables(
    lines: List[str],
    model_stats: List['ModelStats'],
    confusion_matrices: Optional[Dict[str, Dict[str, int]]] = None,
):
    """Generate legacy detailed statistics tables."""
    lines.append("## Detailed Statistics\n")
    lines.append("Granular performance metrics including success counts, efficiency distributions, ")
    lines.append("and statistical summaries (median, IQR, mean) for checks and time.\n")

    # Success rates
    lines.append("### Success Rates\n")
    lines.append("| Model | Successes | Total | Success Rate |")
    lines.append("|-------|-----------|-------|--------------|")
    for stats in model_stats:
        cat_stats = stats.depth_2
        lines.append(f"| {stats.name} | {cat_stats.successes} | {cat_stats.total} | **{cat_stats.success_rate:.1%}** |")
    lines.append("")

    # Pushes statistics
    lines.append("### Checks to Success (Successful Runs Only)\n")
    lines.append("| Model | Median | IQR [25%, 75%] | Mean |")
    lines.append("|-------|--------|----------------|------|")
    for stats in model_stats:
        cat_stats = stats.depth_2
        if cat_stats.pushes:
            p_iqr = cat_stats.pushes_iqr
            lines.append(f"| {stats.name} | {cat_stats.median_pushes:.0f} | [{p_iqr[0]:.0f}, {p_iqr[1]:.0f}] | {cat_stats.mean_pushes:.0f} |")
        else:
            lines.append(f"| {stats.name} | - | - | - |")
    lines.append("")

    # Time statistics
    lines.append("### Time to Success (s) (Successful Runs Only)\n")
    lines.append("| Model | Median | IQR [25%, 75%] | Mean |")
    lines.append("|-------|--------|----------------|------|")
    for stats in model_stats:
        cat_stats = stats.depth_2
        if cat_stats.times:
            t_iqr = cat_stats.time_iqr
            lines.append(f"| {stats.name} | {cat_stats.median_time/1000:.1f} | [{t_iqr[0]/1000:.1f}, {t_iqr[1]/1000:.1f}] | {cat_stats.mean_time/1000:.1f} |")
        else:
            lines.append(f"| {stats.name} | - | - | - |")
    lines.append("")

    # Interaction statistics
    lines.append("## Interaction Statistics (Successful Runs Only)\n")
    lines.append("Measures how often solutions involve object collisions during push execution. ")
    lines.append("**Wall Collision Rate**: Fraction of solutions where pushed object contacts walls. ")
    lines.append("**Movable Collision Rate**: Fraction where pushed object contacts other movable objects.\n")
    lines.append("*Note: Statistics computed over successful runs only. Models with lower success rates may show different interaction patterns due to selection bias (failing on harder instances).*\n")
    lines.append("| Model | Wall Collision Rate | Movable Collision Rate |")
    lines.append("|-------|---------------------|------------------------|")
    for stats in model_stats:
        cat_stats = stats.depth_2
        if cat_stats.successes > 0:
            wall_rate = f"{cat_stats.wall_collision_rate:.1%} ({cat_stats.wall_collisions}/{cat_stats.successes})"
            any_mov = sum(1 for c in cat_stats.movable_collisions_list if c > 0)
            mov_rate = f"{cat_stats.any_movable_collision_rate:.1%} ({any_mov}/{cat_stats.successes})"
            lines.append(f"| {stats.name} | {wall_rate} | {mov_rate} |")
        else:
            lines.append(f"| {stats.name} | - | - |")
    lines.append("")

    # Confusion matrix
    if confusion_matrices:
        lines.append("## Model Result Breakdown\n")
        lines.append("How each model solved 2-push problems:\n")
        lines.append("- **Model 1-Push**: Solved with 1 push (unexpected for true 2-push)")
        lines.append("- **Model 2-Push**: Solved with exactly 2 pushes (optimal)")
        lines.append("- **Model 2+ Push**: Solved with more than 2 pushes")
        lines.append("- **Failed**: Did not find a solution\n")

        lines.append("| Model | 1-Push | 2-Push | 2+ Push | Failed | Total |")
        lines.append("|-------|--------|--------|---------|--------|-------|")

        def pct(val, total):
            return f"{val} ({val/total*100:.0f}%)" if total > 0 else "0"

        for model_name, matrix in confusion_matrices.items():
            oracle_2_total = (matrix['oracle_2_model_1'] + matrix['oracle_2_model_2'] +
                             matrix['oracle_2_model_2plus'] + matrix['oracle_2_failed'])
            lines.append(f"| {model_name} | {pct(matrix['oracle_2_model_1'], oracle_2_total)} | "
                        f"{pct(matrix['oracle_2_model_2'], oracle_2_total)} | "
                        f"{pct(matrix['oracle_2_model_2plus'], oracle_2_total)} | "
                        f"{pct(matrix['oracle_2_failed'], oracle_2_total)} | {oracle_2_total} |")
        lines.append("")


def _report_failure_reasons(lines: List[str], model_stats: List['ModelStats']):
    """Generate failure reasons section."""
    lines.append("## Failure Reasons\n")
    lines.append("Breakdown of why models failed to find solutions. Common reasons include: ")
    lines.append("timeout (search exceeded time limit), no valid primitives (no feasible push actions found), ")
    lines.append("and search exhausted (all candidates evaluated without finding a solution).\n")
    for stats in model_stats:
        if stats.failure_reasons:
            lines.append(f"### {stats.name}\n")
            lines.append("| Reason | Count |")
            lines.append("|--------|-------|")
            for reason, count in sorted(stats.failure_reasons.items(), key=lambda x: -x[1]):
                lines.append(f"| {reason} | {count} |")
            lines.append("")


def generate_markdown_report(
    model_stats: List[ModelStats],
    config: EvalConfig,
    depth_counts: Dict[str, int],
    output_path: str,
    confusion_matrices: Optional[Dict[str, Dict[str, int]]] = None,
    hybrid_stats: Optional[Dict[str, HybridStats]] = None,
    collision_efficiency: Optional[Dict[str, Dict[str, Dict[str, Any]]]] = None,
    difficulty_stratification: Optional[Dict[str, Dict[str, Dict[str, Any]]]] = None,
    hybrid_stats_by_difficulty: Optional[Dict[str, Dict[str, Dict[str, int]]]] = None,
    ra_at_k_stats: Optional[Dict[str, Dict[Optional[int], RAatKStats]]] = None,
    random_baselines: Optional[Dict[str, float]] = None,
    success_at_budget: Optional[Dict[str, Dict[int, Dict[str, Any]]]] = None,
    success_at_time: Optional[Dict[str, Dict[float, Dict[str, Any]]]] = None,
):
    """Generate a markdown report with comparison tables (2-push only)."""
    lines = []
    lines.append("# 2-Push Evaluation Results\n")
    lines.append(f"Generated from evaluation config.\n")

    # Dataset overview
    lines.append("## Dataset Overview\n")
    lines.append("Problems requiring exactly 2 sequential push actions to clear a path to the goal region, ")
    lines.append("as determined by an exhaustive oracle search. Evaluates model ability to solve multi-step ")
    lines.append("manipulation problems where object interactions must be planned in sequence.\n")
    count_2push = depth_counts.get('depth_2', 0)
    lines.append(f"Total 2-push problems evaluated: **{count_2push}**\n")

    # Summary table
    _report_summary_table(lines, model_stats)

    # Success@Budget
    if success_at_budget:
        _report_success_at_budget(lines, success_at_budget, config)

    # Success@Time
    if success_at_time:
        _report_success_at_time(lines, success_at_time, config)

    # ML Grounding
    has_ml_grounding_data = any(stats.depth_2.ml_aligned_counts for stats in model_stats)
    if has_ml_grounding_data:
        _report_ml_grounding(lines, model_stats)

    # RA@K
    if ra_at_k_stats:
        _report_ra_at_k(lines, ra_at_k_stats, config)

    # Hybrid Decomposition
    has_phase_data = hybrid_stats and any(
        hs.solved_by_learned > 0 or hs.solved_by_fallback > 0
        for hs in hybrid_stats.values()
    )
    if has_phase_data:
        _report_hybrid_decomposition(lines, hybrid_stats, hybrid_stats_by_difficulty)

    # Collision Stratification
    if collision_efficiency:
        _report_collision_stratification(lines, collision_efficiency)

    # Difficulty Stratification
    if difficulty_stratification:
        _report_difficulty_stratification(lines, difficulty_stratification)

    # Legacy tables
    _report_legacy_tables(lines, model_stats, confusion_matrices)

    # Failure reasons
    _report_failure_reasons(lines, model_stats)

    # Write to file
    with open(output_path, 'w') as f:
        f.write('\n'.join(lines))

    print(f"Saved: {output_path}")


# =============================================================================
# Main
# =============================================================================

def get_env_key_pairs(data: Dict[str, Dict[str, RegionResult]]) -> set:
    """Get set of (env, key) tuples from data."""
    return {(env, key) for env in data for key in data[env]}


def filter_to_intersection(
    all_data: Dict[str, Dict[str, Dict[str, RegionResult]]],
    reference_data: Dict[str, Dict[str, RegionResult]],
    require_reference_success: bool = True,
) -> Tuple[Dict[str, Dict[str, Dict[str, RegionResult]]], set]:
    """
    Filter all model data to only include triplets present in ALL models
    AND in the reference (for categorization).

    Args:
        all_data: {model_name: {env: {key: RegionResult}}}
        reference_data: {env: {key: RegionResult}}
        require_reference_success: If True, only include pairs where reference succeeded

    Returns filtered data and the intersection set.
    """
    # Get keys from each model
    all_keys = [get_env_key_pairs(data) for data in all_data.values()]

    # Also require presence in reference (for categorization)
    reference_keys = get_env_key_pairs(reference_data)
    all_keys.append(reference_keys)

    # Find intersection
    if not all_keys:
        return {}, set()

    intersection = all_keys[0]
    for keys in all_keys[1:]:
        intersection = intersection & keys

    # Filter to only pairs where reference succeeded (solvable problems)
    if require_reference_success:
        intersection = {
            (env, key) for env, key in intersection
            if reference_data[env][key].success
        }

    # Filter each model's data to intersection
    filtered = {}
    for name, data in all_data.items():
        filtered[name] = {}
        for env, key in intersection:
            if env not in filtered[name]:
                filtered[name][env] = {}
            filtered[name][env][key] = data[env][key]

    return filtered, intersection


def main():
    parser = argparse.ArgumentParser(description="2-Push Evaluation Script")
    parser.add_argument("--config", type=str, required=True,
                       help="Path to YAML config file")
    parser.add_argument("--output-dir", type=str, default=None,
                       help="Override output directory from config")
    parser.add_argument("--no-show", action="store_true",
                       help="Don't show plots interactively")

    args = parser.parse_args()

    # Load config from YAML
    config = EvalConfig.from_yaml(args.config)

    # Override output dir if specified
    if args.output_dir:
        config.output_dir = args.output_dir

    # Validate config
    if config.reference is None and not config.references:
        raise ValueError("Config must specify a 'reference' model for chain depth categorization")

    # Create output directory
    Path(config.output_dir).mkdir(parents=True, exist_ok=True)

    # Load all reference data (for consistency analysis with multiple oracles)
    reference_data_list: List[Dict[str, Dict[str, RegionResult]]] = []
    reference_names: List[str] = []

    print(f"\n{'='*60}")
    print(f"  LOADING REFERENCE/ORACLE DATA")
    print(f"{'='*60}")

    for ref_config in config.references:
        print(f"Loading reference: {ref_config.name}...")
        data, failures = load_pickle_data(f"{ref_config.dir}/**/*.pkl")
        print(f"  Loaded {sum(len(v) for v in data.values())} triplets")
        reference_data_list.append(data)
        reference_names.append(ref_config.name)

    # Use first reference as primary (for backwards compatibility)
    reference_data = reference_data_list[0] if reference_data_list else {}
    reference_failures = {}

    # Run multi-reference consistency analysis if we have multiple oracles
    consistency_data = None
    consistency_categories = None
    consistency_categories_2push = None
    if len(reference_data_list) > 1:
        print(f"\n{'='*60}")
        print(f"  MULTI-REFERENCE CONSISTENCY ANALYSIS")
        print(f"{'='*60}")
        print(f"  Analyzing {len(reference_data_list)} oracle runs for difficulty consistency...")

        consistency_data, common_triplets = compute_multi_reference_consistency(
            reference_data_list, reference_names
        )
        print(f"  Common triplets across all oracles: {len(common_triplets)}")

        # Categorize by consistency (all depths)
        consistency_categories, thresholds = categorize_by_consistency(consistency_data)
        consistency_stats = compute_consistency_stats(
            consistency_data, consistency_categories, thresholds, len(reference_data_list)
        )
        print_consistency_report(consistency_stats, consistency_categories, "All Depths")

        # Categorize by consistency (2-push only)
        consistency_categories_2push, thresholds_2push = categorize_by_consistency(consistency_data, depth_filter=2)
        consistency_stats_2push = compute_consistency_stats(
            consistency_data, consistency_categories_2push, thresholds_2push, len(reference_data_list)
        )
        print_consistency_report(consistency_stats_2push, consistency_categories_2push, "2-Push Only")

        # Categorize by consistency (1-push only)
        consistency_categories_1push, thresholds_1push = categorize_by_consistency(consistency_data, depth_filter=1)
        consistency_stats_1push = compute_consistency_stats(
            consistency_data, consistency_categories_1push, thresholds_1push, len(reference_data_list)
        )
        print_consistency_report(consistency_stats_1push, consistency_categories_1push, "1-Push Only")

        # Save consistency report to markdown
        save_consistency_report_to_markdown(
            consistency_stats, consistency_stats_1push, consistency_stats_2push,
            consistency_categories, consistency_categories_1push, consistency_categories_2push,
            reference_names,
            output_path=f"{config.output_dir}/consistency_report.md"
        )

        # Generate consistency plots
        print(f"\n  Generating consistency plots...")
        plot_consistency_scatter(
            consistency_data, consistency_categories_2push, depth_filter=2,
            output_path=f"{config.output_dir}/consistency_scatter_2push.png"
        )
        plot_push_variance_histogram(
            consistency_data, depth_filter=2,
            output_path=f"{config.output_dir}/consistency_variance_2push.png"
        )
        plot_push_variance_histogram(
            consistency_data, depth_filter=1,
            output_path=f"{config.output_dir}/consistency_variance_1push.png"
        )

    # Build difficulty mapping from multi-reference consistency analysis
    # This uses median pushes across multiple oracles for robust difficulty categorization
    difficulty_mapping_2push = None
    if consistency_categories_2push is not None:
        difficulty_mapping_2push = build_difficulty_mapping(consistency_categories_2push)
        print(f"  Built difficulty mapping from {len(difficulty_mapping_2push)} triplets (2-push)")

    # Load all models (baselines + learned) - references are only for consistency analysis
    all_model_data: Dict[str, Dict[str, Dict[str, RegionResult]]] = {}
    all_model_failures: Dict[str, Dict[str, int]] = {}

    for baseline in config.baselines:
        print(f"Loading baseline: {baseline.name}...")
        data, failures = load_pickle_data(f"{baseline.dir}/**/*.pkl")
        print(f"  Loaded {sum(len(v) for v in data.values())} triplets")
        all_model_data[baseline.name] = data
        all_model_failures[baseline.name] = failures

    for model in config.learned:
        print(f"Loading learned model: {model.name}...")
        data, failures = load_pickle_data(f"{model.dir}/**/*.pkl")
        print(f"  Loaded {sum(len(v) for v in data.values())} triplets")
        all_model_data[model.name] = data
        all_model_failures[model.name] = failures

    # Find intersection across all models + reference
    print("\nComputing intersection of triplets across all models...")

    # Debug: Show triplet counts before intersection
    print("\n  === TRIPLET COUNTS BEFORE INTERSECTION ===")
    reference_keys = get_env_key_pairs(reference_data)
    reference_success_keys = {(env, key) for env, key in reference_keys
                              if reference_data[env][key].success}
    print(f"  Reference (all):     {len(reference_keys)} triplets")
    print(f"  Reference (success): {len(reference_success_keys)} triplets")

    for name, data in all_model_data.items():
        model_keys = get_env_key_pairs(data)
        model_success_keys = {(env, key) for env, key in model_keys
                              if data[env][key].success}
        # How many of this model's triplets overlap with reference success?
        overlap_with_ref = model_keys & reference_success_keys
        print(f"  {name}:")
        print(f"    Total triplets: {len(model_keys)}")
        print(f"    Successful:     {len(model_success_keys)}")
        print(f"    Overlap w/ ref: {len(overlap_with_ref)}")

    filtered_data, intersection = filter_to_intersection(all_model_data, reference_data)
    print(f"\n  === FINAL INTERSECTION ===")
    print(f"  Intersection size: {len(intersection)} triplets")

    # Show what was lost
    if len(reference_success_keys) > len(intersection):
        lost = len(reference_success_keys) - len(intersection)
        pct_lost = lost / len(reference_success_keys) * 100
        print(f"  Lost from reference: {lost} triplets ({pct_lost:.1f}%)")
        print(f"  (These are triplets where oracle succeeded but not all models have matching triplets)")

    # Count by chain depth category
    depth_counts = {'depth_1': 0, 'depth_2': 0}
    for env, key in intersection:
        if env in reference_data and key in reference_data[env]:
            category = categorize_problem(reference_data[env][key])
            if category == '1push':
                depth_counts['depth_1'] += 1
            elif category == '2push':
                depth_counts['depth_2'] += 1
    print(f"  By chain depth: 1-Push={depth_counts['depth_1']}, 2-Push={depth_counts['depth_2']}")

    # Compute stats for each model (using filtered data)
    all_stats: List[ModelStats] = []
    time_data = {}
    push_data = {}
    # Learned-only (ML-only phase) data for hybrid models
    time_data_learned_only = {}
    push_data_learned_only = {}

    # Get set of learned model names for phase tracking
    learned_model_names_set = {m.name for m in config.learned}

    for name in filtered_data:
        model_stats = compute_stats(
            filtered_data[name], reference_data,
            name, all_model_failures.get(name, {})
        )
        all_stats.append(model_stats)

        # Compute time-based success
        time_data[name] = compute_time_based_success(
            filtered_data[name], reference_data, config
        )

        # Compute push-based success
        push_data[name] = compute_push_based_success(
            filtered_data[name], reference_data, config
        )

        # Compute learned-only (ML-only phase) data for hybrid models
        if name in learned_model_names_set:
            time_data_learned_only[name] = compute_time_based_success(
                filtered_data[name], reference_data, config, learned_only=True
            )
            push_data_learned_only[name] = compute_push_based_success(
                filtered_data[name], reference_data, config, learned_only=True
            )
            # Debug: show ML-only success rate
            lo_2push = push_data_learned_only[name].get('2push', {})
            lo_rates = lo_2push.get('rates', [])
            lo_final = lo_rates[-1] if lo_rates else 0
            total_2push = push_data[name].get('2push', {})
            total_rates = total_2push.get('rates', [])
            total_final = total_rates[-1] if total_rates else 0

    # Compute collision-based success stats
    collision_stats = {}
    for name in filtered_data:
        collision_stats[name] = compute_collision_success_stats(
            filtered_data[name], reference_data
        )

    # Compute confusion matrices (oracle chain depth vs model chain depth)
    confusion_matrices = {}
    for name in filtered_data:
        confusion_matrices[name] = compute_chain_depth_confusion_matrix(
            filtered_data[name], reference_data
        )

    # Compute hybrid stats (learned vs fallback decomposition) - 2-push only
    # Only for learned models (not reference/baselines which don't use hybrid approach)
    learned_model_names = {m.name for m in config.learned}
    hybrid_stats_2push = {}
    hybrid_stats_by_difficulty = {}
    for name in filtered_data:
        if name in learned_model_names:
            hybrid_stats_2push[name] = compute_hybrid_stats(
                filtered_data[name], reference_data, depth_filter=2
            )
            hybrid_stats_by_difficulty[name] = compute_hybrid_stats_by_difficulty(
                filtered_data[name], reference_data, depth_filter=2,
                difficulty_mapping=difficulty_mapping_2push
            )

    # Compute collision bucket efficiency
    collision_efficiency = {}
    for name in filtered_data:
        collision_efficiency[name] = compute_collision_bucket_efficiency(
            filtered_data[name], reference_data
        )

    # Compute difficulty stratification (based on multi-reference median pushes if available) - 2-push only
    difficulty_stratification = {}
    for name in filtered_data:
        difficulty_stratification[name] = compute_difficulty_stratification(
            filtered_data[name], reference_data, depth_filter=2,
            difficulty_mapping=difficulty_mapping_2push
        )

    # Compute time and push based success by difficulty (for plots)
    time_data_by_difficulty = {}
    push_data_by_difficulty = {}
    time_data_by_difficulty_learned_only = {}
    push_data_by_difficulty_learned_only = {}
    for name in filtered_data:
        time_data_by_difficulty[name] = compute_time_based_success_by_difficulty(
            filtered_data[name], reference_data, config, depth_filter=2,
            difficulty_mapping=difficulty_mapping_2push
        )
        push_data_by_difficulty[name] = compute_push_based_success_by_difficulty(
            filtered_data[name], reference_data, config, depth_filter=2,
            difficulty_mapping=difficulty_mapping_2push
        )
        # Compute learned-only by difficulty for learned models
        if name in learned_model_names_set:
            time_data_by_difficulty_learned_only[name] = compute_time_based_success_by_difficulty(
                filtered_data[name], reference_data, config, depth_filter=2,
                difficulty_mapping=difficulty_mapping_2push, learned_only=True
            )
            push_data_by_difficulty_learned_only[name] = compute_push_based_success_by_difficulty(
                filtered_data[name], reference_data, config, depth_filter=2,
                difficulty_mapping=difficulty_mapping_2push, learned_only=True
            )

    # Compute ReachableAttachment@K for learned models only
    learned_model_names = {m.name for m in config.learned}
    ra_at_k_stats: Dict[str, Dict[Optional[int], RAatKStats]] = {}
    random_baselines: Dict[str, float] = {}

    for name in filtered_data:
        if name in learned_model_names:
            ra_at_k_stats[name] = {}
            for k in config.ra_at_k_values:
                ra_at_k_stats[name][k] = compute_ra_at_k(
                    filtered_data[name], reference_data, k=k, depth_filter=2
                )
            # Compute random baseline for this model
            random_baselines[name] = compute_random_baseline(
                filtered_data[name], reference_data, depth_filter=2
            )

    # Compute Success@B (constant-compute comparison) for all models
    success_at_budget: Dict[str, Dict[int, Dict[str, Any]]] = {}
    for name in filtered_data:
        success_at_budget[name] = compute_success_at_budget(
            filtered_data[name], reference_data,
            budgets=config.success_at_budget_values,
            depth_filter=2
        )

    # Compute Success@T (constant-time comparison) for all models
    success_at_time: Dict[str, Dict[float, Dict[str, Any]]] = {}
    for name in filtered_data:
        success_at_time[name] = compute_success_at_time_budget(
            filtered_data[name], reference_data,
            time_budgets=config.success_at_time_values,
            depth_filter=2
        )

    # Compute Success@B and Success@T stratified by difficulty
    success_at_budget_by_diff: Dict[str, Dict[str, Dict[int, Dict[str, Any]]]] = {}
    success_at_time_by_diff: Dict[str, Dict[str, Dict[float, Dict[str, Any]]]] = {}
    for name in filtered_data:
        success_at_budget_by_diff[name] = compute_success_at_budget_by_difficulty(
            filtered_data[name], reference_data,
            budgets=config.success_at_budget_values,
            depth_filter=2,
            difficulty_mapping=difficulty_mapping_2push
        )
        success_at_time_by_diff[name] = compute_success_at_time_budget_by_difficulty(
            filtered_data[name], reference_data,
            time_budgets=config.success_at_time_values,
            depth_filter=2,
            difficulty_mapping=difficulty_mapping_2push
        )

    # Print summary
    print_summary(all_stats)

    # Print collision-based success rates
    print("\n" + "=" * 80)
    print("SUCCESS RATE BY COLLISION TYPE (based on oracle solution)")
    print("=" * 80)
    collision_types = ['none', 'wall_only', 'movable_only', 'both']
    collision_labels = {'none': 'No Collision', 'wall_only': 'Wall Only',
                        'movable_only': 'Movable Only', 'both': 'Both'}

    # Print oracle collision distribution (use first model's totals - same for all)
    if collision_stats:
        first_model = list(collision_stats.keys())[0]
        total_problems = sum(collision_stats[first_model][ct]['total'] for ct in collision_types)
        print(f"\nOracle Collision Distribution (N={total_problems}):")
        for ct in collision_types:
            count = collision_stats[first_model][ct]['total']
            pct = count / total_problems * 100 if total_problems > 0 else 0
            print(f"  {collision_labels[ct]:15s}: {count:3d} ({pct:5.1f}%)")

        # Print success rates per model
        for name in collision_stats:
            print(f"\n{name}:")
            for ct in collision_types:
                stats = collision_stats[name][ct]
                rate = stats['successes'] / stats['total'] if stats['total'] > 0 else 0.0
                print(f"  {collision_labels[ct]:15s}: {stats['successes']:3d}/{stats['total']:3d} = {rate:.1%}")

    # Print difficulty stratification
    print("\n" + "=" * 80)
    if difficulty_mapping_2push is not None:
        print("DIFFICULTY STRATIFICATION (based on multi-reference median pushes)")
    else:
        print("DIFFICULTY STRATIFICATION (based on single-reference pushes)")
    print("=" * 80)
    difficulty_levels = ['easy', 'medium', 'hard']
    difficulty_labels_print = {'easy': 'Easy', 'medium': 'Medium', 'hard': 'Hard'}

    if difficulty_stratification:
        # Print oracle push ranges (same for all models)
        first_model = list(difficulty_stratification.keys())[0]
        print("\nOracle Push Ranges:")
        for diff in difficulty_levels:
            r = difficulty_stratification[first_model][diff]['oracle_push_range']
            n = difficulty_stratification[first_model][diff]['total']
            print(f"  {difficulty_labels_print[diff]:8s}: {r[0]:3d} – {r[1]:3d} pushes  (N={n})")

        # Print success rates per model
        for name in difficulty_stratification:
            print(f"\n{name}:")
            for diff in difficulty_levels:
                stats = difficulty_stratification[name][diff]
                rate = stats['successes'] / stats['total'] if stats['total'] > 0 else 0.0
                print(f"  {difficulty_labels_print[diff]:8s}: {stats['successes']:3d}/{stats['total']:3d} = {rate:.1%}")

    # Print confusion matrices (2-push only)
    print("\n" + "=" * 80)
    print("2-PUSH MODEL RESULT BREAKDOWN")
    print("=" * 80)
    for name, matrix in confusion_matrices.items():
        oracle_2_total = (matrix['oracle_2_model_1'] + matrix['oracle_2_model_2'] +
                         matrix['oracle_2_model_2plus'] + matrix['oracle_2_failed'])

        print(f"\n{name} (n={oracle_2_total}):")
        print(f"  Model 1-push: {matrix['oracle_2_model_1']:3d} ({matrix['oracle_2_model_1']/oracle_2_total*100:5.1f}%)" if oracle_2_total > 0 else "  Model 1-push: N/A")
        print(f"  Model 2-push: {matrix['oracle_2_model_2']:3d} ({matrix['oracle_2_model_2']/oracle_2_total*100:5.1f}%)" if oracle_2_total > 0 else "  Model 2-push: N/A")
        print(f"  Model 2+:     {matrix['oracle_2_model_2plus']:3d} ({matrix['oracle_2_model_2plus']/oracle_2_total*100:5.1f}%)" if oracle_2_total > 0 else "  Model 2+: N/A")
        print(f"  Failed:       {matrix['oracle_2_failed']:3d} ({matrix['oracle_2_failed']/oracle_2_total*100:5.1f}%)" if oracle_2_total > 0 else "  Failed: N/A")

    # Print ML grounding stats (2-push only) - only if grounding data exists
    has_grounding_data = any(
        stats.depth_2.ml_aligned_counts for stats in all_stats
    )
    if has_grounding_data:
        print("\n" + "=" * 80)
        print("ML PREDICTION GROUNDING (2-Push Problems)")
        print("=" * 80)
        print("Shows % of aligned ML primitives that point to reachable push edges")

        for stats in all_stats:
            cat_stats = stats.depth_2
            if cat_stats.ml_aligned_counts:
                n_samples = len(cat_stats.ml_aligned_counts)
                print(f"\n{stats.name} (n={n_samples}):")
                print(f"  Grounding:  {cat_stats.mean_ml_aligned_reachable_ratio:.1%} of aligned primitives are reachable")
                print(f"  Totals:     {cat_stats.total_ml_aligned_reachable}/{cat_stats.total_ml_aligned} aligned primitives")

    # Print hybrid decomposition stats (2-push only) - only if phase data exists
    has_phase_data = any(
        hs.solved_by_learned > 0 or hs.solved_by_fallback > 0
        for hs in hybrid_stats_2push.values()
    )
    if has_phase_data:
        print("\n" + "=" * 80)
        print("HYBRID DECOMPOSITION (2-Push Problems)")
        print("=" * 80)
        print("Phase tracking: solved_in_phase == 'ML-only' → LEARNED, 'primitives' → FALLBACK")

        for name, hs in hybrid_stats_2push.items():
            if hs.total == 0:
                continue
            print(f"\n{name} (n={hs.total}):")
            print(f"  Solved by LEARNED:  {hs.solved_by_learned:3d} ({hs.learned_rate:.1%})")
            print(f"  Solved by FALLBACK: {hs.solved_by_fallback:3d} ({hs.fallback_rate:.1%})")
            print(f"  Failed:             {hs.failed:3d} ({(1-hs.success_rate):.1%})")

            if hs.learned_pushes:
                l_iqr = hs.learned_pushes_iqr
                print(f"  Learned checks:     median={hs.learned_median_pushes:.0f} [{l_iqr[0]:.0f}, {l_iqr[1]:.0f}]")
            if hs.fallback_pushes:
                f_iqr = hs.fallback_pushes_iqr
                print(f"  Fallback checks:    median={hs.fallback_median_pushes:.0f} [{f_iqr[0]:.0f}, {f_iqr[1]:.0f}]")
            if hs.checks_before_fallback:
                bf_iqr = hs.checks_before_fallback_iqr
                print(f"  Checks before FB:   median={hs.median_checks_before_fallback:.0f} [{bf_iqr[0]:.0f}, {bf_iqr[1]:.0f}]")

        # Print hybrid stats by difficulty
        if hybrid_stats_by_difficulty:
            print("\n  By Difficulty:")
            difficulty_levels = ['easy', 'medium', 'hard']
            difficulty_labels = {'easy': 'Easy', 'medium': 'Medium', 'hard': 'Hard'}

            for name in hybrid_stats_by_difficulty:
                print(f"\n  {name}:")
                for diff in difficulty_levels:
                    stats = hybrid_stats_by_difficulty[name][diff]
                    n = stats['total']
                    if n > 0:
                        learned_pct = stats['learned'] / n * 100
                        fallback_pct = stats['fallback'] / n * 100
                        failed_pct = stats['failed'] / n * 100
                        print(f"    {difficulty_labels[diff]:8s} (N={n:2d}): Learned={learned_pct:5.1f}% ({stats['learned']:2d}), "
                              f"Fallback={fallback_pct:5.1f}% ({stats['fallback']:2d}), Failed={failed_pct:5.1f}% ({stats['failed']:2d})")
                    else:
                        print(f"    {difficulty_labels[diff]:8s} (N= 0): -")

    # Print ReachableAttachment@K stats (learned models only)
    if ra_at_k_stats:
        print("\n" + "=" * 80)
        print("REACHABLE ATTACHMENT @ K (2-Push Problems)")
        print("=" * 80)
        print("Fraction of top-K ML-ranked primitives with reachable push attachments")
        print("(Higher = ML predictions are better grounded in physical reachability)\n")

        for name in ra_at_k_stats:
            print(f"{name}:")
            for k in config.ra_at_k_values:
                ra = ra_at_k_stats[name][k]
                k_str = f"@{k}" if k is not None else "@All"
                if ra.n_instances > 0:
                    print(f"  RA{k_str:>5}: macro={ra.macro:.1%}, micro={ra.micro:.1%} "
                          f"({ra.total_reachable}/{ra.total_considered}, n={ra.n_instances})")
                else:
                    print(f"  RA{k_str:>5}: no data")
            print()

    # Print Success@B (constant-compute comparison)
    if success_at_budget:
        print("\n" + "=" * 80)
        print("SUCCESS @ BUDGET (2-Push Problems)")
        print("=" * 80)
        print("Success rate at fixed verification budget (constant-compute comparison)")
        print("Budget = max number of simulation-verified push evaluations\n")

        # Print header
        budget_strs = [f"@{b}" for b in config.success_at_budget_values]
        header = f"{'Model':<30} | " + " | ".join(f"{s:>8}" for s in budget_strs)
        print(header)
        print("-" * len(header))

        for name in success_at_budget:
            row = f"{name:<30} |"
            for b in config.success_at_budget_values:
                stats = success_at_budget[name][b]
                row += f" {stats['rate']:>7.1%} |"
            print(row)

        # Print N (same for all models at each budget)
        first_model = list(success_at_budget.keys())[0]
        n_total = success_at_budget[first_model][config.success_at_budget_values[0]]['total']
        print(f"\n(N={n_total} 2-push problems)")

    # Print Success@T (constant-time comparison)
    if success_at_time:
        print("\n" + "=" * 80)
        print("SUCCESS @ TIME (2-Push Problems)")
        print("=" * 80)
        print("Success rate at fixed time budget (constant-time comparison)")
        print("Time = max search time in seconds\n")

        # Format time values as seconds for display
        def format_time(t_ms: float) -> str:
            if t_ms >= 1000:
                return f"@{t_ms/1000:.0f}s"
            return f"@{t_ms:.0f}ms"

        # Print header
        time_strs = [format_time(t) for t in config.success_at_time_values]
        header = f"{'Model':<30} | " + " | ".join(f"{s:>8}" for s in time_strs)
        print(header)
        print("-" * len(header))

        for name in success_at_time:
            row = f"{name:<30} |"
            for t in config.success_at_time_values:
                stats = success_at_time[name][t]
                row += f" {stats['rate']:>7.1%} |"
            print(row)

        # Print N (same for all models at each time budget)
        first_model = list(success_at_time.keys())[0]
        n_total = success_at_time[first_model][config.success_at_time_values[0]]['total']
        print(f"\n(N={n_total} 2-push problems)")

    # Print Success@B stratified by difficulty
    if success_at_budget_by_diff:
        difficulty_labels = {'easy': 'Easy', 'medium': 'Medium', 'hard': 'Hard'}
        print("\n" + "=" * 80)
        print("SUCCESS @ BUDGET BY DIFFICULTY (2-Push Problems)")
        print("=" * 80)
        print("Success rate at fixed verification budget, stratified by problem difficulty")
        print("(Difficulty = oracle push count terciles: Easy=lowest 33%, Hard=highest 33%)\n")

        budget_strs = [f"@{b}" for b in config.success_at_budget_values]

        for diff in ['easy', 'medium', 'hard']:
            first_model = list(success_at_budget_by_diff.keys())[0]
            n_diff = success_at_budget_by_diff[first_model][diff][config.success_at_budget_values[0]]['total']
            print(f"\n{difficulty_labels[diff]} Problems (N={n_diff}):")
            header = f"{'Model':<30} | " + " | ".join(f"{s:>8}" for s in budget_strs)
            print(header)
            print("-" * len(header))

            for name in success_at_budget_by_diff:
                row = f"{name:<30} |"
                for b in config.success_at_budget_values:
                    stats = success_at_budget_by_diff[name][diff][b]
                    row += f" {stats['rate']:>7.1%} |"
                print(row)

    # Print Success@T stratified by difficulty
    if success_at_time_by_diff:
        difficulty_labels = {'easy': 'Easy', 'medium': 'Medium', 'hard': 'Hard'}
        print("\n" + "=" * 80)
        print("SUCCESS @ TIME BY DIFFICULTY (2-Push Problems)")
        print("=" * 80)
        print("Success rate at fixed time budget, stratified by problem difficulty")
        print("(Difficulty = oracle push count terciles: Easy=lowest 33%, Hard=highest 33%)\n")

        def format_time(t_ms: float) -> str:
            if t_ms >= 1000:
                return f"@{t_ms/1000:.0f}s"
            return f"@{t_ms:.0f}ms"

        time_strs = [format_time(t) for t in config.success_at_time_values]

        for diff in ['easy', 'medium', 'hard']:
            first_model = list(success_at_time_by_diff.keys())[0]
            n_diff = success_at_time_by_diff[first_model][diff][config.success_at_time_values[0]]['total']
            print(f"\n{difficulty_labels[diff]} Problems (N={n_diff}):")
            header = f"{'Model':<30} | " + " | ".join(f"{s:>8}" for s in time_strs)
            print(header)
            print("-" * len(header))

            for name in success_at_time_by_diff:
                row = f"{name:<30} |"
                for t in config.success_at_time_values:
                    stats = success_at_time_by_diff[name][diff][t]
                    row += f" {stats['rate']:>7.1%} |"
                print(row)

    # Generate plots
    print("\nGenerating plots...")

    plot_success_rates(
        all_stats,
        config,
        f"{config.output_dir}/success_rates.png"
    )

    if all_stats:
        plot_pushes_boxplot(
            all_stats,
            config,
            f"{config.output_dir}/pushes_boxplot.png"
        )

        plot_time_boxplot(
            all_stats,
            config,
            f"{config.output_dir}/time_boxplot.png"
        )

    if time_data:
        plot_time_vs_success(
            time_data,
            config,
            f"{config.output_dir}/time_vs_success.png",
            learned_only_data=time_data_learned_only if time_data_learned_only else None,
        )

    if push_data:
        plot_pushes_vs_success(
            push_data,
            config,
            f"{config.output_dir}/pushes_vs_success.png",
            learned_only_data=push_data_learned_only if push_data_learned_only else None,
        )

    if all_stats:
        plot_interactions(
            all_stats,
            config,
            f"{config.output_dir}/interactions.png"
        )

    if collision_stats:
        plot_collision_success_rates(
            collision_stats,
            config,
            f"{config.output_dir}/collision_success_rates.png"
        )

    # Plot confusion matrices
    if confusion_matrices:
        plot_confusion_matrix(
            confusion_matrices,
            config,
            f"{config.output_dir}/confusion_matrix.png"
        )

    # Plot hybrid decomposition
    if hybrid_stats_2push:
        plot_hybrid_decomposition(
            hybrid_stats_2push,
            config,
            f"{config.output_dir}/hybrid_decomposition.png"
        )

    # Plot collision bucket efficiency
    if collision_efficiency:
        plot_collision_bucket_efficiency(
            collision_efficiency,
            config,
            f"{config.output_dir}/collision_bucket_efficiency.png"
        )

    # Plot success vs time/pushes by difficulty
    if time_data_by_difficulty:
        plot_time_vs_success_by_difficulty(
            time_data_by_difficulty,
            config,
            f"{config.output_dir}/time_vs_success_by_difficulty.png",
            learned_only_data=time_data_by_difficulty_learned_only,
        )

    if push_data_by_difficulty:
        plot_pushes_vs_success_by_difficulty(
            push_data_by_difficulty,
            config,
            f"{config.output_dir}/pushes_vs_success_by_difficulty.png",
            learned_only_data=push_data_by_difficulty_learned_only,
        )

    # Generate markdown report
    generate_markdown_report(
        all_stats,
        config,
        depth_counts,
        f"{config.output_dir}/results.md",
        confusion_matrices=confusion_matrices,
        hybrid_stats=hybrid_stats_2push,
        collision_efficiency=collision_efficiency,
        difficulty_stratification=difficulty_stratification,
        hybrid_stats_by_difficulty=hybrid_stats_by_difficulty,
        ra_at_k_stats=ra_at_k_stats,
        random_baselines=random_baselines,
        success_at_budget=success_at_budget,
        success_at_time=success_at_time,
    )

    print(f"\nPlots saved to: {config.output_dir}")

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
