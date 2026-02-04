#!/usr/bin/env python3
"""
1-Push Evaluation Script

Compares multiple learned models against search (oracle) and non-learned baselines.
Generates comparison plots for success rates, pushes, and time.

Usage:
    python eval_1push.py --config eval_config.yaml
    python eval_1push.py --config eval_config.yaml --output-dir ./my_plots

Example config (eval_config.yaml):

    # Reference determines easy/medium/hard categorization
    reference:
      name: "Search (Oracle)"
      dir: /path/to/search/results

    # Non-learned baselines
    baselines:
      - name: "No Heuristic"
        dir: /path/to/no_heuristic/results

    # Learned models to compare
    learned:
      - name: "AdaLN"
        dir: /path/to/adaln/results
      - name: "CrossAttn"
        dir: /path/to/crossattn/results

    # Optional settings
    settings:
      exclude_easy: true
      easy_threshold: 0.75
      hard_threshold: 0.25
      output_dir: ./eval_plots
      time_cutoff_max: 6000
      time_step: 100
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

# Set up nicer plot style (paper-ready font sizes)
plt.style.use('seaborn-v0_8-whitegrid')
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.size'] = 16
mpl.rcParams['axes.titlesize'] = 20
mpl.rcParams['axes.titleweight'] = 'bold'
mpl.rcParams['axes.labelsize'] = 18
mpl.rcParams['xtick.labelsize'] = 14
mpl.rcParams['ytick.labelsize'] = 14
mpl.rcParams['legend.fontsize'] = 14
mpl.rcParams['figure.facecolor'] = 'white'
mpl.rcParams['axes.facecolor'] = 'white'
mpl.rcParams['axes.edgecolor'] = '#333333'
mpl.rcParams['axes.linewidth'] = 1.0
mpl.rcParams['grid.alpha'] = 0.3
mpl.rcParams['axes.spines.top'] = False
mpl.rcParams['axes.spines.right'] = False


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class ModelConfig:
    """Configuration for a single model/baseline."""
    name: str
    dir: str
    color: Optional[str] = None


@dataclass
class EvalConfig:
    """Configuration for 1-push evaluation."""
    # References (multiple oracle runs with different seeds for consistency analysis)
    references: List[ModelConfig] = field(default_factory=list)

    # Legacy: single reference (for backwards compatibility)
    reference: ModelConfig = None

    # Baselines (non-learned)
    baselines: List[ModelConfig] = field(default_factory=list)

    # Learned models to compare
    learned: List[ModelConfig] = field(default_factory=list)

    # Filtering
    exclude_easy: bool = True

    # Plot settings
    output_dir: str = "./eval_plots"
    time_cutoff_max: int = 6000  # ms
    time_step: int = 100  # ms
    push_cutoff_max: int = 10  # max number of pushes
    push_step: int = 1  # step size for push cutoffs

    # ReachableAttachment@K thresholds (None = all ranked primitives)
    ra_at_k_values: List[Optional[int]] = field(default_factory=lambda: [10, 50, 100, None])

    # Success@B budget values (number of simulation-verified checks)
    success_at_budget_values: List[int] = field(default_factory=lambda: [5, 10, 20, 50])

    # Success@T time budget values (milliseconds)
    success_at_time_values: List[float] = field(default_factory=lambda: [1000, 3000, 6000, 20000])

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
                ))

        if 'settings' in data:
            settings = data['settings']
            config.exclude_easy = settings.get('exclude_easy', config.exclude_easy)
            config.output_dir = settings.get('output_dir', config.output_dir)
            config.time_cutoff_max = settings.get('time_cutoff_max', config.time_cutoff_max)
            config.time_step = settings.get('time_step', config.time_step)
            config.push_cutoff_max = settings.get('push_cutoff_max', config.push_cutoff_max)
            config.push_step = settings.get('push_step', config.push_step)
            # New settings
            if 'ra_at_k_values' in settings:
                config.ra_at_k_values = settings['ra_at_k_values']
            if 'success_at_budget_values' in settings:
                config.success_at_budget_values = settings['success_at_budget_values']
            if 'success_at_time_values' in settings:
                config.success_at_time_values = settings['success_at_time_values']

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
    chain_depth: int = 1  # Always 1 for 1-push problems
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
    exclude_easy: bool = True,
    reference_data: Optional[Dict[str, Dict[str, RegionResult]]] = None,
) -> Tuple[Dict[str, Dict[str, RegionResult]], Dict[str, int]]:
    """
    Load evaluation data from pickle files.

    Args:
        data_dir: Glob pattern for pickle files
        exclude_easy: Whether to exclude 'easy' environments
        reference_data: If provided, only include env+region+object triplets that exist in reference

    Returns:
        per_env_per_key: {xml_file_name: {region_label::object_id: RegionResult}}
        failure_reasons: {reason: count}
    """
    per_env_per_key: Dict[str, Dict[str, RegionResult]] = {}
    failure_reasons: Dict[str, int] = defaultdict(int)

    for file in glob(data_dir, recursive=True):
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

                if exclude_easy and "easy" in xml_file_name:
                    continue

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

                # Track failure reasons (after dedup to avoid over-counting)
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

                # Extract interaction types
                wall_collision = ep.get('any_wall_collision', False)
                movable_collisions = ep.get('unique_movable_collision_count', 0)

                # Extract phase tracking data
                phase_push_counts = alg_stats.get('phase_push_counts', None)
                solved_in_phase = alg_stats.get('solved_in_phase', '')

                # Extract ML grounding data
                aligned_primitives = alg_stats.get('aligned_primitives', [])
                reachable_edges = set(alg_stats.get('reachable_edges', []))

                # Compute ml_aligned_count and ml_aligned_reachable_count
                ml_aligned_count = len(aligned_primitives)
                ml_aligned_reachable_count = 0
                for prim in aligned_primitives:
                    edge_idx = prim.get('edge_idx')
                    if edge_idx is not None and edge_idx in reachable_edges:
                        ml_aligned_reachable_count += 1

                result = RegionResult(
                    success=solution_found and pushes > 0,
                    pushes=pushes,
                    solutions=solutions,
                    solutions_found=solutions_found,
                    ratio=solutions / pushes if pushes > 0 else 0.0,
                    time_taken=time_taken,
                    failure_reason=failure_reason,
                    xml_file=xml_file_name,
                    region=region_label,
                    object_id=object_id,
                    chain_depth=1,  # Always 1 for 1-push problems
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
class DifficultyStats:
    """Statistics for a difficulty category."""
    pushes: List[int] = field(default_factory=list)
    times: List[float] = field(default_factory=list)
    solutions: List[int] = field(default_factory=list)  # solutions_total (for stats)
    solutions_found: List[int] = field(default_factory=list)  # solutions_found (for distribution)
    successes: int = 0
    total: int = 0
    # Interaction tracking
    wall_collisions: int = 0  # count of successful runs with wall collisions
    movable_collisions_list: List[int] = field(default_factory=list)  # movable collision counts per success
    # ML grounding metrics
    ml_aligned_counts: List[int] = field(default_factory=list)  # Per-instance aligned primitive counts
    ml_aligned_reachable_counts: List[int] = field(default_factory=list)  # Per-instance reachable counts

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
    def median_time(self) -> float:
        return float(np.median(self.times)) if self.times else 0.0

    @property
    def mean_time(self) -> float:
        return float(np.mean(self.times)) if self.times else 0.0

    @property
    def pushes_iqr(self) -> Tuple[float, float]:
        """Return (25th percentile, 75th percentile) for pushes."""
        if not self.pushes:
            return (0.0, 0.0)
        return (float(np.percentile(self.pushes, 25)), float(np.percentile(self.pushes, 75)))

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
    def total_ml_aligned(self) -> int:
        """Total ML-aligned primitives across all instances."""
        return sum(self.ml_aligned_counts) if self.ml_aligned_counts else 0

    @property
    def total_ml_aligned_reachable(self) -> int:
        """Total reachable ML-aligned primitives across all instances."""
        return sum(self.ml_aligned_reachable_counts) if self.ml_aligned_reachable_counts else 0

    @property
    def mean_ml_aligned_reachable_ratio(self) -> float:
        """Mean grounding ratio (fraction of aligned primitives that are reachable)."""
        if not self.ml_aligned_counts:
            return 0.0
        ratios = []
        for aligned, reachable in zip(self.ml_aligned_counts, self.ml_aligned_reachable_counts):
            if aligned > 0:
                ratios.append(reachable / aligned)
        return float(np.mean(ratios)) if ratios else 0.0

    @property
    def micro_ml_aligned_reachable_ratio(self) -> float:
        """Micro-averaged grounding ratio (total reachable / total aligned)."""
        total = self.total_ml_aligned
        if total == 0:
            return 0.0
        return self.total_ml_aligned_reachable / total


# Backwards compatibility alias
CategoryStats = DifficultyStats


@dataclass
class ModelStats:
    """Statistics for a model across all difficulty categories."""
    name: str
    easy: DifficultyStats = field(default_factory=DifficultyStats)
    medium: DifficultyStats = field(default_factory=DifficultyStats)
    hard: DifficultyStats = field(default_factory=DifficultyStats)
    failure_reasons: Dict[str, int] = field(default_factory=dict)

    def get_category(self, name: str) -> CategoryStats:
        return getattr(self, name)

    @property
    def total_successes(self) -> int:
        return self.easy.successes + self.medium.successes + self.hard.successes

    @property
    def total_trials(self) -> int:
        return self.easy.total + self.medium.total + self.hard.total

    @property
    def overall_success_rate(self) -> float:
        return self.total_successes / self.total_trials if self.total_trials > 0 else 0.0


def compute_stats(
    model_data: Dict[str, Dict[str, RegionResult]],
    search_data: Dict[str, Dict[str, RegionResult]],
    model_name: str,
    failure_reasons: Optional[Dict[str, int]] = None,
    difficulty_mapping: Optional[Dict[Tuple[str, str], str]] = None,
) -> ModelStats:
    """
    Compute statistics for a model, categorized by difficulty.

    Args:
        model_data: Model results by env/key
        search_data: Reference/search results by env/key
        model_name: Name for the model
        failure_reasons: Optional failure reason counts
        difficulty_mapping: Optional pre-computed difficulty mapping {(env, key): 'easy'|'medium'|'hard'}
                           If None, difficulty is computed from percentiles of reference pushes.
    """
    stats = ModelStats(name=model_name)
    if failure_reasons:
        stats.failure_reasons = failure_reasons

    # Build difficulty mapping if not provided (data-driven percentiles)
    if difficulty_mapping is None:
        oracle_pushes = []
        problem_keys = []
        for env in search_data:
            if env not in model_data:
                continue
            for key in search_data[env]:
                if key not in model_data[env]:
                    continue
                ref_result = search_data[env][key]
                if ref_result.success:
                    oracle_pushes.append(ref_result.pushes)
                    problem_keys.append((env, key))

        if oracle_pushes:
            thresholds = compute_percentile_thresholds(oracle_pushes)
            difficulty_mapping = {}
            for (env, key), pushes in zip(problem_keys, oracle_pushes):
                difficulty_mapping[(env, key)] = assign_difficulty(pushes, thresholds)
        else:
            difficulty_mapping = {}

    # Compute stats using difficulty mapping
    for env, key, ref_result, model_result in iter_matched_triplets(model_data, search_data):
        difficulty = difficulty_mapping.get((env, key), 'medium')  # Default to medium
        category = stats.get_category(difficulty)

        category.total += 1

        if model_result.success:
            category.successes += 1
            category.pushes.append(model_result.pushes)
            category.times.append(model_result.time_taken)
            category.solutions.append(model_result.solutions)
            category.solutions_found.append(model_result.solutions_found)
            # Track interactions
            if model_result.wall_collision:
                category.wall_collisions += 1
            category.movable_collisions_list.append(model_result.movable_collisions)
            # Track ML grounding metrics
            if model_result.ml_aligned_count > 0:
                category.ml_aligned_counts.append(model_result.ml_aligned_count)
                category.ml_aligned_reachable_counts.append(model_result.ml_aligned_reachable_count)

    return stats


def compute_time_based_success(
    model_data: Dict[str, Dict[str, RegionResult]],
    search_data: Dict[str, Dict[str, RegionResult]],
    config: EvalConfig,
    difficulty_mapping: Optional[Dict[Tuple[str, str], str]] = None,
    learned_only: bool = False,
) -> Dict[str, Dict[str, Any]]:
    """
    Compute success rate as a function of time cutoff.

    Args:
        difficulty_mapping: Pre-computed difficulty mapping. If None, uses data-driven percentiles.
        learned_only: If True, only count successes from ML-only phase (no fallback).

    Returns:
        {category: {'cutoffs': [...], 'rates': [...], 'total': int}}
    """
    cutoffs = np.arange(0, config.time_cutoff_max + config.time_step, config.time_step)

    # Build difficulty mapping if not provided
    if difficulty_mapping is None:
        oracle_pushes = []
        problem_keys = []
        for env, key, ref_result, _ in iter_matched_triplets(model_data, search_data):
            oracle_pushes.append(ref_result.pushes)
            problem_keys.append((env, key))

        if oracle_pushes:
            thresholds = compute_percentile_thresholds(oracle_pushes)
            difficulty_mapping = {k: assign_difficulty(p, thresholds)
                                  for k, p in zip(problem_keys, oracle_pushes)}
        else:
            difficulty_mapping = {}

    # Collect times by category
    times_by_category: Dict[str, List[float]] = {'easy': [], 'medium': [], 'hard': []}
    totals_by_category: Dict[str, int] = {'easy': 0, 'medium': 0, 'hard': 0}

    for env, key, ref_result, model_result in iter_matched_triplets(model_data, search_data):
        cat = difficulty_mapping.get((env, key), 'medium')
        totals_by_category[cat] += 1

        # Check success based on learned_only flag
        if learned_only:
            if model_result.solved_by_learned:
                times_by_category[cat].append(model_result.time_taken)
        else:
            if model_result.success:
                times_by_category[cat].append(model_result.time_taken)

    # Compute rates at each cutoff
    result = {}
    for cat in ['easy', 'medium', 'hard']:
        times = np.array(times_by_category[cat])
        total = totals_by_category[cat]
        rates = []
        for cutoff in cutoffs:
            if total > 0:
                successes = np.sum(times <= cutoff) if len(times) > 0 else 0
                rates.append(successes / total)
            else:
                rates.append(0.0)
        result[cat] = {'cutoffs': cutoffs.tolist(), 'rates': rates, 'total': total}

    return result


def compute_push_based_success(
    model_data: Dict[str, Dict[str, RegionResult]],
    search_data: Dict[str, Dict[str, RegionResult]],
    config: EvalConfig,
    difficulty_mapping: Optional[Dict[Tuple[str, str], str]] = None,
    learned_only: bool = False,
) -> Dict[str, Dict[str, Any]]:
    """
    Compute success rate as a function of push count cutoff.

    Args:
        difficulty_mapping: Pre-computed difficulty mapping. If None, uses data-driven percentiles.
        learned_only: If True, only count successes from ML-only phase (no fallback).

    Returns:
        {category: {'cutoffs': [...], 'rates': [...], 'total': int}}
    """
    cutoffs = list(range(0, config.push_cutoff_max + 1, config.push_step))

    # Build difficulty mapping if not provided
    if difficulty_mapping is None:
        oracle_pushes = []
        problem_keys = []
        for env, key, ref_result, _ in iter_matched_triplets(model_data, search_data):
            oracle_pushes.append(ref_result.pushes)
            problem_keys.append((env, key))

        if oracle_pushes:
            thresholds = compute_percentile_thresholds(oracle_pushes)
            difficulty_mapping = {k: assign_difficulty(p, thresholds)
                                  for k, p in zip(problem_keys, oracle_pushes)}
        else:
            difficulty_mapping = {}

    # Collect pushes by category
    pushes_by_category: Dict[str, List[int]] = {'easy': [], 'medium': [], 'hard': []}
    totals_by_category: Dict[str, int] = {'easy': 0, 'medium': 0, 'hard': 0}

    for env, key, ref_result, model_result in iter_matched_triplets(model_data, search_data):
        cat = difficulty_mapping.get((env, key), 'medium')
        totals_by_category[cat] += 1

        # Check success based on learned_only flag
        if learned_only:
            if model_result.solved_by_learned:
                pushes_by_category[cat].append(model_result.pushes)
        else:
            if model_result.success:
                pushes_by_category[cat].append(model_result.pushes)

    # Compute rates at each cutoff
    result = {}
    for cat in ['easy', 'medium', 'hard']:
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


def compute_collision_success_stats(
    model_data: Dict[str, Dict[str, RegionResult]],
    search_data: Dict[str, Dict[str, RegionResult]],
) -> Dict[str, Dict[str, int]]:
    """
    Compute success rates broken down by collision type.

    Collision categories (based on oracle solution):
    - none: No wall or movable collisions
    - wall_only: Wall collision but no movable collisions
    - movable_only: Movable collisions but no wall collision
    - both: Both wall and movable collisions

    Returns:
        {collision_type: {'successes': int, 'total': int}}
    """
    stats = {
        'none': {'successes': 0, 'total': 0},
        'wall_only': {'successes': 0, 'total': 0},
        'movable_only': {'successes': 0, 'total': 0},
        'both': {'successes': 0, 'total': 0},
    }

    for env, key, ref_result, model_result in iter_matched_triplets(model_data, search_data):
        cat = get_collision_category(ref_result)
        stats[cat]['total'] += 1
        if model_result.success:
            stats[cat]['successes'] += 1

    return stats


def compute_collision_bucket_efficiency(
    model_data: Dict[str, Dict[str, RegionResult]],
    search_data: Dict[str, Dict[str, RegionResult]],
) -> Dict[str, Dict[str, Any]]:
    """
    Compute efficiency stats (checks, time) per collision bucket for solved cases.

    Returns:
        {collision_type: {'pushes': [...], 'times': [...], 'successes': int, 'total': int}}
    """
    stats = {
        'none': {'pushes': [], 'times': [], 'successes': 0, 'total': 0},
        'wall_only': {'pushes': [], 'times': [], 'successes': 0, 'total': 0},
        'movable_only': {'pushes': [], 'times': [], 'successes': 0, 'total': 0},
        'both': {'pushes': [], 'times': [], 'successes': 0, 'total': 0},
    }

    for env, key, ref_result, model_result in iter_matched_triplets(model_data, search_data):
        cat = get_collision_category(ref_result)
        stats[cat]['total'] += 1
        if model_result.success:
            stats[cat]['successes'] += 1
            stats[cat]['pushes'].append(model_result.pushes)
            stats[cat]['times'].append(model_result.time_taken)

    return stats


def compute_difficulty_stratification(
    model_data: Dict[str, Dict[str, RegionResult]],
    search_data: Dict[str, Dict[str, RegionResult]],
    difficulty_mapping: Optional[Dict[Tuple[str, str], str]] = None,
) -> Dict[str, Dict[str, Any]]:
    """
    Stratify problems by difficulty based on oracle push counts.
    Uses 33rd percentile splits: easy (fewest pushes 33%), medium (middle 33%), hard (most pushes 33%).

    Args:
        difficulty_mapping: Pre-computed difficulty mapping. If None, uses data-driven percentiles.

    Returns:
        {difficulty: {'pushes': [...], 'times': [...], 'successes': int, 'total': int,
                      'oracle_push_range': (min, max)}}
    """
    # Build difficulty mapping if not provided
    oracle_pushes_list = []
    problem_keys = []

    for env, key, ref_result, _ in iter_matched_triplets(model_data, search_data):
        oracle_pushes_list.append(ref_result.pushes)
        problem_keys.append((env, key))

    if not oracle_pushes_list:
        return {
            'easy': {'pushes': [], 'times': [], 'successes': 0, 'total': 0, 'oracle_push_range': (0, 0)},
            'medium': {'pushes': [], 'times': [], 'successes': 0, 'total': 0, 'oracle_push_range': (0, 0)},
            'hard': {'pushes': [], 'times': [], 'successes': 0, 'total': 0, 'oracle_push_range': (0, 0)},
        }

    if difficulty_mapping is None:
        thresholds = compute_percentile_thresholds(oracle_pushes_list)
        difficulty_mapping = {k: assign_difficulty(p, thresholds)
                              for k, p in zip(problem_keys, oracle_pushes_list)}

    # Initialize stats
    stats = {
        'easy': {'pushes': [], 'times': [], 'successes': 0, 'total': 0, 'oracle_pushes': []},
        'medium': {'pushes': [], 'times': [], 'successes': 0, 'total': 0, 'oracle_pushes': []},
        'hard': {'pushes': [], 'times': [], 'successes': 0, 'total': 0, 'oracle_pushes': []},
    }

    # Categorize each problem
    for (env, key), oracle_pushes in zip(problem_keys, oracle_pushes_list):
        difficulty = difficulty_mapping.get((env, key), 'medium')
        stats[difficulty]['total'] += 1
        stats[difficulty]['oracle_pushes'].append(oracle_pushes)

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
                int(min(stats[diff]['oracle_pushes'])),
                int(max(stats[diff]['oracle_pushes']))
            )
        else:
            stats[diff]['oracle_push_range'] = (0, 0)
        # Remove oracle_pushes list (not needed in output)
        del stats[diff]['oracle_pushes']

    return stats


# =============================================================================
# Multi-Reference Consistency Analysis
# =============================================================================

@dataclass
class TripletConsistency:
    """Per-triplet consistency across multiple reference runs."""
    key: str  # "env::region::object"
    pushes: List[int]  # Push counts from each reference
    chain_depths: List[int]  # Chain depths from each reference (always [1, 1, ...] for 1-push)
    all_successful: bool  # Whether all references solved this

    @property
    def median_pushes(self) -> float:
        return float(np.median(self.pushes)) if self.pushes else 0.0

    @property
    def mean_pushes(self) -> float:
        return float(np.mean(self.pushes)) if self.pushes else 0.0

    @property
    def std_pushes(self) -> float:
        return float(np.std(self.pushes)) if self.pushes else 0.0

    @property
    def cv_pushes(self) -> float:
        """Coefficient of variation (std/mean)."""
        if self.mean_pushes == 0:
            return 0.0
        return self.std_pushes / self.mean_pushes


@dataclass
class ConsistencyStats:
    """Aggregate consistency statistics."""
    n_triplets: int
    n_all_successful: int
    mean_cv_pushes: float
    threshold_p33: float
    threshold_p66: float
    n_easy: int
    n_medium: int
    n_hard: int


def compute_multi_reference_consistency(
    reference_data_list: List[Dict[str, Dict[str, RegionResult]]],
    reference_names: List[str],
) -> Tuple[Dict[Tuple[str, str], TripletConsistency], Set[Tuple[str, str]]]:
    """
    Analyze consistency across multiple oracle runs.

    Returns:
        consistency_data: {(env, key): TripletConsistency}
        common_triplets: Set of (env, key) tuples present in all references
    """
    # Find common triplets across all references
    all_keys = []
    for ref_data in reference_data_list:
        keys = {(env, key) for env in ref_data for key in ref_data[env]}
        all_keys.append(keys)

    common_triplets = all_keys[0] if all_keys else set()
    for keys in all_keys[1:]:
        common_triplets = common_triplets & keys

    # Build consistency data
    consistency_data: Dict[Tuple[str, str], TripletConsistency] = {}

    for env, key in common_triplets:
        pushes = []
        chain_depths = []
        all_successful = True

        for ref_data in reference_data_list:
            result = ref_data[env][key]
            if result.success:
                pushes.append(result.pushes)
                chain_depths.append(result.chain_depth)
            else:
                all_successful = False

        consistency_data[(env, key)] = TripletConsistency(
            key=f"{env}::{key}",
            pushes=pushes,
            chain_depths=chain_depths,
            all_successful=all_successful,
        )

    return consistency_data, common_triplets


def categorize_by_consistency(
    consistency_data: Dict[Tuple[str, str], TripletConsistency],
) -> Tuple[Dict[str, List[TripletConsistency]], Dict[str, float]]:
    """
    Categorize triplets by difficulty using median pushes from multi-reference analysis.

    Returns:
        categories: {'easy': [...], 'medium': [...], 'hard': [...]}
        thresholds: {'p33': float, 'p66': float}
    """
    # Filter to triplets where all references succeeded
    successful = [tc for tc in consistency_data.values() if tc.all_successful]

    if not successful:
        return {'easy': [], 'medium': [], 'hard': []}, {'p33': 0.0, 'p66': 0.0}

    # Compute thresholds based on median pushes
    median_pushes = [tc.median_pushes for tc in successful]
    thresholds = compute_percentile_thresholds(median_pushes)

    categories: Dict[str, List[TripletConsistency]] = {'easy': [], 'medium': [], 'hard': []}

    for tc in successful:
        difficulty = assign_difficulty(tc.median_pushes, thresholds)
        categories[difficulty].append(tc)

    return categories, thresholds


def build_difficulty_mapping(
    categories: Dict[str, List[TripletConsistency]],
) -> Dict[Tuple[str, str], str]:
    """Build (env, key) -> difficulty mapping from categorized triplets."""
    mapping = {}
    for difficulty, triplets in categories.items():
        for tc in triplets:
            # Parse the key back to (env, key)
            parts = tc.key.split("::", 1)
            if len(parts) == 2:
                env, key = parts
                mapping[(env, key)] = difficulty
    return mapping


def compute_consistency_stats(
    consistency_data: Dict[Tuple[str, str], TripletConsistency],
    categories: Dict[str, List[TripletConsistency]],
    thresholds: Dict[str, float],
    n_references: int,
) -> ConsistencyStats:
    """Compute aggregate consistency statistics."""
    all_successful = [tc for tc in consistency_data.values() if tc.all_successful]
    cv_values = [tc.cv_pushes for tc in all_successful if tc.mean_pushes > 0]

    return ConsistencyStats(
        n_triplets=len(consistency_data),
        n_all_successful=len(all_successful),
        mean_cv_pushes=float(np.mean(cv_values)) if cv_values else 0.0,
        threshold_p33=thresholds.get('p33', 0.0),
        threshold_p66=thresholds.get('p66', 0.0),
        n_easy=len(categories.get('easy', [])),
        n_medium=len(categories.get('medium', [])),
        n_hard=len(categories.get('hard', [])),
    )


def print_consistency_report(
    stats: ConsistencyStats,
    categories: Dict[str, List[TripletConsistency]],
    title: str = "Consistency Analysis",
):
    """Print consistency analysis report to stdout."""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")
    print(f"  Common triplets: {stats.n_triplets}")
    print(f"  All successful:  {stats.n_all_successful}")
    print(f"  Mean CV:         {stats.mean_cv_pushes:.3f}")
    print(f"\n  Thresholds (data-driven percentiles):")
    print(f"    Easy:   ≤ {stats.threshold_p33:.0f} pushes")
    print(f"    Medium: {stats.threshold_p33:.0f} - {stats.threshold_p66:.0f} pushes")
    print(f"    Hard:   > {stats.threshold_p66:.0f} pushes")
    print(f"\n  Distribution:")
    print(f"    Easy:   {stats.n_easy}")
    print(f"    Medium: {stats.n_medium}")
    print(f"    Hard:   {stats.n_hard}")


# =============================================================================
# Hybrid Stats (Learned vs Fallback Decomposition)
# =============================================================================

@dataclass
class HybridStats:
    """Statistics for hybrid (learned + fallback) decomposition."""
    total: int = 0
    solved_by_learned: int = 0
    solved_by_fallback: int = 0
    failed: int = 0
    learned_pushes: List[int] = field(default_factory=list)
    learned_times: List[float] = field(default_factory=list)
    fallback_pushes: List[int] = field(default_factory=list)
    fallback_times: List[float] = field(default_factory=list)
    checks_before_fallback: List[int] = field(default_factory=list)

    @property
    def success_rate(self) -> float:
        return (self.solved_by_learned + self.solved_by_fallback) / self.total if self.total > 0 else 0.0

    @property
    def learned_rate(self) -> float:
        return self.solved_by_learned / self.total if self.total > 0 else 0.0

    @property
    def fallback_rate(self) -> float:
        return self.solved_by_fallback / self.total if self.total > 0 else 0.0

    @property
    def learned_median_pushes(self) -> float:
        return float(np.median(self.learned_pushes)) if self.learned_pushes else 0.0

    @property
    def learned_median_time(self) -> float:
        return float(np.median(self.learned_times)) if self.learned_times else 0.0

    @property
    def fallback_median_pushes(self) -> float:
        return float(np.median(self.fallback_pushes)) if self.fallback_pushes else 0.0

    @property
    def fallback_median_time(self) -> float:
        return float(np.median(self.fallback_times)) if self.fallback_times else 0.0

    @property
    def median_checks_before_fallback(self) -> float:
        return float(np.median(self.checks_before_fallback)) if self.checks_before_fallback else 0.0

    @property
    def learned_pushes_iqr(self) -> Tuple[float, float]:
        if not self.learned_pushes:
            return (0.0, 0.0)
        return (float(np.percentile(self.learned_pushes, 25)),
                float(np.percentile(self.learned_pushes, 75)))

    @property
    def learned_time_iqr(self) -> Tuple[float, float]:
        if not self.learned_times:
            return (0.0, 0.0)
        return (float(np.percentile(self.learned_times, 25)),
                float(np.percentile(self.learned_times, 75)))

    @property
    def fallback_pushes_iqr(self) -> Tuple[float, float]:
        if not self.fallback_pushes:
            return (0.0, 0.0)
        return (float(np.percentile(self.fallback_pushes, 25)),
                float(np.percentile(self.fallback_pushes, 75)))

    @property
    def fallback_time_iqr(self) -> Tuple[float, float]:
        if not self.fallback_times:
            return (0.0, 0.0)
        return (float(np.percentile(self.fallback_times, 25)),
                float(np.percentile(self.fallback_times, 75)))

    @property
    def checks_before_fallback_iqr(self) -> Tuple[float, float]:
        if not self.checks_before_fallback:
            return (0.0, 0.0)
        return (float(np.percentile(self.checks_before_fallback, 25)),
                float(np.percentile(self.checks_before_fallback, 75)))


def compute_hybrid_stats(
    model_data: Dict[str, Dict[str, RegionResult]],
    reference_data: Dict[str, Dict[str, RegionResult]],
) -> HybridStats:
    """Compute hybrid decomposition statistics."""
    stats = HybridStats()

    for env, key, ref_result, model_result in iter_matched_triplets(model_data, reference_data):
        stats.total += 1

        if model_result.solved_by_learned:
            stats.solved_by_learned += 1
            stats.learned_pushes.append(model_result.pushes)
            stats.learned_times.append(model_result.time_taken)
        elif model_result.solved_by_fallback:
            stats.solved_by_fallback += 1
            stats.fallback_pushes.append(model_result.pushes)
            stats.fallback_times.append(model_result.time_taken)
            # Track checks before fallback if available
            if model_result.phase_push_counts:
                ml_checks = model_result.phase_push_counts.get('ML-only', 0)
                stats.checks_before_fallback.append(ml_checks)
        else:
            stats.failed += 1

    return stats


def compute_hybrid_stats_by_difficulty(
    model_data: Dict[str, Dict[str, RegionResult]],
    reference_data: Dict[str, Dict[str, RegionResult]],
    difficulty_mapping: Optional[Dict[Tuple[str, str], str]] = None,
) -> Dict[str, Dict[str, int]]:
    """
    Compute hybrid stats stratified by difficulty.

    Returns:
        {difficulty: {'total': int, 'learned': int, 'fallback': int, 'failed': int}}
    """
    result = {
        'easy': {'total': 0, 'learned': 0, 'fallback': 0, 'failed': 0},
        'medium': {'total': 0, 'learned': 0, 'fallback': 0, 'failed': 0},
        'hard': {'total': 0, 'learned': 0, 'fallback': 0, 'failed': 0},
    }

    # Build difficulty mapping if not provided
    if difficulty_mapping is None:
        oracle_pushes = []
        problem_keys = []
        for env, key, ref_result, _ in iter_matched_triplets(model_data, reference_data):
            oracle_pushes.append(ref_result.pushes)
            problem_keys.append((env, key))
        if oracle_pushes:
            thresholds = compute_percentile_thresholds(oracle_pushes)
            difficulty_mapping = {k: assign_difficulty(p, thresholds)
                                  for k, p in zip(problem_keys, oracle_pushes)}
        else:
            difficulty_mapping = {}

    for env, key, ref_result, model_result in iter_matched_triplets(model_data, reference_data):
        difficulty = difficulty_mapping.get((env, key), 'medium')
        result[difficulty]['total'] += 1

        if model_result.solved_by_learned:
            result[difficulty]['learned'] += 1
        elif model_result.solved_by_fallback:
            result[difficulty]['fallback'] += 1
        else:
            result[difficulty]['failed'] += 1

    return result


# =============================================================================
# RA@K Metrics (ReachableAttachment@K)
# =============================================================================

@dataclass
class RAatKStats:
    """ReachableAttachment@K statistics."""
    k: Optional[int]  # None means all primitives
    macro: float  # Macro-averaged (mean of per-instance ratios)
    micro: float  # Micro-averaged (total reachable / total considered)
    total_reachable: int
    total_considered: int
    n_instances: int


def compute_ra_at_k_single(
    aligned_primitives: List[Dict],
    reachable_edges: Set,
    k: Optional[int] = None,
) -> Tuple[int, int]:
    """
    Compute reachable count at top-K for a single instance.

    Args:
        aligned_primitives: List of primitives with 'edge_idx' and 'votes'
        reachable_edges: Set of reachable edge indices
        k: Number of top primitives to consider (None = all)

    Returns:
        (reachable_count, total_count)
    """
    if not aligned_primitives:
        return 0, 0

    # Sort by votes (descending)
    sorted_prims = sorted(aligned_primitives, key=lambda p: -p.get('votes', 0))

    # Take top-K
    if k is not None:
        sorted_prims = sorted_prims[:k]

    reachable = 0
    for prim in sorted_prims:
        edge_idx = prim.get('edge_idx')
        if edge_idx is not None and edge_idx in reachable_edges:
            reachable += 1

    return reachable, len(sorted_prims)


def compute_ra_at_k(
    model_data: Dict[str, Dict[str, RegionResult]],
    reference_data: Dict[str, Dict[str, RegionResult]],
    k: Optional[int] = None,
) -> RAatKStats:
    """
    Compute ReachableAttachment@K across all instances.

    Args:
        k: Top-K primitives to consider (None = all)
    """
    ratios = []
    total_reachable = 0
    total_considered = 0
    n_instances = 0

    for env, key, ref_result, model_result in iter_matched_triplets(model_data, reference_data):
        if not model_result.aligned_primitives:
            continue

        reachable, total = compute_ra_at_k_single(
            model_result.aligned_primitives,
            model_result.reachable_edges,
            k=k
        )

        if total > 0:
            ratios.append(reachable / total)
            total_reachable += reachable
            total_considered += total
            n_instances += 1

    macro = float(np.mean(ratios)) if ratios else 0.0
    micro = total_reachable / total_considered if total_considered > 0 else 0.0

    return RAatKStats(
        k=k,
        macro=macro,
        micro=micro,
        total_reachable=total_reachable,
        total_considered=total_considered,
        n_instances=n_instances,
    )


def compute_random_baseline(
    model_data: Dict[str, Dict[str, RegionResult]],
    reference_data: Dict[str, Dict[str, RegionResult]],
) -> float:
    """Compute random baseline (expected RA if primitives were randomly ordered)."""
    ratios = []

    for env, key, ref_result, model_result in iter_matched_triplets(model_data, reference_data):
        if model_result.ml_aligned_count > 0:
            ratios.append(model_result.ml_aligned_reachable_ratio)

    return float(np.mean(ratios)) if ratios else 0.0


# =============================================================================
# Success@Budget and Success@Time
# =============================================================================

def _compute_success_at_threshold(
    model_data: Dict[str, Dict[str, RegionResult]],
    reference_data: Dict[str, Dict[str, RegionResult]],
    thresholds: List,
    metric_getter: Callable[[RegionResult], float],
    learned_only: bool = False,
) -> Dict[Any, Dict[str, Any]]:
    """
    Generic helper to compute success rate at specific thresholds.

    Args:
        learned_only: If True, only count successes where solved_by_learned is True.

    Returns:
        {threshold: {'successes': int, 'total': int, 'rate': float}}
    """
    metrics_list = []
    total = 0

    for env, key, ref_result, model_result in iter_matched_triplets(model_data, reference_data):
        total += 1
        is_success = model_result.solved_by_learned if learned_only else model_result.success
        if is_success:
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
    learned_only: bool = False,
) -> Dict[int, Dict[str, Any]]:
    """
    Compute success rate at specific verification budgets (Success@B).

    This is a constant-compute comparison: what success rate does each method
    achieve when limited to B simulation-verified push evaluations?

    Args:
        learned_only: If True, only count successes where solved_by_learned is True.
    """
    return _compute_success_at_threshold(
        model_data, reference_data, budgets,
        metric_getter=lambda r: r.pushes,
        learned_only=learned_only,
    )


def compute_success_at_time_budget(
    model_data: Dict[str, Dict[str, RegionResult]],
    reference_data: Dict[str, Dict[str, RegionResult]],
    time_budgets: List[float],
    learned_only: bool = False,
) -> Dict[float, Dict[str, Any]]:
    """
    Compute success rate at specific time budgets (Success@T).

    This is a constant-time comparison: what success rate does each method
    achieve when limited to T milliseconds?

    Args:
        learned_only: If True, only count successes where solved_by_learned is True.
    """
    return _compute_success_at_threshold(
        model_data, reference_data, time_budgets,
        metric_getter=lambda r: r.time_taken,
        learned_only=learned_only,
    )


def _compute_success_at_threshold_by_difficulty(
    model_data: Dict[str, Dict[str, RegionResult]],
    reference_data: Dict[str, Dict[str, RegionResult]],
    thresholds: List,
    metric_getter: Callable[[RegionResult], float],
    difficulty_mapping: Optional[Dict[Tuple[str, str], str]] = None,
    learned_only: bool = False,
) -> Dict[str, Dict[Any, Dict[str, Any]]]:
    """
    Generic helper to compute success rate at thresholds, stratified by difficulty.

    Args:
        learned_only: If True, only count successes where solved_by_learned is True.

    Returns:
        {difficulty: {threshold: {'successes': int, 'total': int, 'rate': float}}}
    """
    # Build difficulty mapping if not provided
    if difficulty_mapping is None:
        oracle_pushes = []
        problem_keys = []
        for env, key, ref_result, _ in iter_matched_triplets(model_data, reference_data):
            oracle_pushes.append(ref_result.pushes)
            problem_keys.append((env, key))

        if not oracle_pushes:
            empty_result = {t: {'successes': 0, 'total': 0, 'rate': 0.0} for t in thresholds}
            return {'easy': empty_result.copy(), 'medium': empty_result.copy(), 'hard': empty_result.copy()}

        thresh = compute_percentile_thresholds(oracle_pushes)
        difficulty_mapping = {k: assign_difficulty(p, thresh)
                              for k, p in zip(problem_keys, oracle_pushes)}

    # Collect metrics by difficulty
    metrics_by_difficulty: Dict[str, List[float]] = {'easy': [], 'medium': [], 'hard': []}
    totals_by_difficulty: Dict[str, int] = {'easy': 0, 'medium': 0, 'hard': 0}

    for env, key, ref_result, model_result in iter_matched_triplets(model_data, reference_data):
        difficulty = difficulty_mapping.get((env, key), 'medium')
        totals_by_difficulty[difficulty] += 1
        is_success = model_result.solved_by_learned if learned_only else model_result.success
        if is_success:
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
    difficulty_mapping: Optional[Dict[Tuple[str, str], str]] = None,
    learned_only: bool = False,
) -> Dict[str, Dict[int, Dict[str, Any]]]:
    """Compute success rate at specific verification budgets, stratified by difficulty."""
    return _compute_success_at_threshold_by_difficulty(
        model_data, reference_data, budgets,
        metric_getter=lambda r: r.pushes,
        difficulty_mapping=difficulty_mapping,
        learned_only=learned_only,
    )


def compute_success_at_time_budget_by_difficulty(
    model_data: Dict[str, Dict[str, RegionResult]],
    reference_data: Dict[str, Dict[str, RegionResult]],
    time_budgets: List[float],
    difficulty_mapping: Optional[Dict[Tuple[str, str], str]] = None,
    learned_only: bool = False,
) -> Dict[str, Dict[float, Dict[str, Any]]]:
    """Compute success rate at specific time budgets, stratified by difficulty."""
    return _compute_success_at_threshold_by_difficulty(
        model_data, reference_data, time_budgets,
        metric_getter=lambda r: r.time_taken,
        difficulty_mapping=difficulty_mapping,
        learned_only=learned_only,
    )


# =============================================================================
# Plotting
# =============================================================================

def get_model_color(idx: int, config: EvalConfig) -> str:
    """Get color for a model by index."""
    return config.model_colors[idx % len(config.model_colors)]


def plot_success_rates(
    model_stats: List[ModelStats],
    config: EvalConfig,
    output_path: Optional[str] = None,
):
    """Plot success rate comparison across models."""
    categories = ['easy', 'medium', 'hard']
    n_models = len(model_stats)
    x = np.arange(len(categories))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))

    for i, stats in enumerate(model_stats):
        rates = [stats.get_category(cat).success_rate for cat in categories]
        counts = [(stats.get_category(cat).successes, stats.get_category(cat).total) for cat in categories]

        offset = (i - n_models/2 + 0.5) * width
        bars = ax.bar(x + offset, rates, width * 0.9, label=stats.name,
                      color=get_model_color(i, config), edgecolor='white', linewidth=0.5)

        # Add percentage labels
        for bar, rate, (succ, total) in zip(bars, rates, counts):
            ax.annotate(f'{rate:.0%}',
                       xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                       xytext=(0, 4), textcoords="offset points",
                       ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels([c.capitalize() for c in categories])
    ax.set_ylim(0, 1.12)
    ax.set_ylabel('Success Rate')
    ax.set_title('Success Rate')
    ax.axhline(y=1.0, color='#888888', linestyle='--', linewidth=0.8, alpha=0.5)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=min(n_models, 4), frameon=True, fancybox=True)

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.2)
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Saved: {output_path}")
    return fig


def plot_pushes_boxplot(
    model_stats: List[ModelStats],
    config: EvalConfig,
    output_path: Optional[str] = None,
):
    """Plot pushes comparison as boxplots."""
    categories = ['easy', 'medium', 'hard']
    n_models = len(model_stats)

    fig, ax = plt.subplots(figsize=(9, 5))

    positions = []
    data = []
    colors = []

    for cat_idx, cat in enumerate(categories):
        for model_idx, stats in enumerate(model_stats):
            pos = cat_idx * (n_models + 1) + model_idx
            positions.append(pos)
            data.append(stats.get_category(cat).pushes or [0])
            colors.append(get_model_color(model_idx, config))

    bp = ax.boxplot(data, positions=positions, widths=0.6, patch_artist=True, showfliers=False)

    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
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

    # Set x-axis labels
    cat_positions = [(i * (n_models + 1) + (n_models - 1) / 2) for i in range(len(categories))]
    ax.set_xticks(cat_positions)
    ax.set_xticklabels([c.capitalize() for c in categories])

    # Legend
    legend_handles = [plt.Rectangle((0,0),1,1, facecolor=get_model_color(i, config), alpha=0.85)
                      for i in range(n_models)]
    ax.legend(legend_handles, [s.name for s in model_stats], loc='upper center',
              bbox_to_anchor=(0.5, -0.12), ncol=min(n_models, 4), frameon=True, fancybox=True)

    ax.set_ylabel('Pushes to Success')
    ax.set_title('Pushes to Solution')

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.2)
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Saved: {output_path}")
    return fig


def plot_time_boxplot(
    model_stats: List[ModelStats],
    config: EvalConfig,
    output_path: Optional[str] = None,
):
    """Plot time comparison as boxplots."""
    categories = ['easy', 'medium', 'hard']
    n_models = len(model_stats)

    fig, ax = plt.subplots(figsize=(9, 5))

    positions = []
    data = []
    colors = []

    for cat_idx, cat in enumerate(categories):
        for model_idx, stats in enumerate(model_stats):
            pos = cat_idx * (n_models + 1) + model_idx
            positions.append(pos)
            data.append(stats.get_category(cat).times or [0])
            colors.append(get_model_color(model_idx, config))

    bp = ax.boxplot(data, positions=positions, widths=0.6, patch_artist=True, showfliers=False)

    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
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

    cat_positions = [(i * (n_models + 1) + (n_models - 1) / 2) for i in range(len(categories))]
    ax.set_xticks(cat_positions)
    ax.set_xticklabels([c.capitalize() for c in categories])

    legend_handles = [plt.Rectangle((0,0),1,1, facecolor=get_model_color(i, config), alpha=0.85)
                      for i in range(n_models)]
    ax.legend(legend_handles, [s.name for s in model_stats], loc='upper center',
              bbox_to_anchor=(0.5, -0.12), ncol=min(n_models, 4), frameon=True, fancybox=True)

    ax.set_ylabel('Time to Success (ms)')
    ax.set_title('Time to Solution')

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.2)
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Saved: {output_path}")
    return fig


def plot_solutions_distribution(
    model_stats: List[ModelStats],
    config: EvalConfig,
    output_path: Optional[str] = None,
):
    """Plot distribution of solution counts per category (histogram)."""
    categories = ['easy', 'medium', 'hard']

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    for idx, cat in enumerate(categories):
        ax = axes[idx]

        for model_idx, stats in enumerate(model_stats):
            solutions_list = stats.get_category(cat).solutions  # solutions_total_for_neighbour
            if not solutions_list:
                continue

            # Count frequency of each solution count
            from collections import Counter
            counts = Counter(solutions_list)

            # Get sorted solution values and their frequencies
            sol_values = sorted(counts.keys())
            frequencies = [counts[v] for v in sol_values]

            # Bar plot
            x_pos = np.arange(len(sol_values))
            width = 0.8 / len(model_stats)
            offset = (model_idx - len(model_stats)/2 + 0.5) * width

            bars = ax.bar(x_pos + offset, frequencies, width,
                         label=stats.name, color=get_model_color(model_idx, config),
                         edgecolor='black')

            # Add count labels on bars
            for bar, freq in zip(bars, frequencies):
                if freq > 0:
                    ax.annotate(f'{freq}',
                               xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                               xytext=(0, 2), textcoords="offset points",
                               ha='center', va='bottom', fontsize=8)

            ax.set_xticks(x_pos)
            ax.set_xticklabels(sol_values)

        ax.set_xlabel('Number of Solutions Found')
        ax.set_ylabel('Count (env+region pairs)')
        ax.set_title(f'{cat.capitalize()} Category')
        ax.grid(True, axis='y', linestyle='--', alpha=0.6)

    # Shared legend below all subplots
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, -0.02),
               ncol=min(len(labels), 4), frameon=True, fancybox=True)

    plt.suptitle('Solution Distribution', fontsize=20, fontweight='bold')
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_path}")
    return fig


def plot_time_vs_success(
    time_data: Dict[str, Dict[str, Dict[str, List[float]]]],  # {model_name: {category: {cutoffs, rates}}}
    config: EvalConfig,
    output_path: Optional[str] = None,
):
    """Plot success rate vs time cutoff."""
    difficulty_levels = ['easy', 'medium', 'hard']
    difficulty_labels = {'easy': 'Easy', 'medium': 'Medium', 'hard': 'Hard'}
    difficulty_colors = {'easy': '#55A868', 'medium': '#DD8452', 'hard': '#C44E52'}

    # Get N for each category from the first model
    n_by_cat = {}
    for cat_data in time_data.values():
        for cat in difficulty_levels:
            if cat in cat_data and cat not in n_by_cat:
                n_by_cat[cat] = cat_data[cat].get('total', 0)
        break

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)

    for idx, diff in enumerate(difficulty_levels):
        ax = axes[idx]

        for model_idx, (model_name, cat_data) in enumerate(time_data.items()):
            if diff in cat_data:
                cutoffs_ms = cat_data[diff]['cutoffs']
                cutoffs_s = [c / 1000.0 for c in cutoffs_ms]  # Convert to seconds
                rates = cat_data[diff]['rates']
                ax.plot(cutoffs_s, rates, label=model_name,
                       color=get_model_color(model_idx, config), linewidth=2)

        ax.set_xlabel('Time cutoff (s)')
        if idx == 0:
            ax.set_ylabel('Success Rate')
        ax.set_ylim(0, 1.05)
        ax.set_xlim(0, config.time_cutoff_max / 1000.0)
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.set_title(f"{difficulty_labels[diff]} (N={100})", fontsize=18,
                    fontweight='bold', color=difficulty_colors[diff])

    # Shared legend below all subplots
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, -0.02),
               ncol=min(len(labels), 4), fontsize=16)

    fig.suptitle("Success Rate vs Time Budget", fontsize=22, fontweight='bold')
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Saved: {output_path}")
    return fig


def plot_pushes_vs_success(
    push_data: Dict[str, Dict[str, Dict[str, List[float]]]],  # {model_name: {category: {cutoffs, rates, total}}}
    config: EvalConfig,
    output_path: Optional[str] = None,
):
    """Plot success rate vs push count cutoff."""
    difficulty_levels = ['easy', 'medium', 'hard']
    difficulty_labels = {'easy': 'Easy', 'medium': 'Medium', 'hard': 'Hard'}
    difficulty_colors = {'easy': '#55A868', 'medium': '#DD8452', 'hard': '#C44E52'}

    # Get N for each category from the first model
    n_by_cat = {}
    for cat_data in push_data.values():
        for cat in difficulty_levels:
            if cat in cat_data and cat not in n_by_cat:
                n_by_cat[cat] = cat_data[cat].get('total', 0)
        break

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)

    for idx, diff in enumerate(difficulty_levels):
        ax = axes[idx]

        for model_idx, (model_name, cat_data) in enumerate(push_data.items()):
            if diff in cat_data:
                cutoffs = cat_data[diff]['cutoffs']
                rates = cat_data[diff]['rates']
                ax.plot(cutoffs, rates, label=model_name,
                       color=get_model_color(model_idx, config), linewidth=2)

        ax.set_xlabel('Simulation-verified push evaluations')
        if idx == 0:
            ax.set_ylabel('Success Rate')
        ax.set_ylim(0, 1.05)
        ax.set_xlim(0, config.push_cutoff_max)
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.set_title(f"{difficulty_labels[diff]} (N={100})", fontsize=18,
                    fontweight='bold', color=difficulty_colors[diff])

    # Shared legend below all subplots
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, -0.02),
               ncol=min(len(labels), 4), fontsize=16)

    fig.suptitle("Success Rate vs Verification Budget", fontsize=22, fontweight='bold')
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Saved: {output_path}")
    return fig


def plot_wall_collision_rate(
    model_stats: List[ModelStats],
    config: EvalConfig,
    output_path: Optional[str] = None,
):
    """Plot wall collision rate by difficulty category."""
    categories = ['easy', 'medium', 'hard']
    n_models = len(model_stats)

    fig, ax = plt.subplots(figsize=(8, 6))

    x = np.arange(len(categories))
    width = 0.8 / n_models

    for i, stats in enumerate(model_stats):
        rates = [stats.get_category(cat).wall_collision_rate for cat in categories]
        offset = (i - n_models/2 + 0.5) * width
        bars = ax.bar(x + offset, rates, width, label=stats.name,
                      color=get_model_color(i, config), edgecolor='black')

        for bar, rate in zip(bars, rates):
            if rate > 0:
                ax.annotate(f'{rate:.0%}',
                           xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                           xytext=(0, 3), textcoords="offset points",
                           ha='center', va='bottom', fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels([c.capitalize() for c in categories])
    ax.set_ylim(0, 1.15)
    ax.set_ylabel('Wall Collision Rate')
    ax.set_title('Wall Collision Rate')
    ax.grid(True, axis='y', linestyle='--', alpha=0.6)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=min(n_models, 4),
              frameon=True, fancybox=True)

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.2)
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_path}")
    return fig


def plot_movable_collision_rate(
    model_stats: List[ModelStats],
    config: EvalConfig,
    output_path: Optional[str] = None,
):
    """Plot movable collision rate by difficulty category."""
    categories = ['easy', 'medium', 'hard']
    n_models = len(model_stats)

    fig, ax = plt.subplots(figsize=(8, 6))

    x = np.arange(len(categories))
    width = 0.8 / n_models

    for i, stats in enumerate(model_stats):
        rates = [stats.get_category(cat).any_movable_collision_rate for cat in categories]
        offset = (i - n_models/2 + 0.5) * width
        bars = ax.bar(x + offset, rates, width, label=stats.name,
                      color=get_model_color(i, config), edgecolor='black')

        for bar, rate in zip(bars, rates):
            if rate > 0:
                ax.annotate(f'{rate:.0%}',
                           xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                           xytext=(0, 3), textcoords="offset points",
                           ha='center', va='bottom', fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels([c.capitalize() for c in categories])
    ax.set_ylim(0, 1.15)
    ax.set_ylabel('Movable Collision Rate')
    ax.set_title('Movable Collision Rate')
    ax.grid(True, axis='y', linestyle='--', alpha=0.6)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=min(n_models, 4),
              frameon=True, fancybox=True)

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.2)
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
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
                # Put percentage inside bar if tall enough, otherwise above
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
    ax.set_title('Success Rate by Collision Type')
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=min(n_models, 4),
              frameon=True, fancybox=True, shadow=False)
    plt.subplots_adjust(bottom=0.2)

    # Add horizontal line at 100%
    ax.axhline(y=1.0, color='#888888', linestyle='--', linewidth=0.8, alpha=0.5)

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Saved: {output_path}")
    return fig


def print_summary(
    model_stats: List[ModelStats],
    hybrid_stats: Optional[Dict[str, 'HybridStats']] = None,
    learned_model_names: Optional[Set[str]] = None,
):
    """Print summary statistics."""
    print("\n" + "=" * 80)
    print("EVALUATION SUMMARY")
    print("=" * 80)

    categories = ['easy', 'medium', 'hard']
    if learned_model_names is None:
        learned_model_names = set()

    for stats in model_stats:
        print(f"\n{'─' * 40}")
        print(f"Model: {stats.name}")
        print(f"{'─' * 40}")
        print(f"Overall: {stats.total_successes}/{stats.total_trials} = {stats.overall_success_rate:.4f}")

        # Show diffusion-only/fallback breakdown for SAGE models
        if hybrid_stats and stats.name in hybrid_stats and stats.name in learned_model_names:
            hs = hybrid_stats[stats.name]
            print(f"  └─ Diffusion-Only: {hs.solved_by_learned}/{hs.total} = {hs.learned_rate:.4f}, "
                  f"Fallback: {hs.solved_by_fallback}/{hs.total} = {hs.fallback_rate:.4f}")

        for cat in categories:
            cat_stats = stats.get_category(cat)
            print(f"\n  {cat.capitalize()}:")
            print(f"    Success: {cat_stats.successes}/{cat_stats.total} = {cat_stats.success_rate:.4f}")
            if cat_stats.pushes:
                p_iqr = cat_stats.pushes_iqr
                print(f"    Pushes:  median={cat_stats.median_pushes:.0f} [{p_iqr[0]:.0f}, {p_iqr[1]:.0f}], mean={cat_stats.mean_pushes:.1f}")
            if cat_stats.times:
                t_iqr = cat_stats.time_iqr
                print(f"    Time:    median={cat_stats.median_time:.0f}ms [{t_iqr[0]:.0f}, {t_iqr[1]:.0f}], mean={cat_stats.mean_time:.1f}ms")
            if cat_stats.solutions:
                print(f"    Solutions: total={cat_stats.total_solutions}, mean={cat_stats.mean_solutions:.1f}")
            if cat_stats.successes > 0:
                print(f"    Interactions: wall_col={cat_stats.wall_collision_rate:.1%}, movable_col={cat_stats.any_movable_collision_rate:.1%}")

        if stats.failure_reasons:
            print(f"\n  Failure Reasons:")
            for reason, count in sorted(stats.failure_reasons.items(), key=lambda x: -x[1]):
                print(f"    {reason}: {count}")


def generate_markdown_report(
    model_stats: List[ModelStats],
    config: EvalConfig,
    category_counts: Dict[str, int],
    output_path: str,
    collision_efficiency: Optional[Dict[str, Dict[str, Dict[str, Any]]]] = None,
    difficulty_stratification: Optional[Dict[str, Dict[str, Dict[str, Any]]]] = None,
    hybrid_stats: Optional[Dict[str, 'HybridStats']] = None,
    hybrid_stats_by_difficulty: Optional[Dict[str, Dict[str, Dict[str, int]]]] = None,
    ra_at_k_stats: Optional[Dict[str, Dict[Optional[int], 'RAatKStats']]] = None,
    random_baselines: Optional[Dict[str, float]] = None,
    success_at_budget: Optional[Dict[str, Dict[int, Dict[str, Any]]]] = None,
    success_at_time: Optional[Dict[str, Dict[float, Dict[str, Any]]]] = None,
    difficulty_thresholds: Optional[Dict[str, float]] = None,
    success_at_budget_learned: Optional[Dict[str, Dict[int, Dict[str, Any]]]] = None,
    success_at_time_learned: Optional[Dict[str, Dict[float, Dict[str, Any]]]] = None,
    learned_model_names: Optional[Set[str]] = None,
):
    """Generate a markdown report with comparison tables."""
    categories = ['easy', 'medium', 'hard']
    difficulty_labels = {'easy': 'Easy', 'medium': 'Medium', 'hard': 'Hard'}

    lines = []
    lines.append("# 1-Push Evaluation Results\n")
    lines.append(f"Generated from evaluation config.\n")

    # Show difficulty thresholds if available
    if difficulty_thresholds:
        lines.append(f"Difficulty thresholds (data-driven percentiles): "
                     f"Easy ≤ {difficulty_thresholds['p33']:.0f} pushes, "
                     f"Medium ≤ {difficulty_thresholds['p66']:.0f} pushes, "
                     f"Hard > {difficulty_thresholds['p66']:.0f} pushes\n")

    # Dataset overview
    lines.append("## Dataset Overview\n")
    total_pairs = sum(category_counts.values())
    lines.append(f"Total env+region pairs evaluated: **{total_pairs}**\n")
    lines.append("| Category | Count | Percentage |")
    lines.append("|----------|-------|------------|")
    for cat in categories:
        count = category_counts[cat]
        pct = count / total_pairs * 100 if total_pairs > 0 else 0
        lines.append(f"| {cat.capitalize()} | {count} | {pct:.1f}% |")
    lines.append("")

    # Overall success rates
    lines.append("## Overall Success Rates\n")
    lines.append("| Model | Successes | Total | Success Rate |")
    lines.append("|-------|-----------|-------|--------------|")
    for stats in model_stats:
        lines.append(f"| {stats.name} | {stats.total_successes} | {stats.total_trials} | **{stats.overall_success_rate:.1%}** |")
    lines.append("")

    # Success rates by category
    lines.append("## Success Rates by Category\n")
    header = "| Model |"
    separator = "|-------|"
    for cat in categories:
        header += f" {cat.capitalize()} |"
        separator += "--------|"
    lines.append(header)
    lines.append(separator)

    for stats in model_stats:
        row = f"| {stats.name} |"
        for cat in categories:
            cat_stats = stats.get_category(cat)
            row += f" {cat_stats.success_rate:.1%} ({cat_stats.successes}/{cat_stats.total}) |"
        lines.append(row)
    lines.append("")

    # Pushes statistics (successful runs only)
    lines.append("## Pushes to Success (Successful Runs Only)\n")
    lines.append("Format: median [IQR]\n")
    header = "| Model |"
    separator = "|-------|"
    for cat in categories:
        header += f" {cat.capitalize()} |"
        separator += "--------|"
    lines.append(header)
    lines.append(separator)

    for stats in model_stats:
        row = f"| {stats.name} |"
        for cat in categories:
            cat_stats = stats.get_category(cat)
            if cat_stats.pushes:
                p_iqr = cat_stats.pushes_iqr
                row += f" {cat_stats.median_pushes:.0f} [{p_iqr[0]:.0f}, {p_iqr[1]:.0f}] |"
            else:
                row += " - |"
        lines.append(row)
    lines.append("")

    # Time statistics (successful runs only) - in seconds
    lines.append("## Time to Success in seconds (Successful Runs Only)\n")
    lines.append("Format: median [IQR]\n")
    lines.append(header)
    lines.append(separator)

    for stats in model_stats:
        row = f"| {stats.name} |"
        for cat in categories:
            cat_stats = stats.get_category(cat)
            if cat_stats.times:
                t_iqr = cat_stats.time_iqr
                # Convert ms to seconds
                row += f" {cat_stats.median_time/1000:.1f} [{t_iqr[0]/1000:.1f}, {t_iqr[1]/1000:.1f}] |"
            else:
                row += " - |"
        lines.append(row)
    lines.append("")

    # Interaction statistics
    lines.append("## Interaction Statistics (Successful Runs Only)\n")
    lines.append("*Note: Statistics computed over successful runs only. Models with lower success rates may show different interaction patterns due to selection bias (failing on harder instances).*\n")

    lines.append("### Wall Collision Rate\n")
    lines.append("Percentage of successful runs that had collisions with walls.\n")
    lines.append(header)
    lines.append(separator)

    for stats in model_stats:
        row = f"| {stats.name} |"
        for cat in categories:
            cat_stats = stats.get_category(cat)
            if cat_stats.successes > 0:
                row += f" {cat_stats.wall_collision_rate:.1%} ({cat_stats.wall_collisions}/{cat_stats.successes}) |"
            else:
                row += " - |"
        lines.append(row)
    lines.append("")

    lines.append("### Movable Object Collision Rate\n")
    lines.append("Percentage of successful runs that collided with other movable objects.\n")
    lines.append(header)
    lines.append(separator)

    for stats in model_stats:
        row = f"| {stats.name} |"
        for cat in categories:
            cat_stats = stats.get_category(cat)
            if cat_stats.successes > 0:
                any_mov = sum(1 for c in cat_stats.movable_collisions_list if c > 0)
                row += f" {cat_stats.any_movable_collision_rate:.1%} ({any_mov}/{cat_stats.successes}) |"
            else:
                row += " - |"
        lines.append(row)
    lines.append("")

    lines.append("### Mean Movable Collisions\n")
    lines.append("Average number of unique movable objects collided with per successful run.\n")
    lines.append(header)
    lines.append(separator)

    for stats in model_stats:
        row = f"| {stats.name} |"
        for cat in categories:
            cat_stats = stats.get_category(cat)
            if cat_stats.movable_collisions_list:
                row += f" {cat_stats.mean_movable_collisions:.2f} |"
            else:
                row += " - |"
        lines.append(row)
    lines.append("")

    # =========================================================================
    # COLLISION STRATIFICATION (success rate by collision type)
    # =========================================================================
    if collision_efficiency:
        lines.append("## Success Rate by Collision Type\n")
        lines.append("Collision type determined by oracle (search) solution.\n")

        collision_types = ['none', 'wall_only', 'movable_only', 'both']
        collision_labels = {'none': 'No Collision', 'wall_only': 'Wall Only',
                            'movable_only': 'Movable Only', 'both': 'Both'}

        col_header = "| Model |"
        for ct in collision_types:
            col_header += f" {collision_labels[ct]} |"
        lines.append(col_header)

        col_sep = "|-------|"
        for _ in collision_types:
            col_sep += "-------------|"
        lines.append(col_sep)

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
                    med_time = f"{np.median(times)/1000:.1f}"  # Convert ms to seconds
                else:
                    med_checks = "-"
                    med_time = "-"
                lines.append(f"| {name} | {collision_labels[ct]} | {n_solved} | {med_checks} | {med_time} |")
        lines.append("")

    # =========================================================================
    # DIFFICULTY STRATIFICATION (based on oracle push counts)
    # =========================================================================
    if difficulty_stratification:
        lines.append("## Difficulty Stratification (by Oracle Push Counts)\n")
        lines.append("*Problems split into thirds by oracle push counts: Easy (fewest 33%), Medium (middle 33%), Hard (most 33%).*\n")

        difficulty_levels = ['easy', 'medium', 'hard']

        # Get oracle push ranges from first model (same for all)
        first_model = list(difficulty_stratification.keys())[0]
        range_info = []
        for diff in difficulty_levels:
            r = difficulty_stratification[first_model][diff].get('oracle_push_range', (0, 0))
            range_info.append(f"**{difficulty_labels[diff]}**: {r[0]}–{r[1]} pushes")
        lines.append(f"Oracle push ranges: {', '.join(range_info)}\n")

        # Success rate table
        lines.append("### Success Rate by Difficulty\n")
        diff_header = "| Model |"
        for diff in difficulty_levels:
            diff_header += f" {difficulty_labels[diff]} |"
        lines.append(diff_header)

        diff_sep = "|-------|"
        for _ in difficulty_levels:
            diff_sep += "------------|"
        lines.append(diff_sep)

        for name in difficulty_stratification:
            row = f"| {name} |"
            for diff in difficulty_levels:
                stats = difficulty_stratification[name][diff]
                rate = stats['successes'] / stats['total'] if stats['total'] > 0 else 0.0
                row += f" {rate:.1%} ({stats['successes']}/{stats['total']}) |"
            lines.append(row)
        lines.append("")

        # Efficiency by difficulty (solved cases only)
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

    # =========================================================================
    # HYBRID DECOMPOSITION (Learned vs Fallback)
    # =========================================================================
    if hybrid_stats:
        lines.append("## SAGE (Hybrid): Diffusion-Only vs Fallback\n")
        lines.append("*Phase tracking: solved_in_phase='ML-only' → Diffusion-Only, 'primitives' → Fallback*\n")

        lines.append("| Model | Total | Diffusion-Only | Fallback | Failed | Success Rate |")
        lines.append("|-------|-------|----------------|----------|--------|--------------|")

        for name, hs in hybrid_stats.items():
            if hs.total > 0:
                lines.append(f"| {name} | {hs.total} | {hs.solved_by_learned} ({hs.learned_rate:.1%}) | "
                             f"{hs.solved_by_fallback} ({hs.fallback_rate:.1%}) | {hs.failed} | {hs.success_rate:.1%} |")
        lines.append("")

        # Diffusion-Only efficiency section
        lines.append("### Diffusion-Only Cases: Efficiency\n")
        lines.append("*Problems solved by diffusion model phase only.*\n")
        lines.append("| Model | N | Pushes (median [IQR]) | Time (s) (median [IQR]) |")
        lines.append("|-------|---|----------------------|-------------------------|")
        for name, hs in hybrid_stats.items():
            n_learned = hs.solved_by_learned
            if hs.learned_pushes:
                l_iqr = hs.learned_pushes_iqr
                learned_pushes = f"{hs.learned_median_pushes:.0f} [{l_iqr[0]:.0f}, {l_iqr[1]:.0f}]"
                lt_iqr = hs.learned_time_iqr
                learned_time = f"{hs.learned_median_time/1000:.1f} [{lt_iqr[0]/1000:.1f}, {lt_iqr[1]/1000:.1f}]"
            else:
                learned_pushes = "-"
                learned_time = "-"
            lines.append(f"| {name} | {n_learned} | {learned_pushes} | {learned_time} |")
        lines.append("")

        # Fallback efficiency section
        any_fallback = any(hs.solved_by_fallback > 0 for hs in hybrid_stats.values())
        if any_fallback:
            lines.append("### Fallback Cases: Efficiency\n")
            lines.append("*Problems where ML phase exhausted, solved by primitives phase.*\n")
            lines.append("| Model | N | Pushes (median [IQR]) | Time (s) (median [IQR]) |")
            lines.append("|-------|---|----------------------|-------------------------|")
            for name, hs in hybrid_stats.items():
                n_fallback = hs.solved_by_fallback
                if hs.fallback_pushes:
                    f_iqr = hs.fallback_pushes_iqr
                    fallback_pushes = f"{hs.fallback_median_pushes:.0f} [{f_iqr[0]:.0f}, {f_iqr[1]:.0f}]"
                    ft_iqr = hs.fallback_time_iqr
                    fallback_time = f"{hs.fallback_median_time/1000:.1f} [{ft_iqr[0]/1000:.1f}, {ft_iqr[1]/1000:.1f}]"
                else:
                    fallback_pushes = "-"
                    fallback_time = "-"
                lines.append(f"| {name} | {n_fallback} | {fallback_pushes} | {fallback_time} |")
            lines.append("")

        # Add by-difficulty breakdown if available
        if hybrid_stats_by_difficulty:
            lines.append("### SAGE (Hybrid) by Difficulty\n")
            lines.append("| Model | Difficulty | N | Diffusion-Only | Fallback | Failed |")
            lines.append("|-------|------------|---|----------------|----------|--------|")
            for name in hybrid_stats_by_difficulty:
                for diff in categories:
                    stats = hybrid_stats_by_difficulty[name].get(diff, {})
                    n = stats.get('total', 0)
                    if n > 0:
                        diffonly = stats.get('learned', 0)
                        fallback = stats.get('fallback', 0)
                        failed = stats.get('failed', 0)
                        diffonly_pct = diffonly / n * 100
                        fallback_pct = fallback / n * 100
                        failed_pct = failed / n * 100
                        lines.append(f"| {name} | {diff.capitalize()} | {n} | {diffonly_pct:.1f}% ({diffonly}) | {fallback_pct:.1f}% ({fallback}) | {failed_pct:.1f}% ({failed}) |")
            lines.append("")

    # =========================================================================
    # RA@K METRICS
    # =========================================================================
    if ra_at_k_stats:
        lines.append("## Reachable Attachment @ K (RA@K)\n")
        lines.append("*Fraction of top-K ML-ranked primitives with reachable push attachments.*\n")

        # Build header dynamically based on k values
        header = "| Model |"
        for k in config.ra_at_k_values:
            k_str = f"@{k}" if k is not None else "@All"
            header += f" RA{k_str} |"
        header += " Random |"
        lines.append(header)

        sep = "|-------|"
        for _ in config.ra_at_k_values:
            sep += "--------|"
        sep += "--------|"
        lines.append(sep)

        for name in ra_at_k_stats:
            row = f"| {name} |"
            for k in config.ra_at_k_values:
                ra = ra_at_k_stats[name][k]
                if ra.n_instances > 0:
                    row += f" {ra.macro:.1%} |"
                else:
                    row += " - |"
            if random_baselines and name in random_baselines:
                row += f" {random_baselines[name]:.1%} |"
            else:
                row += " - |"
            lines.append(row)
        lines.append("")

    # =========================================================================
    # SUCCESS @ BUDGET
    # =========================================================================
    if success_at_budget:
        lines.append("## Success @ Budget\n")
        lines.append("*Success rate at fixed verification budget (constant-compute comparison).*\n")
        lines.append("*Format: Total (Diffusion-Only) for SAGE models.*\n")

        header = "| Model |"
        for b in config.success_at_budget_values:
            header += f" @{b} |"
        lines.append(header)

        sep = "|-------|"
        for _ in config.success_at_budget_values:
            sep += "-------------|"
        lines.append(sep)

        for name in success_at_budget:
            row = f"| {name} |"
            for b in config.success_at_budget_values:
                stats = success_at_budget[name][b]
                if success_at_budget_learned and name in success_at_budget_learned:
                    learned_stats = success_at_budget_learned[name][b]
                    row += f" {stats['rate']:.1%} ({learned_stats['rate']:.1%}) |"
                else:
                    row += f" {stats['rate']:.1%} |"
            lines.append(row)
        lines.append("")

    # =========================================================================
    # SUCCESS @ TIME
    # =========================================================================
    if success_at_time:
        lines.append("## Success @ Time\n")
        lines.append("*Success rate at fixed time budget (constant-time comparison).*\n")
        lines.append("*Format: Total (Diffusion-Only) for SAGE models.*\n")

        def format_time(t_ms: float) -> str:
            if t_ms >= 1000:
                return f"@{t_ms/1000:.0f}s"
            return f"@{t_ms:.0f}ms"

        header = "| Model |"
        for t in config.success_at_time_values:
            header += f" {format_time(t)} |"
        lines.append(header)

        sep = "|-------|"
        for _ in config.success_at_time_values:
            sep += "-------------|"
        lines.append(sep)

        for name in success_at_time:
            row = f"| {name} |"
            for t in config.success_at_time_values:
                stats = success_at_time[name][t]
                if success_at_time_learned and name in success_at_time_learned:
                    learned_stats = success_at_time_learned[name][t]
                    row += f" {stats['rate']:.1%} ({learned_stats['rate']:.1%}) |"
                else:
                    row += f" {stats['rate']:.1%} |"
            lines.append(row)
        lines.append("")

    # Detailed per-model breakdown
    lines.append("## Detailed Per-Model Statistics\n")

    for stats in model_stats:
        lines.append(f"### {stats.name}\n")
        lines.append("| Category | Success Rate | Med Pushes | Mean Pushes | Med Time | Mean Time | Wall Col | Mov Col |")
        lines.append("|----------|--------------|------------|-------------|----------|-----------|----------|---------|")

        for cat in categories:
            cat_stats = stats.get_category(cat)
            success_str = f"{cat_stats.success_rate:.1%} ({cat_stats.successes}/{cat_stats.total})"
            med_push = f"{cat_stats.median_pushes:.1f}" if cat_stats.pushes else "-"
            mean_push = f"{cat_stats.mean_pushes:.1f}" if cat_stats.pushes else "-"
            med_time = f"{cat_stats.median_time:.0f}" if cat_stats.times else "-"
            mean_time = f"{cat_stats.mean_time:.0f}" if cat_stats.times else "-"
            wall_col = f"{cat_stats.wall_collision_rate:.0%}" if cat_stats.successes > 0 else "-"
            mov_col = f"{cat_stats.any_movable_collision_rate:.0%}" if cat_stats.successes > 0 else "-"
            lines.append(f"| {cat.capitalize()} | {success_str} | {med_push} | {mean_push} | {med_time} | {mean_time} | {wall_col} | {mov_col} |")

        # Add failure reasons if available
        if stats.failure_reasons:
            lines.append("")
            lines.append("**Failure Reasons:**\n")
            lines.append("| Reason | Count |")
            lines.append("|--------|-------|")
            for reason, count in sorted(stats.failure_reasons.items(), key=lambda x: -x[1]):
                lines.append(f"| {reason} | {count} |")
        lines.append("")

    # Write to file
    with open(output_path, 'w') as f:
        f.write('\n'.join(lines))

    print(f"Saved: {output_path}")


# =============================================================================
# Main
# =============================================================================

def get_env_region_keys(data: Dict[str, Dict[str, RegionResult]]) -> set:
    """Get set of (env, region) tuples from data."""
    return {(env, region) for env in data for region in data[env]}


def filter_to_intersection(
    all_data: Dict[str, Dict[str, Dict[str, RegionResult]]],
    reference_data: Dict[str, Dict[str, RegionResult]],
    require_reference_success: bool = True,
) -> Tuple[Dict[str, Dict[str, Dict[str, RegionResult]]], set]:
    """
    Filter all model data to only include env+region pairs present in ALL models
    AND in the reference (for categorization).

    Args:
        all_data: {model_name: {env: {region: RegionResult}}}
        reference_data: {env: {region: RegionResult}}
        require_reference_success: If True, only include pairs where reference succeeded

    Returns filtered data and the intersection set.
    """
    # Get keys from each model
    all_keys = [get_env_region_keys(data) for data in all_data.values()]

    # Also require presence in reference (for categorization)
    reference_keys = get_env_region_keys(reference_data)
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
            (env, region) for env, region in intersection
            if reference_data[env][region].success
        }

    # Filter each model's data to intersection
    filtered = {}
    for name, data in all_data.items():
        filtered[name] = {}
        for env, region in intersection:
            if env not in filtered[name]:
                filtered[name][env] = {}
            filtered[name][env][region] = data[env][region]

    return filtered, intersection


def main():
    parser = argparse.ArgumentParser(description="1-Push Evaluation Script")
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
        raise ValueError("Config must specify a 'reference' model for difficulty categorization")

    # Create output directory
    Path(config.output_dir).mkdir(parents=True, exist_ok=True)

    # Load reference data (single or multiple)
    reference_data_list = []
    reference_names = []

    if len(config.references) > 1:
        # Multiple references: consistency analysis mode
        print(f"Loading {len(config.references)} references for consistency analysis...")
        for ref in config.references:
            print(f"  Loading: {ref.name}...")
            ref_data, _ = load_pickle_data(
                f"{ref.dir}/**/*.pkl",
                exclude_easy=config.exclude_easy,
            )
            print(f"    Loaded {sum(len(v) for v in ref_data.values())} triplets")
            reference_data_list.append(ref_data)
            reference_names.append(ref.name)
        # Use first reference as primary
        reference_data = reference_data_list[0]
    else:
        # Single reference (legacy mode)
        print(f"Loading reference data: {config.reference.name}...")
        print(f"  Using triplets (env, region, object) for evaluation granularity")
        reference_data, _ = load_pickle_data(
            f"{config.reference.dir}/**/*.pkl",
            exclude_easy=config.exclude_easy,
        )
        print(f"  Loaded {sum(len(v) for v in reference_data.values())} triplets")
        reference_data_list = [reference_data]
        reference_names = [config.reference.name]

    # Build difficulty mapping (data-driven percentiles)
    difficulty_mapping: Optional[Dict[Tuple[str, str], str]] = None
    difficulty_thresholds: Optional[Dict[str, float]] = None
    consistency_stats: Optional[ConsistencyStats] = None

    if len(reference_data_list) > 1:
        # Multi-reference consistency analysis
        print("\nRunning multi-reference consistency analysis...")
        consistency_data, common_triplets = compute_multi_reference_consistency(
            reference_data_list, reference_names
        )
        categories, thresholds = categorize_by_consistency(consistency_data)
        difficulty_mapping = build_difficulty_mapping(categories)
        difficulty_thresholds = thresholds
        consistency_stats = compute_consistency_stats(
            consistency_data, categories, thresholds, len(reference_data_list)
        )
        print_consistency_report(consistency_stats, categories, "Multi-Reference Consistency")
    else:
        # Single reference: compute difficulty from oracle pushes
        print("\nComputing difficulty categories from oracle push counts...")
        oracle_pushes = []
        problem_keys = []
        for env in reference_data:
            for key in reference_data[env]:
                ref_result = reference_data[env][key]
                if ref_result.success:
                    oracle_pushes.append(ref_result.pushes)
                    problem_keys.append((env, key))

        if oracle_pushes:
            difficulty_thresholds = compute_percentile_thresholds(oracle_pushes)
            difficulty_mapping = {k: assign_difficulty(p, difficulty_thresholds)
                                  for k, p in zip(problem_keys, oracle_pushes)}
            print(f"  Thresholds (data-driven): Easy ≤ {difficulty_thresholds['p33']:.0f}, "
                  f"Medium ≤ {difficulty_thresholds['p66']:.0f} pushes")

    # Load all models (baselines + learned)
    all_model_data: Dict[str, Dict[str, Dict[str, RegionResult]]] = {}
    all_model_failures: Dict[str, Dict[str, int]] = {}

    for baseline in config.baselines:
        print(f"Loading baseline: {baseline.name}...")
        data, failures = load_pickle_data(
            f"{baseline.dir}/**/*.pkl",
            exclude_easy=config.exclude_easy,
        )
        print(f"  Loaded {sum(len(v) for v in data.values())} triplets")
        all_model_data[baseline.name] = data
        all_model_failures[baseline.name] = failures

    for model in config.learned:
        print(f"Loading learned model: {model.name}...")
        data, failures = load_pickle_data(
            f"{model.dir}/**/*.pkl",
            exclude_easy=config.exclude_easy,
        )
        print(f"  Loaded {sum(len(v) for v in data.values())} triplets")
        all_model_data[model.name] = data
        all_model_failures[model.name] = failures

    # Find intersection across all models + reference
    print("\nComputing intersection of triplets across all models...")

    # Debug: Show triplet counts before intersection
    print("\n  === TRIPLET COUNTS BEFORE INTERSECTION ===")
    reference_keys = get_env_region_keys(reference_data)
    reference_success_keys = {(env, region) for env, region in reference_keys
                              if reference_data[env][region].success}
    print(f"  Reference (all):     {len(reference_keys)} triplets")
    print(f"  Reference (success): {len(reference_success_keys)} triplets")

    for name, data in all_model_data.items():
        model_keys = get_env_region_keys(data)
        model_success_keys = {(env, region) for env, region in model_keys
                              if data[env][region].success}
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

    # Count by category using data-driven difficulty_mapping
    category_counts = {'easy': 0, 'medium': 0, 'hard': 0}
    for env, key in intersection:
        if difficulty_mapping and (env, key) in difficulty_mapping:
            difficulty = difficulty_mapping[(env, key)]
            category_counts[difficulty] += 1
    print(f"  By difficulty: Easy={category_counts['easy']}, Medium={category_counts['medium']}, Hard={category_counts['hard']}")

    # Compute stats for each model (using filtered data and data-driven difficulty)
    all_stats: List[ModelStats] = []
    time_data = {}
    push_data = {}
    time_data_learned_only = {}  # For ML-only success tracking
    push_data_learned_only = {}

    for name in filtered_data:
        model_stats = compute_stats(
            filtered_data[name], reference_data,
            name, all_model_failures.get(name, {}),
            difficulty_mapping=difficulty_mapping
        )
        all_stats.append(model_stats)

        # Compute time-based success
        time_data[name] = compute_time_based_success(
            filtered_data[name], reference_data, config,
            difficulty_mapping=difficulty_mapping
        )

        # Compute push-based success
        push_data[name] = compute_push_based_success(
            filtered_data[name], reference_data, config,
            difficulty_mapping=difficulty_mapping
        )

        # Also compute learned-only versions (if solved_in_phase data is available)
        time_data_learned_only[name] = compute_time_based_success(
            filtered_data[name], reference_data, config,
            difficulty_mapping=difficulty_mapping, learned_only=True
        )
        push_data_learned_only[name] = compute_push_based_success(
            filtered_data[name], reference_data, config,
            difficulty_mapping=difficulty_mapping, learned_only=True
        )

    # Compute collision-based success stats
    collision_stats = {}
    for name in filtered_data:
        collision_stats[name] = compute_collision_success_stats(
            filtered_data[name], reference_data
        )

    # Compute collision bucket efficiency
    collision_efficiency = {}
    for name in filtered_data:
        collision_efficiency[name] = compute_collision_bucket_efficiency(
            filtered_data[name], reference_data
        )

    # Compute difficulty stratification (based on oracle pushes)
    difficulty_stratification = {}
    for name in filtered_data:
        difficulty_stratification[name] = compute_difficulty_stratification(
            filtered_data[name], reference_data, difficulty_mapping=difficulty_mapping
        )

    # Compute stats for reference (oracle) - for solutions plot
    reference_stats = compute_stats(
        reference_data, reference_data,
        config.reference.name, {},
        difficulty_mapping=difficulty_mapping
    )

    # Compute new metrics: Hybrid stats, RA@K, Success@Budget/Time
    hybrid_stats = {}
    hybrid_stats_by_difficulty = {}
    ra_at_k_stats = {}
    random_baselines = {}
    success_at_budget = {}
    success_at_time = {}
    success_at_budget_by_diff = {}
    success_at_time_by_diff = {}
    # Learned-only versions (for learned models with solved_in_phase data)
    success_at_budget_learned = {}
    success_at_time_learned = {}
    success_at_budget_by_diff_learned = {}
    success_at_time_by_diff_learned = {}

    learned_model_names = [m.name for m in config.learned]
    learned_model_names_set = set(learned_model_names)

    for name in filtered_data:
        # Hybrid stats (only meaningful for models with solved_in_phase data)
        hs = compute_hybrid_stats(filtered_data[name], reference_data)
        if hs.solved_by_learned > 0 or hs.solved_by_fallback > 0:
            hybrid_stats[name] = hs
            hybrid_stats_by_difficulty[name] = compute_hybrid_stats_by_difficulty(
                filtered_data[name], reference_data, difficulty_mapping=difficulty_mapping
            )

        # RA@K stats (only for learned models with aligned_primitives data)
        if name in learned_model_names:
            # Check if model has aligned_primitives data
            has_aligned = False
            for env_data in filtered_data[name].values():
                for result in env_data.values():
                    if result.aligned_primitives:
                        has_aligned = True
                        break
                if has_aligned:
                    break

            if has_aligned:
                ra_at_k_stats[name] = {}
                for k in config.ra_at_k_values:
                    ra_at_k_stats[name][k] = compute_ra_at_k(
                        filtered_data[name], reference_data, k=k
                    )
                random_baselines[name] = compute_random_baseline(
                    filtered_data[name], reference_data
                )

        # Success@Budget and Success@Time
        success_at_budget[name] = compute_success_at_budget(
            filtered_data[name], reference_data, config.success_at_budget_values
        )
        success_at_time[name] = compute_success_at_time_budget(
            filtered_data[name], reference_data, config.success_at_time_values
        )
        success_at_budget_by_diff[name] = compute_success_at_budget_by_difficulty(
            filtered_data[name], reference_data, config.success_at_budget_values,
            difficulty_mapping=difficulty_mapping
        )
        success_at_time_by_diff[name] = compute_success_at_time_budget_by_difficulty(
            filtered_data[name], reference_data, config.success_at_time_values,
            difficulty_mapping=difficulty_mapping
        )

        # Learned-only versions (for learned models with hybrid stats)
        if name in learned_model_names_set and name in hybrid_stats:
            success_at_budget_learned[name] = compute_success_at_budget(
                filtered_data[name], reference_data, config.success_at_budget_values,
                learned_only=True
            )
            success_at_time_learned[name] = compute_success_at_time_budget(
                filtered_data[name], reference_data, config.success_at_time_values,
                learned_only=True
            )
            success_at_budget_by_diff_learned[name] = compute_success_at_budget_by_difficulty(
                filtered_data[name], reference_data, config.success_at_budget_values,
                difficulty_mapping=difficulty_mapping,
                learned_only=True
            )
            success_at_time_by_diff_learned[name] = compute_success_at_time_budget_by_difficulty(
                filtered_data[name], reference_data, config.success_at_time_values,
                difficulty_mapping=difficulty_mapping,
                learned_only=True
            )

    # Print summary
    print_summary(all_stats, hybrid_stats=hybrid_stats, learned_model_names=learned_model_names_set)

    # Print collision-based success rates
    print("\n" + "=" * 80)
    print("SUCCESS RATE BY COLLISION TYPE (based on oracle solution)")
    print("=" * 80)
    collision_types = ['none', 'wall_only', 'movable_only', 'both']
    collision_labels = {'none': 'No Collision', 'wall_only': 'Wall Only',
                        'movable_only': 'Movable Only', 'both': 'Both'}

    # Print oracle collision distribution (use first model's totals - same for all)
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
    print("DIFFICULTY STRATIFICATION (based on oracle push counts)")
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

        # Print success rates per model with efficiency metrics
        for name in difficulty_stratification:
            print(f"\n{name}:")
            for diff in difficulty_levels:
                stats = difficulty_stratification[name][diff]
                rate = stats['successes'] / stats['total'] if stats['total'] > 0 else 0.0
                efficiency_str = ""
                if stats['pushes'] and stats['times']:
                    median_pushes = np.median(stats['pushes'])
                    median_time_ms = np.median(stats['times'])
                    efficiency_str = f", median: {median_pushes:.0f} pushes, {median_time_ms/1000:.1f}s"
                print(f"  {difficulty_labels_print[diff]:8s}: {stats['successes']:3d}/{stats['total']:3d} = {rate:.1%}{efficiency_str}")

    # Print hybrid stats (diffusion-only vs fallback decomposition)
    if hybrid_stats:
        print("\n" + "=" * 80)
        print("SAGE (HYBRID): DIFFUSION-ONLY VS FALLBACK")
        print("=" * 80)
        print("Phase tracking: solved_in_phase == 'ML-only' → Diffusion-Only, 'primitives' → Fallback")

        for name, hs in hybrid_stats.items():
            if hs.total == 0:
                continue
            print(f"\n{name} (n={hs.total}):")
            print(f"  Diffusion-Only: {hs.solved_by_learned:3d} ({hs.learned_rate:.1%})")
            print(f"  Fallback:       {hs.solved_by_fallback:3d} ({hs.fallback_rate:.1%})")
            print(f"  Failed:             {hs.failed:3d} ({(1-hs.success_rate):.1%})")

            if hs.learned_pushes:
                l_iqr = hs.learned_pushes_iqr
                print(f"  Diff-Only pushes:   median={hs.learned_median_pushes:.0f} [{l_iqr[0]:.0f}, {l_iqr[1]:.0f}]")
            if hs.learned_times:
                lt_iqr = hs.learned_time_iqr
                print(f"  Diff-Only time:     median={hs.learned_median_time/1000:.1f}s [{lt_iqr[0]/1000:.1f}, {lt_iqr[1]/1000:.1f}]")
            if hs.fallback_pushes:
                f_iqr = hs.fallback_pushes_iqr
                print(f"  Fallback pushes:    median={hs.fallback_median_pushes:.0f} [{f_iqr[0]:.0f}, {f_iqr[1]:.0f}]")
            if hs.fallback_times:
                ft_iqr = hs.fallback_time_iqr
                print(f"  Fallback time:      median={hs.fallback_median_time/1000:.1f}s [{ft_iqr[0]/1000:.1f}, {ft_iqr[1]/1000:.1f}]")
            if hs.checks_before_fallback:
                bf_iqr = hs.checks_before_fallback_iqr
                print(f"  Checks before FB:   median={hs.median_checks_before_fallback:.0f} [{bf_iqr[0]:.0f}, {bf_iqr[1]:.0f}]")

        # Print hybrid stats by difficulty
        if hybrid_stats_by_difficulty:
            print("\n  By Difficulty:")
            for name in hybrid_stats_by_difficulty:
                print(f"\n  {name}:")
                for diff in difficulty_levels:
                    stats = hybrid_stats_by_difficulty[name][diff]
                    n = stats['total']
                    if n > 0:
                        diffonly_pct = stats['learned'] / n * 100
                        fallback_pct = stats['fallback'] / n * 100
                        failed_pct = stats['failed'] / n * 100
                        print(f"    {difficulty_labels_print[diff]:8s} (N={n:2d}): Diff-Only={diffonly_pct:5.1f}% ({stats['learned']:2d}), "
                              f"Fallback={fallback_pct:5.1f}% ({stats['fallback']:2d}), Failed={failed_pct:5.1f}% ({stats['failed']:2d})")
                    else:
                        print(f"    {difficulty_labels_print[diff]:8s} (N= 0): -")

    # Print RA@K stats (SAGE models only)
    if ra_at_k_stats:
        print("\n" + "=" * 80)
        print("REACHABLE ATTACHMENT @ K")
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
            if name in random_baselines:
                print(f"  Random baseline: {random_baselines[name]:.1%}")
            print()

    # Print Success@Budget
    if success_at_budget:
        print("\n" + "=" * 80)
        print("SUCCESS @ BUDGET")
        print("=" * 80)
        print("Success rate at fixed verification budget (constant-compute comparison)")
        print("Budget = max number of simulation-verified push evaluations")
        print("Format: Total (Diffusion-Only) for SAGE models\n")

        budget_strs = [f"@{b}" for b in config.success_at_budget_values]
        header = f"{'Model':<30} | " + " | ".join(f"{s:>16}" for s in budget_strs)
        print(header)
        print("-" * len(header))

        for name in success_at_budget:
            row = f"{name:<30} |"
            for b in config.success_at_budget_values:
                stats = success_at_budget[name][b]
                if name in success_at_budget_learned:
                    learned_stats = success_at_budget_learned[name][b]
                    row += f" {stats['rate']:>5.1%} ({learned_stats['rate']:>5.1%}) |"
                else:
                    row += f" {stats['rate']:>15.1%} |"
            print(row)

        first_model = list(success_at_budget.keys())[0]
        n_total = success_at_budget[first_model][config.success_at_budget_values[0]]['total']
        print(f"\n(N={n_total} problems)")

    # Print Success@Time
    if success_at_time:
        print("\n" + "=" * 80)
        print("SUCCESS @ TIME")
        print("=" * 80)
        print("Success rate at fixed time budget (constant-time comparison)")
        print("Format: Total (Diffusion-Only) for SAGE models\n")

        def format_time(t_ms: float) -> str:
            if t_ms >= 1000:
                return f"@{t_ms/1000:.0f}s"
            return f"@{t_ms:.0f}ms"

        time_strs = [format_time(t) for t in config.success_at_time_values]
        header = f"{'Model':<30} | " + " | ".join(f"{s:>16}" for s in time_strs)
        print(header)
        print("-" * len(header))

        for name in success_at_time:
            row = f"{name:<30} |"
            for t in config.success_at_time_values:
                stats = success_at_time[name][t]
                if name in success_at_time_learned:
                    learned_stats = success_at_time_learned[name][t]
                    row += f" {stats['rate']:>5.1%} ({learned_stats['rate']:>5.1%}) |"
                else:
                    row += f" {stats['rate']:>15.1%} |"
            print(row)

        first_model = list(success_at_time.keys())[0]
        n_total = success_at_time[first_model][config.success_at_time_values[0]]['total']
        print(f"\n(N={n_total} problems)")

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

        plot_solutions_distribution(
            [reference_stats],  # Only oracle/reference
            config,
            f"{config.output_dir}/solutions_distribution.png"
        )

    if time_data:
        plot_time_vs_success(
            time_data,
            config,
            f"{config.output_dir}/time_vs_success.png"
        )

    if push_data:
        plot_pushes_vs_success(
            push_data,
            config,
            f"{config.output_dir}/pushes_vs_success.png"
        )

    if all_stats:
        plot_wall_collision_rate(
            all_stats,
            config,
            f"{config.output_dir}/wall_collision_rate.png"
        )
        plot_movable_collision_rate(
            all_stats,
            config,
            f"{config.output_dir}/movable_collision_rate.png"
        )

    if collision_stats:
        plot_collision_success_rates(
            collision_stats,
            config,
            f"{config.output_dir}/collision_success_rates.png"
        )

    # Generate markdown report
    generate_markdown_report(
        all_stats,
        config,
        category_counts,
        f"{config.output_dir}/results.md",
        collision_efficiency=collision_efficiency,
        difficulty_stratification=difficulty_stratification,
        hybrid_stats=hybrid_stats,
        hybrid_stats_by_difficulty=hybrid_stats_by_difficulty,
        ra_at_k_stats=ra_at_k_stats,
        random_baselines=random_baselines,
        success_at_budget=success_at_budget,
        success_at_time=success_at_time,
        difficulty_thresholds=difficulty_thresholds,
        success_at_budget_learned=success_at_budget_learned,
        success_at_time_learned=success_at_time_learned,
        learned_model_names=learned_model_names_set,
    )

    print(f"\nPlots saved to: {config.output_dir}")

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
