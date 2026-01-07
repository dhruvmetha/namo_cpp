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
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict

import yaml
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

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
    """Configuration for a single model/baseline."""
    name: str
    dir: str
    color: Optional[str] = None


@dataclass
class EvalConfig:
    """Configuration for evaluation."""
    # Reference model (determines difficulty categorization)
    reference: ModelConfig = None

    # Baselines (non-learned)
    baselines: List[ModelConfig] = field(default_factory=list)

    # Learned models
    learned: List[ModelConfig] = field(default_factory=list)

    # Filtering
    exclude_easy: bool = True

    # Thresholds for difficulty categorization (based on reference success ratio)
    easy_threshold: float = 0.75
    hard_threshold: float = 0.25

    # Plot settings
    output_dir: str = "./eval_plots"
    time_cutoff_max: int = 6000  # ms
    time_step: int = 100  # ms
    push_cutoff_max: int = 10  # max number of pushes
    push_step: int = 1  # step size for push cutoffs

    # Colors for models (will cycle if more models than colors)
    # Using a colorblind-friendly palette
    model_colors: List[str] = field(default_factory=lambda: [
        '#4C72B0',  # muted blue
        '#DD8452',  # muted orange
        '#55A868',  # muted green (okay as accent, not with red)
        '#C44E52',  # muted red (okay as accent, not with green)
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

        # Parse reference
        if 'reference' in data:
            ref = data['reference']
            config.reference = ModelConfig(
                name=ref.get('name', 'Reference'),
                dir=ref['dir'],
                color=ref.get('color'),
            )

        # Parse baselines
        if 'baselines' in data:
            for b in data['baselines']:
                config.baselines.append(ModelConfig(
                    name=b.get('name', 'Baseline'),
                    dir=b['dir'],
                    color=b.get('color'),
                ))

        # Parse learned models
        if 'learned' in data:
            for m in data['learned']:
                config.learned.append(ModelConfig(
                    name=m.get('name', 'Learned'),
                    dir=m['dir'],
                    color=m.get('color'),
                ))

        # Parse settings
        if 'settings' in data:
            settings = data['settings']
            config.exclude_easy = settings.get('exclude_easy', config.exclude_easy)
            config.easy_threshold = settings.get('easy_threshold', config.easy_threshold)
            config.hard_threshold = settings.get('hard_threshold', config.hard_threshold)
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
    """Results for a single env+region pair."""
    success: bool = False
    pushes: int = 0
    solutions: int = 0  # solutions_total_for_neighbour (for ratio/categorization)
    solutions_found: int = 0  # solutions_found_for_neighbour (for distribution)
    ratio: float = 0.0
    time_taken: float = 0.0
    failure_reason: str = ""
    ml_goals_raw: List[Any] = field(default_factory=list)
    search_solutions: List[Any] = field(default_factory=list)
    # Interaction types
    wall_collision: bool = False
    movable_collisions: int = 0


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

                result = RegionResult(
                    success=solution_found and pushes > 0,
                    pushes=pushes,
                    solutions=solutions,
                    solutions_found=solutions_found,
                    ratio=solutions / pushes if pushes > 0 else 0.0,
                    time_taken=time_taken,
                    failure_reason=failure_reason,
                    ml_goals_raw=alg_stats.get('ml_goals_raw', []),
                    search_solutions=ep.get('search_solutions', []),
                    wall_collision=wall_collision,
                    movable_collisions=movable_collisions,
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
class CategoryStats:
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


@dataclass
class ModelStats:
    """Statistics for a model across all categories."""
    name: str
    easy: CategoryStats = field(default_factory=CategoryStats)
    medium: CategoryStats = field(default_factory=CategoryStats)
    hard: CategoryStats = field(default_factory=CategoryStats)
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
    config: EvalConfig,
    model_name: str,
    failure_reasons: Optional[Dict[str, int]] = None,
) -> ModelStats:
    """
    Compute statistics for a model, categorized by difficulty.

    Difficulty is determined by the search (oracle) success ratio.
    """
    stats = ModelStats(name=model_name)
    if failure_reasons:
        stats.failure_reasons = failure_reasons

    # Only consider env+region pairs in both model and search
    for env in model_data:
        if env not in search_data:
            continue

        for region in model_data[env]:
            if region not in search_data[env]:
                continue

            search_result = search_data[env][region]
            model_result = model_data[env][region]

            # Determine category based on search ratio
            if search_result.ratio > config.easy_threshold:
                category = stats.easy
            elif search_result.ratio > config.hard_threshold:
                category = stats.medium
            else:
                category = stats.hard

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

    return stats


def compute_time_based_success(
    model_data: Dict[str, Dict[str, RegionResult]],
    search_data: Dict[str, Dict[str, RegionResult]],
    config: EvalConfig,
) -> Dict[str, Dict[str, List[float]]]:
    """
    Compute success rate as a function of time cutoff.

    Returns:
        {category: {'cutoffs': [...], 'rates': [...]}}
    """
    cutoffs = np.arange(0, config.time_cutoff_max + config.time_step, config.time_step)

    # Collect times by category
    times_by_category: Dict[str, List[float]] = {'easy': [], 'medium': [], 'hard': []}
    totals_by_category: Dict[str, int] = {'easy': 0, 'medium': 0, 'hard': 0}

    for env in model_data:
        if env not in search_data:
            continue
        for region in model_data[env]:
            if region not in search_data[env]:
                continue

            search_result = search_data[env][region]
            model_result = model_data[env][region]

            # Determine category
            if search_result.ratio > config.easy_threshold:
                cat = 'easy'
            elif search_result.ratio > config.hard_threshold:
                cat = 'medium'
            else:
                cat = 'hard'

            totals_by_category[cat] += 1
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
        result[cat] = {'cutoffs': cutoffs.tolist(), 'rates': rates}

    return result


def compute_push_based_success(
    model_data: Dict[str, Dict[str, RegionResult]],
    search_data: Dict[str, Dict[str, RegionResult]],
    config: EvalConfig,
) -> Dict[str, Dict[str, List[float]]]:
    """
    Compute success rate as a function of push count cutoff.

    Returns:
        {category: {'cutoffs': [...], 'rates': [...], 'total': int}}
    """
    cutoffs = list(range(0, config.push_cutoff_max + 1, config.push_step))

    # Collect pushes by category
    pushes_by_category: Dict[str, List[int]] = {'easy': [], 'medium': [], 'hard': []}
    totals_by_category: Dict[str, int] = {'easy': 0, 'medium': 0, 'hard': 0}

    for env in model_data:
        if env not in search_data:
            continue
        for region in model_data[env]:
            if region not in search_data[env]:
                continue

            search_result = search_data[env][region]
            model_result = model_data[env][region]

            # Determine category
            if search_result.ratio > config.easy_threshold:
                cat = 'easy'
            elif search_result.ratio > config.hard_threshold:
                cat = 'medium'
            else:
                cat = 'hard'

            totals_by_category[cat] += 1
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
    config: EvalConfig,
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
    stats = {
        'none': {'successes': 0, 'total': 0},
        'wall_only': {'successes': 0, 'total': 0},
        'movable_only': {'successes': 0, 'total': 0},
        'both': {'successes': 0, 'total': 0},
    }

    for env in model_data:
        if env not in search_data:
            continue
        for region in model_data[env]:
            if region not in search_data[env]:
                continue

            search_result = search_data[env][region]
            model_result = model_data[env][region]

            # Only consider cases where search succeeded (solvable problems)
            if not search_result.success:
                continue

            # Determine collision category based on search (oracle) result
            has_wall = search_result.wall_collision
            has_movable = search_result.movable_collisions > 0

            if has_wall and has_movable:
                cat = 'both'
            elif has_wall:
                cat = 'wall_only'
            elif has_movable:
                cat = 'movable_only'
            else:
                cat = 'none'

            stats[cat]['total'] += 1
            if model_result.success:
                stats[cat]['successes'] += 1

    return stats


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
    ax.set_title('Success Rate by Difficulty')
    ax.legend(loc='upper right', frameon=True, fancybox=True)
    ax.axhline(y=1.0, color='#888888', linestyle='--', linewidth=0.8, alpha=0.5)

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
    ax.legend(legend_handles, [s.name for s in model_stats], loc='upper left', frameon=True, fancybox=True)

    ax.set_ylabel('Pushes to Success')
    ax.set_title('Pushes to Success by Difficulty')

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white')
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
    ax.legend(legend_handles, [s.name for s in model_stats], loc='upper left', frameon=True, fancybox=True)

    ax.set_ylabel('Time to Success (ms)')
    ax.set_title('Time to Success by Difficulty')

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white')
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
        ax.legend()

    plt.suptitle('Distribution of Total Solutions per Category (Oracle Search)', fontsize=14)
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {output_path}")
    return fig


def plot_time_vs_success(
    time_data: Dict[str, Dict[str, Dict[str, List[float]]]],  # {model_name: {category: {cutoffs, rates}}}
    config: EvalConfig,
    output_path: Optional[str] = None,
):
    """Plot success rate vs time cutoff."""
    categories = ['easy', 'medium', 'hard']

    # Get N for each category from the first model
    n_by_cat = {}
    for cat_data in time_data.values():
        for cat in categories:
            if cat in cat_data and cat not in n_by_cat:
                n_by_cat[cat] = cat_data[cat].get('total', 0)
        break

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)

    for idx, cat in enumerate(categories):
        ax = axes[idx]

        for model_idx, (model_name, cat_data) in enumerate(time_data.items()):
            if cat in cat_data:
                cutoffs_ms = cat_data[cat]['cutoffs']
                cutoffs_s = [c / 1000.0 for c in cutoffs_ms]  # Convert to seconds
                rates = cat_data[cat]['rates']
                ax.plot(cutoffs_s, rates, label=model_name,
                       color=get_model_color(model_idx, config), linewidth=2)

        n_problems = n_by_cat.get(cat, 0)
        ax.set_title(f"{cat.capitalize()} Regions (N={n_problems})")
        ax.set_xlabel('Time cutoff (s)')
        if idx == 0:
            ax.set_ylabel('Success Rate')
        ax.set_ylim(0, 1.05)
        ax.set_xlim(0, config.time_cutoff_max / 1000.0)
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend()

    plt.suptitle("Success Rate @ Time Cutoff", fontsize=14, fontweight='bold')
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white')
        print(f"Saved: {output_path}")
    return fig


def plot_pushes_vs_success(
    push_data: Dict[str, Dict[str, Dict[str, List[float]]]],  # {model_name: {category: {cutoffs, rates, total}}}
    config: EvalConfig,
    output_path: Optional[str] = None,
):
    """Plot success rate vs push count cutoff."""
    categories = ['easy', 'medium', 'hard']

    # Get N for each category from the first model
    n_by_cat = {}
    for cat_data in push_data.values():
        for cat in categories:
            if cat in cat_data and cat not in n_by_cat:
                n_by_cat[cat] = cat_data[cat].get('total', 0)
        break

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)

    for idx, cat in enumerate(categories):
        ax = axes[idx]

        for model_idx, (model_name, cat_data) in enumerate(push_data.items()):
            if cat in cat_data:
                cutoffs = cat_data[cat]['cutoffs']
                rates = cat_data[cat]['rates']
                ax.plot(cutoffs, rates, label=model_name,
                       color=get_model_color(model_idx, config), linewidth=2)

        n_problems = n_by_cat.get(cat, 0)
        ax.set_title(f"{cat.capitalize()} Regions (N={n_problems})")
        ax.set_xlabel('Simulation-verified push evaluations')
        if idx == 0:
            ax.set_ylabel('Success Rate')
        ax.set_ylim(0, 1.05)
        ax.set_xlim(0, config.push_cutoff_max)
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend(loc='lower right')

    plt.suptitle("Success Rate @ Push Evaluations", fontsize=14, fontweight='bold')
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white')
        print(f"Saved: {output_path}")
    return fig


def plot_interactions(
    model_stats: List[ModelStats],
    config: EvalConfig,
    output_path: Optional[str] = None,
):
    """Plot interaction statistics (wall and movable collisions)."""
    categories = ['easy', 'medium', 'hard']
    n_models = len(model_stats)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Plot 1: Wall collision rate
    ax1 = axes[0]
    x = np.arange(len(categories))
    width = 0.8 / n_models

    for i, stats in enumerate(model_stats):
        rates = [stats.get_category(cat).wall_collision_rate for cat in categories]
        offset = (i - n_models/2 + 0.5) * width
        bars = ax1.bar(x + offset, rates, width, label=stats.name,
                      color=get_model_color(i, config), edgecolor='black')

        for bar, rate in zip(bars, rates):
            if rate > 0:
                ax1.annotate(f'{rate:.0%}',
                           xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                           xytext=(0, 3), textcoords="offset points",
                           ha='center', va='bottom', fontsize=9)

    ax1.set_xticks(x)
    ax1.set_xticklabels([c.capitalize() for c in categories])
    ax1.set_ylim(0, 1.15)
    ax1.set_ylabel('Wall Collision Rate')
    ax1.set_title('Wall Collision Rate by Category\n(among successful runs)')
    ax1.legend(loc='upper right')
    ax1.grid(True, axis='y', linestyle='--', alpha=0.6)

    # Plot 2: Movable collision rate (any collision)
    ax2 = axes[1]

    for i, stats in enumerate(model_stats):
        rates = [stats.get_category(cat).any_movable_collision_rate for cat in categories]
        offset = (i - n_models/2 + 0.5) * width
        bars = ax2.bar(x + offset, rates, width, label=stats.name,
                      color=get_model_color(i, config), edgecolor='black')

        for bar, rate in zip(bars, rates):
            if rate > 0:
                ax2.annotate(f'{rate:.0%}',
                           xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                           xytext=(0, 3), textcoords="offset points",
                           ha='center', va='bottom', fontsize=9)

    ax2.set_xticks(x)
    ax2.set_xticklabels([c.capitalize() for c in categories])
    ax2.set_ylim(0, 1.15)
    ax2.set_ylabel('Movable Collision Rate')
    ax2.set_title('Movable Object Collision Rate by Category\n(among successful runs)')
    ax2.legend(loc='upper right')
    ax2.grid(True, axis='y', linestyle='--', alpha=0.6)

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
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
    ax.set_title('Success Rate by Collision Type Required')
    ax.legend(loc='upper right', frameon=True, fancybox=True, shadow=False)

    # Add horizontal line at 100%
    ax.axhline(y=1.0, color='#888888', linestyle='--', linewidth=0.8, alpha=0.5)

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white')
        print(f"Saved: {output_path}")
    return fig


def print_summary(model_stats: List[ModelStats]):
    """Print summary statistics."""
    print("\n" + "=" * 80)
    print("EVALUATION SUMMARY")
    print("=" * 80)

    categories = ['easy', 'medium', 'hard']

    for stats in model_stats:
        print(f"\n{'─' * 40}")
        print(f"Model: {stats.name}")
        print(f"{'─' * 40}")
        print(f"Overall: {stats.total_successes}/{stats.total_trials} = {stats.overall_success_rate:.4f}")

        for cat in categories:
            cat_stats = stats.get_category(cat)
            print(f"\n  {cat.capitalize()}:")
            print(f"    Success: {cat_stats.successes}/{cat_stats.total} = {cat_stats.success_rate:.4f}")
            if cat_stats.pushes:
                print(f"    Pushes:  median={cat_stats.median_pushes:.1f}, mean={cat_stats.mean_pushes:.1f}")
            if cat_stats.times:
                print(f"    Time:    median={cat_stats.median_time:.1f}ms, mean={cat_stats.mean_time:.1f}ms")
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
):
    """Generate a markdown report with comparison tables."""
    categories = ['easy', 'medium', 'hard']

    lines = []
    lines.append("# 1-Push Evaluation Results\n")
    lines.append(f"Generated from evaluation config.\n")

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
    lines.append("### Median Pushes\n")
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
                row += f" {cat_stats.median_pushes:.1f} |"
            else:
                row += " - |"
        lines.append(row)
    lines.append("")

    lines.append("### Mean Pushes\n")
    lines.append(header)
    lines.append(separator)

    for stats in model_stats:
        row = f"| {stats.name} |"
        for cat in categories:
            cat_stats = stats.get_category(cat)
            if cat_stats.pushes:
                row += f" {cat_stats.mean_pushes:.1f} |"
            else:
                row += " - |"
        lines.append(row)
    lines.append("")

    # Time statistics (successful runs only)
    lines.append("## Time to Success in ms (Successful Runs Only)\n")
    lines.append("### Median Time\n")
    lines.append(header)
    lines.append(separator)

    for stats in model_stats:
        row = f"| {stats.name} |"
        for cat in categories:
            cat_stats = stats.get_category(cat)
            if cat_stats.times:
                row += f" {cat_stats.median_time:.0f} |"
            else:
                row += " - |"
        lines.append(row)
    lines.append("")

    lines.append("### Mean Time\n")
    lines.append(header)
    lines.append(separator)

    for stats in model_stats:
        row = f"| {stats.name} |"
        for cat in categories:
            cat_stats = stats.get_category(cat)
            if cat_stats.times:
                row += f" {cat_stats.mean_time:.0f} |"
            else:
                row += " - |"
        lines.append(row)
    lines.append("")

    # Interaction statistics
    lines.append("## Interaction Statistics (Successful Runs Only)\n")
    lines.append("These metrics show collision rates among successful runs.\n")

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
    if config.reference is None:
        raise ValueError("Config must specify a 'reference' model for difficulty categorization")

    # Create output directory
    Path(config.output_dir).mkdir(parents=True, exist_ok=True)

    # Load reference data (for difficulty categorization only)
    print(f"Loading reference data: {config.reference.name} (for categorization)...")
    print(f"  Using triplets (env, region, object) for evaluation granularity")
    reference_data, _ = load_pickle_data(
        f"{config.reference.dir}/**/*.pkl",
        exclude_easy=config.exclude_easy,
    )
    print(f"  Loaded {sum(len(v) for v in reference_data.values())} triplets")

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

    # Count by category
    category_counts = {'easy': 0, 'medium': 0, 'hard': 0}
    for env, region in intersection:
        if env in reference_data and region in reference_data[env]:
            ratio = reference_data[env][region].ratio
            if ratio > config.easy_threshold:
                category_counts['easy'] += 1
            elif ratio > config.hard_threshold:
                category_counts['medium'] += 1
            else:
                category_counts['hard'] += 1
    print(f"  By category: Easy={category_counts['easy']}, Medium={category_counts['medium']}, Hard={category_counts['hard']}")

    # Compute stats for each model (using filtered data)
    all_stats: List[ModelStats] = []
    time_data = {}
    push_data = {}

    for name in filtered_data:
        model_stats = compute_stats(
            filtered_data[name], reference_data, config,
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

    # Compute collision-based success stats
    collision_stats = {}
    for name in filtered_data:
        collision_stats[name] = compute_collision_success_stats(
            filtered_data[name], reference_data, config
        )

    # Compute stats for reference (oracle) - for solutions plot
    reference_stats = compute_stats(
        reference_data, reference_data, config,
        config.reference.name, {}
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

    # Generate markdown report
    generate_markdown_report(
        all_stats,
        config,
        category_counts,
        f"{config.output_dir}/results.md"
    )

    print(f"\nPlots saved to: {config.output_dir}")

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
