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

    # Colors for models (will cycle if more models than colors)
    model_colors: List[str] = field(default_factory=lambda: [
        '#2ecc71',  # green
        '#e74c3c',  # red
        '#3498db',  # blue
        '#9b59b6',  # purple
        '#f39c12',  # orange
        '#1abc9c',  # teal
        '#e67e22',  # dark orange
        '#16a085',  # dark teal
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
        reference_data: If provided, only include env+region pairs that exist in reference

    Returns:
        per_env_per_region: {xml_file_name: {region_label: RegionResult}}
        failure_reasons: {reason: count}
    """
    per_env_per_region: Dict[str, Dict[str, RegionResult]] = {}
    failure_reasons: Dict[str, int] = defaultdict(int)

    for file in glob(data_dir, recursive=True):
        try:
            with open(file, "rb") as f:
                data = pickle.load(f)

            episode_results = data.get('episode_results', [])
            if not episode_results:
                continue

            region_done = set()

            for ep in episode_results:
                xml_file = ep.get('xml_file', '')
                xml_file_name = "_".join(xml_file.split('/')[-4:])

                if exclude_easy and "easy" in xml_file_name:
                    continue

                alg_stats = ep.get('algorithm_stats', {})
                region_label = alg_stats.get('neighbour_region_label')

                if region_label is None:
                    continue

                # If reference provided, only include matching pairs
                if reference_data is not None:
                    if xml_file_name not in reference_data:
                        continue
                    if region_label not in reference_data[xml_file_name]:
                        continue
                    if not reference_data[xml_file_name][region_label].success:
                        continue

                # Track failure reasons
                failure_reason = alg_stats.get('failure_reason', 'unknown')
                failure_reasons[failure_reason] += 1

                # Only process each region once per file
                if region_label in region_done:
                    continue
                region_done.add(region_label)

                # Initialize env dict if needed
                if xml_file_name not in per_env_per_region:
                    per_env_per_region[xml_file_name] = {}

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

                per_env_per_region[xml_file_name][region_label] = result

        except Exception as e:
            print(f"Error loading {file}: {e}")
            continue

    return per_env_per_region, dict(failure_reasons)


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
    width = 0.8 / n_models

    fig, ax = plt.subplots(figsize=(12, 10))

    for i, stats in enumerate(model_stats):
        rates = [stats.get_category(cat).success_rate for cat in categories]

        offset = (i - n_models/2 + 0.5) * width
        bars = ax.bar(x + offset, rates, width, label=stats.name,
                      color=get_model_color(i, config), edgecolor='black')

        # Add percentage labels
        for bar, rate in zip(bars, rates):
            ax.annotate(f'{rate:.0%}',
                       xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                       xytext=(0, 3), textcoords="offset points",
                       ha='center', va='bottom', fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels([c.capitalize() for c in categories])
    ax.set_ylim(0, 1.15)
    ax.set_ylabel('Success Rate')
    ax.set_title('Success Rate by Difficulty Category')
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.08), ncol=n_models, frameon=True)
    ax.grid(True, axis='y', linestyle='--', alpha=0.6)

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
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

    fig, ax = plt.subplots(figsize=(12, 8))

    positions = []
    data = []
    colors = []
    labels_added = set()

    for cat_idx, cat in enumerate(categories):
        for model_idx, stats in enumerate(model_stats):
            pos = cat_idx * (n_models + 1) + model_idx
            positions.append(pos)
            data.append(stats.get_category(cat).pushes or [0])
            colors.append(get_model_color(model_idx, config))

    bp = ax.boxplot(data, positions=positions, widths=0.6, patch_artist=True, showfliers=False)

    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    for median in bp['medians']:
        median.set_color('red')
        median.set_linewidth(2)

    # Set x-axis labels
    cat_positions = [(i * (n_models + 1) + (n_models - 1) / 2) for i in range(len(categories))]
    ax.set_xticks(cat_positions)
    ax.set_xticklabels([c.capitalize() for c in categories])

    # Legend
    legend_handles = [plt.Rectangle((0,0),1,1, facecolor=get_model_color(i, config), alpha=0.7)
                      for i in range(n_models)]
    ax.legend(legend_handles, [s.name for s in model_stats], loc='upper right')

    ax.set_ylabel('Pushes to Success')
    ax.set_title('Pushes to Success by Difficulty Category')
    ax.grid(True, axis='y', linestyle='--', alpha=0.6)

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150)
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

    fig, ax = plt.subplots(figsize=(12, 8))

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
        patch.set_alpha(0.7)

    for median in bp['medians']:
        median.set_color('red')
        median.set_linewidth(2)

    cat_positions = [(i * (n_models + 1) + (n_models - 1) / 2) for i in range(len(categories))]
    ax.set_xticks(cat_positions)
    ax.set_xticklabels([c.capitalize() for c in categories])

    legend_handles = [plt.Rectangle((0,0),1,1, facecolor=get_model_color(i, config), alpha=0.7)
                      for i in range(n_models)]
    ax.legend(legend_handles, [s.name for s in model_stats], loc='upper right')

    ax.set_ylabel('Time to Success (ms)')
    ax.set_title('Time to Success by Difficulty Category')
    ax.grid(True, axis='y', linestyle='--', alpha=0.6)

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150)
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

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)

    for idx, cat in enumerate(categories):
        ax = axes[idx]

        for model_idx, (model_name, cat_data) in enumerate(time_data.items()):
            if cat in cat_data:
                cutoffs = cat_data[cat]['cutoffs']
                rates = cat_data[cat]['rates']
                ax.plot(cutoffs, rates, label=model_name,
                       color=get_model_color(model_idx, config), linewidth=2)

        ax.set_title(f"{cat.capitalize()} Regions")
        ax.set_xlabel('Time Cutoff (ms)')
        if idx == 0:
            ax.set_ylabel('Success Rate')
        ax.set_ylim(0, 1.05)
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend()

    plt.suptitle("Success Rate @ Time Cutoff")
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150)
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
    reference_data, _ = load_pickle_data(
        f"{config.reference.dir}/**/*.pkl",
        exclude_easy=config.exclude_easy,
    )
    print(f"  Loaded {sum(len(v) for v in reference_data.values())} env+region pairs")

    # Load all models (baselines + learned)
    all_model_data: Dict[str, Dict[str, Dict[str, RegionResult]]] = {}
    all_model_failures: Dict[str, Dict[str, int]] = {}

    for baseline in config.baselines:
        print(f"Loading baseline: {baseline.name}...")
        data, failures = load_pickle_data(
            f"{baseline.dir}/**/*.pkl",
            exclude_easy=config.exclude_easy,
        )
        print(f"  Loaded {sum(len(v) for v in data.values())} env+region pairs")
        all_model_data[baseline.name] = data
        all_model_failures[baseline.name] = failures

    for model in config.learned:
        print(f"Loading learned model: {model.name}...")
        data, failures = load_pickle_data(
            f"{model.dir}/**/*.pkl",
            exclude_easy=config.exclude_easy,
        )
        print(f"  Loaded {sum(len(v) for v in data.values())} env+region pairs")
        all_model_data[model.name] = data
        all_model_failures[model.name] = failures

    # Find intersection across all models + reference
    print("\nComputing intersection of env+region pairs across all models...")
    filtered_data, intersection = filter_to_intersection(all_model_data, reference_data)
    print(f"  Intersection size: {len(intersection)} env+region pairs")

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

    # Compute stats for reference (oracle) - for solutions plot
    reference_stats = compute_stats(
        reference_data, reference_data, config,
        config.reference.name, {}
    )

    # Print summary
    print_summary(all_stats)

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

    if all_stats:
        plot_interactions(
            all_stats,
            config,
            f"{config.output_dir}/interactions.png"
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
