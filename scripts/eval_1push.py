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
    solutions: int = 0
    ratio: float = 0.0
    time_taken: float = 0.0
    failure_reason: str = ""
    ml_goals_raw: List[Any] = field(default_factory=list)
    search_solutions: List[Any] = field(default_factory=list)


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
                solution_found = ep.get('solution_found', False)
                time_taken = ep.get('search_time_ms', 0)

                result = RegionResult(
                    success=solution_found and pushes > 0,
                    pushes=pushes,
                    solutions=solutions,
                    ratio=solutions / pushes if pushes > 0 else 0.0,
                    time_taken=time_taken,
                    failure_reason=failure_reason,
                    ml_goals_raw=alg_stats.get('ml_goals_raw', []),
                    search_solutions=ep.get('search_solutions', []),
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
    successes: int = 0
    total: int = 0

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

        if stats.failure_reasons:
            print(f"\n  Failure Reasons:")
            for reason, count in sorted(stats.failure_reasons.items(), key=lambda x: -x[1]):
                print(f"    {reason}: {count}")


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

    if time_data:
        plot_time_vs_success(
            time_data,
            config,
            f"{config.output_dir}/time_vs_success.png"
        )

    print(f"\nPlots saved to: {config.output_dir}")

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
