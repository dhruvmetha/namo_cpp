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
from typing import Dict, List, Optional
from collections import defaultdict

import yaml
import numpy as np
import matplotlib.pyplot as plt


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
    # Reference (search) determines problem categorization
    reference: ModelConfig = None

    # Learned models to compare
    learned: List[ModelConfig] = field(default_factory=list)

    # Plot settings
    output_dir: str = "./eval_2push_plots"
    time_cutoff_max: int = 10000  # ms
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

        # Parse reference (search)
        if 'reference' in data:
            ref = data['reference']
            config.reference = ModelConfig(
                name=ref.get('name', 'Search'),
                dir=ref['dir'],
                color=ref.get('color'),
            )

        # Parse learned models (list)
        if 'learned' in data:
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

        return config


# =============================================================================
# Data Loading
# =============================================================================

@dataclass
class RegionResult:
    """Results for a single env+region pair."""
    success: bool = False
    pushes: int = 0
    time_taken: float = 0.0
    failure_reason: str = ""
    xml_file: str = ""
    region: str = ""
    chain_depth: int = 0


def load_pickle_data(data_dir: str) -> Dict[str, Dict[str, RegionResult]]:
    """
    Load evaluation data from pickle files.

    Returns:
        {xml_file_name: {region_label: RegionResult}}
    """
    results: Dict[str, Dict[str, RegionResult]] = {}

    for file in glob(data_dir, recursive=True):
        if 'collection_summary' in file:
            continue

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

                alg_stats = ep.get('algorithm_stats', {})
                region_label = alg_stats.get('neighbour_region_label')

                if region_label is None:
                    continue

                # Only process each region once per file
                if region_label in region_done:
                    continue
                region_done.add(region_label)

                if xml_file_name not in results:
                    results[xml_file_name] = {}

                pushes = alg_stats.get('pushes_total_for_neighbour', 0)
                solution_found = ep.get('solution_found', False)
                time_taken = ep.get('search_time_ms', 0)
                failure_reason = alg_stats.get('failure_reason', '')
                chain_depth = alg_stats.get('chain_depth', 0)

                results[xml_file_name][region_label] = RegionResult(
                    success=solution_found and pushes > 0,
                    pushes=pushes,
                    time_taken=time_taken,
                    failure_reason=failure_reason,
                    xml_file=xml_file,
                    region=region_label,
                    chain_depth=chain_depth,
                )

        except Exception as e:
            print(f"Error loading {file}: {e}")
            continue

    return results


# =============================================================================
# Analysis
# =============================================================================

@dataclass
class DepthStats:
    """Statistics for a specific chain depth."""
    successes: int = 0
    total: int = 0
    pushes: List[int] = field(default_factory=list)
    times: List[float] = field(default_factory=list)

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
    """Statistics for a model."""
    name: str
    # Overall stats
    successes: int = 0
    total: int = 0
    pushes: List[int] = field(default_factory=list)
    times: List[float] = field(default_factory=list)
    failure_reasons: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    # Per chain-depth stats
    depth_1: DepthStats = field(default_factory=DepthStats)
    depth_2: DepthStats = field(default_factory=DepthStats)

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
    data: Dict[str, Dict[str, RegionResult]],
    reference_data: Dict[str, Dict[str, RegionResult]],
    name: str,
    common_pairs: set,
    filter_learned_success: bool = False,
    learned_data: Optional[Dict[str, Dict[str, RegionResult]]] = None,
) -> ModelStats:
    """Compute statistics for a model on the common env+region pairs.

    Uses reference_data to determine problem category:
    - 1push: search succeeded with chain_depth=1
    - 2push: chain_depth=2 OR search exhausted budget on chain_depth=1
    - blocked: excluded (not solvable)

    If filter_learned_success=True, only include pairs where learned succeeded.
    """
    stats = ModelStats(name=name)

    for env, region in common_pairs:
        if env not in data or region not in data[env]:
            continue
        if env not in reference_data or region not in reference_data[env]:
            continue

        result = data[env][region]
        ref_result = reference_data[env][region]

        category = categorize_problem(ref_result)

        # Skip blocked and unsolvable problems
        if category in ('blocked', 'unsolvable'):
            continue

        # If filtering to learned success, skip if learned failed
        if filter_learned_success and learned_data is not None:
            if env not in learned_data or region not in learned_data[env]:
                continue
            if not learned_data[env][region].success:
                continue

        # Overall stats
        stats.total += 1
        if result.success:
            stats.successes += 1
            stats.pushes.append(result.pushes)
            stats.times.append(result.time_taken)
        elif result.failure_reason:
            stats.failure_reasons[result.failure_reason] += 1

        # Per-depth stats
        depth_stats = stats.depth_1 if category == '1push' else stats.depth_2
        depth_stats.total += 1
        if result.success:
            depth_stats.successes += 1
            depth_stats.pushes.append(result.pushes)
            depth_stats.times.append(result.time_taken)

    return stats


def compute_time_based_success(
    data: Dict[str, Dict[str, RegionResult]],
    reference_data: Dict[str, Dict[str, RegionResult]],
    common_pairs: set,
    config: EvalConfig,
    filter_learned_success: bool = False,
    learned_data: Optional[Dict[str, Dict[str, RegionResult]]] = None,
) -> Dict[str, Dict[str, List[float]]]:
    """Compute success rate as a function of time cutoff, split by category."""
    cutoffs = np.arange(0, config.time_cutoff_max + config.time_step, config.time_step)

    # Collect times by category (1push, 2push, all)
    times_by_cat = {'1push': [], '2push': [], 'all': []}
    totals_by_cat = {'1push': 0, '2push': 0, 'all': 0}

    for env, region in common_pairs:
        if env not in data or region not in data[env]:
            continue
        if env not in reference_data or region not in reference_data[env]:
            continue

        ref_result = reference_data[env][region]
        category = categorize_problem(ref_result)

        # Skip blocked and unsolvable problems
        if category in ('blocked', 'unsolvable'):
            continue

        # If filtering to learned success, skip if learned failed
        if filter_learned_success and learned_data is not None:
            if env not in learned_data or region not in learned_data[env]:
                continue
            if not learned_data[env][region].success:
                continue

        totals_by_cat[category] += 1
        totals_by_cat['all'] += 1

        if data[env][region].success:
            times_by_cat[category].append(data[env][region].time_taken)
            times_by_cat['all'].append(data[env][region].time_taken)

    # Compute rates at each cutoff
    result = {}
    for cat in ['1push', '2push', 'all']:
        times = np.array(times_by_cat[cat])
        total = totals_by_cat[cat]
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


def plot_success_comparison(
    all_stats: List[ModelStats],
    config: EvalConfig,
    output_dir: str,
):
    """Plot success rate comparison - separate plots for 1-push, 2-push, and overall."""
    categories = [
        ('1-Push', 'depth_1', 'success_1push.png'),
        ('2-Push', 'depth_2', 'success_2push.png'),
        ('Overall', None, 'success_overall.png'),
    ]

    for title, depth_attr, filename in categories:
        fig, ax = plt.subplots(figsize=(max(8, len(all_stats) * 2), 6))

        models = [s.name for s in all_stats]
        if depth_attr:
            rates = [getattr(s, depth_attr).success_rate for s in all_stats]
            counts = [getattr(s, depth_attr).total for s in all_stats]
        else:
            rates = [s.success_rate for s in all_stats]
            counts = [s.total for s in all_stats]
        colors = [get_model_color(i, config) for i in range(len(all_stats))]

        bars = ax.bar(models, rates, color=colors, edgecolor='black', width=0.5)

        for bar, rate, count in zip(bars, rates, counts):
            ax.annotate(f'{rate:.1%}\n(n={count})',
                       xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                       xytext=(0, 3), textcoords="offset points",
                       ha='center', va='bottom', fontsize=12, fontweight='bold')

        ax.set_ylim(0, 1.2)
        ax.set_ylabel('Success Rate', fontsize=12)
        ax.set_title(f'{title}: Success Rate', fontsize=14)
        ax.grid(True, axis='y', linestyle='--', alpha=0.6)
        plt.xticks(rotation=15, ha='right')

        plt.tight_layout()
        output_path = f"{output_dir}/{filename}"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {output_path}")
        plt.close(fig)


def plot_pushes_comparison(
    all_stats: List[ModelStats],
    config: EvalConfig,
    output_dir: str,
):
    """Plot pushes comparison - separate plots for 1-push, 2-push, and overall."""
    categories = [
        ('1-Push', 'depth_1', 'pushes_1push.png'),
        ('2-Push', 'depth_2', 'pushes_2push.png'),
        ('Overall', None, 'pushes_overall.png'),
    ]

    for title, depth_attr, filename in categories:
        fig, ax = plt.subplots(figsize=(max(8, len(all_stats) * 2), 6))

        if depth_attr:
            data = [getattr(s, depth_attr).pushes or [0] for s in all_stats]
            cats = [getattr(s, depth_attr) for s in all_stats]
        else:
            data = [s.pushes or [0] for s in all_stats]
            cats = all_stats
        labels = [s.name for s in all_stats]
        colors = [get_model_color(i, config) for i in range(len(all_stats))]

        bp = ax.boxplot(data, tick_labels=labels, patch_artist=True, showfliers=False)

        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        for median in bp['medians']:
            median.set_color('red')
            median.set_linewidth(2)

        # Add median annotations
        for i, cat in enumerate(cats):
            if cat.pushes:
                ax.annotate(f'med={cat.median_pushes:.1f}',
                           xy=(i + 1, cat.median_pushes),
                           xytext=(0, 5), textcoords='offset points',
                           fontsize=9, ha='center')

        ax.set_ylabel('Pushes to Success', fontsize=12)
        ax.set_title(f'{title}: Pushes to Success', fontsize=14)
        ax.grid(True, axis='y', linestyle='--', alpha=0.6)
        plt.xticks(rotation=15, ha='right')

        plt.tight_layout()
        output_path = f"{output_dir}/{filename}"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {output_path}")
        plt.close(fig)


def plot_time_comparison(
    all_stats: List[ModelStats],
    config: EvalConfig,
    output_dir: str,
):
    """Plot time comparison - separate plots for 1-push, 2-push, and overall."""
    categories = [
        ('1-Push', 'depth_1', 'time_1push.png'),
        ('2-Push', 'depth_2', 'time_2push.png'),
        ('Overall', None, 'time_overall.png'),
    ]

    for title, depth_attr, filename in categories:
        fig, ax = plt.subplots(figsize=(max(8, len(all_stats) * 2), 6))

        if depth_attr:
            data = [getattr(s, depth_attr).times or [0] for s in all_stats]
            cats = [getattr(s, depth_attr) for s in all_stats]
        else:
            data = [s.times or [0] for s in all_stats]
            cats = all_stats
        labels = [s.name for s in all_stats]
        colors = [get_model_color(i, config) for i in range(len(all_stats))]

        bp = ax.boxplot(data, tick_labels=labels, patch_artist=True, showfliers=False)

        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        for median in bp['medians']:
            median.set_color('red')
            median.set_linewidth(2)

        # Add median annotations
        for i, cat in enumerate(cats):
            if cat.times:
                ax.annotate(f'med={cat.median_time:.0f}ms',
                           xy=(i + 1, cat.median_time),
                           xytext=(0, 5), textcoords='offset points',
                           fontsize=9, ha='center')

        ax.set_ylabel('Time to Success (ms)', fontsize=12)
        ax.set_title(f'{title}: Time to Success', fontsize=14)
        ax.grid(True, axis='y', linestyle='--', alpha=0.6)
        plt.xticks(rotation=15, ha='right')

        plt.tight_layout()
        output_path = f"{output_dir}/{filename}"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {output_path}")
        plt.close(fig)


def plot_time_vs_success(
    time_data: Dict[str, Dict[str, Dict[str, List[float]]]],  # {model_name: {category: {cutoffs, rates}}}
    config: EvalConfig,
    output_dir: str,
):
    """Plot success rate vs time cutoff - separate plots for each category."""
    categories = [
        ('1push', '1-Push', 'time_vs_success_1push.png'),
        ('2push', '2-Push', 'time_vs_success_2push.png'),
        ('all', 'Overall', 'time_vs_success_overall.png'),
    ]

    for cat, title, filename in categories:
        fig, ax = plt.subplots(figsize=(10, 6))

        for model_idx, (model_name, cat_data) in enumerate(time_data.items()):
            if cat in cat_data:
                ax.plot(cat_data[cat]['cutoffs'], cat_data[cat]['rates'],
                       label=model_name, color=get_model_color(model_idx, config), linewidth=2)

        ax.set_xlabel('Time Cutoff (ms)', fontsize=12)
        ax.set_ylabel('Success Rate', fontsize=12)
        ax.set_title(f'{title}: Success Rate @ Time Cutoff', fontsize=14)
        ax.set_ylim(0, 1.05)
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend(fontsize=11)

        plt.tight_layout()
        output_path = f"{output_dir}/{filename}"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {output_path}")
        plt.close(fig)


def plot_time_vs_success_full(
    all_model_data: Dict[str, Dict[str, Dict[str, RegionResult]]],  # {name: {env: {region: result}}}
    reference_data: Dict[str, Dict[str, RegionResult]],
    common_pairs: set,
    config: EvalConfig,
    output_dir: str,
):
    """Plot success rate vs time cutoff using full time range."""
    # Collect all times to determine max
    all_times = {'1push': [], '2push': [], 'all': []}

    for env, region in common_pairs:
        if env not in reference_data or region not in reference_data[env]:
            continue
        ref_result = reference_data[env][region]
        category = categorize_problem(ref_result)
        if category in ('blocked', 'unsolvable'):
            continue

        # Collect times from all models
        for model_data in all_model_data.values():
            if env in model_data and region in model_data[env] and model_data[env][region].success:
                all_times[category].append(model_data[env][region].time_taken)
                all_times['all'].append(model_data[env][region].time_taken)

    categories = [
        ('1push', '1-Push', 'time_vs_success_full_1push.png'),
        ('2push', '2-Push', 'time_vs_success_full_2push.png'),
        ('all', 'Overall', 'time_vs_success_full_overall.png'),
    ]

    for cat, title, filename in categories:
        if not all_times[cat]:
            continue

        max_time = max(all_times[cat]) * 1.1
        step = max(100, int(max_time / 200))  # ~200 points
        cutoffs = np.arange(0, max_time + step, step)

        fig, ax = plt.subplots(figsize=(10, 6))

        for model_idx, (model_name, model_data) in enumerate(all_model_data.items()):
            # Compute totals and times for this model
            total = 0
            times = []

            for env, region in common_pairs:
                if env not in reference_data or region not in reference_data[env]:
                    continue
                ref_result = reference_data[env][region]
                category_check = categorize_problem(ref_result)
                if category_check in ('blocked', 'unsolvable'):
                    continue
                if cat != 'all' and category_check != cat:
                    continue

                total += 1
                if env in model_data and region in model_data[env] and model_data[env][region].success:
                    times.append(model_data[env][region].time_taken)

            t = np.array(times)
            rates = []
            for cutoff in cutoffs:
                if total > 0:
                    successes = np.sum(t <= cutoff) if len(t) > 0 else 0
                    rates.append(successes / total)
                else:
                    rates.append(0.0)
            ax.plot(cutoffs / 1000, rates, label=model_name,
                   color=get_model_color(model_idx, config), linewidth=2)

        ax.set_xlabel('Time Cutoff (seconds)', fontsize=12)
        ax.set_ylabel('Success Rate', fontsize=12)
        ax.set_title(f'{title}: Success Rate @ Time Cutoff (Full Range)', fontsize=14)
        ax.set_ylim(0, 1.05)
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend(fontsize=11)

        plt.tight_layout()
        output_path = f"{output_dir}/{filename}"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {output_path}")
        plt.close(fig)


def print_summary(all_stats: List[ModelStats]):
    """Print summary statistics."""
    print("\n" + "=" * 70)
    print("2-PUSH EVALUATION SUMMARY")
    print("=" * 70)

    for stats in all_stats:
        print(f"\n{'─' * 50}")
        print(f"Model: {stats.name}")
        print(f"{'─' * 50}")

        # Per-depth stats
        for depth_name, depth_stats in [('1-Push', stats.depth_1), ('2-Push', stats.depth_2)]:
            if depth_stats.total > 0:
                print(f"\n  {depth_name}:")
                print(f"    Success: {depth_stats.successes}/{depth_stats.total} = {depth_stats.success_rate:.2%}")
                if depth_stats.pushes:
                    print(f"    Pushes:  median={depth_stats.median_pushes:.1f}, mean={depth_stats.mean_pushes:.1f}")
                if depth_stats.times:
                    print(f"    Time:    median={depth_stats.median_time:.0f}ms, mean={depth_stats.mean_time:.0f}ms")

        # Overall stats
        print(f"\n  Overall:")
        print(f"    Success: {stats.successes}/{stats.total} = {stats.success_rate:.2%}")
        if stats.pushes:
            print(f"    Pushes:  median={stats.median_pushes:.1f}, mean={stats.mean_pushes:.1f}")
        if stats.times:
            print(f"    Time:    median={stats.median_time:.0f}ms, mean={stats.mean_time:.0f}ms")

        if stats.failure_reasons:
            print(f"\n  Failure Reasons:")
            for reason, count in sorted(stats.failure_reasons.items(), key=lambda x: -x[1]):
                print(f"    {reason}: {count}")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="2-Push Evaluation Script")
    parser.add_argument("--config", type=str, required=True,
                       help="Path to YAML config file")
    parser.add_argument("--output-dir", type=str, default=None,
                       help="Override output directory from config")
    parser.add_argument("--no-show", action="store_true",
                       help="Don't show plots interactively")

    args = parser.parse_args()

    # Load config
    config = EvalConfig.from_yaml(args.config)

    if args.output_dir:
        config.output_dir = args.output_dir

    if config.reference is None:
        raise ValueError("Config must specify a 'reference' model")
    if not config.learned:
        raise ValueError("Config must specify at least one 'learned' model")

    Path(config.output_dir).mkdir(parents=True, exist_ok=True)

    # Load reference data
    print(f"Loading reference data: {config.reference.name}...")
    reference_data = load_pickle_data(f"{config.reference.dir}/**/*.pkl")
    reference_pairs = sum(len(v) for v in reference_data.values())
    print(f"  Loaded {len(reference_data)} envs, {reference_pairs} env+region pairs")

    # Load all learned models
    all_model_data: Dict[str, Dict[str, Dict[str, RegionResult]]] = {}

    # Include reference model first (so it appears first in plots)
    all_model_data[config.reference.name] = reference_data

    for model in config.learned:
        print(f"Loading learned model: {model.name}...")
        data = load_pickle_data(f"{model.dir}/**/*.pkl")
        pairs = sum(len(v) for v in data.values())
        print(f"  Loaded {len(data)} envs, {pairs} env+region pairs")
        all_model_data[model.name] = data

    # Find common env+region pairs across ALL models
    all_sets = []
    for data in all_model_data.values():
        all_sets.append({(env, region) for env in data for region in data[env]})

    common_pairs = all_sets[0]
    for s in all_sets[1:]:
        common_pairs = common_pairs & s
    print(f"\nCommon env+region pairs: {len(common_pairs)}")

    # Count categories
    blocked_count = sum(
        1 for env, region in common_pairs
        if categorize_problem(reference_data[env][region]) == 'blocked'
    )
    unsolvable_count = sum(
        1 for env, region in common_pairs
        if categorize_problem(reference_data[env][region]) == 'unsolvable'
    )
    solvable = len(common_pairs) - blocked_count - unsolvable_count
    print(f"  Blocked (no reachable objects): {blocked_count}")
    print(f"  Unsolvable (search exhausted): {unsolvable_count}")
    print(f"  Solvable (reference succeeded): {solvable}")

    if not common_pairs:
        print("ERROR: No common env+region pairs found!")
        return

    # Compute stats for all models (use reference_data for chain_depth categorization)
    all_stats: List[ModelStats] = []
    time_data: Dict[str, Dict[str, Dict[str, List[float]]]] = {}

    for model_name, model_data in all_model_data.items():
        stats = compute_stats(model_data, reference_data, model_name, common_pairs)
        all_stats.append(stats)

        # Compute time-based success
        time_data[model_name] = compute_time_based_success(
            model_data, reference_data, common_pairs, config
        )

    # Print summary
    print_summary(all_stats)

    # Generate plots
    print("\nGenerating plots...")

    plot_success_comparison(all_stats, config, config.output_dir)
    plot_pushes_comparison(all_stats, config, config.output_dir)
    plot_time_comparison(all_stats, config, config.output_dir)
    plot_time_vs_success(time_data, config, config.output_dir)
    plot_time_vs_success_full(all_model_data, reference_data, common_pairs, config, config.output_dir)

    # Generate learned-success plots for each learned model
    for learned_model in config.learned:
        learned_name = learned_model.name
        learned_data = all_model_data[learned_name]

        # Create subdirectory for this model's learned-success plots
        subdir_name = learned_model.success_dir or f"{learned_name.replace(' ', '_').lower()}_success"
        learned_success_dir = f"{config.output_dir}/{subdir_name}"
        Path(learned_success_dir).mkdir(parents=True, exist_ok=True)

        print(f"\nGenerating {learned_name} success-only plots...")

        # Compute stats filtered to where this learned model succeeded
        filtered_stats: List[ModelStats] = []
        filtered_time_data: Dict[str, Dict[str, Dict[str, List[float]]]] = {}

        for model_name, model_data in all_model_data.items():
            stats = compute_stats(
                model_data, reference_data, model_name, common_pairs,
                filter_learned_success=True, learned_data=learned_data
            )
            filtered_stats.append(stats)

            filtered_time_data[model_name] = compute_time_based_success(
                model_data, reference_data, common_pairs, config,
                filter_learned_success=True, learned_data=learned_data
            )

        # Generate plots for this filter
        plot_success_comparison(filtered_stats, config, learned_success_dir)
        plot_pushes_comparison(filtered_stats, config, learned_success_dir)
        plot_time_comparison(filtered_stats, config, learned_success_dir)
        plot_time_vs_success(filtered_time_data, config, learned_success_dir)

    print(f"\nPlots saved to: {config.output_dir}")

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
