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

        # Parse reference (search)
        if 'reference' in data:
            ref = data['reference']
            config.reference = ModelConfig(
                name=ref.get('name', 'Search'),
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

    # Only consider triplets in both model and reference
    for env in model_data:
        if env not in reference_data:
            continue

        for key in model_data[env]:
            if key not in reference_data[env]:
                continue

            ref_result = reference_data[env][key]
            model_result = model_data[env][key]

            category = categorize_problem(ref_result)

            # Skip blocked and unsolvable problems
            if category in ('blocked', 'unsolvable'):
                continue

            # Determine depth category
            if category == '1push':
                depth_stats = stats.depth_1
            else:  # 2push
                depth_stats = stats.depth_2

            depth_stats.total += 1

            if model_result.success:
                depth_stats.successes += 1
                depth_stats.pushes.append(model_result.pushes)
                depth_stats.times.append(model_result.time_taken)
                depth_stats.solutions.append(model_result.solutions)
                depth_stats.solutions_found.append(model_result.solutions_found)
                # Track interactions
                if model_result.wall_collision:
                    depth_stats.wall_collisions += 1
                depth_stats.movable_collisions_list.append(model_result.movable_collisions)

    return stats


def compute_time_based_success(
    model_data: Dict[str, Dict[str, RegionResult]],
    reference_data: Dict[str, Dict[str, RegionResult]],
    config: EvalConfig,
) -> Dict[str, Dict[str, List[float]]]:
    """
    Compute success rate as a function of time cutoff.

    Returns:
        {category: {'cutoffs': [...], 'rates': [...]}}
    """
    cutoffs = np.arange(0, config.time_cutoff_max + config.time_step, config.time_step)

    # Collect times by category
    times_by_category: Dict[str, List[float]] = {'1push': [], '2push': []}
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
    stats = {
        'none': {'successes': 0, 'total': 0},
        'wall_only': {'successes': 0, 'total': 0},
        'movable_only': {'successes': 0, 'total': 0},
        'both': {'successes': 0, 'total': 0},
    }

    for env in model_data:
        if env not in reference_data:
            continue
        for key in model_data[env]:
            if key not in reference_data[env]:
                continue

            ref_result = reference_data[env][key]
            model_result = model_data[env][key]

            # Only consider cases where search succeeded (solvable problems)
            if not ref_result.success:
                continue

            # Determine collision category based on search (oracle) result
            has_wall = ref_result.wall_collision
            has_movable = ref_result.movable_collisions > 0

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
        'oracle_1_model_1': 0,
        'oracle_1_model_2': 0,
        'oracle_1_model_2plus': 0,  # More than 2 pushes
        'oracle_2_model_1': 0,
        'oracle_2_model_2': 0,
        'oracle_2_model_2plus': 0,  # More than 2 pushes
        'oracle_1_failed': 0,
        'oracle_2_failed': 0,
    }

    for env in model_data:
        if env not in reference_data:
            continue
        for key in model_data[env]:
            if key not in reference_data[env]:
                continue

            ref_result = reference_data[env][key]
            model_result = model_data[env][key]

            # Only consider solvable problems (oracle succeeded)
            if not ref_result.success:
                continue

            oracle_depth = ref_result.chain_depth

            if oracle_depth == 1:
                if model_result.success:
                    model_depth = model_result.chain_depth
                    if model_depth == 1:
                        matrix['oracle_1_model_1'] += 1
                    elif model_depth == 2:
                        matrix['oracle_1_model_2'] += 1
                    else:
                        matrix['oracle_1_model_2plus'] += 1
                else:
                    matrix['oracle_1_failed'] += 1
            elif oracle_depth == 2:
                if model_result.success:
                    model_depth = model_result.chain_depth
                    if model_depth == 1:
                        matrix['oracle_2_model_1'] += 1
                    elif model_depth == 2:
                        matrix['oracle_2_model_2'] += 1
                    else:
                        matrix['oracle_2_model_2plus'] += 1
                else:
                    matrix['oracle_2_failed'] += 1

    return matrix


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


def plot_pushes_boxplot(
    model_stats: List[ModelStats],
    config: EvalConfig,
    output_path: Optional[str] = None,
):
    """Plot pushes comparison as boxplots (2-push only)."""
    n_models = len(model_stats)
    names = [s.name for s in model_stats]

    fig, ax = plt.subplots(figsize=(8, 5))

    data = [s.depth_2.pushes or [0] for s in model_stats]
    bp = ax.boxplot(data, tick_labels=names, patch_artist=True, showfliers=False)

    for i, patch in enumerate(bp['boxes']):
        patch.set_facecolor(get_model_color(i, config))
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

    ax.tick_params(axis='x', rotation=15)
    ax.set_ylabel('Pushes to Success')
    ax.set_title('2-Push Problems - Pushes Distribution')

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
    """Plot time comparison as boxplots (2-push only)."""
    n_models = len(model_stats)
    names = [s.name for s in model_stats]

    fig, ax = plt.subplots(figsize=(8, 5))

    data = [s.depth_2.times or [0] for s in model_stats]
    bp = ax.boxplot(data, tick_labels=names, patch_artist=True, showfliers=False)

    for i, patch in enumerate(bp['boxes']):
        patch.set_facecolor(get_model_color(i, config))
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

    ax.tick_params(axis='x', rotation=15)
    ax.set_ylabel('Time to Success (ms)')
    ax.set_title('2-Push Problems - Time Distribution')

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white')
        print(f"Saved: {output_path}")
    return fig


def plot_time_vs_success(
    time_data: Dict[str, Dict[str, Dict[str, List[float]]]],  # {model_name: {category: {cutoffs, rates}}}
    config: EvalConfig,
    output_path: Optional[str] = None,
):
    """Plot success rate vs time cutoff (2-push only)."""
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
            ax.plot(cutoffs_s, rates, label=model_name,
                   color=get_model_color(model_idx, config), linewidth=2)

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
) -> Dict[str, Dict[str, List[float]]]:
    """
    Compute success rate as a function of push count cutoff.

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


def plot_pushes_vs_success(
    push_data: Dict[str, Dict[str, Dict[str, List[float]]]],  # {model_name: {category: {cutoffs, rates, total}}}
    config: EvalConfig,
    output_path: Optional[str] = None,
):
    """Plot success rate vs push count cutoff (2-push only)."""
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
            ax.plot(cutoffs, rates, label=model_name,
                   color=get_model_color(model_idx, config), linewidth=2)

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

        if stats.failure_reasons:
            print(f"\n  Failure Reasons:")
            for reason, count in sorted(stats.failure_reasons.items(), key=lambda x: -x[1]):
                print(f"    {reason}: {count}")


def generate_markdown_report(
    model_stats: List[ModelStats],
    config: EvalConfig,
    depth_counts: Dict[str, int],
    output_path: str,
    confusion_matrices: Optional[Dict[str, Dict[str, int]]] = None,
):
    """Generate a markdown report with comparison tables (2-push only)."""
    lines = []
    lines.append("# 2-Push Evaluation Results\n")
    lines.append(f"Generated from evaluation config.\n")

    # Dataset overview
    lines.append("## Dataset Overview\n")
    count_2push = depth_counts.get('depth_2', 0)
    lines.append(f"Total 2-push problems evaluated: **{count_2push}**\n")

    # Success rates
    lines.append("## Success Rates\n")
    lines.append("| Model | Successes | Total | Success Rate |")
    lines.append("|-------|-----------|-------|--------------|")
    for stats in model_stats:
        cat_stats = stats.depth_2
        lines.append(f"| {stats.name} | {cat_stats.successes} | {cat_stats.total} | **{cat_stats.success_rate:.1%}** |")
    lines.append("")

    # Pushes statistics
    lines.append("## Pushes to Success (Successful Runs Only)\n")
    lines.append("| Model | Median | Mean |")
    lines.append("|-------|--------|------|")
    for stats in model_stats:
        cat_stats = stats.depth_2
        if cat_stats.pushes:
            lines.append(f"| {stats.name} | {cat_stats.median_pushes:.1f} | {cat_stats.mean_pushes:.1f} |")
        else:
            lines.append(f"| {stats.name} | - | - |")
    lines.append("")

    # Time statistics
    lines.append("## Time to Success in ms (Successful Runs Only)\n")
    lines.append("| Model | Median | Mean |")
    lines.append("|-------|--------|------|")
    for stats in model_stats:
        cat_stats = stats.depth_2
        if cat_stats.times:
            lines.append(f"| {stats.name} | {cat_stats.median_time:.0f} | {cat_stats.mean_time:.0f} |")
        else:
            lines.append(f"| {stats.name} | - | - |")
    lines.append("")

    # Interaction statistics
    lines.append("## Interaction Statistics (Successful Runs Only)\n")
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

    # Confusion Matrix section (2-push only)
    if confusion_matrices:
        lines.append("## Model Result Breakdown\n")
        lines.append("How each model solved 2-push problems:\n")
        lines.append("- **Model 1-Push**: Solved with 1 push (unexpected for true 2-push)")
        lines.append("- **Model 2-Push**: Solved with exactly 2 pushes (optimal)")
        lines.append("- **Model 2+ Push**: Solved with more than 2 pushes")
        lines.append("- **Failed**: Did not find a solution\n")

        lines.append("| Model | 1-Push | 2-Push | 2+ Push | Failed | Total |")
        lines.append("|-------|--------|--------|---------|--------|-------|")

        for model_name, matrix in confusion_matrices.items():
            oracle_2_total = (matrix['oracle_2_model_1'] + matrix['oracle_2_model_2'] +
                             matrix['oracle_2_model_2plus'] + matrix['oracle_2_failed'])

            def pct(val, total):
                return f"{val} ({val/total*100:.0f}%)" if total > 0 else "0"

            lines.append(f"| {model_name} | {pct(matrix['oracle_2_model_1'], oracle_2_total)} | "
                        f"{pct(matrix['oracle_2_model_2'], oracle_2_total)} | "
                        f"{pct(matrix['oracle_2_model_2plus'], oracle_2_total)} | "
                        f"{pct(matrix['oracle_2_failed'], oracle_2_total)} | {oracle_2_total} |")
        lines.append("")

    # Failure reasons per model
    lines.append("## Failure Reasons\n")
    for stats in model_stats:
        if stats.failure_reasons:
            lines.append(f"### {stats.name}\n")
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
    if config.reference is None:
        raise ValueError("Config must specify a 'reference' model for chain depth categorization")

    # Create output directory
    Path(config.output_dir).mkdir(parents=True, exist_ok=True)

    # Load reference data (for chain depth categorization AND as oracle baseline)
    print(f"Loading reference data: {config.reference.name} (for categorization + oracle baseline)...")
    print(f"  Using triplets (env, region, object) for evaluation granularity")
    reference_data, reference_failures = load_pickle_data(f"{config.reference.dir}/**/*.pkl")
    print(f"  Loaded {sum(len(v) for v in reference_data.values())} triplets")

    # Load all models (reference/oracle + baselines + learned)
    # Reference goes first so it appears first in plots
    all_model_data: Dict[str, Dict[str, Dict[str, RegionResult]]] = {}
    all_model_data[config.reference.name] = reference_data
    all_model_failures: Dict[str, Dict[str, int]] = {}
    all_model_failures[config.reference.name] = reference_failures

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

    # Plot confusion matrices
    if confusion_matrices:
        plot_confusion_matrix(
            confusion_matrices,
            config,
            f"{config.output_dir}/confusion_matrix.png"
        )

    # Generate markdown report
    generate_markdown_report(
        all_stats,
        config,
        depth_counts,
        f"{config.output_dir}/results.md",
        confusion_matrices=confusion_matrices
    )

    print(f"\nPlots saved to: {config.output_dir}")

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
