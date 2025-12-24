#!/usr/bin/env python3
"""
Standalone script to generate histogram of solutions_total_for_neighbour
categorized by difficulty (easy/medium/hard based on ratio).
"""

import pickle
from glob import glob
from collections import Counter, defaultdict
import numpy as np
import matplotlib.pyplot as plt

# Config
DATA_DIR = "/common/users/dm1487/namo_data/dec2/aug9_envs/1_push_train"
EASY_THRESHOLD = 0.9
HARD_THRESHOLD = 0.1
OUTPUT_PATH = "./solutions_found_histogram.png"

def load_data(data_dir):
    """Load data and count episodes (solutions) per env+region."""
    from collections import defaultdict

    files = glob(f"{data_dir}/**/*.pkl", recursive=True)
    print(f"Found {len(files)} pickle files")

    # Track episodes per (xml_name, region) and their stats
    episode_counts = defaultdict(int)
    region_stats = {}  # (xml_name, region) -> (pushes, solutions_total)

    for f in files:
        try:
            with open(f, 'rb') as fp:
                data = pickle.load(fp)

            for ep in data.get('episode_results', []):
                xml_file = ep.get('xml_file', '')
                xml_name = "_".join(xml_file.split('/')[-4:])

                # Skip easy envs
                if "easy" in xml_name:
                    continue

                alg_stats = ep.get('algorithm_stats', {})
                region_label = alg_stats.get('neighbour_region_label')
                solution_found = ep.get('solution_found', False)

                if region_label is None or not solution_found:
                    continue

                key = (xml_name, region_label)
                episode_counts[key] += 1

                # Store stats (same for all episodes of this region)
                if key not in region_stats:
                    pushes = alg_stats.get('pushes_total_for_neighbour', 0)
                    solutions_total = alg_stats.get('solutions_total_for_neighbour', 0)
                    region_stats[key] = (pushes, solutions_total)

        except Exception as e:
            continue

    # Now categorize and collect episode counts
    results = {'easy': [], 'medium': [], 'hard': []}

    for key, count in episode_counts.items():
        if key not in region_stats:
            continue
        pushes, solutions_total = region_stats[key]
        if pushes == 0:
            continue

        ratio = solutions_total / pushes

        if ratio > EASY_THRESHOLD:
            category = 'easy'
        elif ratio > HARD_THRESHOLD:
            category = 'medium'
        else:
            category = 'hard'

        results[category].append(count)

    return results


def plot_histogram(results, output_path):
    """Plot histogram of solutions_total per category."""
    categories = ['easy', 'medium', 'hard']

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    for idx, cat in enumerate(categories):
        ax = axes[idx]
        data = results[cat]

        if not data:
            ax.set_title(f'{cat.capitalize()} Category (no data)')
            continue

        # Count frequency of each solutions_total value
        counts = Counter(data)
        sol_values = sorted(counts.keys())
        frequencies = [counts[v] for v in sol_values]

        # Create bar plot
        bars = ax.bar(range(len(sol_values)), frequencies, color='#2ecc71', edgecolor='black')

        # Add count labels on bars
        for bar, freq in zip(bars, frequencies):
            if freq > 0:
                ax.annotate(f'{freq}',
                           xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                           xytext=(0, 2), textcoords="offset points",
                           ha='center', va='bottom', fontsize=8)

        ax.set_xticks(range(len(sol_values)))
        ax.set_xticklabels(sol_values)
        ax.set_xlabel('Number of Solutions Recorded')
        ax.set_ylabel('Count (env+region pairs)')
        ax.set_title(f'{cat.capitalize()} Category (n={len(data)})')
        ax.grid(True, axis='y', linestyle='--', alpha=0.6)

    plt.suptitle('Distribution of Recorded Solutions per Region by Difficulty Category\n(1_push_train data)', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")


def main():
    print("Loading data...")
    results = load_data(DATA_DIR)

    print(f"\nCounts per category:")
    for cat in ['easy', 'medium', 'hard']:
        print(f"  {cat.capitalize()}: {len(results[cat])} env+region pairs")

    print("\nGenerating histogram...")
    plot_histogram(results, OUTPUT_PATH)


if __name__ == "__main__":
    main()
