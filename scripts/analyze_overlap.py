#!/usr/bin/env python3
"""Analyze overlap between robot and goal regions in 2-push dataset."""

import h5py
import numpy as np
from tqdm import tqdm

def analyze_overlaps(h5_path: str):
    """Analyze overlapping robot and goal regions in the dataset."""

    with h5py.File(h5_path, 'r') as f:
        total_samples = f['solution_depth'].shape[0]
        solution_depth = f['solution_depth'][:]

        # Find 2-step solutions
        two_step_mask = solution_depth.flatten() == 2
        two_step_indices = np.where(two_step_mask)[0]
        num_two_step = len(two_step_indices)

        print(f"Total samples: {total_samples}")
        print(f"2-step solutions: {num_two_step} ({100*num_two_step/total_samples:.2f}%)")
        print()

        # Analyze overlaps for 2-step samples
        overlap_with_goal_sample = 0
        overlap_with_goal_a1 = 0
        overlap_with_goal_a2 = 0
        any_overlap = 0

        overlap_percentages_goal_sample = []
        overlap_percentages_a1 = []
        overlap_percentages_a2 = []

        batch_size = 1000

        print("Analyzing overlaps for 2-step solutions...")
        for batch_start in tqdm(range(0, num_two_step, batch_size)):
            batch_end = min(batch_start + batch_size, num_two_step)
            batch_indices = two_step_indices[batch_start:batch_end]

            # Load batch data
            robot_region = f['local_robot_region'][batch_indices]
            goal_sample_region = f['local_goal_sample_region'][batch_indices]
            goal_mask_a1 = f['local_goal_mask_a1'][batch_indices]
            goal_mask_a2 = f['local_goal_mask_a2'][batch_indices]

            for i in range(len(batch_indices)):
                robot = robot_region[i] > 0.5
                goal_sample = goal_sample_region[i] > 0.5
                goal_a1 = goal_mask_a1[i] > 0.5
                goal_a2 = goal_mask_a2[i] > 0.5

                # Compute overlaps
                overlap_gs = np.logical_and(robot, goal_sample)
                overlap_a1 = np.logical_and(robot, goal_a1)
                overlap_a2 = np.logical_and(robot, goal_a2)

                has_overlap_gs = overlap_gs.sum() > 0
                has_overlap_a1 = overlap_a1.sum() > 0
                has_overlap_a2 = overlap_a2.sum() > 0

                if has_overlap_gs:
                    overlap_with_goal_sample += 1
                    # Calculate overlap percentage relative to goal region
                    if goal_sample.sum() > 0:
                        pct = 100 * overlap_gs.sum() / goal_sample.sum()
                        overlap_percentages_goal_sample.append(pct)

                if has_overlap_a1:
                    overlap_with_goal_a1 += 1
                    if goal_a1.sum() > 0:
                        pct = 100 * overlap_a1.sum() / goal_a1.sum()
                        overlap_percentages_a1.append(pct)

                if has_overlap_a2:
                    overlap_with_goal_a2 += 1
                    if goal_a2.sum() > 0:
                        pct = 100 * overlap_a2.sum() / goal_a2.sum()
                        overlap_percentages_a2.append(pct)

                if has_overlap_gs or has_overlap_a1 or has_overlap_a2:
                    any_overlap += 1

        print("\n" + "="*60)
        print("OVERLAP ANALYSIS FOR 2-STEP SOLUTIONS")
        print("="*60)

        print(f"\n1. Overlap with goal_sample_region:")
        print(f"   Samples with overlap: {overlap_with_goal_sample}/{num_two_step} ({100*overlap_with_goal_sample/num_two_step:.2f}%)")
        if overlap_percentages_goal_sample:
            print(f"   Mean overlap %: {np.mean(overlap_percentages_goal_sample):.2f}%")
            print(f"   Median overlap %: {np.median(overlap_percentages_goal_sample):.2f}%")
            print(f"   Max overlap %: {np.max(overlap_percentages_goal_sample):.2f}%")

        print(f"\n2. Overlap with goal_mask_a1 (first action goal):")
        print(f"   Samples with overlap: {overlap_with_goal_a1}/{num_two_step} ({100*overlap_with_goal_a1/num_two_step:.2f}%)")
        if overlap_percentages_a1:
            print(f"   Mean overlap %: {np.mean(overlap_percentages_a1):.2f}%")
            print(f"   Median overlap %: {np.median(overlap_percentages_a1):.2f}%")
            print(f"   Max overlap %: {np.max(overlap_percentages_a1):.2f}%")

        print(f"\n3. Overlap with goal_mask_a2 (second action goal):")
        print(f"   Samples with overlap: {overlap_with_goal_a2}/{num_two_step} ({100*overlap_with_goal_a2/num_two_step:.2f}%)")
        if overlap_percentages_a2:
            print(f"   Mean overlap %: {np.mean(overlap_percentages_a2):.2f}%")
            print(f"   Median overlap %: {np.median(overlap_percentages_a2):.2f}%")
            print(f"   Max overlap %: {np.max(overlap_percentages_a2):.2f}%")

        print(f"\n4. ANY overlap (robot with any goal region):")
        print(f"   Samples with any overlap: {any_overlap}/{num_two_step} ({100*any_overlap/num_two_step:.2f}%)")

        print("\n" + "="*60)

        return {
            'total_samples': total_samples,
            'two_step_samples': num_two_step,
            'overlap_goal_sample': overlap_with_goal_sample,
            'overlap_a1': overlap_with_goal_a1,
            'overlap_a2': overlap_with_goal_a2,
            'any_overlap': any_overlap,
        }


if __name__ == "__main__":
    h5_path = "/common/users/dm1487/namo_data/h5_files/dec2/aug9_envs/2_push_train_corrected_overlaps_2.h5"
    results = analyze_overlaps(h5_path)
