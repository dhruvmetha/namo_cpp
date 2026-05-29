#!/bin/bash
# Filter NPZs with robot_region/goal_sample_region overlap (degenerate cases
# where robot was already in the goal-sample region per the snapshot).
# Moves overlap NPZs to sibling `<phase_mask_dir>_overlap_filtered/` dirs,
# preserving env subdir structure. Non-destructive (reversible by moving back).
#
# Picker scripts that glob the phase mask dirs naturally see only clean NPZs.
#
#SBATCH --job-name=v3-filter-overlaps
#SBATCH --partition=main-redhat
#SBATCH --cpus-per-task=32
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/dm1487/logs/v3-filter-overlaps-%j.out
#SBATCH --error=/scratch/dm1487/logs/v3-filter-overlaps-%j.err

set -euo pipefail
NAMO_REPO=${NAMO_REPO:-/cache/home/dm1487/projects/namo/namo_cpp}
source "$NAMO_REPO/scripts/amarel/activate.sh"

echo "host=$(hostname)  job=${SLURM_JOB_ID:-N/A}"
echo "start=$(date -Is)"

cd "$NAMO_REPO"
PYTHONPATH="$NAMO_REPO/python:$NAMO_REPO/build_python" python \
    python/namo/visualization/mask_generation/filter_npz_overlaps.py \
    /scratch/dm1487/outputs/v3_phase1_masks \
    /scratch/dm1487/outputs/v3_phase2_masks \
    /scratch/dm1487/outputs/v3_phase3_masks \
    /scratch/dm1487/outputs/v3_phase4_masks \
    /scratch/dm1487/outputs/v3_phase5_masks \
    /scratch/dm1487/outputs/v3_aug9_phase1_masks \
    /scratch/dm1487/outputs/v3_aug9_phase2_masks \
    /scratch/dm1487/outputs/v3_aug9_phase3_masks \
    /scratch/dm1487/outputs/v3_aug9_phase4_masks \
    /scratch/dm1487/outputs/v3_aug9_phase5A_masks \
    /scratch/dm1487/outputs/v3_aug9_phase5B_masks \
    /scratch/dm1487/outputs/v3_aug9_phase5C_masks \
    /scratch/dm1487/outputs/v3_aug9_phase5D_masks \
    /scratch/dm1487/outputs/v3_aug9_phase5E_masks \
    --workers "$SLURM_CPUS_PER_TASK"

echo "end=$(date -Is)"
