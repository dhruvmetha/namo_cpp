#!/bin/bash
# Mine residual phase candidates from a phase output directory.
# Args:
#   $1 = phase output dir
#   $2 = residual manifest output path
#
# Excludes goal_region_not_in_snapshot (fundamentally unsolvable, waste of compute).
#
#SBATCH --partition=main-redhat
#SBATCH --cpus-per-task=64
#SBATCH --mem=64G
#SBATCH --time=00:10:00
#SBATCH --output=/scratch/dm1487/logs/mine-%j.out
#SBATCH --error=/scratch/dm1487/logs/mine-%j.err
set -euo pipefail
NAMO_REPO=${NAMO_REPO:-/cache/home/dm1487/projects/namo/namo_cpp}
cd "$NAMO_REPO"
source scripts/amarel/activate.sh
python scripts/build_phase2_manifest.py \
    --phase1-dir "$1" \
    --output     "$2" \
    --workers   "$SLURM_CPUS_PER_TASK" \
    --include-reasons all_pushes_failed no_reachable_objects no_reachable_edges
