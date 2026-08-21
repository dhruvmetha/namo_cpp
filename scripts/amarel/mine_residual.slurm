#!/bin/bash
# Mine residual phase candidates from a phase output directory.
# Args:
#   $1 = phase output dir
#   $2 = residual manifest output path
#
# Excludes goal_region_not_in_snapshot (fundamentally unsolvable, waste of compute).
#
#SBATCH --partition=main
#SBATCH --cpus-per-task=64
#SBATCH --mem=64G
#SBATCH --time=00:10:00
#SBATCH --output=logs/mine-%j.out
#SBATCH --error=logs/mine-%j.err
set -euo pipefail
NAMO_REPO="${NAMO_REPO:?source env.<machine>.sh first}"
cd "$NAMO_REPO"
source scripts/amarel/activate.sh
python scripts/build_phase2_manifest.py \
    --phase1-dir "$1" \
    --output     "$2" \
    --workers   "$SLURM_CPUS_PER_TASK" \
    --include-reasons all_pushes_failed no_reachable_objects no_reachable_edges
