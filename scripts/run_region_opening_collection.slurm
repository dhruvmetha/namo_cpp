#!/bin/bash
#SBATCH --job-name=namo_region_collect
#SBATCH --output=logs/region_collect_%j.out
#SBATCH --error=logs/region_collect_%j.err
#SBATCH --time=24:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=64
#SBATCH --partition=main

# ============================================================
# SLURM script for region opening data collection
#
# Usage:
#   sbatch scripts/run_region_opening_collection.slurm
#   sbatch --export=START_IDX=0,END_IDX=50 scripts/run_region_opening_collection.slurm
#
# Monitor:
#   squeue -u $USER              # Check job status
#   tail -f logs/region_collect_<job_id>.out  # Watch output
#   scancel <job_id>             # Cancel job
# ============================================================

set -euo pipefail

# Print job info
echo "============================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "Start time: $(date)"
echo "============================================"

# Change to repo directory
cd /common/home/dm1487/robotics_research/ktamp/namo

# Activate environment
source /common/users/dm1487/envs/mjxrl/bin/activate

# Set defaults (can be overridden via --export)
START_IDX=${START_IDX:-0}
END_IDX=${END_IDX:-100}
WORKERS=${WORKERS:-$SLURM_CPUS_PER_TASK}

echo "Collecting indices $START_IDX to $END_IDX with $WORKERS workers"
echo "============================================"

# Run the collection script
./scripts/run_region_opening_collection.sh \
  --start-idx "$START_IDX" \
  --end-idx "$END_IDX" \
  --workers "$WORKERS"

echo "============================================"
echo "End time: $(date)"
echo "Job completed successfully"
