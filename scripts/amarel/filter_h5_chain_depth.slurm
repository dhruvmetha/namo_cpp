#!/bin/bash
# Filter an existing H5 by maximum push-step count and write a new H5 plus JSON summary.
#
# Inputs (env vars, override at submit time):
#   INPUT_H5         source HDF5 file
#   OUTPUT_H5        filtered HDF5 file
#   NPZ_ROOT         original NPZ root used to build the H5 (required for minimal H5s)
#   MAX_PUSH_STEPS   keep only scenarios with at most this many push steps
#   SUMMARY_JSON     optional JSON summary path
#   COPY_BATCH_SIZE  HDF5 row-copy batch size
#
# Default settings below are tuned for the current batch750 minimal H5.

#SBATCH --job-name=namo-h5-filter
#SBATCH --partition=main-redhat
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/tdn39/Robotics/namo/namo_cpp/slurm_logs/h5filter-%j.out
#SBATCH --error=/scratch/tdn39/Robotics/namo/namo_cpp/slurm_logs/h5filter-%j.err

set -euo pipefail

NAMO_REPO=${NAMO_REPO:-/scratch/tdn39/Robotics/namo/namo_cpp}
SAGE_REPO=${SAGE_REPO:-/scratch/tdn39/Robotics/namo/sage_learning}
mkdir -p /scratch/tdn39/Robotics/namo/namo_cpp/slurm_logs
source "$NAMO_REPO/scripts/amarel/activate.sh"

if hostname | grep -q '^halk'; then
    echo "ERROR: refusing to run H5 filter on incompatible halk node $(hostname)." >&2
    exit 2
fi

INPUT_H5=${INPUT_H5:-/scratch/tdn39/Robotics/namo/namo_cpp/collection_runs/batch750/h5/batch750_local_minimal.h5}
OUTPUT_H5=${OUTPUT_H5:-/scratch/tdn39/Robotics/namo/namo_cpp/collection_runs/batch750/h5/batch750_local_minimal_max5push.h5}
NPZ_ROOT=${NPZ_ROOT:-/scratch/tdn39/Robotics/namo/namo_cpp/collection_runs/batch750/masks/all_npz}
MAX_PUSH_STEPS=${MAX_PUSH_STEPS:-${MAX_CHAIN_DEPTH:-5}}
COPY_BATCH_SIZE=${COPY_BATCH_SIZE:-256}
SUMMARY_JSON=${SUMMARY_JSON:-${OUTPUT_H5%.h5}.summary.json}

echo "host=$(hostname)  job=${SLURM_JOB_ID:-N/A}"
echo "input:   $INPUT_H5"
echo "output:  $OUTPUT_H5"
echo "npzroot: $NPZ_ROOT"
echo "steps:   $MAX_PUSH_STEPS"
echo "batch:   $COPY_BATCH_SIZE"
echo "start:   $(date -Is)"

PYTHONPATH="$SAGE_REPO" python "$SAGE_REPO/scripts/filter_hdf5_by_chain_depth.py" \
    "$INPUT_H5" \
    "$OUTPUT_H5" \
    --npz-root "$NPZ_ROOT" \
    --max-push-steps "$MAX_PUSH_STEPS" \
    --summary-json "$SUMMARY_JSON" \
    --copy-batch-size "$COPY_BATCH_SIZE"

echo "end:     $(date -Is)"
ls -lh "$OUTPUT_H5" "$SUMMARY_JSON"
