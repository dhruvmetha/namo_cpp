#!/bin/bash
# Build the unified minimal H5 dataset from all phase NPZ dirs.
# Serial conversion (single H5 writer), 4 cores for h5py + I/O headroom.
#
# Inputs (env vars, override at submit time):
#   STAGE_DIR    parent dir containing symlinks to phase mask dirs
#                (default: $NAMO_H5/all_npz)
#   OUTPUT_H5    destination file
#                (default: $NAMO_H5/car_v1_aug9_all.h5)
#
#SBATCH --job-name=namo-h5-build
#SBATCH --partition=main-redhat
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/h5build-%j.out
#SBATCH --error=logs/h5build-%j.err

set -euo pipefail
NAMO_REPO="${NAMO_REPO:?source env.<machine>.sh first}"
SAGE_REPO="${SAGE_REPO:?source env.<machine>.sh first}"
source "$NAMO_REPO/scripts/amarel/activate.sh"

STAGE_DIR=${STAGE_DIR:-"$NAMO_H5/all_npz"}
OUTPUT_H5=${OUTPUT_H5:-"$NAMO_H5/car_v1_aug9_all.h5"}

echo "host=$(hostname)  job=${SLURM_JOB_ID:-N/A}"
echo "stage:  $STAGE_DIR"
echo "output: $OUTPUT_H5"
echo "start:  $(date -Is)"

PYTHONPATH="$SAGE_REPO" python "$SAGE_REPO/scripts/convert_to_hdf5.py" \
    "$STAGE_DIR" \
    "$OUTPUT_H5" \
    --minimal

echo "end:    $(date -Is)"
ls -lh "$OUTPUT_H5"
