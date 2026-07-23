#!/bin/bash
#SBATCH --job-name=c0h5
#SBATCH --account=general
#SBATCH --partition=main-redhat
#SBATCH --exclude=halk[0001-0159]
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=03:00:00
set -euo pipefail

: "${NAMO_REPO:?}"
: "${RESULT_ROOT:?}"
: "${SHARD_COUNT:?}"
cd "$NAMO_REPO"
source env.amarel.sh >/dev/null 2>&1
export PYTHONPATH="$NAMO_REPO/build_python:$NAMO_REPO/python:$NAMO_REPO/scripts:$NAMO_REPO/scripts/sandbox:$NAMO_REPO/scripts/pipeline:${PYTHONPATH:-}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 OPENCV_NUM_THREADS=1

shard_idx=${SLURM_ARRAY_TASK_ID:-0}
pkl_glob=${PKL_GLOB:-$RESULT_ROOT/collect/**/*_results.pkl}
out_dir=${H5_OUT_DIR:-$RESULT_ROOT/h5}
mkdir -p "$out_dir"

"$NAMO_PYTHON" scripts/pipeline/build_rung2_h5.py \
  --pkl-glob "$pkl_glob" \
  --out "$out_dir/candidates_$(printf '%03d' "$shard_idx").h5" \
  --shard-idx "$shard_idx" \
  --shard-count "$SHARD_COUNT" \
  --colossus0
