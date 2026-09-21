#!/bin/bash
# Render MORE's collected trees into per-shard H5 files, one array task per group.
#
# The build is embarrassingly parallel across collection shards: each renders its own
# states and writes its own H5, and the trainer reads the set. Rendering measured at
# roughly 30 states in 6 seconds including startup, so this is minutes rather than the
# hours a single core would take.
#
# Usage (from the repo root on Amarel):
#   ROOT=$NAMO_SCRATCH/more_collect_20260921_v2 NSHARDS=20 \
#     sbatch --array=0-19 scripts/slurm/more_build_h5.slurm
#
#SBATCH --job-name=more-h5
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%A_%a.out
set -euo pipefail

: "${ROOT:?set ROOT to the campaign dir holding raw/}"
: "${NSHARDS:?set NSHARDS to the array size}"
NAMO_REPO=${NAMO_REPO:-${SLURM_SUBMIT_DIR:?no NAMO_REPO and no SLURM_SUBMIT_DIR}}
cd "$NAMO_REPO"
source scripts/amarel/activate.sh
source env.amarel.sh
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
       NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 OPENCV_FOR_THREADS_NUM=1
export CUDA_VISIBLE_DEVICES=""          # rendering is the visualizer, never the model

I=${SLURM_ARRAY_TASK_ID:?}
mapfile -t ALL < <(ls "$ROOT"/raw/*.jsonl | sort)
MINE=()
for ((k=I; k<${#ALL[@]}; k+=NSHARDS)); do MINE+=("${ALL[$k]}"); done   # round-robin
echo "MORE-H5 start $(date) host=$(hostname) task=$I files=${#MINE[@]} of ${#ALL[@]}"
if [ ${#MINE[@]} -eq 0 ]; then echo "no files for this task"; exit 0; fi

mkdir -p "$ROOT/h5"
"$NAMO_PYTHON" scripts/rl_loop/more_build_h5.py \
  --shards "${MINE[@]}" \
  --out "$ROOT/h5/part_$(printf '%04d' "$I").h5" \
  ${NO_REACH_NEGATIVE:+--no-reach-negative}

echo "MORE-H5 done $(date) task=$I"
