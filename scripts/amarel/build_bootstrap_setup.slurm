#!/bin/bash
# Build the bootstrap-SETUP H5 from exhaustive training labels (s0 renders only, no sims). SHARDED by episode-index
# (render is ~1.8s/ep due to the wavefront rebuild -> a single task blows the wall; shards write incrementally + safe).
# Env: VSUMMARY(density|depth), OUT_DIR. Opt: SHARD(260). Each task writes OUT_DIR/shard_<id>.h5.
#SBATCH --job-name=bootbuild
#SBATCH --partition=main,main-redhat
#SBATCH --array=0-19
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=01:30:00
#SBATCH --output=logs/bootbuild-%A_%a.out
#SBATCH --error=logs/bootbuild-%A_%a.err
set -euo pipefail
cd "$NAMO_REPO"
: "${MJ_PATH:?source env.<machine>.sh first}"
export LD_LIBRARY_PATH="$MJ_PATH/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$PWD/build_python:$PWD/python:$PWD/scripts:$PWD/scripts/sandbox:$PWD/scripts/pipeline:$SAGE_REPO"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 CUDA_VISIBLE_DEVICES=""
: "${VSUMMARY:?}"; : "${OUT_DIR:?}"; SHARD=${SHARD:-260}
CKPT=${CKPT:-"$NAMO_SCRATCH/sage_outputs/scorer/qfull_nohz_v3_v4hq_s1/namo-classifier/wl8k6iyv/checkpoints/epoch012-val_loss0.6896.ckpt"}
START=$(( SLURM_ARRAY_TASK_ID * SHARD )); END=$(( START + SHARD )); mkdir -p "$OUT_DIR"
"$NAMO_PYTHON" -u scripts/sandbox/build_bootstrap_setup.py \
    --ckpt "$CKPT" --vsummary "$VSUMMARY" --start "$START" --end "$END" --out-h5 "$OUT_DIR/shard_${SLURM_ARRAY_TASK_ID}.h5"
