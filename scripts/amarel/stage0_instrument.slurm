#!/bin/bash
# STAGE 0 instrumentation (realized setup/finish ranks vs the exhaustive pairmap), sharded by xml-index.
# Env: CKPT, OUT_DIR. Opt: SHARD(160), HSETUP(2). Read-only on training.
#SBATCH --job-name=stage0
#SBATCH --partition=main,main
#SBATCH --array=0-12
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=02:00:00
#SBATCH --output=logs/stage0-%A_%a.out
#SBATCH --error=logs/stage0-%A_%a.err
set -euo pipefail
cd "$NAMO_REPO"
: "${MJ_PATH:?source env.<machine>.sh first}"
export LD_LIBRARY_PATH="$MJ_PATH/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$PWD/build_python:$PWD/python:$PWD/scripts:$PWD/scripts/sandbox:$PWD/scripts/pipeline:$SAGE_REPO"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 CUDA_VISIBLE_DEVICES=""
: "${CKPT:?}"; : "${OUT_DIR:?}"; SHARD=${SHARD:-160}; HSETUP=${HSETUP:-2}
START=$(( SLURM_ARRAY_TASK_ID * SHARD )); END=$(( START + SHARD )); mkdir -p "$OUT_DIR"
"$NAMO_PYTHON" -u scripts/sandbox/stage0_instrument.py \
    --ckpt "$CKPT" --hsetup "$HSETUP" --start "$START" --end "$END" --out "$OUT_DIR/shard_${SLURM_ARRAY_TASK_ID}.json"
