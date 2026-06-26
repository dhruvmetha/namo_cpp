#!/bin/bash
# STAGE 0 instrumentation (realized setup/finish ranks vs the exhaustive pairmap), sharded by xml-index.
# Env: CKPT, OUT_DIR. Opt: SHARD(160), HSETUP(2). Read-only on training.
#SBATCH --job-name=stage0
#SBATCH --partition=main,main-redhat
#SBATCH --array=0-12
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/dm1487/logs/stage0-%A_%a.out
#SBATCH --error=/scratch/dm1487/logs/stage0-%A_%a.err
set -euo pipefail
cd /cache/home/dm1487/projects/namo/namo_cpp
export MJ_PATH=/scratch/dm1487/mujoco/mujoco-3.2.7
export LD_LIBRARY_PATH="$MJ_PATH/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$PWD/build_python:$PWD/python:$PWD/scripts:$PWD/scripts/sandbox:$PWD/scripts/pipeline:/cache/home/dm1487/projects/namo/sage_learning"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 CUDA_VISIBLE_DEVICES=""
: "${CKPT:?}"; : "${OUT_DIR:?}"; SHARD=${SHARD:-160}; HSETUP=${HSETUP:-2}
START=$(( SLURM_ARRAY_TASK_ID * SHARD )); END=$(( START + SHARD )); mkdir -p "$OUT_DIR"
/scratch/dm1487/envs/namo/bin/python -u scripts/sandbox/stage0_instrument.py \
    --ckpt "$CKPT" --hsetup "$HSETUP" --start "$START" --end "$END" --out "$OUT_DIR/shard_${SLURM_ARRAY_TASK_ID}.json"
