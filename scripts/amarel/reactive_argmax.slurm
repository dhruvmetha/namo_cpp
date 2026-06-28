#!/bin/bash
# PURE reactive (argmax setup@H2 -> argmax finish@H1), sharded by xml-index. Env: CKPT, OUT_DIR. Opt: SHARD(80).
#SBATCH --job-name=reactarg
#SBATCH --partition=main,main-redhat
#SBATCH --array=0-12
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=01:00:00
#SBATCH --output=logs/reactarg-%A_%a.out
#SBATCH --error=logs/reactarg-%A_%a.err
set -euo pipefail
cd "$NAMO_REPO"
: "${MJ_PATH:?source env.<machine>.sh first}"
export LD_LIBRARY_PATH="$MJ_PATH/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$PWD/build_python:$PWD/python:$PWD/scripts:$PWD/scripts/sandbox:$PWD/scripts/pipeline:$SAGE_REPO"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 CUDA_VISIBLE_DEVICES=""
: "${CKPT:?}"; : "${OUT_DIR:?}"; SHARD=${SHARD:-80}
START=$(( SLURM_ARRAY_TASK_ID * SHARD )); END=$(( START + SHARD )); mkdir -p "$OUT_DIR"
"$NAMO_PYTHON" -u scripts/sandbox/eval_reactive_argmax.py \
    --ckpt "$CKPT" --prior "${PRIOR:-q}" --seed "${SEED:-7000}" --start "$START" --end "$END" --out "$OUT_DIR/shard_${SLURM_ARRAY_TASK_ID}.json"
