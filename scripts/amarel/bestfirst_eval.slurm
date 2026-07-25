#!/bin/bash
# Value-guided greedy BEST-FIRST search eval, sharded by scene over CPUs (objective = min sims).
# Required env: CKPT, MANIFEST, OUT_DIR ; optional HMAX(2) SIM_BUDGET(30) PRIOR(model) AGG(mean5) COMBINE(blend) SHARD(25)
#SBATCH --job-name=bestfirst
#SBATCH --partition=main,main-redhat
#SBATCH --array=0-39
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=03:00:00
#SBATCH --output=logs/bestf-%A_%a.out
#SBATCH --error=logs/bestf-%A_%a.err
set -euo pipefail
cd "$NAMO_REPO"
: "${MJ_PATH:?source env.<machine>.sh first}"
export LD_LIBRARY_PATH="$MJ_PATH/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$PWD/build_python:$PWD/python:$PWD/scripts:$PWD/scripts/sandbox:$SAGE_REPO"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2

: "${CKPT:?}"; : "${MANIFEST:?}"; : "${OUT_DIR:?}"
HMAX=${HMAX:-2}; SIM_BUDGET=${SIM_BUDGET:-900}; PRIOR=${PRIOR:-model}; AGG=${AGG:-mean5}; COMBINE=${COMBINE:-blend}; SHARD=${SHARD:-13}
KEY=${KEY:-$("$NAMO_PYTHON" -m namo.eval_sets pure2push_manifest)}
SEED_BASE=${SEED_BASE:-7000}; DIVE_BONUS=${DIVE_BONUS:-0}
mkdir -p "$OUT_DIR"
START=$(( SLURM_ARRAY_TASK_ID * SHARD )); END=$(( START + SHARD ))
echo "task ${SLURM_ARRAY_TASK_ID}: [$START,$END) hmax=$HMAX budget=$SIM_BUDGET prior=$PRIOR seed=$SEED_BASE dive=$DIVE_BONUS key=$(basename $KEY) on $(hostname)"
"$NAMO_PYTHON" -u scripts/sandbox/eval_bestfirst.py \
    --ckpt "$CKPT" --manifest "$MANIFEST" --hmax "$HMAX" --sim-budget "$SIM_BUDGET" --key "$KEY" \
    --prior "$PRIOR" --agg "$AGG" --combine "$COMBINE" --seed-base "$SEED_BASE" --dive-bonus "$DIVE_BONUS" ${RAW:+--raw} --start "$START" --end "$END" \
    --out "$OUT_DIR/shard_${SLURM_ARRAY_TASK_ID}.json" --leaf-out "$OUT_DIR/shard_${SLURM_ARRAY_TASK_ID}.jsonl"
echo "task ${SLURM_ARRAY_TASK_ID} DONE"
