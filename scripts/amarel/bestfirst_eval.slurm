#!/bin/bash
# Value-guided greedy BEST-FIRST search eval, sharded by scene over CPUs (objective = min sims).
# Required env: CKPT, OUT_DIR ; optional KEY, MANIFEST override, HMAX(2), SIM_BUDGET(30), PRIOR(model),
# AGG(mean5), COMBINE(blend), DISCOUNT(off), TAU(1), GAMMA(.65), RAW(0), SEED_BASE(7000), SHARD(13).
#SBATCH --job-name=bestfirst
#SBATCH --partition=main
#SBATCH --array=0-39
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=03:00:00
#SBATCH --output=logs/bestf-%A_%a.out
#SBATCH --error=logs/bestf-%A_%a.err
set -euo pipefail
cd "$NAMO_REPO"
: "${MJ_PATH:?source env.<machine>.sh first}"
export LD_LIBRARY_PATH="$MJ_PATH/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$PWD/build_python:$PWD/python:$PWD/scripts:$PWD/scripts/sandbox:$SAGE_REPO"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1

: "${CKPT:?}"; : "${OUT_DIR:?}"
HMAX=${HMAX:-2}; SIM_BUDGET=${SIM_BUDGET:-900}; PRIOR=${PRIOR:-model}; AGG=${AGG:-mean5}; COMBINE=${COMBINE:-blend}; SHARD=${SHARD:-13}
DISCOUNT=${DISCOUNT:-off}; TAU=${TAU:-1.0}; GAMMA=${GAMMA:-0.65}; RAW=${RAW:-0}
KEY=${KEY:-$("$NAMO_PYTHON" -m namo.eval_sets pure2push_manifest)}
ONLY_KEY=${ONLY_KEY:-}
SEED_BASE=${SEED_BASE:-7000}; DIVE_BONUS=${DIVE_BONUS:-0}
mkdir -p "$OUT_DIR"
START=$(( SLURM_ARRAY_TASK_ID * SHARD )); END=$(( START + SHARD ))
MANIFEST_ARGS=()
if [[ -n "${MANIFEST:-}" ]]; then MANIFEST_ARGS=(--manifest "$MANIFEST"); fi
RAW_ARGS=()
if [[ "$RAW" == "1" ]]; then RAW_ARGS=(--raw); fi
ONLY_ARGS=()
if [[ -n "$ONLY_KEY" ]]; then ONLY_ARGS=(--only-key "$ONLY_KEY"); fi
echo "task ${SLURM_ARRAY_TASK_ID}: [$START,$END) hmax=$HMAX budget=$SIM_BUDGET prior=$PRIOR seed=$SEED_BASE combine=$COMBINE discount=$DISCOUNT tau=$TAU dedupe=on jam_prune=on key=$(basename "$KEY") on $(hostname)"
"$NAMO_PYTHON" -u scripts/sandbox/eval_bestfirst.py \
    --ckpt "$CKPT" "${MANIFEST_ARGS[@]}" --hmax "$HMAX" --sim-budget "$SIM_BUDGET" --key "$KEY" "${ONLY_ARGS[@]}" \
    --prior "$PRIOR" --agg "$AGG" --combine "$COMBINE" --discount "$DISCOUNT" --tau "$TAU" --gamma "$GAMMA" \
    --seed-base "$SEED_BASE" --dive-bonus "$DIVE_BONUS" "${RAW_ARGS[@]}" --start "$START" --end "$END" \
    --out "$OUT_DIR/shard_${SLURM_ARRAY_TASK_ID}.json" --leaf-out "$OUT_DIR/shard_${SLURM_ARRAY_TASK_ID}.jsonl"
echo "task ${SLURM_ARRAY_TASK_ID} DONE"
