#!/bin/bash
# REACTIVE ROLLOUT eval (the faithful deploy-matched horizon-Q number), sharded by scene over CPUs.
# Per array task: a slice of the manifest -> eval_rollout.py (model drives, sim only committed pushes,
# grade by reachability). Merge shard JSONs after with a tiny reducer (sum solve@k over n).
#
# Required env at submit:  CKPT, MANIFEST, OUT_DIR ; optional BUDGET (2), TOPK (10), SHARD (25)
#SBATCH --job-name=rollout
#SBATCH --partition=main,main
#SBATCH --array=0-39
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=03:00:00
#SBATCH --output=logs/rollout-%A_%a.out
#SBATCH --error=logs/rollout-%A_%a.err
set -euo pipefail

cd "$NAMO_REPO"
: "${MJ_PATH:?source env.<machine>.sh first}"
export LD_LIBRARY_PATH="$MJ_PATH/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$PWD/build_python:$PWD/python:$PWD/scripts:$PWD/scripts/sandbox:$SAGE_REPO"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2

: "${CKPT:?must set CKPT}"
: "${MANIFEST:?must set MANIFEST}"
: "${OUT_DIR:?must set OUT_DIR}"
BUDGET=${BUDGET:-2}; TOPK=${TOPK:-10}; SHARD=${SHARD:-25}
PRIOR=${PRIOR:-q}; W2=${W2:-1}; FLAT_H1=${FLAT_H1:-0}
flat_arg=""; [ "$FLAT_H1" = "1" ] && flat_arg="--flat-h1"
mkdir -p "$OUT_DIR"
START=$(( SLURM_ARRAY_TASK_ID * SHARD )); END=$(( START + SHARD ))
echo "task ${SLURM_ARRAY_TASK_ID}: scenes [$START,$END) budget=$BUDGET prior=$PRIOR w2=$W2 flat_h1=$FLAT_H1 on $(hostname)"

"$NAMO_PYTHON" -u scripts/sandbox/eval_rollout.py \
    --ckpt "$CKPT" --manifest "$MANIFEST" --budget "$BUDGET" --topk "$TOPK" \
    --prior "$PRIOR" --w2 "$W2" $flat_arg \
    --start "$START" --end "$END" \
    --out "$OUT_DIR/shard_${SLURM_ARRAY_TASK_ID}.json" \
    --leaf-out "$OUT_DIR/shard_${SLURM_ARRAY_TASK_ID}.jsonl"
echo "task ${SLURM_ARRAY_TASK_ID} DONE"
