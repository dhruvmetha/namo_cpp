#!/bin/bash
# Exhaustively collect one focused pure-2push episode per array task.
# Required env: MANIFEST, OUT_DIR, CONFIG_YAML, CKPT. Optional: SHARD(1), WORKERS(1).
#SBATCH --job-name=fill2p-gt
#SBATCH --partition=main
#SBATCH --array=0-36
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/fill2p-gt-%A_%a.out
#SBATCH --error=logs/fill2p-gt-%A_%a.err
set -euo pipefail
cd "${NAMO_REPO:?}"
source scripts/amarel/activate.sh >/dev/null 2>&1
cd "$NAMO_REPO"
export PYTHONPATH="$PWD/build_python:$PWD/python:$PWD/scripts:$PWD/scripts/sandbox:$PWD/scripts/pipeline:${SAGE_REPO:-}:${PYTHONPATH:-}"
export NAMO_PIN_THREADS=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 OPENCV_NUM_THREADS=1

: "${MANIFEST:?}"
: "${OUT_DIR:?}"
: "${CONFIG_YAML:?}"
: "${CKPT:?}"
SHARD=${SHARD:-1}
WORKERS=${WORKERS:-1}
start=$((SLURM_ARRAY_TASK_ID * SHARD))
end=$((start + SHARD))

"$NAMO_PYTHON" python/namo/data_collection/modular_parallel_collection.py \
  --config-yaml "$CONFIG_YAML" \
  --manifest "$MANIFEST" --start-idx "$start" --end-idx "$end" --workers "$WORKERS" \
  --scorer-ckpt "$CKPT" --ml-device cpu --target-goal-region \
  --output-dir "$OUT_DIR" --unique-run-dir --run-name "fill_gt_t${SLURM_ARRAY_TASK_ID}"
echo "FILL2P task $SLURM_ARRAY_TASK_ID DONE @ $(date -Is)"
