#!/bin/bash
# Canonical hmax=2 best-first search for one registered eval set, sharded by room.
# Submit as --array=0-31 and set CKPT, OUT, and EVAL_SET=onepush_manifest|pure2push_manifest.
#SBATCH --job-name=hmax2_eval
#SBATCH --partition=unlimited
#SBATCH --time=03:00:00
#SBATCH --mem=4G
#SBATCH --array=0-31
#SBATCH --output=%x_%A_%a.out
set -euo pipefail

: "${CKPT:?set CKPT=<checkpoint path>}"
: "${OUT:?set OUT=<evaluation output directory>}"
: "${EVAL_SET:?set EVAL_SET=onepush_manifest|pure2push_manifest}"

REPO=${NAMO_REPO:-/common/home/dm1487/robotics_research/ktamp/namo}
cd "$REPO"
source env.ilab.sh >/dev/null 2>&1
export PYTHONPATH="${NAMO_BINDINGS:-$REPO/build_python}:$REPO/python"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 OPENCV_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=""
# Pin the margin to the value every registry row was made at. Without this the run picks up
# whatever config/wavefront_inflation.yaml currently holds, which moved to 1 mm on 2026-09-05
# and would silently produce numbers that do not compare to anything recorded.
export NAMO_CFG="$REPO/config/margin_5mm/namo_config_complete_skill15_car_1x.yaml"
PY=${NAMO_PYTHON:-python}
KEY=$($PY -m namo.eval_sets "$EVAL_SET")
N=$($PY -c 'import json, sys; print(len(json.load(open(sys.argv[1]))))' "$KEY")
I=$SLURM_ARRAY_TASK_ID
SHARDS=${SHARDS:-32}
S=$((I * N / SHARDS))
EN=$(((I + 1) * N / SHARDS))

mkdir -p "$OUT"
MARGIN=$(grep -o "base_inflation_margin_m: *[0-9.]*" "$(dirname "$NAMO_CFG")/wavefront_inflation.yaml" | awk '{print $2}')
echo "HMAX2 start $(date) host=$(hostname) eval_set=$EVAL_SET ckpt=$CKPT shard=$I range=[$S:$EN] margin=${MARGIN}m"
"$PY" scripts/sandbox/eval_bestfirst.py --ckpt "$CKPT" --key "$KEY" --hmax 2 --sim-budget 900 \
  --prior model --agg mean5 --combine q --discount off --start "$S" --end "$EN" \
  --out "$OUT/shard_${I}.json" --leaf-out "$OUT/shard_${I}.jsonl"
echo "HMAX2 done $(date) shard=$I"
