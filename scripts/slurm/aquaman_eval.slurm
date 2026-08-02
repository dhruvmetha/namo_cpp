#!/bin/bash
# Aquaman-0 canonical eval: BOTH tiers at hmax=2, discount OFF (matches registry row
# deploy-nodiscount-hmax2-v1; see EXP-2026-08-02-bootstrap-value-loop). Clone of
# eval_d20_model.slurm with the 1push leg moved to eval_bestfirst @ hmax=2
# [USER 2026-08-02: 1push tier at hmax=2 always; no discount].
# Submit with --array=0-71 and set CKPT, OUT. 1push = tasks 0-39 (1323 eps / 34 per shard),
# 2push = tasks 40-71 (32 shards).
#SBATCH --job-name=aq0_eval
#SBATCH --partition=unlimited
#SBATCH --time=03:00:00
#SBATCH --mem=16G
#SBATCH --output=logs/%x_%A_%a.out
set -euo pipefail

: "${CKPT:?set CKPT=<checkpoint path>}"
: "${OUT:?set OUT=<evaluation output root>}"

REPO=${NAMO_REPO:-/common/home/dm1487/robotics_research/ktamp/namo}
cd "$REPO"
source env.ilab.sh >/dev/null 2>&1
export PYTHONPATH="${NAMO_BINDINGS:-$REPO/build_python}:$REPO/python"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 OPENCV_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=""
PY="${NAMO_PYTHON:-python}"
command -v "$PY" >/dev/null 2>&1 || PY=/common/users/dm1487/envs/mjxrl/bin/python
ID=$SLURM_ARRAY_TASK_ID

echo "AQ0 EVAL start $(date) host=$(hostname) CKPT=$CKPT ID=$ID"
if [ "$ID" -lt 40 ]; then
  KEY=$("$PY" -m namo.eval_sets onepush_manifest)
  N1=$("$PY" -c 'import json,sys; print(len(json.load(open(sys.argv[1]))))' "$KEY")
  S=$((ID * N1 / 40)); EN=$(((ID + 1) * N1 / 40))
  mkdir -p "$OUT/1push_hmax2"
  "$PY" scripts/sandbox/eval_bestfirst.py --ckpt "$CKPT" --key "$KEY" --hmax 2 --sim-budget 900 \
    --prior model --agg mean5 --combine q --discount off --start "$S" --end "$EN" \
    --out "$OUT/1push_hmax2/shard_${ID}.json" --leaf-out "$OUT/1push_hmax2/shard_${ID}.jsonl"
else
  I=$((ID - 40))
  N2=$($PY -c 'import json; from namo import eval_sets; print(len(json.load(open(eval_sets.PURE2PUSH))))')
  S=$((I * N2 / 32)); EN=$(((I + 1) * N2 / 32))
  mkdir -p "$OUT/2push"
  "$PY" scripts/sandbox/eval_bestfirst.py --ckpt "$CKPT" --hmax 2 --sim-budget 900 \
    --prior model --agg mean5 --combine q --discount off --start "$S" --end "$EN" \
    --out "$OUT/2push/shard_${I}.json" --leaf-out "$OUT/2push/shard_${I}.jsonl"
fi
echo "AQ0 EVAL done $(date) ID=$ID"
