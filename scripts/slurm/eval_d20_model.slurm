#!/bin/bash
# Canonical d20 comparison harness: 1push fixed-cut search (6 shards) + pure-2push search (32 shards).
# Submit with --array=0-37 and set CKPT, OUT, and optionally NAMO_REPO.
#SBATCH --job-name=d20_eval
#SBATCH --partition=unlimited
#SBATCH --time=03:00:00
#SBATCH --mem=16G
#SBATCH --array=0-37
#SBATCH --output=%x_%A_%a.out
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
NOHZ=${NOHZ:-"$NAMO_OUTPUTS/scorer/qfull_nohz_v3_v4hq_s1/namo-classifier/wl8k6iyv/checkpoints/epoch012-val_loss0.6896.ckpt"}
ID=$SLURM_ARRAY_TASK_ID
KEY=${KEY:-$("$PY" -m namo.eval_sets onepush_manifest)}

echo "EVAL start $(date) host=$(hostname) CKPT=$CKPT ID=$ID"
if [ "$ID" -lt 6 ]; then
  BND=(0 221 442 663 884 1105 1323)
  S=${BND[$ID]}
  EN=${BND[$((ID + 1))]}
  mkdir -p "$OUT/1push"
  "$PY" scripts/sandbox/time_bestfirst.py --hmax 1 --budget 300 \
    --key "$KEY" \
    --hz-ckpt "$CKPT" --nohz-ckpt "$NOHZ" \
    --models Hz --start "$S" --end "$EN" --out "$OUT/1push/shard_${S}_${EN}.jsonl"
else
  I=$((ID - 6))
  S=$((I * 1018 / 32))
  EN=$(((I + 1) * 1018 / 32))
  mkdir -p "$OUT/2push"
  "$PY" scripts/sandbox/eval_bestfirst.py --ckpt "$CKPT" --hmax 2 --sim-budget 900 \
    --prior model --agg mean5 --combine q --start "$S" --end "$EN" \
    --out "$OUT/2push/shard_${I}.json" --leaf-out "$OUT/2push/shard_${I}.jsonl"
fi
echo "EVAL done $(date) ID=$ID"
