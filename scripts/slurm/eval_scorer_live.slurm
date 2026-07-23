#!/bin/bash
#SBATCH --job-name=scorer_live
#SBATCH --partition=unlimited
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/common/users/dm1487/scratch_namo/eval/push_depth/logs/%x_%j.out
set -euo pipefail

: "${OUT:?set OUT=<result.json>}"

if [ -z "${CKPT:-}" ]; then
  : "${TRAIN_LOG:?set CKPT=<checkpoint> or TRAIN_LOG=<completed train.log>}"
  CKPT=$(sed -n 's/^\[train_q2\] best ckpt: //p' "$TRAIN_LOG" | tail -1)
  : "${CKPT:?completed training log has no best-checkpoint marker: $TRAIN_LOG}"
fi
test -s "$CKPT"

REPO=${NAMO_REPO:-/common/home/dm1487/robotics_research/ktamp/namo}
SAGE_OVERRIDE=${SAGE_REPO:-}
cd "$REPO"
source env.ilab.sh >/dev/null 2>&1
if [ -n "$SAGE_OVERRIDE" ]; then export SAGE_REPO="$SAGE_OVERRIDE"; fi

PY=/common/users/dm1487/envs/mjxrl/bin/python
export PATH="$(dirname "$PY"):$PATH"
BINDINGS=${NAMO_BUILD_PYTHON:-$REPO/build_python}
if ! compgen -G "$BINDINGS/namo_rl*.so" >/dev/null; then
  GIT_COMMON=$(git -C "$REPO" rev-parse --git-common-dir)
  BINDINGS="$(cd "$(dirname "$GIT_COMMON")" && pwd)/build_python"
fi
export PYTHONPATH="$BINDINGS:$REPO/python:$REPO/scripts:$REPO/scripts/sandbox:$REPO/scripts/pipeline:${SAGE_REPO:-}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 OPENCV_NUM_THREADS=1
export TMPDIR=/tmp/namo_${SLURM_JOB_ID}
mkdir -p "$TMPDIR" "$(dirname "$OUT")"

START=${START:-0}
END=${END:--1}
LEAF=${LEAF:-${OUT%.json}.jsonl}
echo "SCORER LIVE start=$(date) host=$(hostname) range=[$START:$END] ckpt=$CKPT"
"$PY" -u scripts/eval_scorer.py \
  --ckpt "$CKPT" --network edge_crossattn --num-depths 5 \
  --live-canonical --start "$START" --end "$END" --leaf-out "$LEAF" --out "$OUT"
echo "SCORER LIVE done=$(date) rows=$(wc -l < "$LEAF") out=$OUT"
