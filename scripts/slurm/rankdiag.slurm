#!/bin/bash
#SBATCH --job-name=rankdiag
#SBATCH --partition=unlimited
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=%x_%j.out
set -euo pipefail

: "${BASE_CKPT:?set BASE_CKPT}"
: "${TREAT_CKPT:?set TREAT_CKPT}"
: "${H5:?set H5}"
: "${OUT:?set OUT}"

REPO=${NAMO_REPO:-/common/home/dm1487/robotics_research/ktamp/namo}
cd "$REPO"
source env.ilab.sh >/dev/null 2>&1
PY="${NAMO_PYTHON:-python}"
command -v "$PY" >/dev/null 2>&1 || PY=/common/users/dm1487/envs/mjxrl/bin/python
export PYTHONPATH="${NAMO_BINDINGS:-$REPO/build_python}:$REPO/python:$REPO/scripts:$REPO/scripts/sandbox"
LIMIT_ARGS=()
if [ -n "${LIMIT:-}" ]; then
  LIMIT_ARGS=(--limit "$LIMIT")
fi

"$PY" scripts/rl_loop/score_round2_eval.py \
  --ckpt "$BASE_CKPT" --ckpt "$TREAT_CKPT" \
  --h5 "$H5" --out "$OUT" --batch-size "${BATCH:-256}" "${LIMIT_ARGS[@]}"
