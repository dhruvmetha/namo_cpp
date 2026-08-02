#!/bin/bash
# Aquaman-0 depth arm: hmax=3 on subset180 (EXP-2026-07-25 protocol: discount off, budget 900),
# comparable to that card's model/random h3 baselines. Submit --array=0-29 with CKPT, OUT.
#SBATCH --job-name=aq0_depth
#SBATCH --partition=unlimited
#SBATCH --time=03:00:00
#SBATCH --mem=16G
#SBATCH --exclude=ilab1
#SBATCH --output=logs/%x_%A_%a.out
set -euo pipefail
: "${CKPT:?}"; : "${OUT:?}"
REPO=${NAMO_REPO:-/common/home/dm1487/robotics_research/ktamp/namo}
cd "$REPO"; source env.ilab.sh >/dev/null 2>&1
export PYTHONPATH="${NAMO_BINDINGS:-$REPO/build_python}:$REPO/python"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 OPENCV_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=""
PY="${NAMO_PYTHON:-python}"; command -v "$PY" >/dev/null 2>&1 || PY=/common/users/dm1487/envs/mjxrl/bin/python
MAN=/common/users/dm1487/scratch_namo/curriculum2/beast/round3/eval/hmax_depth/subset180.txt
I=$SLURM_ARRAY_TASK_ID; S=$((I * 6)); EN=$((S + 6))
mkdir -p "$OUT"
"$PY" scripts/sandbox/eval_bestfirst.py --ckpt "$CKPT" --manifest "$MAN" --hmax 3 --sim-budget 900 \
  --prior model --agg mean5 --combine q --discount off --start "$S" --end "$EN" \
  --out "$OUT/shard_${I}.json" --leaf-out "$OUT/shard_${I}.jsonl"
echo "depth shard $I done"
