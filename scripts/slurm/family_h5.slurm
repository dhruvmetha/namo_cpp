#!/bin/bash
# FAMILY corpus H5 render array [EXP-2026-08-09]: build_rung2_h5.py --family-select over the base
# shards. Per-shard H5s land in $OUT; pooling happens after. ~200 shards x ~1-1.5h.
#SBATCH --job-name=family_h5
#SBATCH --partition=main
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=4:00:00
#SBATCH --output=logs/%x_%A_%a.out
set -uo pipefail
: "${OUT:?}"; : "${SHARDS:?}"
REPO=${NAMO_REPO:?set NAMO_REPO}
cd "$REPO" && source env.amarel.sh >/dev/null 2>&1
PY="${NAMO_PYTHON:?}"
export PYTHONPATH="$REPO/build_python:$REPO/python:${PYTHONPATH:-}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
mkdir -p "$OUT"
"$PY" scripts/pipeline/build_rung2_h5.py --family-select \
  --pkl-glob "${PKL_GLOB:?}" \
  --shard-idx "${SLURM_ARRAY_TASK_ID:-0}" --shard-count "$SHARDS" \
  --out "$OUT/family_shard_${SLURM_ARRAY_TASK_ID:-0}.h5"
