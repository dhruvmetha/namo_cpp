#!/bin/bash
#SBATCH --job-name=kh-easy2
#SBATCH --partition=unlimited
#SBATCH --nodelist=rlab7
#SBATCH --mem=8G
#SBATCH --time=01:00:00

set -euo pipefail

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1

REPO=${NAMO_REPO:-${SLURM_SUBMIT_DIR:?}}
cd "$REPO"
source env.ilab.sh

: "${OUT_DIR:?set OUT_DIR to a fresh output directory}"
LIMIT=${LIMIT:-50}
MAX_ATTEMPTS=${MAX_ATTEMPTS:-500}
SEED=${SEED:-220822}

SECONDS=0
"$NAMO_PYTHON" scripts/pipeline/compose_keyhole_modules.py \
  --horizons 1push 1push \
  --tiers easy easy \
  --template set2/benchmark_5 \
  --out-dir "$OUT_DIR" \
  --limit "$LIMIT" \
  --max-attempts "$MAX_ATTEMPTS" \
  --min-separation 0.30 \
  --seed "$SEED" \
  --replay-donor-actions
echo "elapsed_seconds=$SECONDS output=$OUT_DIR"
