#!/bin/bash
#SBATCH --job-name=khctx
#SBATCH --partition=unlimited
#SBATCH --nodelist=rlab7
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --array=0-3%4
#SBATCH --output=/common/users/dm1487/slurm/%x_%A_%a.out

set -euo pipefail

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1

REPO=${NAMO_REPO:-${SLURM_SUBMIT_DIR:?}}
cd "$REPO"
source env.ilab.sh

: "${TAG:?set TAG to a unique output tag}"
SOURCE_ROOT=${SOURCE_ROOT:-$NAMO_SCRATCH/eval/same_template_keyhole_20260824/scale25_b5_v1}
LIMIT=${LIMIT:-3}
MAX_ATTEMPTS=${MAX_ATTEMPTS:-500}
VARIANTS_PER_HOP=${VARIANTS_PER_HOP:-64}
SEED_BASE=${SEED_BASE:-90500}

pair_names=(mm mh hm hh)
pair=${pair_names[$SLURM_ARRAY_TASK_ID]}
out="$NAMO_SCRATCH/eval/$TAG/contact/$pair"

SECONDS=0
"$NAMO_PYTHON" scripts/pipeline/compose_keyhole_modules.py \
  --augment-contact-manifest "$SOURCE_ROOT/$pair/manifest.jsonl" \
  --out-dir "$out" \
  --limit "$LIMIT" \
  --max-attempts "$MAX_ATTEMPTS" \
  --contact-variants-per-hop "$VARIANTS_PER_HOP" \
  --seed "$((SEED_BASE + SLURM_ARRAY_TASK_ID))" \
  --allow-shortfall
echo "elapsed_seconds=$SECONDS output=$out"
