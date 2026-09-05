#!/bin/bash
#SBATCH --job-name=khreval
#SBATCH --partition=unlimited
#SBATCH --nodelist=rlab7
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=/common/users/dm1487/slurm/%x_%j.out

set -euo pipefail

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1

REPO=${NAMO_REPO:-${SLURM_SUBMIT_DIR:?}}
cd "$REPO"
source env.ilab.sh

: "${SOURCE_MANIFEST:?set SOURCE_MANIFEST to the manifest to replay}"
: "${OUT_DIR:?set OUT_DIR to the revalidation output directory}"

"$NAMO_PYTHON" scripts/pipeline/compose_keyhole_modules.py \
  --revalidate-manifest "$SOURCE_MANIFEST" \
  --out-dir "$OUT_DIR"
