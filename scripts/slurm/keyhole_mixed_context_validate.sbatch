#!/bin/bash
#SBATCH --job-name=khctxval
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

: "${ROOT:?set ROOT to the approval-pilot artifact root}"

pair_names=(mm mh hm hh)
pair_titles=("Medium + Medium" "Medium + Hard" "Hard + Medium" "Hard + Hard")
pair=${pair_names[$SLURM_ARRAY_TASK_ID]}
title=${pair_titles[$SLURM_ARRAY_TASK_ID]}
source_manifest="$ROOT/selected/$pair/manifest.jsonl"
validated="$ROOT/revalidated/$pair"
renders="$ROOT/renders/$pair"

"$NAMO_PYTHON" scripts/pipeline/compose_keyhole_modules.py \
  --revalidate-manifest "$source_manifest" \
  --out-dir "$validated"

mapfile -t xmls < "$validated/xmls.txt"
"$NAMO_PYTHON" scripts/render_env_with_regions.py "${xmls[@]}" \
  --out-dir "$renders" \
  --montage-path "$ROOT/renders/${pair}_montage.png" \
  --montage-title "$title"
