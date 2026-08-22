#!/bin/bash
#SBATCH --job-name=khmods
#SBATCH --partition=main
#SBATCH --account=general
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:15:00
#SBATCH --array=0-39%40

set -euo pipefail

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1

REPO=${NAMO_REPO:-${SLURM_SUBMIT_DIR:?}}
cd "$REPO"
source env.amarel.sh

: "${TAG:?set TAG to a unique output tag}"
LIMIT=${LIMIT:-5}
MAX_ATTEMPTS=${MAX_ATTEMPTS:-100}
MIN_SEPARATION=${MIN_SEPARATION:-0.30}
SEED_BASE=${SEED_BASE:-220822}

templates=(
  set1/benchmark_1 set1/benchmark_2 set1/benchmark_3 set1/benchmark_4 set1/benchmark_5
  set2/benchmark_1 set2/benchmark_2 set2/benchmark_3 set2/benchmark_4 set2/benchmark_5
)
left_tiers=(medium medium hard hard)
right_tiers=(medium hard medium hard)
pair_names=(medium_medium medium_hard hard_medium hard_hard)

task=${SLURM_ARRAY_TASK_ID}
pair_index=$((task / 10))
template_index=$((task % 10))
template=${templates[$template_index]}
left=${left_tiers[$pair_index]}
right=${right_tiers[$pair_index]}
pair=${pair_names[$pair_index]}
template_slug=${template//\//_}
out="$NAMO_SCRATCH/eval/$TAG/$pair/$template_slug"

SECONDS=0
python scripts/pipeline/compose_keyhole_modules.py \
  --horizons 1push 1push \
  --tiers "$left" "$right" \
  --template "$template" \
  --out-dir "$out" \
  --limit "$LIMIT" \
  --max-attempts "$MAX_ATTEMPTS" \
  --min-separation "$MIN_SEPARATION" \
  --seed "$((SEED_BASE + task))" \
  --replay-donor-actions \
  --allow-shortfall
echo "elapsed_seconds=$SECONDS output=$out"
