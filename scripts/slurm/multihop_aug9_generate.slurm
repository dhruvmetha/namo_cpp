#!/bin/bash
# Generate exact two-hop XMLs from all ten aug9_car_v3 templates.
# Array indexing is batch-major: array task ID % 10 selects the template.
#
# Example pilot:
#   OUT=/scratch/... NUM_ENVS=2 sbatch --array=0-9 scripts/slurm/multihop_aug9_generate.slurm
# Example scale (10 seed batches/template, 10 requested layouts/task):
#   OUT=/scratch/... NUM_ENVS=10 sbatch --array=0-99 scripts/slurm/multihop_aug9_generate.slurm
#SBATCH --job-name=mh2_gen
#SBATCH --partition=main-redhat
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=logs/%x_%A_%a.out
set -euo pipefail

: "${OUT:?set OUT to the generation output root}"

NAMO_REPO=${NAMO_REPO:-${SLURM_SUBMIT_DIR:?no NAMO_REPO and no SLURM_SUBMIT_DIR}}
ENV_CREATOR_REPO=${ENV_CREATOR_REPO:-$(dirname "$NAMO_REPO")/mujoco_env_creator}
NUM_ENVS=${NUM_ENVS:-10}
SEED_BASE=${SEED_BASE:-650000000}
# Initial shortest region-path length to filter for. Defaults to 2 so every prior
# two-hop invocation reproduces exactly; set to 3 or 4 for the deeper pools.
EXACT_HOP=${EXACT_HOP:-2}

templates=(
  set1/benchmark_1.xml
  set1/benchmark_2.xml
  set1/benchmark_3.xml
  set1/benchmark_4.xml
  set1/benchmark_5.xml
  set2/benchmark_1.xml
  set2/benchmark_2.xml
  set2/benchmark_3.xml
  set2/benchmark_4.xml
  set2/benchmark_5.xml
)

task_id=${SLURM_ARRAY_TASK_ID:-0}
template_id=$((task_id % ${#templates[@]}))
batch_id=$((task_id / ${#templates[@]}))
template_rel=${templates[$template_id]}
start_seed=$((SEED_BASE + template_id * 1000000 + batch_id * 10000))
run_offset=$((batch_id * NUM_ENVS))

cd "$NAMO_REPO"
source env.amarel.sh
export PYTHONPATH="$NAMO_REPO/build_python:$NAMO_REPO/python"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
PY=${NAMO_PYTHON:-python}
CONFIG="$NAMO_REPO/config/namo_config_complete_skill15_car_1x.yaml"
TEMPLATE_ROOT="$ENV_CREATOR_REPO/templates/aug9_car_v3"
mkdir -p "$OUT" "$NAMO_REPO/logs"

echo "GEN start $(date) host=$(hostname) task=$task_id template=$template_rel batch=$batch_id seed=$start_seed requested=$NUM_ENVS"
"$PY" "$ENV_CREATOR_REPO/generate_envs.py" "$TEMPLATE_ROOT/$template_rel" \
  --namo-config "$CONFIG" \
  --config "$NAMO_REPO/config/multihop_aug9_generation.yaml" \
  --num-envs "$NUM_ENVS" \
  --output-dir "$OUT" \
  --num-workers "$SLURM_CPUS_PER_TASK" \
  --start-seed "$start_seed" \
  --run-id-offset "$run_offset" \
  --num-objects-json "$TEMPLATE_ROOT/num_objects.json" \
  --object-size-range 0.06 0.16 \
  --object-half-height 0.05 \
  --goal-size 0.02 \
  --clearance-radius 0.0 \
  --min-goal-distance 0.0 \
  --no-require-adjacent \
  --exact-hop "$EXACT_HOP" \
  --samples-per-pair 1 \
  --min-region-area 0.02
echo "GEN done $(date) task=$task_id"
