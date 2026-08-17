#!/bin/bash
# Sharded HY5U or uniform-random Full-NAMO evaluation for exact-two-hop XMLs.
# Every keyhole gets a fresh 300-simulation budget; hmax remains two pushes.
#
# Example:
#   MANIFEST=/scratch/.../xmls.txt OUT=/scratch/.../eval N_SHARDS=20 PRIOR=model \
#     CKPT=/cache/.../HY5U_s2.ckpt \
#     sbatch --array=0-19 scripts/slurm/multihop_aug9_eval.slurm
#SBATCH --job-name=mh2_eval
#SBATCH --partition=main-redhat
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --output=logs/%x_%A_%a.out
set -euo pipefail

: "${MANIFEST:?set MANIFEST to the generated XML manifest}"
: "${OUT:?set OUT to the sharded evaluation output root}"
: "${N_SHARDS:?set N_SHARDS to the submitted array length}"
: "${CKPT:?set CKPT to the HY5U checkpoint}"
PRIOR=${PRIOR:-model}
SEARCH_SEED=${SEARCH_SEED:-42}

NAMO_REPO=${NAMO_REPO:-${SLURM_SUBMIT_DIR:?no NAMO_REPO and no SLURM_SUBMIT_DIR}}
task_id=${SLURM_ARRAY_TASK_ID:-0}
total=$(wc -l < "$MANIFEST")
start=$((task_id * total / N_SHARDS))
end=$(((task_id + 1) * total / N_SHARDS))

if [ "$start" -ge "$end" ]; then
  echo "EVAL empty shard task=$task_id range=[$start,$end) total=$total"
  exit 0
fi

cd "$NAMO_REPO"
source env.amarel.sh
export PYTHONPATH="$NAMO_REPO/build_python:$NAMO_REPO/python"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
PY=${NAMO_PYTHON:-python}
CONFIG="$NAMO_REPO/config/namo_config_complete_skill15_car_1x.yaml"
shard_out="$OUT/shard_$(printf '%04d' "$task_id")"
mkdir -p "$shard_out" "$NAMO_REPO/logs"
sed -n "$((start + 1)),${end}p" "$MANIFEST" > "$shard_out/input.txt"

echo "EVAL start $(date) host=$(hostname) task=$task_id range=[$start,$end) total=$total prior=$PRIOR seed=$SEARCH_SEED"
"$PY" -m namo.solvability_runner \
  --manifest "$shard_out/input.txt" \
  --path-length 2 \
  --config-file "$CONFIG" \
  --goal-strategy scorer \
  --region-max-chain-depth 2 \
  --primitive-data-dir "$NAMO_REPO/data" \
  --primitive-prefix 1x_car_d5_ \
  --region-success-min-reachable 20 \
  --goals-per-region 100 \
  --seed "$SEARCH_SEED" \
  --simulation-budget "${SIM_BUDGET:-900}" \
  --simulation-budget-scope keyhole \
  --local-search best_first \
  --best-first-prior "$PRIOR" \
  --scorer-ckpt "$CKPT" \
  --ml-device cpu \
  --workers "$SLURM_CPUS_PER_TASK" \
  --output-dir "$shard_out"
echo "EVAL done $(date) task=$task_id"
