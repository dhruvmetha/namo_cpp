#!/bin/bash
# v3 cascade collection — reusable for phases 2..5E.
# Parameterized via env vars (set at sbatch time):
#   MANIFEST     full path to the phase manifest (one XML path per line)
#   OUTPUT_DIR   where per-shard pkls land (e.g. $NAMO_OUTPUTS/v3_phase2)
#   DEPTH        --region-max-chain-depth (2 for phases 2-5)
#   KROLL        --rollout-samples-per-state (20 phase2, 50 phase3+)
#   NUM_SHARDS   array width (must match --array=0-(NUM_SHARDS-1) at submit)
#   GOAL_SEED    optional --seed for phase-5 seed sweep (default unset)
#
# NO exhaustive mode (phases 2-5 use normal early-stop pruning).
# d5 primitives + 1x car config, target-goal-region, pattern B (record 1 random).
#
#SBATCH --job-name=v3-casc
#SBATCH --partition=main-redhat
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/v3-casc-%A_%a.out
#SBATCH --error=logs/v3-casc-%A_%a.err

set -euo pipefail
NAMO_REPO="${NAMO_REPO:?source env.<machine>.sh first}"
source "$NAMO_REPO/scripts/amarel/activate.sh"

: "${MANIFEST:?set MANIFEST}"
: "${OUTPUT_DIR:?set OUTPUT_DIR}"
: "${DEPTH:?set DEPTH}"
: "${KROLL:?set KROLL}"
: "${NUM_SHARDS:?set NUM_SHARDS}"

mkdir -p "$NAMO_LOGS" "$OUTPUT_DIR"

TOTAL_ENVS=$(wc -l < "$MANIFEST")
SHARD_SIZE=$(( (TOTAL_ENVS + NUM_SHARDS - 1) / NUM_SHARDS ))
TASK_ID=${SLURM_ARRAY_TASK_ID:-0}
START_IDX=$(( TASK_ID * SHARD_SIZE ))
END_IDX=$(( START_IDX + SHARD_SIZE ))
if [ $END_IDX -gt $TOTAL_ENVS ]; then END_IDX=$TOTAL_ENVS; fi
if [ $START_IDX -ge $TOTAL_ENVS ]; then echo "empty shard $TASK_ID"; exit 0; fi

SHARD_DIR="$OUTPUT_DIR/shard_${TASK_ID}"
mkdir -p "$SHARD_DIR"
SHARD_MANIFEST="$SHARD_DIR/shard_manifest.txt"
sed -n "$((START_IDX+1)),${END_IDX}p" "$MANIFEST" > "$SHARD_MANIFEST"
N=$(wc -l < "$SHARD_MANIFEST")

SHARD_ENVS="$SHARD_DIR/envs"
mkdir -p "$SHARD_ENVS"
cnt=0
while IFS= read -r xml; do
    base=$(basename "$xml" .xml)
    parent=$(basename "$(dirname "$xml")")
    ln -sf "$xml" "$SHARD_ENVS/${parent}_${base}_${cnt}.xml"
    cnt=$((cnt+1))
done < "$SHARD_MANIFEST"

echo "host=$(hostname) job=${SLURM_JOB_ID:-N/A}_${TASK_ID} cpus=${SLURM_CPUS_PER_TASK}"
echo "manifest=$MANIFEST shard=[$START_IDX,$END_IDX) envs=$N depth=$DEPTH K=$KROLL"
echo "start=$(date -Is)"

SEED_ARG=""
if [ -n "${GOAL_SEED:-}" ]; then SEED_ARG="--shuffle-seed $GOAL_SEED"; fi  # seeds RandomRolloutGoalStrategy (region_opening.py:613)

# Phase 1 only: enable exhaustive mode via an override config (true pattern B).
EXH_ARG=""
if [ -n "${EXHAUSTIVE_CONFIG:-}" ]; then EXH_ARG="--config-yaml $EXHAUSTIVE_CONFIG"; fi

cd "$NAMO_REPO"
PYTHONPATH="$NAMO_REPO/build_python:$NAMO_REPO/python" \
python -m namo.data_collection.modular_parallel_collection \
    $EXH_ARG \
    --algorithm region_opening \
    --output-dir "$SHARD_DIR/pkls" \
    --xml-dir "$SHARD_ENVS" \
    --start-idx 0 --end-idx "$N" \
    --workers "$SLURM_CPUS_PER_TASK" \
    --episodes-per-env 1 \
    --region-max-chain-depth "$DEPTH" \
    --region-max-solutions-per-neighbor 9999 \
    --region-max-recorded-solutions-per-neighbor 1 \
    --rollout-samples-per-state "$KROLL" \
    --primitive-data-dir "$NAMO_REPO/data" \
    --primitive-prefix "1x_car_d5_" \
    --goal-strategy random_rollout \
    --target-goal-region \
    --goals-per-region 50 \
    --region-allow-collisions \
    --region-min-reachable-fraction 0.2 \
    --search-timeout 120 \
    $SEED_ARG \
    --config-file config/namo_config_complete_skill15_car_1x.yaml

echo "end=$(date -Is)"
echo "PKLs: $(find "$SHARD_DIR/pkls" -name '*.pkl' | wc -l)"
