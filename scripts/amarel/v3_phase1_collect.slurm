#!/bin/bash
# v3 phase-1 data collection — sharded array.
# Reads /scratch/dm1487/manifests/v3_feb_top250k.txt (250K XMLs),
# slices by SLURM_ARRAY_TASK_ID into 60 shards of ~4167 envs each.
#
# Phase 1 = single-push opening with exhaustive_mode + random_rollout + target-goal.
# Pattern B (uniform random over valid solutions per attempt).
#
#SBATCH --job-name=v3-p1
#SBATCH --partition=main-redhat
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=/scratch/dm1487/logs/v3-p1-%A_%a.out
#SBATCH --error=/scratch/dm1487/logs/v3-p1-%A_%a.err
#SBATCH --array=0-59

set -euo pipefail
NAMO_REPO=${NAMO_REPO:-/cache/home/dm1487/projects/namo/namo_cpp}
# OUTPUT_DIR overridable so the same script drives multiple corpora
# (e.g. test_aug9_phase1, test_feb_phase1). Default preserves v3_feb behavior.
OUTPUT_DIR=${OUTPUT_DIR:-/scratch/dm1487/outputs/v3_phase1}
mkdir -p /scratch/dm1487/logs "$OUTPUT_DIR"
source "$NAMO_REPO/scripts/amarel/activate.sh"

MANIFEST=${MANIFEST:-/scratch/dm1487/manifests/v3_feb_top250k.txt}
TOTAL_ENVS=$(wc -l < "$MANIFEST")
NUM_SHARDS=60
SHARD_SIZE=$(( (TOTAL_ENVS + NUM_SHARDS - 1) / NUM_SHARDS ))   # ceil division
TASK_ID=${SLURM_ARRAY_TASK_ID:-0}
START_IDX=$(( TASK_ID * SHARD_SIZE ))
END_IDX=$(( START_IDX + SHARD_SIZE ))
if [ $END_IDX -gt $TOTAL_ENVS ]; then
    END_IDX=$TOTAL_ENVS
fi

# Slice manifest into shard-local file
SHARD_DIR=$OUTPUT_DIR/shard_${TASK_ID}
mkdir -p "$SHARD_DIR"
SHARD_MANIFEST="$SHARD_DIR/shard_manifest.txt"
sed -n "$((START_IDX+1)),${END_IDX}p" "$MANIFEST" > "$SHARD_MANIFEST"
N=$(wc -l < "$SHARD_MANIFEST")

# Stage XMLs into a shard env dir (symlinks — the modular_parallel_collection
# CLI expects --xml-dir to be a flat directory, not a manifest of paths)
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
echo "manifest=$MANIFEST  shard=[$START_IDX,$END_IDX)  envs=$N  envs_dir=$SHARD_ENVS"
echo "start=$(date -Is)"

# Exhaustive config: enables region_exhaustive_mode for true pattern B
EXHAUSTIVE_CONFIG=/scratch/dm1487/v3_smoke_100/exhaustive_config.yaml

cd "$NAMO_REPO"
PYTHONPATH="$NAMO_REPO/build_python:$NAMO_REPO/python" \
python -m namo.data_collection.modular_parallel_collection \
    --config-yaml "$EXHAUSTIVE_CONFIG" \
    --algorithm region_opening \
    --output-dir "$SHARD_DIR/pkls" \
    --xml-dir "$SHARD_ENVS" \
    --start-idx 0 --end-idx "$N" \
    --workers "$SLURM_CPUS_PER_TASK" \
    --episodes-per-env 1 \
    --region-max-chain-depth 1 \
    --region-max-solutions-per-neighbor 9999 \
    --region-max-recorded-solutions-per-neighbor 1 \
    --rollout-samples-per-state 600 \
    --primitive-data-dir "$NAMO_REPO/data" \
    --primitive-prefix "1x_car_d5_" \
    --goal-strategy random_rollout \
    --target-goal-region \
    --goals-per-region "${GOALS_PER_REGION:-50}" \
    --region-allow-collisions \
    --region-min-reachable-fraction 0.2 \
    --search-timeout 60 \
    --config-file config/namo_config_complete_skill15_car_1x.yaml

echo "end=$(date -Is)"
echo "PKLs: $(find "$SHARD_DIR/pkls" -name '*.pkl' | wc -l)"
