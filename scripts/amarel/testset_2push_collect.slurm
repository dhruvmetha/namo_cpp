#!/bin/bash
# CANONICAL TEST SET — exhaustive depth-2 ground-truth collection (car).
# Produces the 2-push answer key: which FIRST-push (edge,depth) enables a solving
# second push, per scene. Also re-derives depth-1 F (cross-check vs the validset).
#
# Reads the geometry-verified canonical scene manifest (REAL paths), shards by
# SLURM_ARRAY_TASK_ID, stages symlinks, runs region_opening with:
#   region_max_chain_depth=2 + region_exhaustive_mode + record-ALL-solutions
#   collisions-ALLOWED + target-goal-region   (the PINNED settings)
#
# DEFAULT scope = ALL 2173 canonical scenes (canonical_scenes.txt) -> uniform exhaustive
# depth-1 AND depth-2 labels for the WHOLE set (one source; 1-push tier gets cross-checked +
# gains F1'). To collect ONLY the 985 genuine-2-push subset, override:
#   MANIFEST=$HOME_DIR/manifests/twopush_scenes.txt NUM_SHARDS=16 sbatch testset_2push_collect.slurm
#
#SBATCH --job-name=ts-2push
#SBATCH --partition=main-redhat
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=/scratch/dm1487/datasets/namo_testset_v1/logs/ts-2push-%A_%a.out
#SBATCH --error=/scratch/dm1487/datasets/namo_testset_v1/logs/ts-2push-%A_%a.err
#SBATCH --array=0-31

set -euo pipefail
NAMO_REPO=${NAMO_REPO:-/cache/home/dm1487/projects/namo/namo_cpp}
HOME_DIR=${HOME_DIR:-/scratch/dm1487/datasets/namo_testset_v1}
mkdir -p "$HOME_DIR/logs"
source "$NAMO_REPO/scripts/amarel/activate.sh"

MANIFEST=${MANIFEST:-$HOME_DIR/manifests/canonical_scenes.txt}
TOTAL_ENVS=$(wc -l < "$MANIFEST")
NUM_SHARDS=${NUM_SHARDS:-32}
SHARD_SIZE=$(( (TOTAL_ENVS + NUM_SHARDS - 1) / NUM_SHARDS ))   # ceil
TASK_ID=${SLURM_ARRAY_TASK_ID:-0}
START_IDX=$(( TASK_ID * SHARD_SIZE ))
END_IDX=$(( START_IDX + SHARD_SIZE ))
if [ $END_IDX -gt $TOTAL_ENVS ]; then END_IDX=$TOTAL_ENVS; fi
if [ $START_IDX -ge $TOTAL_ENVS ]; then echo "shard $TASK_ID beyond manifest, nothing to do"; exit 0; fi

PKL_SUBDIR=${PKL_SUBDIR:-pkls_2push_v2}
SHARD_DIR=$HOME_DIR/${PKL_SUBDIR}/shard_${TASK_ID}
mkdir -p "$SHARD_DIR"
SHARD_MANIFEST="$SHARD_DIR/shard_manifest.txt"
sed -n "$((START_IDX+1)),${END_IDX}p" "$MANIFEST" > "$SHARD_MANIFEST"

# Stage scenes into a flat env dir (the CLI's --xml-dir expects a directory).
# Names embed parent dir to keep them unique across runs.
SHARD_ENVS="$SHARD_DIR/envs"
mkdir -p "$SHARD_ENVS"
cnt=0
while IFS= read -r xml; do
    [ -z "$xml" ] && continue
    base=$(basename "$xml" .xml)
    parent=$(basename "$(dirname "$xml")")
    gp=$(basename "$(dirname "$(dirname "$xml")")")
    ln -sf "$xml" "$SHARD_ENVS/${gp}_${parent}_${base}_${cnt}.xml"
    cnt=$((cnt+1))
done < "$SHARD_MANIFEST"
N=$cnt

echo "host=$(hostname) job=${SLURM_JOB_ID:-NA}_${TASK_ID} cpus=${SLURM_CPUS_PER_TASK}"
echo "manifest=$MANIFEST shard=[$START_IDX,$END_IDX) envs=$N"
echo "start=$(date -Is)"

cd "$NAMO_REPO"
CONFIG_YAML=${CONFIG_YAML:-$HOME_DIR/configs/exhaustive_depth2.yaml}
echo "config=$CONFIG_YAML"
PYTHONPATH="$NAMO_REPO/build_python:$NAMO_REPO/python" \
python -m namo.data_collection.modular_parallel_collection \
    --config-yaml "$CONFIG_YAML" \
    --algorithm region_opening \
    --output-dir "$SHARD_DIR/pkls" \
    --xml-dir "$SHARD_ENVS" \
    --start-idx 0 --end-idx "$N" \
    --workers "$SLURM_CPUS_PER_TASK" \
    --episodes-per-env 1 \
    --region-max-chain-depth 2 \
    --region-max-solutions-per-neighbor 9999 \
    --region-max-recorded-solutions-per-neighbor 9999 \
    --primitive-data-dir "$NAMO_REPO/data" \
    --primitive-prefix "1x_car_d5_" \
    --goal-strategy primitive \
    --target-goal-region \
    --goals-per-region ${GOALS_PER_REGION:-100} \
    --region-allow-collisions \
    --region-min-reachable-fraction 0.2 \
    --search-timeout ${SEARCH_TIMEOUT:-1800} \
    --config-file config/namo_config_complete_skill15_car_1x.yaml

echo "end=$(date -Is)"
echo "PKLs: $(find "$SHARD_DIR/pkls" -name '*.pkl' | wc -l)"
