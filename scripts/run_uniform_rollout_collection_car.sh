#!/usr/bin/env bash
# Run a UniformRolloutSampler shard on the diff-drive-car env pool.
#
# Usage: ./scripts/run_uniform_rollout_collection_car.sh <START_IDX> <END_IDX> <OUTPUT_DIR>
# Example:
#   ./scripts/run_uniform_rollout_collection_car.sh 0 9950 \
#     /common/users/dm1487/namo_data/uniform_rollout_car_v0

set -euo pipefail

if [ $# -lt 3 ]; then
    echo "Usage: $0 <START_IDX> <END_IDX> <OUTPUT_DIR>"
    exit 1
fi

START_IDX="$1"
END_IDX="$2"
OUTPUT_DIR="$3"

MANIFEST="$(dirname "$0")/manifests/car_envs_100k.txt"
if [ ! -f "$MANIFEST" ]; then
    echo "Manifest not found: $MANIFEST"
    echo "Run: ./scripts/generate_car_envs_100k_manifest.sh first."
    exit 1
fi

cd "$(dirname "$0")/.."

python python/namo/data_collection/modular_parallel_collection.py \
    --algorithm uniform_rollout_sampler \
    --manifest "$MANIFEST" \
    --start-idx "$START_IDX" \
    --end-idx "$END_IDX" \
    --workers 100 \
    --output-dir "$OUTPUT_DIR" \
    --config-file config/namo_config_car.yaml \
    --primitive-prefix car_ \
    --sampler-max-chain-depth 1 \
    --sampler-region-goal-samples 5 \
    --sampler-num-depths 10
