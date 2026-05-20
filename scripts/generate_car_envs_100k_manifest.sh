#!/usr/bin/env bash
# Build the manifest file for the diff-drive-car env pool at
# /common/users/dm1487/corl2026/namo/envs_100k/.
#
# Output: scripts/manifests/car_envs_100k.txt — one XML path per line.

set -euo pipefail

ENV_ROOT="/common/users/dm1487/corl2026/namo/envs_100k"
OUTPUT_DIR="$(dirname "$0")/manifests"
OUTPUT_FILE="$OUTPUT_DIR/car_envs_100k.txt"

mkdir -p "$OUTPUT_DIR"

cd "$(dirname "$0")/.."

# Generate the manifest from the env pool.
python scripts/generate_xml_manifest.py \
    --input-dir "$ENV_ROOT" \
    --output "$OUTPUT_FILE"

echo ""
echo "Manifest generated: $OUTPUT_FILE"
wc -l "$OUTPUT_FILE"
