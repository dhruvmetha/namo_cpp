#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   ./scripts/run_region_opening_visual.sh --xml-file /path/to/env.xml [extra-args]
# Examples:
#   ./scripts/run_region_opening_visual.sh --xml-file ../ml4kp_ktamp/resources/models/.../env.xml --show-solution auto
#   ./scripts/run_region_opening_visual.sh --xml-file env.xml --region-max-chain-depth 2 --region-max-solutions-per-neighbor 5

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
PYTHON_BIN="python"
VIS_SCRIPT="$REPO_DIR/python/namo/visualization/visual_test_single.py"
YAML_CONFIG="$REPO_DIR/python/namo/data_collection/region_opening_collection.yaml"

exec "$PYTHON_BIN" "$VIS_SCRIPT" \
  --config-yaml "$YAML_CONFIG" \
  --algorithm region_opening \
  "$@"


