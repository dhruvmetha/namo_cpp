#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   ./scripts/run_region_opening_collection.sh [extra-args]
# Example:
#   ./scripts/run_region_opening_collection.sh --start-idx 0 --end-idx 20 --workers 4

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
PYTHON_BIN="python"
COLLECT_SCRIPT="$REPO_DIR/python/namo/data_collection/modular_parallel_collection.py"
YAML_CONFIG="$REPO_DIR/python/namo/data_collection/region_opening_collection.yaml"

exec "$PYTHON_BIN" "$COLLECT_SCRIPT" \
  --config-yaml "$YAML_CONFIG" \
  --algorithm region_opening \
  "$@"


