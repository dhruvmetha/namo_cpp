#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   ./scripts/run_region_opening_ml_sequential.sh [extra-args]
# Example:
#   ./scripts/run_region_opening_ml_sequential.sh --start-idx 0 --end-idx 10

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
PYTHON_BIN="python"
COLLECT_SCRIPT="$REPO_DIR/python/namo/data_collection/sequential_ml_collection.py"
YAML_CONFIG="$REPO_DIR/python/namo/data_collection/region_opening_ml_collection.yaml"

# Ensure python path includes repo root
export PYTHONPATH="${REPO_DIR}/python:${PYTHONPATH:-}"

echo "🚀 Running Sequential ML Data Collection"
echo "📜 Script: $COLLECT_SCRIPT"
echo "⚙️  Config: $YAML_CONFIG"
echo "=================================================="

exec "$PYTHON_BIN" "$COLLECT_SCRIPT" \
  --config-yaml "$YAML_CONFIG" \
  "$@"
