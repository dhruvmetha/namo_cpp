#!/bin/bash
# Full F-characterization run: all 1767 test envs, 100 workers, exhaustive mode
# Run in screen on rlab7: screen -S f_char bash run_f_char_full.sh

set -e

export MJ_PATH=/common/users/dm1487/ktamp/mujoco
export PYTHONPATH=/common/home/dm1487/robotics_research/ktamp/namo/build_python_mjxrl_rlab7:$PYTHONPATH

NAMO_DIR=/common/home/dm1487/robotics_research/ktamp/namo
PYTHON=/common/users/dm1487/envs/mjxrl/bin/python
OUTPUT_DIR=/common/users/dm1487/namo_data/f_characterization/1_push_exhaustive_full

cd "$NAMO_DIR"

echo "Starting full F-characterization run"
echo "Output: $OUTPUT_DIR"
echo "Workers: 100"
echo "Envs: 1767 (full test set)"
echo "Start time: $(date)"
echo ""

$PYTHON python/namo/data_collection/modular_parallel_collection.py \
    --config-yaml python/namo/data_collection/region_opening_exhaustive.yaml \
    --output-dir "$OUTPUT_DIR" \
    --workers 100 \
    --start-idx 0 --end-idx 1767

echo ""
echo "Finished at: $(date)"
