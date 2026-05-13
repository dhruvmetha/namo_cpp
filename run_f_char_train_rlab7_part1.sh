#!/bin/bash
# F-char training data: envs 774-3000 (fill gaps) + 3000-6000 (new data)
# on rlab7 (256 CPUs, 100 workers)
set -e
export MJ_PATH=/common/users/dm1487/ktamp/mujoco
export PYTHONPATH=/common/home/dm1487/robotics_research/ktamp/namo/build_python_mjxrl_rlab7:$PYTHONPATH

NAMO_DIR=/common/home/dm1487/robotics_research/ktamp/namo
PYTHON=/common/users/dm1487/envs/mjxrl/bin/python

cd "$NAMO_DIR"

echo "F-char TRAINING data: envs 774-6000 on rlab7 (100 workers)"
echo "Start time: $(date)"

# Fill the gap: 774-1500
echo "=== Part 1: envs 774-1500 ==="
$PYTHON python/namo/data_collection/modular_parallel_collection.py \
    --config-yaml python/namo/data_collection/region_opening_exhaustive_train.yaml \
    --output-dir /common/users/dm1487/namo_data/f_characterization/1_push_exhaustive_train \
    --workers 100 \
    --start-idx 774 --end-idx 1500

# Fill the gap: 2277-3000
echo "=== Part 2: envs 2277-3000 ==="
$PYTHON python/namo/data_collection/modular_parallel_collection.py \
    --config-yaml python/namo/data_collection/region_opening_exhaustive_train.yaml \
    --output-dir /common/users/dm1487/namo_data/f_characterization/1_push_exhaustive_train \
    --workers 100 \
    --start-idx 2277 --end-idx 3000

# New data: 3000-6000
echo "=== Part 3: envs 3000-6000 ==="
$PYTHON python/namo/data_collection/modular_parallel_collection.py \
    --config-yaml python/namo/data_collection/region_opening_exhaustive_train.yaml \
    --output-dir /common/users/dm1487/namo_data/f_characterization/1_push_exhaustive_train \
    --workers 100 \
    --start-idx 3000 --end-idx 6000

echo ""
echo "Finished at: $(date)"
