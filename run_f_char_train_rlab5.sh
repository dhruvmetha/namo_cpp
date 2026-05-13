#!/bin/bash
# F-characterization training data: envs 0-1500 on rlab5 (80 CPUs, 60 workers)
set -e
export MJ_PATH=/common/users/dm1487/ktamp/mujoco
export PYTHONPATH=/common/home/dm1487/robotics_research/ktamp/namo/build_python_mjxrl_rlab5:$PYTHONPATH

NAMO_DIR=/common/home/dm1487/robotics_research/ktamp/namo
PYTHON=/common/users/dm1487/envs/mjxrl/bin/python

cd "$NAMO_DIR"

echo "F-char TRAINING data: envs 0-1500 on rlab5"
echo "Start time: $(date)"

$PYTHON python/namo/data_collection/modular_parallel_collection.py \
    --config-yaml python/namo/data_collection/region_opening_exhaustive_train.yaml \
    --output-dir /common/users/dm1487/namo_data/f_characterization/1_push_exhaustive_train \
    --workers 60 \
    --start-idx 0 --end-idx 1500

echo "Finished at: $(date)"
