#!/bin/bash
# F-characterization training data: envs 1500-3000 on rlab6 (48 CPUs, 40 workers)
set -e
export MJ_PATH=/common/users/dm1487/ktamp/mujoco
export PYTHONPATH=/common/home/dm1487/robotics_research/ktamp/namo/build_python_mjxrl_rlab6:$PYTHONPATH

NAMO_DIR=/common/home/dm1487/robotics_research/ktamp/namo
PYTHON=/common/users/dm1487/envs/mjxrl/bin/python

cd "$NAMO_DIR"

echo "F-char TRAINING data: envs 1500-3000 on rlab6"
echo "Start time: $(date)"

$PYTHON python/namo/data_collection/modular_parallel_collection.py \
    --config-yaml python/namo/data_collection/region_opening_exhaustive_train.yaml \
    --output-dir /common/users/dm1487/namo_data/f_characterization/1_push_exhaustive_train \
    --workers 40 \
    --start-idx 1500 --end-idx 3000

echo "Finished at: $(date)"
