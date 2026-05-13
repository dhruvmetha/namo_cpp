#!/bin/bash
set -e
export MJ_PATH=/common/users/dm1487/ktamp/mujoco
NAMO_DIR=/common/home/dm1487/robotics_research/ktamp/namo
cd "$NAMO_DIR"
bash build_python_mjxrl.sh
