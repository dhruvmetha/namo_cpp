# env.robotlearning.sh — per-machine paths for the RobotLearning workstation.
# Source from the repo root: `source env.robotlearning.sh` (see docs/PORTABILITY.md §1)
export NAMO_REPO="$PWD"                                                      # this repo (namo_cpp); sourced from repo root
export NAMO_SCRATCH=/home/dhruv/projects_dhruv/namo/scratch_namo             # base; datasets/h5/manifests/outputs derive from this
export SAGE_REPO="$(dirname "$PWD")/sage_learning"                           # sister folder of this repo
export MJ_PATH=/home/dhruv/2024/projects/mujoco_nightly                      # source build -> libs live under build/lib
export NAMO_GLOBAL_SEED=42
export NAMO_PYTHON="${NAMO_PYTHON:-/home/dhruv/miniconda3/envs/namo312/bin/python}"  # namo312: python 3.12.13, the SOABI build_python/ targets
# derived roots (mirror namo.paths; override individually only if your layout differs):
export NAMO_DATASETS="${NAMO_DATASETS:-$NAMO_SCRATCH/datasets}"
export NAMO_H5="${NAMO_H5:-$NAMO_SCRATCH/h5}"
export NAMO_MANIFESTS="${NAMO_MANIFESTS:-$NAMO_SCRATCH/manifests}"
export NAMO_OUTPUTS="${NAMO_OUTPUTS:-$NAMO_SCRATCH/outputs}"
export NAMO_LOGS="${NAMO_LOGS:-$NAMO_SCRATCH/logs}"
# runtime:
export PATH="$(dirname "$NAMO_PYTHON"):$PATH"
export PYTHONPATH="$PWD/build_python:$PWD/python:$PWD/scripts:$PWD/scripts/sandbox:$PWD/scripts/pipeline:$SAGE_REPO"
export LD_LIBRARY_PATH="$MJ_PATH/build/lib:$MJ_PATH/lib:${LD_LIBRARY_PATH:-}"
echo "[env.robotlearning] NAMO_SCRATCH=$NAMO_SCRATCH SAGE_REPO=$SAGE_REPO MJ_PATH=$MJ_PATH"
