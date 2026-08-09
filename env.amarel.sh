# env.amarel.sh — per-machine paths for Amarel (the original box; these match the hardcoded defaults).
# Source from the repo root: `source env.amarel.sh`
export NAMO_REPO="$PWD"                                   # this repo (namo_cpp); sourced from repo root
export NAMO_SCRATCH=/scratch/dm1487
export SAGE_REPO=/cache/home/dm1487/projects/namo/sage_learning
export MJ_PATH=/scratch/dm1487/mujoco/mujoco-3.2.7
export NAMO_GLOBAL_SEED=42
export NAMO_PYTHON=/scratch/dm1487/envs/namo/bin/python   # interpreter used by slurm/sh scripts
# derived roots (mirror namo.paths; override individually only if your layout differs):
export NAMO_DATASETS="${NAMO_DATASETS:-$NAMO_SCRATCH/datasets}"
export NAMO_H5="${NAMO_H5:-$NAMO_SCRATCH/h5}"
export NAMO_MANIFESTS="${NAMO_MANIFESTS:-$NAMO_SCRATCH/manifests}"
export NAMO_OUTPUTS="${NAMO_OUTPUTS:-$NAMO_SCRATCH/outputs}"
export NAMO_LOGS="${NAMO_LOGS:-$NAMO_SCRATCH/logs}"
# Put the env's bin on PATH. Sourcing this used to set NAMO_PYTHON but leave bare `cmake`,
# `python3` etc. resolving to the SYSTEM ones -- so ./build_python_bindings.sh died with
# "cmake: command not found" and picked python3.9 on a compute node, even though both live in
# $(dirname $NAMO_PYTHON). Every script worked around it separately (see train.slurm gotcha 1).
export PATH="$(dirname "$NAMO_PYTHON"):$PATH"
export PYTHONPATH="$PWD/build_python:$PWD/python:$PWD/scripts:$PWD/scripts/sandbox:$PWD/scripts/pipeline:$SAGE_REPO"
export LD_LIBRARY_PATH="$MJ_PATH/lib:${LD_LIBRARY_PATH:-}"
echo "[env.amarel] NAMO_SCRATCH=$NAMO_SCRATCH"
