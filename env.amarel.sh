# env.amarel.sh — per-machine paths for Amarel (the original box; these match the hardcoded defaults).
export NAMO_SCRATCH=/scratch/dm1487
export SAGE_REPO=/cache/home/dm1487/projects/namo/sage_learning
export MJ_PATH=/scratch/dm1487/mujoco/mujoco-3.2.7
export NAMO_GLOBAL_SEED=42
export PYTHONPATH="$PWD/build_python:$PWD/python:$PWD/scripts:$PWD/scripts/sandbox:$PWD/scripts/pipeline:$SAGE_REPO"
export LD_LIBRARY_PATH="$MJ_PATH/lib:${LD_LIBRARY_PATH:-}"
echo "[env.amarel] NAMO_SCRATCH=$NAMO_SCRATCH"
