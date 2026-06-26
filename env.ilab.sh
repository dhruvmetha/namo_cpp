# env.ilab.sh — per-machine paths for Rutgers CS ilab. Source from the repo root: `source env.ilab.sh`
# (no python-dotenv auto-load; the code reads these from the environment — see docs/PORTABILITY.md §1)
export NAMO_SCRATCH=/common/users/dm1487/scratch_namo                       # base; datasets/h5/manifests/outputs derive from this
export SAGE_REPO=/common/users/dm1487/fresh_start/projects/namo/sage_learning
export MJ_PATH=/common/users/dm1487/fresh_start/mujoco/mujoco-3.2.7         # adjust to where MuJoCo 3.2.7 actually lives
export NAMO_GLOBAL_SEED=42
# runtime:
export PYTHONPATH="$PWD/build_python:$PWD/python:$PWD/scripts:$PWD/scripts/sandbox:$PWD/scripts/pipeline:$SAGE_REPO"
export LD_LIBRARY_PATH="$MJ_PATH/lib:${LD_LIBRARY_PATH:-}"
echo "[env.ilab] NAMO_SCRATCH=$NAMO_SCRATCH SAGE_REPO=$SAGE_REPO MJ_PATH=$MJ_PATH"
