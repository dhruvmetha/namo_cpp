# env.ilab.sh — per-machine paths for Rutgers CS ilab. Source from the repo root: `source env.ilab.sh`
# (no python-dotenv auto-load; the code reads these from the environment — see docs/PORTABILITY.md §1)
export NAMO_REPO="$PWD"                                                      # this repo (namo_cpp); sourced from repo root
export NAMO_SCRATCH=/common/users/dm1487/scratch_namo                       # base; datasets/h5/manifests/outputs derive from this
export SAGE_REPO="$(dirname "$PWD")/sage_learning"                          # sister folder of this repo (source from repo root)
export MJ_PATH=/common/users/dm1487/ktamp/mujoco                            # MuJoCo 3.2.8 (libmujoco.so.3.2.8, header 328) — NB Amarel runs 3.2.7
export NAMO_GLOBAL_SEED=42
export NAMO_PYTHON=python                                                    # interpreter for slurm/sh; adjust to your env (e.g. /path/to/conda/envs/namo/bin/python)
# derived roots (mirror namo.paths; override individually only if your layout differs):
export NAMO_DATASETS="${NAMO_DATASETS:-$NAMO_SCRATCH/datasets}"
export NAMO_H5="${NAMO_H5:-$NAMO_SCRATCH/h5}"
export NAMO_MANIFESTS="${NAMO_MANIFESTS:-$NAMO_SCRATCH/manifests}"
export NAMO_OUTPUTS="${NAMO_OUTPUTS:-$NAMO_SCRATCH/outputs}"
export NAMO_LOGS="${NAMO_LOGS:-$NAMO_SCRATCH/logs}"
# runtime:
export PYTHONPATH="$PWD/build_python:$PWD/python:$PWD/scripts:$PWD/scripts/sandbox:$PWD/scripts/pipeline:$SAGE_REPO"
export LD_LIBRARY_PATH="$MJ_PATH/build/lib:$MJ_PATH/lib:${LD_LIBRARY_PATH:-}"   # ilab MuJoCo is a source build -> build/lib
echo "[env.ilab] NAMO_SCRATCH=$NAMO_SCRATCH SAGE_REPO=$SAGE_REPO MJ_PATH=$MJ_PATH"
