#!/bin/bash
# Source this file inside an `srun --pty bash` shell (or a compute-node tmux)
# to get a working NAMO environment on Amarel.
#
#   source scripts/amarel/activate.sh
#
# Not for use on login nodes (modules + cuda etc. require a compute allocation).

if [ -z "${SLURM_JOB_ID:-}" ]; then
  echo "⚠️  No SLURM_JOB_ID set. You're either on a login node or in a stale shell."
  echo "    Grab a compute node first:"
  echo "      unset SLURM_JOB_ID"
  echo "      srun --partition=main --cpus-per-task=4 --mem=8G --time=2:00:00 --pty bash"
  echo "    then source this file."
  return 1 2>/dev/null || exit 1
fi

module use /projects/community/modulefiles
module load gcc/14.2.0-cermak cmake/3.31.8-rdp135

source /cache/home/dm1487/miniforge3/etc/profile.d/conda.sh
conda activate /scratch/dm1487/envs/namo

export MJ_PATH=/scratch/dm1487/mujoco/mujoco-3.2.7
export LD_LIBRARY_PATH="$MJ_PATH/lib:/scratch/dm1487/envs/namo/lib:${LD_LIBRARY_PATH:-}"

REPO=/cache/home/dm1487/projects/namo/namo_cpp
export PYTHONPATH="$REPO/build_python_mjxrl_amarel2:$REPO/python:${PYTHONPATH:-}"

# Prevent BLAS oversubscription when running multiprocessing.Pool.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONUNBUFFERED=1

cd "$REPO"
echo "✓ NAMO env active on $(hostname). PYTHONPATH set. CWD=$REPO"
