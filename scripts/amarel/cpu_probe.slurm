#!/bin/bash
# Generic CPU runner for sandbox PROBE scripts (model runs on CPU; no GPU). Submit instead of srun-ing into an
# interactive alloc (which gets reclaimed + piggybacks the user's session). Required env: SCRIPT. Optional: ARGS.
#   SCRIPT=scripts/sandbox/check_h2_finish.py ARGS="--n 70" sbatch --job-name=h2probe scripts/amarel/cpu_probe.slurm
#SBATCH --job-name=cpu_probe
#SBATCH --partition=main,main-redhat
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=00:40:00
#SBATCH --output=/scratch/dm1487/logs/probe-%j.out
#SBATCH --error=/scratch/dm1487/logs/probe-%j.err
set -euo pipefail
cd /cache/home/dm1487/projects/namo/namo_cpp
export MJ_PATH=/scratch/dm1487/mujoco/mujoco-3.2.7
export LD_LIBRARY_PATH="$MJ_PATH/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$PWD/build_python:$PWD/python:$PWD/scripts:$PWD/scripts/sandbox:$PWD/scripts/pipeline:/cache/home/dm1487/projects/namo/sage_learning"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=""
: "${SCRIPT:?set SCRIPT=scripts/sandbox/...py}"
echo "probe: $SCRIPT ${ARGS:-} on $(hostname)"
/scratch/dm1487/envs/namo/bin/python -u "$SCRIPT" ${ARGS:-}
echo "PROBE_DONE"
