#!/bin/bash
#SBATCH --job-name=c0gen
#SBATCH --account=general
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --time=03:00:00
#SBATCH --output=/scratch/dm1487/curriculum2_amarel/colossus0_1m/logs/gen_%A_%a.out
#SBATCH --error=/scratch/dm1487/curriculum2_amarel/colossus0_1m/logs/gen_%A_%a.err
set -euo pipefail

: "${NAMO_REPO:?}"
: "${JOBLIST:?}"
cd "$NAMO_REPO"
source env.amarel.sh >/dev/null 2>&1
export PYTHONPATH="$NAMO_REPO/build_python:$NAMO_REPO/python:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="" NAMO_PIN_THREADS=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 OPENCV_NUM_THREADS=1

total=$(wc -l < "$JOBLIST")
tasks=${SLURM_ARRAY_TASK_COUNT:?}
start=$((SLURM_ARRAY_TASK_ID * total / tasks + 1))
end=$(((SLURM_ARRAY_TASK_ID + 1) * total / tasks))
workers=${SLURM_CPUS_ON_NODE:-32}
echo "host=$(hostname) commands=[$start,$end] workers=$workers start=$(date -Is)"
sed -n "${start},${end}p" "$JOBLIST" | xargs -P "$workers" -I CMD bash -c 'CMD' >/dev/null
echo "done=$(date -Is)"
