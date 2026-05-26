#!/bin/bash
#SBATCH --job-name=namo-collect
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/dm1487/logs/namo-collect-%j.out
#SBATCH --error=/scratch/dm1487/logs/namo-collect-%j.err

set -euo pipefail
mkdir -p /scratch/dm1487/logs

module use /projects/community/modulefiles
module load gcc/14.2.0-cermak
module load cmake/3.31.8-rdp135

source /cache/home/dm1487/miniforge3/etc/profile.d/conda.sh
conda activate /scratch/dm1487/envs/namo

export MJ_PATH=/scratch/dm1487/mujoco/mujoco-3.2.7
export LD_LIBRARY_PATH="$MJ_PATH/lib:/scratch/dm1487/envs/namo/lib:${LD_LIBRARY_PATH:-}"

REPO=/cache/home/dm1487/projects/namo/namo_cpp
BUILD_DIR=$REPO/build_python
export PYTHONPATH="$BUILD_DIR:$REPO/python:${PYTHONPATH:-}"

# Sanity-check the canonical build exists before workers fork; otherwise
# every worker fails identically and the array burns a 24h wall clock.
if ! ls "$BUILD_DIR"/namo_rl*.so >/dev/null 2>&1; then
    echo "ERROR: no namo_rl*.so in $BUILD_DIR — build it once with"
    echo "  NAMO_MARCH=x86-64-v3 ./build_python_bindings.sh"
    exit 1
fi

cd "$REPO"

# Pin BLAS thread counts; the work is parallelized by multiprocessing.Pool,
# not intra-op. Letting BLAS spawn threads on top oversubscribes.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

# All overridable via env vars at submit time. Example:
#   START_IDX=0 END_IDX=100 sbatch scripts/amarel/run_amarel_collect.slurm
CONFIG_YAML=${CONFIG_YAML:-python/namo/data_collection/region_opening_amarel_car.yaml}
START_IDX=${START_IDX:-0}
END_IDX=${END_IDX:-1000}
OUTPUT_DIR=${OUTPUT_DIR:-/scratch/dm1487/outputs}
MANIFEST=${MANIFEST:-/scratch/dm1487/manifests/car_envs.txt}

echo "host=$(hostname) job=${SLURM_JOB_ID:-N/A} cpus=${SLURM_CPUS_PER_TASK} start=$(date -Is)"
echo "config_yaml=$CONFIG_YAML  manifest=$MANIFEST  shard=[$START_IDX,$END_IDX)"

python python/namo/data_collection/modular_parallel_collection.py \
    --config-yaml "$CONFIG_YAML" \
    --workers "$SLURM_CPUS_PER_TASK" \
    --output-dir "$OUTPUT_DIR" \
    --start-idx "$START_IDX" \
    --end-idx "$END_IDX" \
    --manifest "$MANIFEST" \
    --unique-run-dir

echo "end=$(date -Is)"
