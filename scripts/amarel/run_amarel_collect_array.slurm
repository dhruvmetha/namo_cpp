#!/bin/bash
#SBATCH --job-name=namo-collect-arr
#SBATCH --partition=main
#SBATCH --array=0-29
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/dm1487/logs/namo-collect-%A_%a.out
#SBATCH --error=/scratch/dm1487/logs/namo-collect-%A_%a.err

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
# every shard fails identically and the array burns a 24h wall clock.
if ! ls "$BUILD_DIR"/namo_rl*.so >/dev/null 2>&1; then
    echo "ERROR: no namo_rl*.so in $BUILD_DIR — build it once with"
    echo "  NAMO_MARCH=x86-64-v3 ./build_python_bindings.sh"
    exit 1
fi

cd "$REPO"

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

# Shard envs across array tasks. car_envs has 29849 XMLs → with --array=0-29
# and SHARD_SIZE=1000, you cover [0, 30000) which spans the whole dataset.
CONFIG_YAML=${CONFIG_YAML:-python/namo/data_collection/region_opening_amarel_car.yaml}
SHARD_SIZE=${SHARD_SIZE:-1000}
GLOBAL_START=${GLOBAL_START:-0}
OUTPUT_DIR=${OUTPUT_DIR:-/scratch/dm1487/outputs}
MANIFEST=${MANIFEST:-/scratch/dm1487/manifests/car_envs.txt}

START_IDX=$(( GLOBAL_START + SLURM_ARRAY_TASK_ID * SHARD_SIZE ))
END_IDX=$(( START_IDX + SHARD_SIZE ))

# EXTRA_ARGS lets the submitter override YAML defaults without editing config.
# Example: EXTRA_ARGS="--region-max-chain-depth 1 --search-timeout 120" sbatch ...
EXTRA_ARGS=${EXTRA_ARGS:-}

echo "host=$(hostname) job=${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID} shard=[$START_IDX,$END_IDX) cpus=${SLURM_CPUS_PER_TASK}"
echo "extra_args=$EXTRA_ARGS"

python python/namo/data_collection/modular_parallel_collection.py \
    --config-yaml "$CONFIG_YAML" \
    --workers "$SLURM_CPUS_PER_TASK" \
    --output-dir "$OUTPUT_DIR" \
    --start-idx "$START_IDX" \
    --end-idx "$END_IDX" \
    --manifest "$MANIFEST" \
    --unique-run-dir \
    --run-name "arr${SLURM_ARRAY_JOB_ID}_t${SLURM_ARRAY_TASK_ID}" \
    $EXTRA_ARGS
