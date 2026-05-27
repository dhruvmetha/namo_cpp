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
# (sbatch parses #SBATCH directives before shell expansion, so the log paths
#  above can't reference $NAMO_LOGS — they're hardcoded to the same default.)

set -euo pipefail

# Resolve repo & source canonical env (modules, conda, MJ_PATH, PYTHONPATH,
# BLAS pins, NAMO_* data vars). Override NAMO_REPO before `sbatch` if relocating.
NAMO_REPO=${NAMO_REPO:-/cache/home/dm1487/projects/namo/namo_cpp}
mkdir -p /scratch/dm1487/logs
source "$NAMO_REPO/scripts/amarel/activate.sh"

# Sanity-check the canonical build exists before workers fork; otherwise
# every shard fails identically and the array burns a 24h wall clock.
if ! ls "$NAMO_REPO/build_python"/namo_rl*.so >/dev/null 2>&1; then
    echo "ERROR: no namo_rl*.so in $NAMO_REPO/build_python — build it once with"
    echo "  NAMO_MARCH=x86-64-v3 ./build_python_bindings.sh"
    exit 1
fi

# Shard envs across array tasks. Default: 30 shards × 1000 envs.
# Adjust SHARD_SIZE and --array=N at submit time for other dataset sizes.
CONFIG_YAML=${CONFIG_YAML:-python/namo/data_collection/region_opening_amarel_car.yaml}
SHARD_SIZE=${SHARD_SIZE:-1000}
GLOBAL_START=${GLOBAL_START:-0}
OUTPUT_DIR=${OUTPUT_DIR:-$NAMO_OUTPUTS}
MANIFEST=${MANIFEST:?must set MANIFEST=<path> at submit time (e.g. $NAMO_MANIFESTS/car_envs_v1.txt)}

START_IDX=$(( GLOBAL_START + SLURM_ARRAY_TASK_ID * SHARD_SIZE ))
END_IDX=$(( START_IDX + SHARD_SIZE ))

# EXTRA_ARGS lets the submitter override YAML defaults without editing config.
# Example: EXTRA_ARGS="--region-max-chain-depth 1 --search-timeout 120" sbatch ...
EXTRA_ARGS=${EXTRA_ARGS:-}

echo "host=$(hostname) job=${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID} shard=[$START_IDX,$END_IDX) cpus=${SLURM_CPUS_PER_TASK}"
echo "manifest=$MANIFEST  output=$OUTPUT_DIR  extra_args=$EXTRA_ARGS"

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
