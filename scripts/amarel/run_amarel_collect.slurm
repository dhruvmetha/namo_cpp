#!/bin/bash
#SBATCH --job-name=namo-collect
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/namo-collect-%j.out
#SBATCH --error=logs/namo-collect-%j.err
# (sbatch parses #SBATCH directives before shell expansion, so the log paths
#  above can't reference $NAMO_LOGS — they're hardcoded to the same default.)

set -euo pipefail

# Resolve repo & source the canonical env (modules, conda, MJ_PATH, PYTHONPATH,
# BLAS pins, NAMO_* data vars). Override NAMO_REPO before `sbatch` if relocating.
NAMO_REPO="${NAMO_REPO:?source env.<machine>.sh first}"
mkdir -p "$NAMO_LOGS"
source "$NAMO_REPO/scripts/amarel/activate.sh"

# Sanity-check the canonical build exists before workers fork; otherwise
# every worker fails identically and the run burns wall clock.
if ! ls "$NAMO_REPO/build_python"/namo_rl*.so >/dev/null 2>&1; then
    echo "ERROR: no namo_rl*.so in $NAMO_REPO/build_python — build it once with"
    echo "  NAMO_MARCH=x86-64-v3 ./build_python_bindings.sh"
    exit 1
fi

# Submit-time overrides. Example:
#   MANIFEST=$NAMO_MANIFESTS/car_envs_v1.txt START_IDX=0 END_IDX=100 \
#   sbatch scripts/amarel/run_amarel_collect.slurm
CONFIG_YAML=${CONFIG_YAML:-python/namo/data_collection/region_opening_amarel_car.yaml}
START_IDX=${START_IDX:-0}
END_IDX=${END_IDX:-1000}
OUTPUT_DIR=${OUTPUT_DIR:-$NAMO_OUTPUTS}
MANIFEST=${MANIFEST:?must set MANIFEST=<path> at submit time (e.g. $NAMO_MANIFESTS/car_envs_v1.txt)}

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
