#!/bin/bash
# Sharded batch_collection smoke. Reads a full PKL manifest, slices it by
# SLURM_ARRAY_TASK_ID, runs batch_collection --pkl-list on that slice.
#
# Required env vars at submit time:
#   PKL_MANIFEST  full PKL list (one path per line)
#   SHARD_SIZE    envs per shard (use to match --array=)
#   OUTPUT_DIR    NPZ output root; each shard writes to $OUTPUT_DIR/shard_$TASK/
#
# Optional:
#   BC_EXTRA_ARGS extra args to batch_collection (e.g., "--local-only --filter-overlaps")
#   BC_WORKERS    workers per shard (default $SLURM_CPUS_PER_TASK)
#SBATCH --partition=main
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/bcsmoke-%A_%a.out
#SBATCH --error=logs/bcsmoke-%A_%a.err

set -euo pipefail
mkdir -p "$NAMO_LOGS"

NAMO_REPO="${NAMO_REPO:?source env.<machine>.sh first}"
source "$NAMO_REPO/scripts/amarel/activate.sh"

: "${PKL_MANIFEST:?must set PKL_MANIFEST}"
: "${SHARD_SIZE:?must set SHARD_SIZE}"
: "${OUTPUT_DIR:?must set OUTPUT_DIR}"

shard_workers=${BC_WORKERS:-$SLURM_CPUS_PER_TASK}
extra=${BC_EXTRA_ARGS:-}
task_id=${SLURM_ARRAY_TASK_ID:-0}
start_line=$(( task_id * SHARD_SIZE + 1 ))
end_line=$(( (task_id + 1) * SHARD_SIZE ))
shard_dir="$OUTPUT_DIR/shard_${task_id}"
shard_list="$shard_dir.pkl_list"
mkdir -p "$shard_dir"

# Slice the manifest for this shard
sed -n "${start_line},${end_line}p" "$PKL_MANIFEST" > "$shard_list"
n_lines=$(wc -l < "$shard_list")

echo "host=$(hostname) array=${SLURM_ARRAY_JOB_ID}_${task_id}"
echo "slice=[$start_line,$end_line] ($n_lines PKLs assigned)"
echo "out=$shard_dir"
echo "extra=$extra"

if [ "$n_lines" -eq 0 ]; then
    echo "  empty slice, nothing to do"
    exit 0
fi

python -m namo.visualization.mask_generation.batch_collection \
    --pkl-list "$shard_list" \
    --output-dir "$shard_dir" \
    --workers "$shard_workers" \
    $extra

echo "shard done. npz count:"
find "$shard_dir" -name "*.npz" | wc -l
