#!/bin/bash
# Build a PKL manifest from a collection phase output dir and submit sharded
# NPZ (mask) generation via run_batch_collection_smoke.slurm.
#
# Handles both the flat (modular_data_*/) and v3 sharded
# (shard_*/pkls/modular_data_*/) collection layouts.
#
# NPZ defaults (batch_collection.py): car config
# (namo_config_complete_skill15_car_1x.yaml), dual-crop (wide 1.2 m + tight
# 0.5 m), 5 mm wavefront — no extra flags needed for v3 car data.
#
# Usage:
#   scripts/amarel/submit_npz_gen.sh <phase_output_dir> <npz_output_dir> [pkls_per_shard]
#
# Example:
#   scripts/amarel/submit_npz_gen.sh \
#       "$NAMO_OUTPUTS/v3_phase1" \
#       "$NAMO_OUTPUTS/v3_phase1_masks"
set -euo pipefail

PHASE_DIR=${1:?usage: submit_npz_gen.sh <phase_output_dir> <npz_output_dir> [pkls_per_shard]}
OUT_DIR=${2:?usage: submit_npz_gen.sh <phase_output_dir> <npz_output_dir> [pkls_per_shard]}
PER_SHARD=${3:-12000}

REPO="${NAMO_REPO:?source env.<machine>.sh first}"
MANIFESTS="$NAMO_MANIFESTS"
mkdir -p "$MANIFESTS" "$OUT_DIR" "$NAMO_LOGS"

TAG=$(basename "$PHASE_DIR")
MANIFEST="$MANIFESTS/${TAG}_pkls.txt"

# Collect result PKLs (both layouts); excludes collection_summary_*.pkl
{
    find "$PHASE_DIR" -path '*/modular_data_*/*_results.pkl' 2>/dev/null || true
} | sort > "$MANIFEST"

N=$(wc -l < "$MANIFEST")
if [ "$N" -eq 0 ]; then
    echo "ERROR: no *_results.pkl found under $PHASE_DIR" >&2
    exit 1
fi

SHARDS=$(( (N + PER_SHARD - 1) / PER_SHARD ))
[ "$SHARDS" -lt 1 ] && SHARDS=1
LAST=$(( SHARDS - 1 ))

echo "phase=$TAG  pkls=$N  shards=$SHARDS  per_shard=$PER_SHARD"
echo "manifest=$MANIFEST"
echo "out=$OUT_DIR"

cd "$REPO" || exit 1   # so the slurm's repo-relative #SBATCH --output=logs/... resolves to $REPO/logs
PKL_MANIFEST="$MANIFEST" SHARD_SIZE="$PER_SHARD" OUTPUT_DIR="$OUT_DIR" \
    sbatch --array=0-"$LAST" --job-name="npz-$TAG" \
        "$REPO/scripts/amarel/run_batch_collection_smoke.slurm"
