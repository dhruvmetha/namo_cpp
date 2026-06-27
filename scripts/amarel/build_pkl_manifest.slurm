#!/bin/bash
# Build a PKL manifest (list of *_results.pkl) from one or more collection
# output dirs, as a SLURM job (keeps the recursive find off the login node).
# Handles flat (modular_data_*/) and sharded (shard_*/pkls/modular_data_*/)
# layouts. Excludes collection_summary_*.pkl.
#
# Required env vars at submit:
#   PHASE_DIRS  space-separated list of collection output dirs
#   OUTPUT      manifest path to write
#
# Example:
#   PHASE_DIRS="$NAMO_OUTPUTS/v3_phase5A $NAMO_OUTPUTS/v3_phase5B" \
#   OUTPUT=$NAMO_MANIFESTS/v3_phase5_pkls.txt \
#     sbatch scripts/amarel/build_pkl_manifest.slurm
#
#SBATCH --job-name=build-pkl-manifest
#SBATCH --partition=main-redhat
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=logs/build-pkl-manifest-%j.out
#SBATCH --error=logs/build-pkl-manifest-%j.err

set -euo pipefail
: "${PHASE_DIRS:?set PHASE_DIRS (space-separated)}"
: "${OUTPUT:?set OUTPUT}"
mkdir -p "$(dirname "$OUTPUT")" "$NAMO_LOGS"

echo "host=$(hostname) dirs=$PHASE_DIRS output=$OUTPUT"
echo "start=$(date -Is)"

: > "$OUTPUT"
for d in $PHASE_DIRS; do
    find "$d" -path '*/modular_data_*/*_results.pkl' 2>/dev/null >> "$OUTPUT" || true
done
sort -u -o "$OUTPUT" "$OUTPUT"

echo "end=$(date -Is)"
echo "PKLs in manifest: $(wc -l < "$OUTPUT")"
