#!/bin/bash
# Post-aug9-cascade: for every aug9 phase output dir, submit sharded NPZ
# generation via run_batch_collection_smoke.slurm. Runs as a chained driver
# after the cascade driver completes.
#
# Required env vars at submit:
#   CORPUS_TAG  e.g. v3_aug9
#
#SBATCH --job-name=v3-aug9-npz-driver
#SBATCH --partition=main-redhat
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=08:00:00
#SBATCH --output=/scratch/dm1487/logs/v3-aug9-npz-driver-%j.out
#SBATCH --error=/scratch/dm1487/logs/v3-aug9-npz-driver-%j.err

set -euo pipefail
NAMO_REPO=${NAMO_REPO:-/cache/home/dm1487/projects/namo/namo_cpp}
cd "$NAMO_REPO"
source scripts/amarel/activate.sh

: "${CORPUS_TAG:?set CORPUS_TAG (e.g. v3_aug9)}"
OUTPUTS=/scratch/dm1487/outputs
MANIFESTS=/scratch/dm1487/manifests
COLLECT="$NAMO_REPO/scripts/amarel/run_batch_collection_smoke.slurm"
PER_SHARD=12000

submit_npz() {
    # $1 = phase tag (e.g. phase1)
    local ph=$1
    local phase_dir="$OUTPUTS/${CORPUS_TAG}_${ph}"
    local mask_dir="$OUTPUTS/${CORPUS_TAG}_${ph}_masks"
    [ -d "$phase_dir" ] || { echo "  skip $ph (no $phase_dir)"; return; }

    local manifest="$MANIFESTS/${CORPUS_TAG}_${ph}_pkls.txt"
    find "$phase_dir" -path '*/modular_data_*/*_results.pkl' 2>/dev/null | sort > "$manifest"
    local n=$(wc -l < "$manifest")
    [ "$n" -eq 0 ] && { echo "  skip $ph (no PKLs)"; return; }

    local shards=$(( (n + PER_SHARD - 1) / PER_SHARD ))
    local last=$(( shards - 1 ))
    mkdir -p "$mask_dir"
    echo "  $ph: $n PKLs -> $shards shards"
    local jid=$(PKL_MANIFEST="$manifest" SHARD_SIZE="$PER_SHARD" OUTPUT_DIR="$mask_dir" \
        sbatch --array=0-"$last" --job-name="npz-${CORPUS_TAG}_${ph}" "$COLLECT" | awk '{print $4}')
    echo "  $ph: job $jid"
    NPZ_JIDS="$NPZ_JIDS $jid"
}

echo "===== aug9 post-cascade NPZ driver: $CORPUS_TAG  (driver job ${SLURM_JOB_ID:-NA}) ====="
date -Is
NPZ_JIDS=""

for ph in phase1 phase2 phase3 phase4 phase5A phase5B phase5C phase5D phase5E; do
    submit_npz "$ph"
done

NPZ_JIDS=$(echo "$NPZ_JIDS" | sed 's/^ *//;s/ /,/g')
echo "all aug9 NPZ jobs: $NPZ_JIDS"
echo "$NPZ_JIDS" > /tmp/aug9_npz_jids.txt

if [ -n "$NPZ_JIDS" ]; then
    echo "waiting for NPZ jobs to finish..."
    while [ "$(squeue -j "$NPZ_JIDS" -h 2>/dev/null | wc -l)" -gt 0 ]; do sleep 60; done
fi

echo ""; echo "===== aug9 NPZ DONE @ $(date -Is) ====="
for ph in phase1 phase2 phase3 phase4 phase5A phase5B phase5C phase5D phase5E; do
    d="$OUTPUTS/${CORPUS_TAG}_${ph}_masks"
    [ -d "$d" ] && echo "  $ph: $(find "$d" -name '*.npz' 2>/dev/null | wc -l) NPZs"
done
