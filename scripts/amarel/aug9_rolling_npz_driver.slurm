#!/bin/bash
# Rolling NPZ + H5 driver for aug9 v3 (CORL crunch). Submits NPZ for each
# aug9 phase as soon as its collection finishes — phase 1+2 immediately
# (already done), 3/4/5A-E as they land. Final step: balanced H5 build
# dependent on all NPZ jobs.
#
# Run with no SLURM dependency — starts firing NPZ jobs immediately.
#
#SBATCH --job-name=v3-aug9-rolling-npz
#SBATCH --partition=main
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=08:00:00
#SBATCH --output=logs/v3-aug9-rolling-npz-%j.out
#SBATCH --error=logs/v3-aug9-rolling-npz-%j.err

set -euo pipefail
NAMO_REPO="${NAMO_REPO:?source env.<machine>.sh first}"
SAGE_REPO="${SAGE_REPO:?source env.<machine>.sh first}"
source "$NAMO_REPO/scripts/amarel/activate.sh"

CORPUS_TAG=v3_aug9
OUTPUTS="$NAMO_OUTPUTS"
MANIFESTS="$NAMO_MANIFESTS"
COLLECT="$NAMO_REPO/scripts/amarel/run_batch_collection_smoke.slurm"
PER_SHARD=12000
NPZ_JIDS=""

submit_npz() {
    # $1 = phase tag (e.g. phase3)
    local ph=$1
    local phase_dir="$OUTPUTS/${CORPUS_TAG}_${ph}"
    local mask_dir="$OUTPUTS/${CORPUS_TAG}_${ph}_masks"
    if [ ! -d "$phase_dir" ]; then
        echo "  $ph: phase dir not present yet, skip"
        return
    fi
    if [ -d "$mask_dir" ] && [ -n "$(find "$mask_dir" -name '*.npz' 2>/dev/null -print -quit)" ]; then
        echo "  $ph: already has NPZs, skip"
        return
    fi
    local manifest="$MANIFESTS/${CORPUS_TAG}_${ph}_pkls.txt"
    find "$phase_dir" -path '*/modular_data_*/*_results.pkl' 2>/dev/null | sort > "$manifest"
    local n=$(wc -l < "$manifest")
    if [ "$n" -eq 0 ]; then echo "  $ph: 0 PKLs, skip"; return; fi
    local shards=$(( (n + PER_SHARD - 1) / PER_SHARD ))
    local last=$(( shards - 1 ))
    mkdir -p "$mask_dir"
    local jid=$(PKL_MANIFEST="$manifest" SHARD_SIZE="$PER_SHARD" OUTPUT_DIR="$mask_dir" \
        sbatch --array=0-"$last" --job-name="npz-${CORPUS_TAG}_${ph}" "$COLLECT" \
        | awk '{print $4}')
    echo "  $ph NPZ: job $jid  ($n PKLs, $shards shards)"
    NPZ_JIDS="$NPZ_JIDS $jid"
}

wait_for_collection_phase() {
    # $1 = phase tag — wait for any sbatch with job-name v3_aug9-p<N> to clear
    local ph=$1
    local jn="${CORPUS_TAG}-p${ph#phase}"
    echo "  waiting for collection phase $ph (job-name $jn) ..."
    while [ "$(squeue -u dm1487 -h -o '%j' | grep -c "^${jn}\$")" -gt 0 ]; do sleep 30; done
    echo "  $ph collection done @ $(date -Is)"
}

echo "===== rolling NPZ driver (job ${SLURM_JOB_ID:-NA}) start $(date -Is) ====="

# Phase 1 + 2 collections already done -> NPZ immediately, in parallel.
echo "--- phase 1 + 2 NPZ (already-done collections) ---"
submit_npz phase1
submit_npz phase2

# Phases 3 -> 4 -> 5A-E happen sequentially in the cascade driver.
# Watch each, fire its NPZ as soon as it finishes.
for ph in phase3 phase4 phase5A phase5B phase5C phase5D phase5E; do
    wait_for_collection_phase "$ph"
    submit_npz "$ph"
done

# Strip leading space, comma-join
NPZ_JIDS=$(echo "$NPZ_JIDS" | sed 's/^ *//;s/ /,/g')
echo ""; echo "===== all aug9 NPZ jobs submitted: $NPZ_JIDS ====="

# Fire balanced H5 build dependent on all NPZ jobs (afterany so a single shard
# failing doesn't block the whole H5; convert_to_hdf5 tolerates partial input).
HJID=$(sbatch --dependency=afterany:$NPZ_JIDS \
    "$SAGE_REPO/scripts/build_v3_balanced_h5.slurm" | awk '{print $4}')
echo "balanced H5 build: $HJID (afterany:$NPZ_JIDS)"
echo "$HJID" > /tmp/v3_balanced_h5_jid.txt

# Wait for all to finish so we can summarize counts at the end.
echo "waiting for all NPZ jobs to finish..."
while [ "$(squeue -j $NPZ_JIDS -h 2>/dev/null | wc -l)" -gt 0 ]; do sleep 60; done
echo "all aug9 NPZ done @ $(date -Is)"
for p in phase1 phase2 phase3 phase4 phase5A phase5B phase5C phase5D phase5E; do
    d="$OUTPUTS/${CORPUS_TAG}_${p}_masks"
    [ -d "$d" ] && echo "  $p: $(find "$d" -name '*.npz' 2>/dev/null | wc -l) NPZs"
done

echo ""; echo "===== rolling driver complete $(date -Is) ====="
