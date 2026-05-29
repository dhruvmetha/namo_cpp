#!/bin/bash
# Patch driver: aug9 phases 1-3 NPZ are already done (jobs 55139633/52/821).
# This fires NPZ for phases 4 + 5A-E once the cascade driver fully completes,
# then submits the balanced H5 build dependent on all aug9 phase-4/5 NPZ jobs.
#
# Submit as afterok:<cascade_driver_jid>:
#   sbatch --dependency=afterok:55136071 scripts/amarel/aug9_post_phase4_5_npz.slurm
#
#SBATCH --job-name=v3-aug9-post-p45-npz
#SBATCH --partition=main-redhat
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/dm1487/logs/v3-aug9-post-p45-npz-%j.out
#SBATCH --error=/scratch/dm1487/logs/v3-aug9-post-p45-npz-%j.err

set -euo pipefail
NAMO_REPO=${NAMO_REPO:-/cache/home/dm1487/projects/namo/namo_cpp}
SAGE_REPO=${SAGE_REPO:-/cache/home/dm1487/projects/namo/sage_learning}
source "$NAMO_REPO/scripts/amarel/activate.sh"

CORPUS_TAG=v3_aug9
OUTPUTS=/scratch/dm1487/outputs
MANIFESTS=/scratch/dm1487/manifests
COLLECT="$NAMO_REPO/scripts/amarel/run_batch_collection_smoke.slurm"
PER_SHARD=12000
NPZ_JIDS=""

submit_npz() {
    local ph=$1
    local phase_dir="$OUTPUTS/${CORPUS_TAG}_${ph}"
    local mask_dir="$OUTPUTS/${CORPUS_TAG}_${ph}_masks"
    if [ ! -d "$phase_dir" ]; then echo "  $ph: phase dir not present, skip"; return; fi
    if [ -d "$mask_dir" ] && [ -n "$(find "$mask_dir" -name '*.npz' 2>/dev/null -print -quit)" ]; then
        echo "  $ph: already has NPZs, skip"; return
    fi
    local manifest="$MANIFESTS/${CORPUS_TAG}_${ph}_pkls.txt"
    find "$phase_dir" -path '*/modular_data_*/*_results.pkl' 2>/dev/null | sort > "$manifest"
    local n=$(wc -l < "$manifest")
    if [ "$n" -eq 0 ]; then echo "  $ph: 0 PKLs"; return; fi
    local shards=$(( (n + PER_SHARD - 1) / PER_SHARD ))
    local last=$(( shards - 1 ))
    mkdir -p "$mask_dir"
    local jid=$(PKL_MANIFEST="$manifest" SHARD_SIZE="$PER_SHARD" OUTPUT_DIR="$mask_dir" \
        sbatch --array=0-"$last" --job-name="npz-${CORPUS_TAG}_${ph}" "$COLLECT" | awk '{print $4}')
    echo "  $ph NPZ: job $jid  ($n PKLs, $shards shards)"
    NPZ_JIDS="$NPZ_JIDS $jid"
}

echo "===== post-phase4/5 NPZ driver (job ${SLURM_JOB_ID:-NA}) start $(date -Is) ====="

# Phase 4 NPZ was already launched by the afterany:55139872 chain (job 55140727).
# Seed NPZ_JIDS with it so the final H5 build waits for it too.
EXISTING_P4_NPZ_JID=${EXISTING_P4_NPZ_JID:-55140727}
if [ -n "$EXISTING_P4_NPZ_JID" ]; then
    echo "  seeding NPZ_JIDS with existing phase 4 NPZ array $EXISTING_P4_NPZ_JID"
    NPZ_JIDS="$EXISTING_P4_NPZ_JID"
fi

# Submit NPZ for phase 5A-E (cascade just finished all collection). Phase 4
# auto-skips because the submit_npz function checks for existing NPZs.
for ph in phase4 phase5A phase5B phase5C phase5D phase5E; do
    submit_npz "$ph"
done

NPZ_JIDS=$(echo "$NPZ_JIDS" | sed 's/^ *//;s/ /,/g')
echo "===== aug9 phase-4/5 NPZ jobs: $NPZ_JIDS ====="

# H5 build dependent on these (phase 1-3 NPZ already done).
if [ -n "$NPZ_JIDS" ]; then
    HJID=$(sbatch --dependency=afterany:$NPZ_JIDS \
        "$SAGE_REPO/scripts/build_v3_balanced_h5.slurm" | awk '{print $4}')
    echo "balanced H5 build: $HJID (afterany:$NPZ_JIDS)"
    echo "$HJID" > /tmp/v3_balanced_h5_jid.txt
else
    HJID=$(sbatch "$SAGE_REPO/scripts/build_v3_balanced_h5.slurm" | awk '{print $4}')
    echo "balanced H5 build: $HJID (no deps — phase 4/5 NPZ skipped)"
    echo "$HJID" > /tmp/v3_balanced_h5_jid.txt
fi

# Wait so we can summarize at the end.
[ -n "$NPZ_JIDS" ] && while [ "$(squeue -j $NPZ_JIDS -h 2>/dev/null | wc -l)" -gt 0 ]; do sleep 60; done
echo "all phase 4/5 NPZ done @ $(date -Is)"
for p in phase1 phase2 phase3 phase4 phase5A phase5B phase5C phase5D phase5E; do
    d="$OUTPUTS/${CORPUS_TAG}_${p}_masks"
    [ -d "$d" ] && echo "  $p: $(find "$d" -name '*.npz' 2>/dev/null | wc -l) NPZs"
done
