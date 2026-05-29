#!/bin/bash
# Sequential v3 cascade driver: waits for phase-1, then mines+runs phases
# 2 -> 3 -> 4 -> 5A-E. Runs inside a SLURM allocation so it survives client
# disconnects. Each child collection array is submitted, polled to completion,
# then the next phase is mined and submitted.
#
# Uses the v3 car collection script (v3_cascade_collect.slurm): d5 primitives,
# car config, target-goal-region, depth/K per phase. Mining via
# build_phase2_manifest.py (sharded-aware) + build_phase4_manifest.py (union).
#
# Required env vars at submit:
#   CORPUS_TAG   e.g. v3_aug9  (output dirs: <OUTPUTS>/<CORPUS_TAG>_phaseN)
#   PHASE1_JID   job id of the already-submitted phase-1 array to wait on
#
# Phase-1 must already be submitted separately (it needs the exhaustive config).
#
#SBATCH --job-name=v3-cascade-driver
#SBATCH --partition=main-redhat
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --output=/scratch/dm1487/logs/v3-cascade-driver-%j.out
#SBATCH --error=/scratch/dm1487/logs/v3-cascade-driver-%j.err

set -euo pipefail
NAMO_REPO=${NAMO_REPO:-/cache/home/dm1487/projects/namo/namo_cpp}
cd "$NAMO_REPO"
source scripts/amarel/activate.sh

: "${CORPUS_TAG:?set CORPUS_TAG (e.g. v3_aug9)}"
: "${PHASE1_JID:?set PHASE1_JID (phase-1 array job id to wait on)}"

MANIFESTS=/scratch/dm1487/manifests
OUTPUTS=/scratch/dm1487/outputs
COLLECT="$NAMO_REPO/scripts/amarel/v3_cascade_collect.slurm"
W=${SLURM_CPUS_PER_TASK:-16}

wait_for_job() { local j=$1; while [ "$(squeue -j "$j" -h 2>/dev/null | wc -l)" -gt 0 ]; do sleep 60; done; }
shards_for()   { python -c "import math;print(min(60,max(1,math.ceil($1/1500))))"; }

# submit_collection <manifest> <outdir> <depth> <K> <name>  (blocks until done)
submit_collection() {
    local manifest=$1 outdir=$2 depth=$3 K=$4 name=$5
    local n shards last jid
    n=$(wc -l < "$manifest")
    if [ "$n" -eq 0 ]; then echo "  [$name] empty manifest, skipping"; return; fi
    shards=$(shards_for "$n"); last=$(( shards - 1 ))
    echo "  [$name] envs=$n shards=$shards depth=$depth K=$K -> $outdir"
    jid=$(MANIFEST="$manifest" OUTPUT_DIR="$outdir" DEPTH="$depth" KROLL="$K" NUM_SHARDS="$shards" \
        sbatch --array=0-"$last" --job-name="$name" "$COLLECT" | awk '{print $4}')
    echo "  [$name] job=$jid  $(date -Is)"
    wait_for_job "$jid"
    echo "  [$name] done  $(date -Is)"
}

echo "===== v3 cascade driver: $CORPUS_TAG  (driver job ${SLURM_JOB_ID:-NA}) ====="
echo "waiting for phase-1 ($PHASE1_JID) ..."; date -Is
wait_for_job "$PHASE1_JID"
echo "phase-1 complete."; date -Is

# ---- Phase 2: mine phase-1 fails, depth-2 K=20 ----
echo "===== phase 2 ====="; date -Is
python scripts/build_phase2_manifest.py \
    --phase1-dir "$OUTPUTS/${CORPUS_TAG}_phase1" \
    --output     "$MANIFESTS/${CORPUS_TAG}_phase2.txt" --workers "$W"
submit_collection "$MANIFESTS/${CORPUS_TAG}_phase2.txt" "$OUTPUTS/${CORPUS_TAG}_phase2" 2 20 "${CORPUS_TAG}-p2"

# ---- Phase 3: mine phase-2 fails, depth-2 K=50 ----
echo "===== phase 3 ====="; date -Is
python scripts/build_phase2_manifest.py \
    --phase1-dir "$OUTPUTS/${CORPUS_TAG}_phase2" \
    --output     "$MANIFESTS/${CORPUS_TAG}_phase3.txt" --workers "$W"
submit_collection "$MANIFESTS/${CORPUS_TAG}_phase3.txt" "$OUTPUTS/${CORPUS_TAG}_phase3" 2 50 "${CORPUS_TAG}-p3"

# ---- Phase 4: union of partial-fail (p1,p2,p3) + p3 still-failing, depth-2 K=50 ----
echo "===== phase 4 ====="; date -Is
python scripts/build_phase4_manifest.py \
    --phase1-dir "$OUTPUTS/${CORPUS_TAG}_phase1" \
    --phase2-dir "$OUTPUTS/${CORPUS_TAG}_phase2" \
    --phase3-dir "$OUTPUTS/${CORPUS_TAG}_phase3" \
    --output     "$MANIFESTS/${CORPUS_TAG}_phase4.txt" --workers "$W"
submit_collection "$MANIFESTS/${CORPUS_TAG}_phase4.txt" "$OUTPUTS/${CORPUS_TAG}_phase4" 2 50 "${CORPUS_TAG}-p4"

# ---- Phase 5A-E: seed sweep on phase-4 residual (empirical include-list) ----
echo "===== phase 5 ====="; date -Is
python scripts/build_phase2_manifest.py \
    --phase1-dir "$OUTPUTS/${CORPUS_TAG}_phase4" \
    --output     "$MANIFESTS/${CORPUS_TAG}_phase5_init.txt" \
    --include-reasons all_pushes_failed ml_no_goals_extracted no_reachable_edges \
    --workers "$W"
N5=$(wc -l < "$MANIFESTS/${CORPUS_TAG}_phase5_init.txt")
if [ "$N5" -gt 0 ]; then
    shards=$(shards_for "$N5"); last=$(( shards - 1 ))
    declare -A SEEDS=([A]=101 [B]=102 [C]=103 [D]=104 [E]=105)
    jids=""
    for p in A B C D E; do
        jid=$(MANIFEST="$MANIFESTS/${CORPUS_TAG}_phase5_init.txt" \
            OUTPUT_DIR="$OUTPUTS/${CORPUS_TAG}_phase5${p}" DEPTH=2 KROLL=50 \
            NUM_SHARDS="$shards" GOAL_SEED="${SEEDS[$p]}" \
            sbatch --array=0-"$last" --job-name="${CORPUS_TAG}-p5${p}" "$COLLECT" | awk '{print $4}')
        echo "  phase5$p (seed=${SEEDS[$p]}): job $jid"
        jids="$jids,$jid"
    done
    jids=${jids#,}
    echo "  waiting for all phase-5 passes: $jids"
    while [ "$(squeue -j "$jids" -h 2>/dev/null | wc -l)" -gt 0 ]; do sleep 60; done
fi

echo "===== $CORPUS_TAG cascade COMPLETE @ $(date -Is) ====="
for ph in phase1 phase2 phase3 phase4 phase5A phase5B phase5C phase5D phase5E; do
    d="$OUTPUTS/${CORPUS_TAG}_${ph}"
    [ -d "$d" ] && echo "  $ph: $(find "$d" -path '*/modular_data_*/*_results.pkl' 2>/dev/null | wc -l) PKLs"
done
