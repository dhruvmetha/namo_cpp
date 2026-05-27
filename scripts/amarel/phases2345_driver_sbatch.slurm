#!/bin/bash
# Sequential driver for phases 2 → 3 → 4 → 5. Runs inside a SLURM allocation
# so it outlives any client/srun disconnect. Each phase:
#   (1) mines a manifest from the prior phase's PKL output
#   (2) submits a collection sbatch array sized to fit the manifest
#   (3) polls squeue until done
#   (4) chains to the next phase
#
# Phase spec (matches build_phase4_manifest.py docstring):
#   Phase 2: depth-2, K=20  (mines phase-1 fails)
#   Phase 3: depth-2, K=50  (mines phase-2 fails)
#   Phase 4: depth-2, K=50  (mines p1/p2/p3 partial-fail + p3 still-failing)
#   Phase 5: depth-2, K=50 + seed sweep A..E (delegated to phase5_driver_sbatch.slurm)
#
# All passes use --goal-strategy random_rollout --target-goal-region.
#
#SBATCH --job-name=namo-p2345
#SBATCH --partition=main-redhat
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=72:00:00
#SBATCH --output=/scratch/dm1487/logs/phases2345_driver-%j.out
#SBATCH --error=/scratch/dm1487/logs/phases2345_driver-%j.err

set -euo pipefail
mkdir -p /scratch/dm1487/logs

NAMO_REPO=${NAMO_REPO:-/cache/home/dm1487/projects/namo/namo_cpp}
source "$NAMO_REPO/scripts/amarel/activate.sh"

MAX_SHARDS=60
TARGET_SHARD_SIZE=100
STOP_THRESHOLD=50   # if a phase's manifest <= this, skip the phase
MANIFESTS=/scratch/dm1487/manifests
OUTPUTS=/scratch/dm1487/outputs

# ----- helpers ---------------------------------------------------------------
wait_for_job() {
    local jid=$1
    while [ "$(squeue -j $jid -h 2>/dev/null | wc -l)" -gt 0 ]; do
        sleep 60
    done
}

submit_collection() {
    # $1=manifest  $2=output_dir  $3=K  $4=depth  $5=job_name_short
    local manifest=$1 outdir=$2 K=$3 depth=$4 short=$5
    local n_envs shards shard_size last_array
    n_envs=$(wc -l < "$manifest")
    shards=$(python -c "import math; print(min($MAX_SHARDS, max(1, math.ceil($n_envs / $TARGET_SHARD_SIZE))))")
    shard_size=$(python -c "import math; print(math.ceil($n_envs / $shards))")
    last_array=$(( shards - 1 ))
    echo "    envs=$n_envs shards=$shards shard_size=$shard_size K=$K depth=$depth"
    rm -rf "$outdir"
    local jid
    jid=$(SHARD_SIZE=$shard_size MANIFEST=$manifest OUTPUT_DIR=$outdir \
        EXTRA_ARGS="--goal-strategy random_rollout --region-max-chain-depth $depth --rollout-samples-per-state $K --target-goal-region" \
        sbatch --partition=main-redhat --array=0-${last_array} --cpus-per-task=32 --mem=100G --time=08:00:00 \
            --job-name=namo-${short} \
            "$NAMO_REPO/scripts/amarel/run_amarel_collect_array.slurm" | awk '{print $4}')
    echo "    collection job: $jid"
    wait_for_job "$jid"
    echo "    $short collection done @ $(date -Is)"
}

mine_simple() {
    # $1=phase output dir  $2=output manifest path
    sbatch_jid=$(sbatch "$NAMO_REPO/scripts/amarel/mine_residual.slurm" "$1" "$2" | awk '{print $4}')
    echo "    miner job: $sbatch_jid"
    while [ "$(squeue -j $sbatch_jid -h 2>/dev/null | wc -l)" -gt 0 ]; do
        sleep 20
    done
}

# ----- phase 2: mine phase-1, run depth-2 K=20 -------------------------------
echo "===== phase 2 ====="; date -Is
P2_MANIFEST=$MANIFESTS/car_envs_v1_aug9_phase2.txt
P2_OUTDIR=$OUTPUTS/car_v1_aug9_phase2
echo "  mining phase-2 manifest from phase-1 ..."
mine_simple "$OUTPUTS/car_v1_aug9_phase1" "$P2_MANIFEST"
n=$(wc -l < "$P2_MANIFEST")
echo "  phase-2 manifest: $n envs"
if [ "$n" -gt "$STOP_THRESHOLD" ]; then
    submit_collection "$P2_MANIFEST" "$P2_OUTDIR" 20 2 "p2"
else
    echo "  skipping phase 2 (residual <= $STOP_THRESHOLD)"
fi

# ----- phase 3: mine phase-2, run depth-2 K=50 -------------------------------
echo ""; echo "===== phase 3 ====="; date -Is
P3_MANIFEST=$MANIFESTS/car_envs_v1_aug9_phase3.txt
P3_OUTDIR=$OUTPUTS/car_v1_aug9_phase3
echo "  mining phase-3 manifest from phase-2 ..."
if [ -d "$P2_OUTDIR" ]; then
    mine_simple "$P2_OUTDIR" "$P3_MANIFEST"
else
    cp "$P2_MANIFEST" "$P3_MANIFEST"
fi
n=$(wc -l < "$P3_MANIFEST")
echo "  phase-3 manifest: $n envs"
if [ "$n" -gt "$STOP_THRESHOLD" ]; then
    submit_collection "$P3_MANIFEST" "$P3_OUTDIR" 50 2 "p3"
else
    echo "  skipping phase 3 (residual <= $STOP_THRESHOLD)"
fi

# ----- phase 4: union miner (p1/p2/p3 partial-fail + p3 still-failing) -------
echo ""; echo "===== phase 4 ====="; date -Is
P4_MANIFEST=$MANIFESTS/car_envs_v1_aug9_phase4.txt
P4_OUTDIR=$OUTPUTS/car_v1_aug9_phase4
if [ -d "$P3_OUTDIR" ]; then
    echo "  building phase-4 union manifest ..."
    python "$NAMO_REPO/scripts/build_phase4_manifest.py" \
        --phase1-dir "$OUTPUTS/car_v1_aug9_phase1" \
        --phase2-dir "$P2_OUTDIR" \
        --phase3-dir "$P3_OUTDIR" \
        --output     "$P4_MANIFEST" \
        --workers    "${SLURM_CPUS_PER_TASK:-8}"
    n=$(wc -l < "$P4_MANIFEST")
    echo "  phase-4 manifest: $n envs"
    if [ "$n" -gt "$STOP_THRESHOLD" ]; then
        submit_collection "$P4_MANIFEST" "$P4_OUTDIR" 50 2 "p4"
    else
        echo "  skipping phase 4 (manifest <= $STOP_THRESHOLD)"
    fi
else
    echo "  phase-3 outdir missing; skipping phase 4"
fi

# ----- phase 5: mine phase-4, hand to phase5_driver_sbatch.slurm -------------
echo ""; echo "===== phase 5 ====="; date -Is
P5_INIT_MANIFEST=$MANIFESTS/car_envs_v1_aug9_phase5_init.txt
if [ -d "$P4_OUTDIR" ]; then
    echo "  mining phase-5 initial manifest from phase-4 ..."
    mine_simple "$P4_OUTDIR" "$P5_INIT_MANIFEST"
else
    cp "$P3_MANIFEST" "$P5_INIT_MANIFEST" 2>/dev/null || cp "$P2_MANIFEST" "$P5_INIT_MANIFEST"
fi
n=$(wc -l < "$P5_INIT_MANIFEST")
echo "  phase-5 initial manifest: $n envs"
if [ "$n" -gt "$STOP_THRESHOLD" ]; then
    echo "  submitting phase-5 driver ..."
    p5_jid=$(START_PASS=A INITIAL_MANIFEST="$P5_INIT_MANIFEST" \
        sbatch "$NAMO_REPO/scripts/amarel/phase5_driver_sbatch.slurm" | awk '{print $4}')
    echo "  phase-5 driver job: $p5_jid"
    wait_for_job "$p5_jid"
    echo "  phase-5 driver done @ $(date -Is)"
else
    echo "  skipping phase 5 (residual <= $STOP_THRESHOLD)"
fi

echo ""; echo "===== phases 2345 complete @ $(date -Is) ====="
