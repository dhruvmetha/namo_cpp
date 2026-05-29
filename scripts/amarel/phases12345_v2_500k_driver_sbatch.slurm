#!/bin/bash
# Sequential driver for v2 500k corpus: phase 1 → 2 → 3 → 4 → 5(A-E).
# Phase 1 = broad depth-1 sweep on the full 500k manifest.
# Phases 2-5 mine residuals from prior phases (same cascade as v1).
#
# Output layout (under $OUTPUTS):
#   car_v2_500k_phase1/   modular_data_<host>_<jobid>_t<task>/  (depth=1, K=all)
#   car_v2_500k_phase2/   ...                                     (depth=2, K=20)
#   car_v2_500k_phase3/   ...                                     (depth=2, K=50)
#   car_v2_500k_phase4/   ...                                     (depth=2, K=50, partial-fail union)
#   car_v2_500k_phase5{A..E}/   seed sweep (depth=2, K=50)
#
#SBATCH --job-name=namo-v2-500k-pipe
#SBATCH --partition=main-redhat
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=72:00:00
#SBATCH --output=/scratch/dm1487/logs/phases12345_v2_500k_driver-%j.out
#SBATCH --error=/scratch/dm1487/logs/phases12345_v2_500k_driver-%j.err

set -euo pipefail
mkdir -p /scratch/dm1487/logs

NAMO_REPO=${NAMO_REPO:-/cache/home/dm1487/projects/namo/namo_cpp}
source "$NAMO_REPO/scripts/amarel/activate.sh"

MAX_SHARDS=60
TARGET_SHARD_SIZE_P1=8500   # phase 1: ceil(500k / 60) = 8334
TARGET_SHARD_SIZE=100
STOP_THRESHOLD=50
MANIFESTS=/scratch/dm1487/manifests
OUTPUTS=/scratch/dm1487/outputs

wait_for_job() {
    local jid=$1
    while [ "$(squeue -j $jid -h 2>/dev/null | wc -l)" -gt 0 ]; do sleep 60; done
}

submit_collection() {
    # $1=manifest  $2=output_dir  $3=K  $4=depth  $5=short  $6=target_shard_size
    local manifest=$1 outdir=$2 K=$3 depth=$4 short=$5 tss=${6:-$TARGET_SHARD_SIZE}
    local n_envs shards shard_size last_array
    n_envs=$(wc -l < "$manifest")
    shards=$(python -c "import math; print(min($MAX_SHARDS, max(1, math.ceil($n_envs / $tss))))")
    shard_size=$(python -c "import math; print(math.ceil($n_envs / $shards))")
    last_array=$(( shards - 1 ))
    echo "    envs=$n_envs shards=$shards shard_size=$shard_size K=$K depth=$depth"
    rm -rf "$outdir"
    local K_arg=""
    if [ "$K" != "all" ]; then K_arg="--rollout-samples-per-state $K"; fi
    local jid
    jid=$(SHARD_SIZE=$shard_size MANIFEST=$manifest OUTPUT_DIR=$outdir \
        EXTRA_ARGS="--goal-strategy random_rollout --region-max-chain-depth $depth $K_arg --target-goal-region" \
        sbatch --partition=main-redhat --array=0-${last_array} --cpus-per-task=32 --mem=100G --time=24:00:00 \
            --job-name=namo-${short} \
            "$NAMO_REPO/scripts/amarel/run_amarel_collect_array.slurm" | awk '{print $4}')
    echo "    collection job: $jid"
    wait_for_job "$jid"
    echo "    $short collection done @ $(date -Is)"
}

mine_simple() {
    sbatch_jid=$(sbatch "$NAMO_REPO/scripts/amarel/mine_residual.slurm" "$1" "$2" | awk '{print $4}')
    echo "    miner job: $sbatch_jid"
    wait_for_job "$sbatch_jid"
}

# ===== phase 1: full 500k, depth-1, K=all =====
echo "===== phase 1 ====="; date -Is
P1_MANIFEST=$MANIFESTS/car_envs_v2_500k.txt
P1_OUTDIR=$OUTPUTS/car_v2_500k_phase1
n=$(wc -l < "$P1_MANIFEST")
echo "  phase-1 manifest: $n envs"
submit_collection "$P1_MANIFEST" "$P1_OUTDIR" all 1 "v2p1" "$TARGET_SHARD_SIZE_P1"

# ===== phase 2: mine phase-1, depth-2 K=20 =====
echo ""; echo "===== phase 2 ====="; date -Is
P2_MANIFEST=$MANIFESTS/car_envs_v2_500k_phase2.txt
P2_OUTDIR=$OUTPUTS/car_v2_500k_phase2
echo "  mining phase-2 manifest from phase-1 ..."
mine_simple "$P1_OUTDIR" "$P2_MANIFEST"
n=$(wc -l < "$P2_MANIFEST")
echo "  phase-2 manifest: $n envs"
[ "$n" -gt "$STOP_THRESHOLD" ] && submit_collection "$P2_MANIFEST" "$P2_OUTDIR" 20 2 "v2p2" || echo "  skip"

# ===== phase 3: mine phase-2, depth-2 K=50 =====
echo ""; echo "===== phase 3 ====="; date -Is
P3_MANIFEST=$MANIFESTS/car_envs_v2_500k_phase3.txt
P3_OUTDIR=$OUTPUTS/car_v2_500k_phase3
mine_simple "$P2_OUTDIR" "$P3_MANIFEST"
n=$(wc -l < "$P3_MANIFEST")
echo "  phase-3 manifest: $n envs"
[ "$n" -gt "$STOP_THRESHOLD" ] && submit_collection "$P3_MANIFEST" "$P3_OUTDIR" 50 2 "v2p3" || echo "  skip"

# ===== phase 4: union (p1+p2+p3 partial-fail + p3 still-failing) =====
echo ""; echo "===== phase 4 ====="; date -Is
P4_MANIFEST=$MANIFESTS/car_envs_v2_500k_phase4.txt
P4_OUTDIR=$OUTPUTS/car_v2_500k_phase4
python "$NAMO_REPO/scripts/build_phase4_manifest.py" \
    --phase1-dir "$P1_OUTDIR" --phase2-dir "$P2_OUTDIR" --phase3-dir "$P3_OUTDIR" \
    --output "$P4_MANIFEST" --workers 8
n=$(wc -l < "$P4_MANIFEST")
echo "  phase-4 manifest: $n envs"
[ "$n" -gt "$STOP_THRESHOLD" ] && submit_collection "$P4_MANIFEST" "$P4_OUTDIR" 50 2 "v2p4" || echo "  skip"

# ===== phase 5: seed sweep =====
echo ""; echo "===== phase 5 ====="; date -Is
P5_INIT_MANIFEST=$MANIFESTS/car_envs_v2_500k_phase5_init.txt
mine_simple "$P4_OUTDIR" "$P5_INIT_MANIFEST"
n=$(wc -l < "$P5_INIT_MANIFEST")
echo "  phase-5 initial manifest: $n envs"
if [ "$n" -gt "$STOP_THRESHOLD" ]; then
    # Build a v2-500k phase-5 driver dynamically by setting INITIAL_MANIFEST
    p5_jid=$(START_PASS=A INITIAL_MANIFEST="$P5_INIT_MANIFEST" \
        CORPUS_TAG=car_v2_500k \
        sbatch "$NAMO_REPO/scripts/amarel/phase5_v2_500k_driver_sbatch.slurm" | awk '{print $4}')
    echo "  phase-5 driver job: $p5_jid"
    wait_for_job "$p5_jid"
fi

echo ""; echo "===== v2 500k cascade complete @ $(date -Is) ====="
