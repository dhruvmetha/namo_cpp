#!/bin/bash
# PARALLEL phase-5 driver: fires all 5 seed-sweep passes (A-E) SIMULTANEOUSLY
# on the SAME phase-4 residual manifest. Each pass uses a different seed.
# Same env may solve in multiple passes — kept intentionally, contributes
# diverse 2-push chains per env.
#
# Required env vars at submit:
#   INITIAL_MANIFEST  phase-4 residual manifest
#
# Optional:
#   CORPUS_TAG        prefix for output dirs (default car_v2_500k)
#
# Outputs:
#   /scratch/dm1487/outputs/${CORPUS_TAG}_phase5{A..E}/  collection PKLs
#
#SBATCH --partition=main-redhat
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/dm1487/logs/phase5_v2_500k_sbatch_driver-%j.out
#SBATCH --error=/scratch/dm1487/logs/phase5_v2_500k_sbatch_driver-%j.err

set -euo pipefail
mkdir -p /scratch/dm1487/logs

NAMO_REPO=${NAMO_REPO:-/cache/home/dm1487/projects/namo/namo_cpp}
source "$NAMO_REPO/scripts/amarel/activate.sh"

CORPUS_TAG=${CORPUS_TAG:-car_v2_500k}
manifest="${INITIAL_MANIFEST:?must set INITIAL_MANIFEST}"
n_envs=$(wc -l < "$manifest")

MAX_SHARDS=60
TARGET_SHARD_SIZE=100
shards=$(python -c "import math; print(min($MAX_SHARDS, max(1, math.ceil($n_envs / $TARGET_SHARD_SIZE))))")
shard_size=$(python -c "import math; print(math.ceil($n_envs / $shards))")
last_array=$(( shards - 1 ))

echo "===== phase-5 PARALLEL driver ====="
echo "  driver job:       ${SLURM_JOB_ID:-N/A}"
echo "  corpus tag:       $CORPUS_TAG"
echo "  initial manifest: $manifest  ($n_envs envs)"
echo "  per pass:         $shards shards x $shard_size envs"
echo "  date:             $(date -Is)"

declare -A SEEDS=([A]=101 [B]=102 [C]=103 [D]=104 [E]=105)

# Fire all 5 passes in parallel
declare -A PASS_JIDS
for pass in A B C D E; do
    seed="${SEEDS[$pass]}"
    out_dir=/scratch/dm1487/outputs/${CORPUS_TAG}_phase5${pass}
    rm -rf "$out_dir"
    echo ""
    echo "----- submitting pass $pass (seed=$seed) -----"
    job_id=$(SHARD_SIZE=$shard_size \
        MANIFEST=$manifest \
        OUTPUT_DIR=$out_dir \
        EXTRA_ARGS="--goal-strategy random_rollout --region-max-chain-depth 2 --rollout-samples-per-state 50 --target-goal-region --shuffle-seed $seed" \
        sbatch --partition=main-redhat --array=0-${last_array} --cpus-per-task=32 --mem=100G --time=04:00:00 \
            --job-name=namo-v2-p5${pass} \
            "$NAMO_REPO/scripts/amarel/run_amarel_collect_array.slurm" | awk '{print $4}')
    echo "  pass $pass: job $job_id"
    PASS_JIDS[$pass]=$job_id
done

echo ""
echo "===== all 5 passes submitted concurrently ====="
for pass in A B C D E; do echo "  pass $pass: ${PASS_JIDS[$pass]}"; done

# Wait for all 5 to finish (running in parallel)
all_jids="${PASS_JIDS[A]},${PASS_JIDS[B]},${PASS_JIDS[C]},${PASS_JIDS[D]},${PASS_JIDS[E]}"
while [ "$(squeue -j $all_jids -h 2>/dev/null | wc -l)" -gt 0 ]; do
    sleep 60
done

echo ""
echo "===== phase-5 parallel complete @ $(date -Is) ====="
for pass in A B C D E; do
    out_dir=/scratch/dm1487/outputs/${CORPUS_TAG}_phase5${pass}
    cnt=$(find "$out_dir" -name "*_results.pkl" 2>/dev/null | wc -l)
    echo "  pass $pass: $cnt PKLs"
done
