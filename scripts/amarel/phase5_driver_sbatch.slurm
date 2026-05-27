#!/bin/bash
# Sequential phase-5 driver, running INSIDE a SLURM allocation so it survives
# any client/srun disconnect. Submits collection + mining sbatch jobs and polls
# squeue until each finishes, then chains to the next pass.
#
# Required env vars at submit:
#   START_PASS    first pass to run (default 'C', valid {A,B,C,D,E})
#   INITIAL_MANIFEST  manifest to use for the first pass
#
# Outputs:
#   /scratch/dm1487/outputs/car_v1_aug9_phase5<X>/  (collection PKLs)
#   /scratch/dm1487/manifests/car_envs_v1_aug9_post_phase5<X>_residual.txt
#   /scratch/dm1487/logs/phase5_sbatch_driver-%j.out  (this driver's log)
#
#SBATCH --partition=main-redhat
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/dm1487/logs/phase5_sbatch_driver-%j.out
#SBATCH --error=/scratch/dm1487/logs/phase5_sbatch_driver-%j.err

set -euo pipefail
mkdir -p /scratch/dm1487/logs

NAMO_REPO=${NAMO_REPO:-/cache/home/dm1487/projects/namo/namo_cpp}
source "$NAMO_REPO/scripts/amarel/activate.sh"

STOP_THRESHOLD=50
MAX_SHARDS=60
TARGET_SHARD_SIZE=100

start_pass="${START_PASS:-C}"
manifest="${INITIAL_MANIFEST:?must set INITIAL_MANIFEST}"

# Schedule: pass letter → seed (101..105)
declare -A SEEDS=([A]=101 [B]=102 [C]=103 [D]=104 [E]=105)

# Pass order
all_passes=(A B C D E)
start_idx=-1
for i in "${!all_passes[@]}"; do
    if [ "${all_passes[$i]}" = "$start_pass" ]; then
        start_idx=$i; break
    fi
done
[ "$start_idx" -lt 0 ] && { echo "ERROR: invalid START_PASS=$start_pass"; exit 1; }

echo "===== phase-5 sbatch driver ====="
echo "  driver job:      ${SLURM_JOB_ID:-N/A}"
echo "  starting from:   $start_pass"
echo "  initial manifest: $manifest"
echo "  date:            $(date -Is)"

for ((i=start_idx; i<${#all_passes[@]}; i++)); do
    pass="${all_passes[$i]}"
    seed="${SEEDS[$pass]}"
    n_envs=$(wc -l < "$manifest")
    echo ""
    echo "===== pass $pass (seed=$seed) on $n_envs envs ====="
    date -Is

    if [ "$n_envs" -le "$STOP_THRESHOLD" ]; then
        echo "  residual $n_envs <= $STOP_THRESHOLD — stopping early"
        break
    fi

    shards=$(python -c "import math; print(min($MAX_SHARDS, max(1, math.ceil($n_envs / $TARGET_SHARD_SIZE))))")
    shard_size=$(python -c "import math; print(math.ceil($n_envs / $shards))")
    last_array=$(( shards - 1 ))
    echo "  shards=$shards shard_size=$shard_size"

    out_dir=/scratch/dm1487/outputs/car_v1_aug9_phase5${pass}
    rm -rf "$out_dir"

    job_id=$(SHARD_SIZE=$shard_size \
        MANIFEST=$manifest \
        OUTPUT_DIR=$out_dir \
        EXTRA_ARGS="--goal-strategy random_rollout --region-max-chain-depth 2 --rollout-samples-per-state 50 --target-goal-region --shuffle-seed $seed" \
        sbatch --partition=main-redhat --array=0-${last_array} --cpus-per-task=32 --mem=100G --time=04:00:00 \
            --job-name=namo-p5${pass} \
            "$NAMO_REPO/scripts/amarel/run_amarel_collect_array.slurm" | awk '{print $4}')
    echo "  collection job: $job_id"

    while [ "$(squeue -j $job_id -h 2>/dev/null | wc -l)" -gt 0 ]; do
        sleep 60
    done
    echo "  pass $pass collection done @ $(date -Is)"

    next_manifest=/scratch/dm1487/manifests/car_envs_v1_aug9_post_phase5${pass}_residual.txt
    miner_job=$(sbatch "$NAMO_REPO/scripts/amarel/mine_residual.slurm" "$out_dir" "$next_manifest" | awk '{print $4}')
    echo "  miner job: $miner_job"
    while [ "$(squeue -j $miner_job -h 2>/dev/null | wc -l)" -gt 0 ]; do
        sleep 20
    done

    n_after=$(wc -l < "$next_manifest" 2>/dev/null || echo 0)
    n_caught=$(( n_envs - n_after ))
    pct=$(python -c "print(f'{100*$n_caught/$n_envs:.1f}')" 2>/dev/null || echo "?")
    echo "  pass $pass: caught $n_caught / $n_envs (${pct}%); residual now $n_after"
    manifest=$next_manifest
done

echo ""
echo "===== phase-5 complete ====="
echo "  final residual:  $(wc -l < "$manifest" 2>/dev/null || echo 'n/a') envs"
echo "  final manifest:  $manifest"
echo "  end time:        $(date -Is)"
