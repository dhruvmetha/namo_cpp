#!/bin/bash
# _full_search 1push gap-fill: best-first SEARCH on the ONEPUSH key (hmax=1, budget 900, n=1323 episodes).
# This horizon was never run on Amarel (only 2push fullsearch exists). Mirrors the 2push campaign:
#   NoHz-v3 model (3 ckpt-seeds s1/s2/s3, MODELS=NoHz, rng=7) vs random (10 rng-seeds 0-9, MODELS=random).
# Runs on iLab `unlimited`, CPU inference (fast_scorer verified working off-Amarel on shared sage_learning).
# 1push pool per object is small (<=~35 pushes) so budget 900 is never binding — search exhausts the pool.
# Output: fullsearch_1push/nohz_s{1,2,3}/shard_0.jsonl  +  fullsearch_1push/rand_s{0..9}/shard_0.jsonl
set -euo pipefail
REPO=/common/home/dm1487/robotics_research/ktamp/namo; cd "$REPO"
S=/common/users/dm1487/scratch_namo/sage_outputs/scorer
OUT=/common/users/dm1487/scratch_namo/eval/fullsearch_1push
KEY=$(PYTHONPATH="$REPO/build_python:$REPO/python:${PYTHONPATH:-}" "${NAMO_PYTHON:-python}" -m namo.eval_sets onepush_manifest)
# Pin to a high-core, low-contention node so unlimited-partition preemption doesn't churn these short jobs.
NODE="${NODELIST:-rlab7}"
declare -A NOHZ=(
  [s1]=$S/qfull_nohz_v3_v4hq_s1/namo-classifier/wl8k6iyv/checkpoints/epoch012-val_loss0.6896.ckpt
  [s2]=$S/qfull_nohz_v3_v4hq_s2/namo-classifier/kzph0acr/checkpoints/epoch012-val_loss0.6922.ckpt
  [s3]=$S/qfull_nohz_v3_v4hq_s3/namo-classifier/dlopoael/checkpoints/epoch012-val_loss0.6874.ckpt )
  # NOTE: shared-FS s3 has only ep012 (Amarel/2push-s3 used ep011, not synced here). Same dlopoael run, +1 epoch.

# --- 3 NoHz model seeds (deterministic; MODELS=NoHz) ---
for sd in s1 s2 s3; do
  sbatch --array=0-0 --nodelist="$NODE" \
    --export="OUT_DIR=$OUT/nohz_${sd},SHARD=1400,BUDGET=900,HMAX=1,KEY=$KEY,MODELS=NoHz,NOHZ_CKPT=${NOHZ[$sd]},RNG_SEED=7" \
    scripts/ilab/fullsearch_bestfirst_ilab.slurm | sed "s/^/  nohz $sd -> /"
done

# --- 10 random rng-seeds (MODELS=random; nohz ckpt is only a planner shell -> prim only, no scoring) ---
for rs in 0 1 2 3 4 5 6 7 8 9; do
  sbatch --array=0-0 --nodelist="$NODE" \
    --export="OUT_DIR=$OUT/rand_s${rs},SHARD=1400,BUDGET=900,HMAX=1,KEY=$KEY,MODELS=random,NOHZ_CKPT=${NOHZ[s1]},RNG_SEED=${rs}" \
    scripts/ilab/fullsearch_bestfirst_ilab.slurm | sed "s/^/  rand s${rs} -> /"
done
echo "launched 13 jobs (3 NoHz ckpt-seeds + 10 random rng-seeds), iLab unlimited CPU, budget 900 hmax 1, onepush n=1323"
