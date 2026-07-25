#!/bin/bash
# _full_search experiment: instrumented best-first campaign, budget 900, hmax 2, key pure2push (n=1018).
# NoHz-v3 model (3 ckpt-seeds s1/s2/s3) vs random (10 rng-seeds 0-9), q-ranker (combine='q').
# Every job is --exclusive AND pinned to emeraldrapids (icelake was drained on the degraded cluster), so
#   model & random pool onto identical hardware => the success-vs-TIME comparison is same-machine-fair,
#   and the instrumented sims/depth_hist/solve_ranks are machine-independent regardless.
# array 0-2 x SHARD=340 = 1020 >= 1018 episodes.  13 jobs x 3 shards = 39 exclusive emerald nodes.
# Output: fullsearch/nohz_s{1,2,3}/shard_*.jsonl  +  fullsearch/rand_s{0..9}/shard_*.jsonl
set -euo pipefail
REPO="${NAMO_REPO:-/cache/home/dm1487/projects/namo/namo_cpp}"; cd "$REPO"
S=/scratch/dm1487/sage_outputs/scorer
PURE2PUSH_KEY=$(PYTHONPATH="$REPO/build_python:$REPO/python:${PYTHONPATH:-}" "${NAMO_PYTHON:-python}" -m namo.eval_sets pure2push_manifest)
OUT=/scratch/dm1487/eval/fullsearch
CON="${CONSTRAINT:-emeraldrapids}"
declare -A NOHZ=(
  [s1]=$S/qfull_nohz_v3_v4hq_s1/namo-classifier/wl8k6iyv/checkpoints/epoch012-val_loss0.6896.ckpt
  [s2]=$S/qfull_nohz_v3_v4hq_s2/namo-classifier/kzph0acr/checkpoints/epoch012-val_loss0.6922.ckpt
  [s3]=$S/qfull_nohz_v3_v4hq_s3/namo-classifier/dlopoael/checkpoints/epoch011-val_loss0.6897.ckpt )

# --- 3 NoHz model seeds (deterministic; MODELS=NoHz) ---
for sd in s1 s2 s3; do
  sbatch --constraint="$CON" --array=0-2 \
    --export="ALL,OUT_DIR=$OUT/nohz_${sd},SHARD=340,BUDGET=900,HMAX=2,KEY=$PURE2PUSH_KEY,MODELS=NoHz,NOHZ_CKPT=${NOHZ[$sd]},RNG_SEED=7,NAMO_REPO=$REPO" \
    scripts/amarel/time_bestfirst_shard.slurm | sed "s/^/  nohz $sd -> /"
done

# --- 10 random rng-seeds (MODELS=random; nohz ckpt is only a planner shell -> prim only, no scoring) ---
for rs in 0 1 2 3 4 5 6 7 8 9; do
  sbatch --constraint="$CON" --array=0-2 \
    --export="ALL,OUT_DIR=$OUT/rand_s${rs},SHARD=340,BUDGET=900,HMAX=2,KEY=$PURE2PUSH_KEY,MODELS=random,NOHZ_CKPT=${NOHZ[s1]},RNG_SEED=${rs},NAMO_REPO=$REPO" \
    scripts/amarel/time_bestfirst_shard.slurm | sed "s/^/  rand s${rs} -> /"
done
echo "launched 13 arrays (3 NoHz ckpt-seeds + 10 random rng-seeds), emeraldrapids exclusive, budget 900, pure2push"
