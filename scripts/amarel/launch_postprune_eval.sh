#!/bin/bash
# Canonical post-pruning search matrix: optional setup-only ranker + random RNG seeds 7000/8000/9000,
# on BOTH registered eval sets, always hmax=2 / budget=900 / combine=q. Search discount is configurable.
set -euo pipefail
REPO="${NAMO_REPO:-/cache/home/dm1487/projects/namo/namo_postprune_eval}"
cd "$REPO"
source env.amarel.sh >/dev/null 2>&1
: "${CKPT:?set CKPT to the synced setup-only checkpoint}"
OUT_ROOT=${OUT_ROOT:-/cache/home/dm1487/eval/postprune_hmax2/full}
DISCOUNT=${DISCOUNT:-conf}
TAU=${TAU:-0.15}
RUN_MODEL=${RUN_MODEL:-1}
SHARD_1P=${SHARD_1P:-34}
SHARD_2P=${SHARD_2P:-26}
MAX_PARALLEL=${MAX_PARALLEL:-40}
ONEPUSH_KEY=${ONEPUSH_KEY:-$("$NAMO_PYTHON" -m namo.eval_sets onepush_manifest)}
TWOPUSH_KEY=${TWOPUSH_KEY:-$("$NAMO_PYTHON" -m namo.eval_sets pure2push_manifest)}
ONEPUSH_SCENES=$("$NAMO_PYTHON" -c 'import json,sys; print(len(json.load(open(sys.argv[1]))))' "$ONEPUSH_KEY")
TWOPUSH_SCENES=$("$NAMO_PYTHON" -c 'import json,sys; print(len(json.load(open(sys.argv[1]))))' "$TWOPUSH_KEY")

launch() {
  local label=$1 key=$2 count=$3 shard=$4 prior=$5 seed=$6
  local last=$(( (count + shard - 1) / shard - 1 ))
  local out="$OUT_ROOT/$label"
  local job
  # SBATCH_EXTRA: wall-clock protocol passes "--exclusive --constraint=icelake" so every shard owns its
  # node and every shard runs on ONE CPU microarchitecture -- the only way pooled times stay comparable.
  # Empty (default) = the ordinary shared-node sim-count run, unchanged.
  job=$(sbatch --parsable ${SBATCH_EXTRA:-} --array="0-${last}%${MAX_PARALLEL}" \
    --export="ALL,MANIFEST=,NAMO_REPO=$REPO,CKPT=$CKPT,KEY=$key,OUT_DIR=$out,HMAX=2,SIM_BUDGET=900,PRIOR=$prior,AGG=mean5,COMBINE=q,DISCOUNT=$DISCOUNT,TAU=$TAU,SEED_BASE=$seed,SHARD=$shard" \
    scripts/amarel/bestfirst_eval.slurm)
  printf '%-24s %s\n' "$label" "$job"
}

if [[ "$RUN_MODEL" == "1" ]]; then
  launch model_1push "$ONEPUSH_KEY" "$ONEPUSH_SCENES" "$SHARD_1P" model 7000
  launch model_2push "$TWOPUSH_KEY" "$TWOPUSH_SCENES" "$SHARD_2P" model 7000
fi
for seed in 7000 8000 9000; do
  launch "random_s${seed}_1push" "$ONEPUSH_KEY" "$ONEPUSH_SCENES" "$SHARD_1P" uniform "$seed"
  launch "random_s${seed}_2push" "$TWOPUSH_KEY" "$TWOPUSH_SCENES" "$SHARD_2P" uniform "$seed"
done
