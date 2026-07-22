#!/bin/bash
set -euo pipefail

: "${NAMO_REPO:?source env.amarel.sh before launching}"
: "${NAMO_ENV_CREATOR:?source env.amarel.sh before launching}"
ROOT=${ROOT:-/scratch/dm1487/curriculum2_amarel/colossus0_1m}
SOURCE_COLOSSUS=/scratch/dm1487/curriculum2_amarel/colossus
TESTSET_SOURCE=/scratch/dm1487/curriculum2_amarel/collect2/testset_xmls.txt
GEN_SHARDS=240
SCENES_TOTAL=1000000
SCENES_PER_WAVE=164500
SHARD_SIZE=350
MAX_WAVE_TASKS=470

if [[ -e "$ROOT/STARTED" ]]; then
  echo "refusing to reuse an existing run root: $ROOT" >&2
  exit 1
fi
mkdir -p "$ROOT/logs" "$ROOT/manifests" "$ROOT/gen" "$ROOT/collect"
date -Is > "$ROOT/STARTED"
cp --no-clobber "$SOURCE_COLOSSUS/d20_finish_ranker.ckpt" "$ROOT/d20_finish_ranker.ckpt"
cp --no-clobber "$TESTSET_SOURCE" "$ROOT/testset_xmls.txt"

log() { echo "[$(date -Is)] $*"; }
wait_job() {
  local job_id=$1
  while squeue -h -j "$job_id" | grep -q .; do sleep 60; done
  if sacct -X -j "$job_id" -n -o State | grep -Eq 'FAILED|CANCELLED|TIMEOUT|OUT_OF_MEMORY|NODE_FAIL'; then
    log "job $job_id failed"
    return 1
  fi
}
wait_submit_slots() {
  local needed=$1
  while true; do
    local queued
    queued=$(squeue -h -u dm1487 | wc -l)
    if ((queued + needed <= 500)); then return; fi
    log "waiting for submit slots: queued=$queued needed=$needed"
    sleep 120
  done
}

JOBLIST="$ROOT/manifests/generation_commands.txt"
"$NAMO_PYTHON" "$NAMO_REPO/scripts/amarel/colossus0_build_gen_joblist.py" \
  --generator-root "$NAMO_ENV_CREATOR" \
  --namo-config "$NAMO_REPO/config/namo_config_complete_skill15_car_1x.yaml" \
  --output-root "$ROOT" --out "$JOBLIST" --python "$NAMO_PYTHON" \
  --aug9-base-envs 360000 --feb-base-envs 800000 --chunk-envs 20

wait_submit_slots "$GEN_SHARDS"
GEN_JOB=$(sbatch --parsable --array="0-$((GEN_SHARDS - 1))" \
  --export=ALL,NAMO_REPO="$NAMO_REPO",JOBLIST="$JOBLIST" \
  "$NAMO_REPO/scripts/amarel/colossus0_generate.sbatch")
log "generation job=$GEN_JOB"
wait_job "$GEN_JOB"

FINAL_JOB=$(sbatch --parsable --export=ALL,NAMO_REPO="$NAMO_REPO",ROOT="$ROOT" \
  "$NAMO_REPO/scripts/amarel/colossus0_finalize.sbatch")
log "finalize job=$FINAL_JOB"
wait_job "$FINAL_JOB"

offset=0
wave=0
while ((offset < SCENES_TOTAL)); do
  end=$((offset + SCENES_PER_WAVE))
  if ((end > SCENES_TOTAL)); then end=$SCENES_TOTAL; fi
  count=$((end - offset))
  tasks=$(((count + SHARD_SIZE - 1) / SHARD_SIZE))
  if ((tasks > MAX_WAVE_TASKS)); then
    echo "wave task count exceeds cap: $tasks" >&2
    exit 1
  fi
  wait_submit_slots "$tasks"
  job=$(sbatch --parsable --array="0-$((tasks - 1))%$MAX_WAVE_TASKS" \
    --export=ALL,NAMO_REPO="$NAMO_REPO",ROOT="$ROOT",BATCH_START="$offset",BATCH_END="$end",WAVE="$wave" \
    "$NAMO_REPO/scripts/amarel/colossus0_collect.sbatch")
  log "collection wave=$wave job=$job scenes=[$offset,$end) tasks=$tasks"
  wait_job "$job"
  offset=$end
  wave=$((wave + 1))
done
date -Is > "$ROOT/COLLECTION_DONE"
log "all $SCENES_TOTAL XMLs collected"
