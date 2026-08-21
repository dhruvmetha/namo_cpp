#!/bin/bash
#SBATCH --job-name=c0continue
#SBATCH --account=general
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=08:00:00
set -euo pipefail

: "${NAMO_REPO:?}"
: "${RESULT_ROOT:?}"
: "${YIELD_REPORT:?}"
cd "$NAMO_REPO"
source env.amarel.sh >/dev/null 2>&1

target_xmls=${TARGET_XMLS:-225000}
offset=${INITIAL_XMLS:-10000}
wave_size=${WAVE_SIZE:-164500}
shard_size=${SHARD_SIZE:-350}
max_tasks=${MAX_WAVE_TASKS:-470}
minimum_positive=${MINIMUM_PILOT_POSITIVE:-7408}

read -r h5_shards rows positive negatives < <(
  "$NAMO_PYTHON" -c '
import json, sys
d = json.load(open(sys.argv[1]))
c = d["candidate_counts"]
print(len(d["candidate_h5s"]), c["rows"], c["positive"], c["negative_root"] + c["negative_finish"])
' "$YIELD_REPORT"
)
if [[ "$h5_shards" -ne 32 || "$rows" -lt 100000 || "$positive" -lt "$minimum_positive" || "$negatives" -lt 33334 ]]; then
  echo "yield gate failed: shards=$h5_shards rows=$rows positive=$positive negatives=$negatives" >&2
  exit 1
fi
projected_positive=$((positive * target_xmls / offset))
if [[ "$projected_positive" -lt 166666 ]]; then
  echo "225k projection misses positive target: $projected_positive" >&2
  exit 1
fi

existing=$(find "$RESULT_ROOT/collect" -type f -name '*_results.pkl' | wc -l)
if [[ "$existing" -ne "$offset" ]]; then
  echo "expected exactly $offset existing result PKLs, found $existing" >&2
  exit 1
fi

if [[ ${DRY_RUN:-0} == 1 ]]; then
  dry_offset=$offset
  dry_wave=1
  while ((dry_offset < target_xmls)); do
    dry_end=$((dry_offset + wave_size))
    if ((dry_end > target_xmls)); then dry_end=$target_xmls; fi
    dry_count=$((dry_end - dry_offset))
    dry_tasks=$(((dry_count + shard_size - 1) / shard_size))
    if ((dry_tasks > max_tasks)); then
      echo "wave task count exceeds cap: $dry_tasks" >&2
      exit 1
    fi
    echo "dry-run wave=$dry_wave XMLs=[$dry_offset,$dry_end) tasks=$dry_tasks"
    dry_offset=$dry_end
    dry_wave=$((dry_wave + 1))
  done
  echo "dry-run gate passed: shards=$h5_shards rows=$rows positive=$positive negatives=$negatives projected_positive=$projected_positive"
  exit 0
fi

mkdir "$RESULT_ROOT/auto_continue_225k_started"

wait_submit_slots() {
  local needed=$1
  while true; do
    local queued
    queued=$(squeue -h -r -u "$USER" | wc -l)
    if ((queued + needed <= 500)); then return; fi
    sleep 60
  done
}

wait_job() {
  local job_id=$1
  while true; do
    if squeue -h -j "$job_id" | grep -q .; then
      sleep 60
      continue
    fi
    local states
    states=$(sacct -X -j "$job_id" -n -P -o State | sed '/^$/d')
    if [[ -z "$states" ]] || grep -Eq 'PENDING|RUNNING|CONFIGURING|COMPLETING|REQUEUED|RESIZING|SUSPENDED' <<<"$states"; then
      sleep 60
      continue
    fi
    if grep -Evq '^COMPLETED\+?$' <<<"$states"; then
      echo "collection wave failed: $job_id" >&2
      return 1
    fi
    return
  done
}

wave=1
while ((offset < target_xmls)); do
  end=$((offset + wave_size))
  if ((end > target_xmls)); then end=$target_xmls; fi
  count=$((end - offset))
  tasks=$(((count + shard_size - 1) / shard_size))
  if ((tasks > max_tasks)); then
    echo "wave task count exceeds cap: $tasks" >&2
    exit 1
  fi
  wait_submit_slots "$tasks"
  job=$(sbatch --parsable --array="0-$((tasks - 1))%$max_tasks" --time=03:00:00 \
    --output="$RESULT_ROOT/logs/collect_w${wave}_%A_%a.out" \
    --error="$RESULT_ROOT/logs/collect_w${wave}_%A_%a.err" \
    --export=ALL,NAMO_REPO="$NAMO_REPO",RESULT_ROOT="$RESULT_ROOT",ROOT="$RESULT_ROOT",BATCH_START="$offset",BATCH_END="$end",WAVE="$wave",SHARD_SIZE="$shard_size" \
    "$NAMO_REPO/scripts/amarel/colossus0_collect.sbatch")
  echo "collection wave=$wave job=$job XMLs=[$offset,$end) tasks=$tasks"
  wait_job "$job"
  offset=$end
  wave=$((wave + 1))
done

final_count=$(find "$RESULT_ROOT/collect" -type f -name '*_results.pkl' | wc -l)
if [[ "$final_count" -ne "$target_xmls" ]]; then
  echo "expected $target_xmls result PKLs, found $final_count" >&2
  exit 1
fi
census_job=$(sbatch --parsable \
  --output="$RESULT_ROOT/logs/census225k_%j.out" \
  --error="$RESULT_ROOT/logs/census225k_%j.err" \
  --export=ALL,NAMO_REPO="$NAMO_REPO",RESULT_ROOT="$RESULT_ROOT" \
  "$NAMO_REPO/scripts/amarel/colossus0_census.sbatch")
echo "all $target_xmls XMLs collected; census job=$census_job"
