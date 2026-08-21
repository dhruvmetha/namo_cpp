#!/bin/bash
#SBATCH --job-name=c0repair
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
cd "$NAMO_REPO"
source env.amarel.sh >/dev/null 2>&1

target_xmls=${TARGET_XMLS:-225000}
mapfile -t missing < <(
  "$NAMO_PYTHON" - "$RESULT_ROOT/collect" "$target_xmls" <<'PY'
import os
import re
import sys

root, target = sys.argv[1], int(sys.argv[2])
seen = set()
pattern = re.compile(r"_env_(\d+)_results[.]pkl$")
for directory, _, files in os.walk(root):
    for name in files:
        match = pattern.search(name)
        if match:
            seen.add(int(match.group(1)))
for index in range(target):
    if index not in seen:
        print(index)
PY
)

if ((${#missing[@]} == 0)); then
  echo "no missing indices; primary census dependency remains authoritative"
  exit 0
fi
echo "repairing ${#missing[@]} missing indices: ${missing[*]}"

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
      echo "repair failed: $job_id" >&2
      return 1
    fi
    return
  done
}

jobs=()
for index in "${missing[@]}"; do
  job=$(sbatch --parsable --array=0-0 --time=06:00:00 \
    --output="$RESULT_ROOT/logs/repair_${index}_%A_%a.out" \
    --error="$RESULT_ROOT/logs/repair_${index}_%A_%a.err" \
    --export=ALL,NAMO_REPO="$NAMO_REPO",ROOT="$RESULT_ROOT",BATCH_START="$index",BATCH_END="$((index + 1))",WAVE=repair,SHARD_SIZE=1 \
    "$NAMO_REPO/scripts/amarel/colossus0_collect.sbatch")
  jobs+=("$job")
done
for job in "${jobs[@]}"; do wait_job "$job"; done

final_count=$(find "$RESULT_ROOT/collect" -type f -name '*_results.pkl' | wc -l)
if [[ "$final_count" -ne "$target_xmls" ]]; then
  echo "expected $target_xmls result PKLs after repair, found $final_count" >&2
  exit 1
fi
if [[ -e "$RESULT_ROOT/census_225k.json" || -e "$RESULT_ROOT/pure2_225k.json" ]]; then
  echo "census output already exists; not submitting a duplicate"
  exit 0
fi
census_job=$(sbatch --parsable \
  --output="$RESULT_ROOT/logs/census225k_repair_%j.out" \
  --error="$RESULT_ROOT/logs/census225k_repair_%j.err" \
  --export=ALL,NAMO_REPO="$NAMO_REPO",RESULT_ROOT="$RESULT_ROOT",EXPECTED_RESULTS="$target_xmls" \
  "$NAMO_REPO/scripts/amarel/colossus0_census.sbatch")
echo "repair complete; census job=$census_job"
