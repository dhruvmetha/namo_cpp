#!/bin/bash
# Background CS-side handoff: wait for architecture training, transfer best checkpoints, then launch Amarel eval.
set -euo pipefail

: "${AMAREL_REPO:?set AMAREL_REPO to the dedicated Amarel NAMO clone}"
: "${AMAREL_SAGE:?set AMAREL_SAGE to the matching dedicated Sage clone}"
: "${AMAREL_BINDINGS:?set AMAREL_BINDINGS to the canonical Amarel build_python directory}"
: "${AMAREL_CKPT_ROOT:?set AMAREL_CKPT_ROOT to a new checkpoint destination}"
: "${AMAREL_EVAL_ROOT:?set AMAREL_EVAL_ROOT to a new evaluation destination}"
: "${EXPECTED_NAMO_SHA:?set EXPECTED_NAMO_SHA to the committed evaluation code revision}"
: "${EXPECTED_SAGE_SHA:?set EXPECTED_SAGE_SHA to the committed model code revision}"

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$REPO"
source env.ilab.sh >/dev/null 2>&1

BASE=${HY5U_ARCH_BASE:-$NAMO_SCRATCH/aquaman/round0/architecture_ablations_20260904}
ARM_SPEC=${HY5U_ARCH_ARMS:-HY5U_global HY5U_no_local}
SUPERVISOR_LOG=${SUPERVISOR_LOG:-$BASE/supervisor.log}
TRAIN_SUPERVISOR_PID=${TRAIN_SUPERVISOR_PID:-1655534}
POLL_SECONDS=${POLL_SECONDS:-300}
AMAREL_HOST=${AMAREL_HOST:-amarel}
STAGE=$BASE/selected_checkpoints
LOCAL_EVAL_COPY=${LOCAL_EVAL_COPY:-$BASE/eval_amarel}
mkdir -p "$STAGE" "$LOCAL_EVAL_COPY"

echo "HANDOFF start $(date) supervisor=$TRAIN_SUPERVISOR_PID poll_seconds=$POLL_SECONDS"
while ! grep -q '^FLEET DONE ' "$SUPERVISOR_LOG" 2>/dev/null; do
  if ! kill -0 "$TRAIN_SUPERVISOR_PID" 2>/dev/null; then
    echo "training supervisor exited without FLEET DONE" >&2
    exit 1
  fi
  sleep "$POLL_SECONDS"
done
echo "HANDOFF training complete $(date)"

select_best() {
  local run_dir=$1
  "$NAMO_PYTHON" - "$run_dir" <<'PY'
import re
import sys
from pathlib import Path

root = Path(sys.argv[1])
matches = []
for path in root.glob("checkpoints/epoch*-val_loss*.ckpt"):
    match = re.search(r"epoch(\d+)-val_loss([0-9.]+)\.ckpt$", path.name)
    if match:
        matches.append((float(match.group(2)), int(match.group(1)), path))
if not matches:
    raise SystemExit(f"no validation checkpoint in {root}")
print(min(matches)[2])
PY
}

arms=($ARM_SPEC)
: > "$STAGE/checkpoints.tsv"
for arm in "${arms[@]}"; do
  for seed in 1 2 3; do
    run_dir=$BASE/models/${arm}_s${seed}
    grep -q '^TRAIN DONE ' "$run_dir/wrapper.log"
    grep -q '^\[epoch 011\]' "$run_dir/train.log"
    best=$(select_best "$run_dir")
    dest=$STAGE/${arm}_s${seed}.ckpt
    cp "$best" "$dest"
    printf '%s\t%s\t%s\t%s\n' "$arm" "$seed" "$best" "$(sha256sum "$dest" | awk '{print $1}')" | tee -a "$STAGE/checkpoints.tsv"
  done
done

ssh "$AMAREL_HOST" "mkdir -p '$AMAREL_CKPT_ROOT'"
rsync -a "$STAGE/" "$AMAREL_HOST:$AMAREL_CKPT_ROOT/"

remote_namo=$(ssh "$AMAREL_HOST" "set -e; cd '$AMAREL_REPO'; git fetch origin feat/horizon-q-redesign >/dev/null; git checkout --detach '$EXPECTED_NAMO_SHA' >/dev/null; git rev-parse HEAD")
remote_sage=$(ssh "$AMAREL_HOST" "set -e; cd '$AMAREL_SAGE'; git fetch origin feat/horizon-q >/dev/null; git checkout --detach '$EXPECTED_SAGE_SHA' >/dev/null; git rev-parse HEAD")
[ "$remote_namo" = "$EXPECTED_NAMO_SHA" ] || { echo "Amarel NAMO $remote_namo != $EXPECTED_NAMO_SHA" >&2; exit 1; }
[ "$remote_sage" = "$EXPECTED_SAGE_SHA" ] || { echo "Amarel Sage $remote_sage != $EXPECTED_SAGE_SHA" >&2; exit 1; }

local_hashes=$(cd "$STAGE" && sha256sum ./*.ckpt | sort)
remote_hashes=$(ssh "$AMAREL_HOST" "cd '$AMAREL_CKPT_ROOT' && sha256sum ./*.ckpt | sort")
[ "$local_hashes" = "$remote_hashes" ] || { echo "checkpoint hash mismatch after transfer" >&2; exit 1; }
echo "HANDOFF transfer verified $(date) namo=$remote_namo sage=$remote_sage"

launch_log=$LOCAL_EVAL_COPY/launch.log
ssh "$AMAREL_HOST" "cd '$AMAREL_REPO' && HY5U_ARCH_ARMS='$ARM_SPEC' CKPT_ROOT='$AMAREL_CKPT_ROOT' OUT_ROOT='$AMAREL_EVAL_ROOT' SAGE_REPO='$AMAREL_SAGE' NAMO_BINDINGS='$AMAREL_BINDINGS' bash scripts/rl_loop/launch_hy5u_arch_eval_amarel.sh" | tee "$launch_log"

mapfile -t agg_jobs < <(awk '/^AGG_JOB / {for (i=1;i<=NF;i++) if ($i ~ /^job=/) {sub(/^job=/, "", $i); print $i}}' "$launch_log")
expected_agg_jobs=$((${#arms[@]} * 3))
[ "${#agg_jobs[@]}" -eq "$expected_agg_jobs" ] || { echo "expected $expected_agg_jobs aggregate jobs, found ${#agg_jobs[@]}" >&2; exit 1; }
job_csv=$(IFS=,; echo "${agg_jobs[*]}")
echo "HANDOFF evaluation queued $(date) aggregate_jobs=$job_csv"

while ssh "$AMAREL_HOST" "squeue -h -j '$job_csv'" | grep -q .; do
  sleep 600
done
states=$(ssh "$AMAREL_HOST" "sacct -X -n -P -j '$job_csv' --format=JobIDRaw,State")
printf '%s\n' "$states" | tee "$LOCAL_EVAL_COPY/aggregate_job_states.tsv"
if printf '%s\n' "$states" | grep -Ev '^[0-9]+\|COMPLETED\|?$' | grep -q .; then
  echo "one or more aggregate jobs did not complete" >&2
  exit 1
fi

for arm in "${arms[@]}"; do
  for seed in 1 2 3; do
    mkdir -p "$LOCAL_EVAL_COPY/full/${arm}_s${seed}"
    rsync -a "$AMAREL_HOST:$AMAREL_EVAL_ROOT/full/${arm}_s${seed}/aggregate.json" "$LOCAL_EVAL_COPY/full/${arm}_s${seed}/aggregate.json"
  done
done
rsync -a "$AMAREL_HOST:$AMAREL_EVAL_ROOT/jobs.tsv" "$LOCAL_EVAL_COPY/jobs.tsv"
echo "HANDOFF DONE $(date) local_aggregates=$LOCAL_EVAL_COPY"
