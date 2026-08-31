#!/bin/bash
# Wait for the three target-box smokes, then train the matched HY5U ablation fleet on direct CS GPUs.
set -euo pipefail

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$REPO"
source env.ilab.sh >/dev/null 2>&1

R0=${HY5U_R0:-$NAMO_SCRATCH/aquaman/round0}
H5=${HY5U_H5:-$R0/hybrid_train_v1.h5}
SMOKE_BASE=${HY5U_SMOKE_BASE:-$R0/ablations_20260831/smoke}
FULL_BASE=${HY5U_FULL_BASE:-$R0/ablations_20260831/models}
SMOKE_TIMEOUT_MIN=${SMOKE_TIMEOUT_MIN:-120}
WORKERS=${WORKERS:-6}
GPU_LIST=(${GPU_LIST:-1 2 3 4 5})

arms=(HY5U_no_family HY5U_regression HY5U_independent)

config_for() {
  case "$1" in
    HY5U_no_family)  CFG_EGMM=0;   CFG_RANK=0.1; CFG_LOWER=0.05; CFG_EDGE=1 ;;
    HY5U_regression) CFG_EGMM=0;   CFG_RANK=0;   CFG_LOWER=0;    CFG_EDGE=1 ;;
    HY5U_independent) CFG_EGMM=0.1; CFG_RANK=0.1; CFG_LOWER=0.05; CFG_EDGE=0 ;;
    *) echo "unknown arm: $1" >&2; return 1 ;;
  esac
}

smokes_ready() {
  local arm log
  for arm in "${arms[@]}"; do
    log=$SMOKE_BASE/${arm}_s1/wrapper.log
    if grep -q "train exited before completion marker\|Traceback (most recent call last)" "$log" 2>/dev/null; then
      echo "SMOKE FAILED arm=$arm log=$log" >&2
      return 2
    fi
    grep -q "TRAIN DONE" "$log" 2>/dev/null || return 1
  done
}

echo "SUPERVISOR start $(date) host=$(hostname) commit=$(git rev-parse --short HEAD)"
for ((minute=0; minute<SMOKE_TIMEOUT_MIN; minute++)); do
  if smokes_ready; then
    echo "SMOKES PASSED $(date) elapsed_minutes=$minute"
    break
  else
    status=$?
    if [ "$status" -eq 2 ]; then exit 1; fi
  fi
  if (( minute % 5 == 0 )); then
    for arm in "${arms[@]}"; do
      log=$SMOKE_BASE/${arm}_s1/train.log
      echo "smoke minute=$minute arm=$arm $(grep -E '\[epoch|eval_scorer-load check' "$log" 2>/dev/null | tail -1)"
    done
  fi
  sleep 60
done
smokes_ready || { echo "SMOKE TIMEOUT after ${SMOKE_TIMEOUT_MIN} minutes" >&2; exit 1; }

run_one() {
  local arm=$1 seed=$2 gpu=$3 out=$FULL_BASE/${arm}_s${seed}
  config_for "$arm"
  if [ -e "$out/train.log" ] || [ -d "$out/checkpoints" ]; then
    echo "refusing to overwrite existing run: $out" >&2
    return 1
  fi
  mkdir -p "$out"
  echo "FULL launch $(date) arm=$arm seed=$seed gpu=$gpu out=$out"
  env CUDA_VISIBLE_DEVICES="$gpu" SLURM_JOB_ID="manual_${arm}_s${seed}" \
    NAMO_REPO="$REPO" H5="$H5" OUT="$out" \
    TRAIN_SCRIPT=scripts/rl_loop/train_q2_round2.py \
    EPOCHS=12 BATCH=256 WORKERS="$WORKERS" SEED="$seed" POSTCHECK_LIMIT=64 \
    NAMO_GAMMA=0.5 NAMO_UNREACH_W=0.1 NAMO_GROUP_EPISODES=1 \
    EGMM_LAMBDA="$CFG_EGMM" RANK_LAMBDA="$CFG_RANK" LOWER_RANK_LAMBDA="$CFG_LOWER" \
    NAMO_EDGE_SELF_ATTN="$CFG_EDGE" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    bash "$REPO/scripts/slurm/train.slurm" > "$out/wrapper.log" 2>&1
  grep -q "TRAIN DONE" "$out/wrapper.log"
  echo "FULL passed $(date) arm=$arm seed=$seed gpu=$gpu"
}

jobs=(
  "HY5U_no_family 1"
  "HY5U_regression 1"
  "HY5U_independent 1"
  "HY5U_no_family 2"
  "HY5U_regression 2"
  "HY5U_independent 2"
  "HY5U_no_family 3"
  "HY5U_regression 3"
  "HY5U_independent 3"
)

offset=0
while [ "$offset" -lt "${#jobs[@]}" ]; do
  pids=()
  labels=()
  for slot in "${!GPU_LIST[@]}"; do
    idx=$((offset + slot))
    [ "$idx" -lt "${#jobs[@]}" ] || break
    read -r arm seed <<< "${jobs[$idx]}"
    run_one "$arm" "$seed" "${GPU_LIST[$slot]}" &
    pids+=("$!")
    labels+=("${arm}_s${seed}")
  done
  failed=0
  for idx in "${!pids[@]}"; do
    if ! wait "${pids[$idx]}"; then
      echo "FULL FAILED ${labels[$idx]}" >&2
      failed=1
    fi
  done
  [ "$failed" -eq 0 ] || exit 1
  offset=$((offset + ${#pids[@]}))
done

echo "FLEET DONE $(date)"
