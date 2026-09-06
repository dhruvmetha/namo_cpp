#!/bin/bash
# Smoke and train selected current-HY5U architecture controls on direct CS GPUs.
set -euo pipefail

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$REPO"
source env.ilab.sh >/dev/null 2>&1

R0=${HY5U_R0:-$NAMO_SCRATCH/aquaman/round0}
H5=${HY5U_H5:-$R0/hybrid_train_v1.h5}
BASE=${HY5U_ARCH_BASE:-$R0/architecture_ablations_20260904}
WORKERS=${WORKERS:-6}
GPU_LIST=(${GPU_LIST:-1 2})
arms=(${HY5U_ARCH_ARMS:-HY5U_global HY5U_no_local})
PARALLEL_SEEDS=${PARALLEL_SEEDS:-0}

config_for() {
  case "$1" in
    HY5U_global)   CFG_GLOBAL=1; CFG_LOCAL=1; CFG_EDGE_EMBED=1 ;;
    HY5U_no_local) CFG_GLOBAL=0; CFG_LOCAL=0; CFG_EDGE_EMBED=1 ;;
    HY5U_no_edge)  CFG_GLOBAL=0; CFG_LOCAL=1; CFG_EDGE_EMBED=0 ;;
    *) echo "unknown arm: $1" >&2; return 1 ;;
  esac
}

run_one() {
  local phase=$1
  local arm=$2
  local seed=$3
  local gpu=$4
  local epochs=$5
  local out=$BASE/$phase/${arm}_s${seed}
  config_for "$arm"
  if [ -e "$out/train.log" ] || [ -d "$out/checkpoints" ]; then
    echo "refusing to overwrite existing run: $out" >&2
    return 1
  fi
  mkdir -p "$out"
  echo "$phase launch $(date) arm=$arm seed=$seed gpu=$gpu epochs=$epochs out=$out"
  env CUDA_VISIBLE_DEVICES="$gpu" SLURM_JOB_ID="manual_arch_${phase}_${arm}_s${seed}" \
    NAMO_REPO="$REPO" H5="$H5" OUT="$out" \
    TRAIN_SCRIPT=scripts/rl_loop/train_q2_round2.py \
    EPOCHS="$epochs" BATCH=256 WORKERS="$WORKERS" SEED="$seed" POSTCHECK_LIMIT=64 \
    NAMO_GAMMA=0.5 NAMO_UNREACH_W=0.1 NAMO_GROUP_EPISODES=1 \
    EGMM_LAMBDA=0.1 RANK_LAMBDA=0.1 LOWER_RANK_LAMBDA=0.05 \
    NAMO_EDGE_SELF_ATTN=1 NAMO_GLOBAL_READOUT="$CFG_GLOBAL" NAMO_USE_LOCAL="$CFG_LOCAL" \
    NAMO_USE_EDGE_EMBED="$CFG_EDGE_EMBED" \
    NAMO_ACTION_MOTION=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    bash "$REPO/scripts/slurm/train.slurm" > "$out/wrapper.log" 2>&1
  grep -q "TRAIN DONE" "$out/wrapper.log"
  echo "$phase passed $(date) arm=$arm seed=$seed gpu=$gpu"
}

run_wave() {
  local phase=$1 epochs=$2 seed=$3
  local pids=() labels=() idx arm
  for idx in "${!arms[@]}"; do
    arm=${arms[$idx]}
    run_one "$phase" "$arm" "$seed" "${GPU_LIST[$idx]}" "$epochs" &
    pids+=("$!")
    labels+=("${arm}_s${seed}")
  done
  local failed=0
  for idx in "${!pids[@]}"; do
    if ! wait "${pids[$idx]}"; then
      echo "$phase FAILED ${labels[$idx]}" >&2
      failed=1
    fi
  done
  [ "$failed" -eq 0 ] || exit 1
}

[ "${#GPU_LIST[@]}" -ge "${#arms[@]}" ] || { echo "GPU_LIST must contain at least one GPU index per arm" >&2; exit 2; }
[ -f "$H5" ] || { echo "missing H5: $H5" >&2; exit 1; }
mkdir -p "$BASE"

echo "SUPERVISOR start $(date) host=$(hostname) namo=$(git rev-parse HEAD) sage=$(git -C "$SAGE_REPO" rev-parse HEAD)"
smoke_start=$(date +%s)
run_wave smoke 1 1
smoke_seconds=$(($(date +%s) - smoke_start))
echo "SMOKES PASSED $(date) elapsed_seconds=$smoke_seconds"

if [ "$PARALLEL_SEEDS" = 1 ]; then
  n_runs=$((${#arms[@]} * 3))
  [ "${#GPU_LIST[@]}" -eq "$n_runs" ] || { echo "PARALLEL_SEEDS=1 requires one GPU index per arm-seed run" >&2; exit 2; }
  pids=(); labels=(); gpu_i=0
  for seed in 1 2 3; do
    for arm in "${arms[@]}"; do
      run_one models "$arm" "$seed" "${GPU_LIST[$gpu_i]}" 12 &
      pids+=("$!"); labels+=("${arm}_s${seed}"); gpu_i=$((gpu_i + 1))
    done
  done
  failed=0
  for idx in "${!pids[@]}"; do
    if ! wait "${pids[$idx]}"; then
      echo "models FAILED ${labels[$idx]}" >&2
      failed=1
    fi
  done
  [ "$failed" -eq 0 ] || exit 1
else
  for seed in 1 2 3; do
    run_wave models 12 "$seed"
  done
fi

echo "FLEET DONE $(date)"
