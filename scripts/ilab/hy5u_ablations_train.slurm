#!/bin/bash
# Scheduler-owned HY5U ablation smoke, full three-seed fleet, and exact-state continuation on CS iLab.
#SBATCH --job-name=hy5u_ab
#SBATCH --partition=unlimited
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%A_%a.out
set -euo pipefail

REPO=${NAMO_REPO:-/common/home/dm1487/robotics_research/ktamp/namo}
cd "$REPO"
source env.ilab.sh >/dev/null 2>&1

MODE=${MODE:-smoke}
R0=${HY5U_R0:-$NAMO_SCRATCH/aquaman/round0}
H5=${HY5U_H5:-$R0/hybrid_train_v1.h5}
TASK=${SLURM_ARRAY_TASK_ID:-0}

arms=(HY5U_no_family HY5U_regression HY5U_independent)
RESUME_FROM=
if [ "$MODE" = "smoke" ]; then
  [ "$TASK" -lt 3 ] || { echo "smoke array must be 0-2" >&2; exit 2; }
  ARM=${arms[$TASK]}
  SEED=1
  EPOCHS=1
  OUT=$R0/ablations_20260831/smoke_ilab_v6/${ARM}_s1
elif [ "$MODE" = "full" ]; then
  [ "$TASK" -lt 9 ] || { echo "full array must be 0-8" >&2; exit 2; }
  ARM=${arms[$((TASK / 3))]}
  SEED=$((TASK % 3 + 1))
  EPOCHS=12
  OUT=$R0/ablations_20260831/models/${ARM}_s${SEED}
elif [ "$MODE" = "resume_smoke" ]; then
  [ "$TASK" -eq 0 ] || { echo "resume_smoke array must contain only task 0" >&2; exit 2; }
  ARM=HY5U_no_family
  SEED=1
  EPOCHS=12
  OUT=$R0/ablations_20260831/resume_smoke_v1/${ARM}_s${SEED}
  RESUME_FROM=$R0/ablations_20260831/models/${ARM}_s${SEED}/checkpoints/last.ckpt
elif [ "$MODE" = "resume" ]; then
  [ "$TASK" -lt 9 ] || { echo "resume array must be 0-8" >&2; exit 2; }
  ARM=${arms[$((TASK / 3))]}
  SEED=$((TASK % 3 + 1))
  EPOCHS=12
  OUT=$R0/ablations_20260831/models/${ARM}_s${SEED}
  RESUME_FROM=$OUT/checkpoints/last.ckpt
else
  echo "unknown MODE=$MODE" >&2
  exit 2
fi

case "$ARM" in
  HY5U_no_family)   EGMM=0;   RANK=0.1; LOWER=0.05; EDGE=1 ;;
  HY5U_regression)  EGMM=0;   RANK=0;   LOWER=0;    EDGE=1 ;;
  HY5U_independent) EGMM=0.1; RANK=0.1; LOWER=0.05; EDGE=0 ;;
esac

[ -f "$H5" ] || { echo "missing H5: $H5" >&2; exit 1; }
if [ -n "$RESUME_FROM" ]; then
  [ -f "$RESUME_FROM" ] || { echo "missing resume checkpoint: $RESUME_FROM" >&2; exit 1; }
else
  [ ! -e "$OUT/train.log" ] || { echo "refusing to overwrite: $OUT" >&2; exit 1; }
fi
mkdir -p "$OUT"
if [ "$MODE" = "resume" ]; then
  RUN_LOG=$OUT/resume_${SLURM_JOB_ID}.log
  [ ! -e "$RUN_LOG" ] || { echo "refusing to overwrite resume log: $RUN_LOG" >&2; exit 1; }
else
  RUN_LOG=$OUT/train.log
fi

echo "ABLATION mode=$MODE arm=$ARM seed=$SEED task=$TASK job=$SLURM_JOB_ID host=$(hostname) commit=$(git rev-parse HEAD) resume_from=${RESUME_FROM:-none} run_log=$RUN_LOG"
STAGE_DIR=${HY5U_STAGE_DIR:-/tmp}
WORKERS=${HY5U_WORKERS:-6}
srun --nodes=1 --ntasks=1 --kill-on-bad-exit=1 env NAMO_REPO="$REPO" NAMO_ENV_SCRIPT=env.ilab.sh \
  H5="$H5" OUT="$OUT" TRAIN_SCRIPT=scripts/rl_loop/train_q2_round2.py \
  RESUME_FROM="$RESUME_FROM" TRAIN_LOG="$RUN_LOG" \
  EPOCHS="$EPOCHS" BATCH=256 WORKERS="$WORKERS" SEED="$SEED" POSTCHECK_LIMIT=64 \
  NAMO_GAMMA=0.5 NAMO_UNREACH_W=0.1 NAMO_GROUP_EPISODES=1 \
  EGMM_LAMBDA="$EGMM" RANK_LAMBDA="$RANK" LOWER_RANK_LAMBDA="$LOWER" \
  NAMO_EDGE_SELF_ATTN="$EDGE" NAMO_STAGE_DIR="$STAGE_DIR" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  bash scripts/slurm/train.slurm
grep -q "eval_scorer-load check" "$RUN_LOG" || { echo "missing scorer-load marker after srun" >&2; exit 1; }
compgen -G "$OUT/checkpoints/*.ckpt" >/dev/null || { echo "missing checkpoint after srun" >&2; exit 1; }
