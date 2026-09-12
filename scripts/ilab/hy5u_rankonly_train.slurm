#!/bin/bash
# Scheduler-owned HY5U rank-only smoke and three-seed fleet on CS iLab. [EXP-2026-09-12]
# One arm only; the loss differs from HY5U by the trainer, not by a lambda, so TRAIN_SCRIPT changes.
#SBATCH --job-name=hy5u_ro
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
ARM=HY5U_rank_only
RESUME_FROM=

if [ "$MODE" = "smoke" ]; then
  [ "$TASK" -eq 0 ] || { echo "smoke array must contain only task 0" >&2; exit 2; }
  SEED=1; EPOCHS=1
  OUT=$R0/rankonly_20260912/smoke/${ARM}_s1
elif [ "$MODE" = "full" ]; then
  [ "$TASK" -lt 3 ] || { echo "full array must be 0-2" >&2; exit 2; }
  SEED=$((TASK + 1)); EPOCHS=12
  OUT=$R0/rankonly_20260912/models/${ARM}_s${SEED}
elif [ "$MODE" = "resume" ]; then
  [ "$TASK" -lt 3 ] || { echo "resume array must be 0-2" >&2; exit 2; }
  SEED=$((TASK + 1)); EPOCHS=12
  OUT=$R0/rankonly_20260912/models/${ARM}_s${SEED}
  RESUME_FROM=$OUT/checkpoints/last.ckpt
else
  echo "unknown MODE=$MODE" >&2; exit 2
fi

[ -f "$H5" ] || { echo "missing H5: $H5" >&2; exit 1; }
if [ -n "$RESUME_FROM" ]; then
  [ -f "$RESUME_FROM" ] || { echo "missing resume checkpoint: $RESUME_FROM" >&2; exit 1; }
  RUN_LOG=$OUT/resume_${SLURM_JOB_ID}.log
  [ ! -e "$RUN_LOG" ] || { echo "refusing to overwrite resume log: $RUN_LOG" >&2; exit 1; }
else
  [ ! -e "$OUT/train.log" ] || { echo "refusing to overwrite: $OUT" >&2; exit 1; }
  RUN_LOG=$OUT/train.log
fi
mkdir -p "$OUT"

echo "RANKONLY mode=$MODE arm=$ARM seed=$SEED task=$TASK job=$SLURM_JOB_ID host=$(hostname) commit=$(git rev-parse HEAD) resume_from=${RESUME_FROM:-none} run_log=$RUN_LOG"
# iLab has NOWHERE node-local to stage this 4.2 GB H5, so read it straight off NFS.
# /dev/shm works until systemd RemoveIPC clears it, which happens when the user's last login
# session on the node ends; an ssh to ilab2 to run squeue was enough to kill job 297444 mid-epoch.
# /tmp is a 98 GB NVMe partition but carries a 2 GiB per-user quota, so the copy dies at 2 GiB
# (job 297445, "Disk quota exceeded", silent fallback to NFS).
#
# WORKERS=0 for the same reason: spawned DataLoader workers rebuild POSIX semaphores out of
# /dev/shm, so a wipe kills them too (297445's SemLock._rebuild FileNotFoundError storm). This is
# the config that completed all three arms of EXP-2026-08-31 on iLab. It costs ~45 min/epoch.
WORKERS=${HY5U_WORKERS:-0}
srun --nodes=1 --ntasks=1 --kill-on-bad-exit=1 env NAMO_REPO="$REPO" NAMO_ENV_SCRIPT=env.ilab.sh \
  H5="$H5" OUT="$OUT" TRAIN_SCRIPT=scripts/rl_loop/train_q2_rankonly.py \
  RESUME_FROM="$RESUME_FROM" TRAIN_LOG="$RUN_LOG" \
  EPOCHS="$EPOCHS" BATCH=256 WORKERS="$WORKERS" SEED="$SEED" POSTCHECK_LIMIT=64 \
  NAMO_GAMMA=0.5 NAMO_UNREACH_W=0.1 NAMO_GROUP_EPISODES=1 \
  EGMM_LAMBDA=0.1 RANK_LAMBDA=0.1 LOWER_RANK_LAMBDA=0.05 \
  NAMO_EDGE_SELF_ATTN=1 NAMO_STAGE_SHM="${HY5U_STAGE_SHM:-0}" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  bash scripts/slurm/train.slurm
grep -q "eval_scorer-load check" "$RUN_LOG" || { echo "missing scorer-load marker after srun" >&2; exit 1; }
compgen -G "$OUT/checkpoints/*.ckpt" >/dev/null || { echo "missing checkpoint after srun" >&2; exit 1; }
