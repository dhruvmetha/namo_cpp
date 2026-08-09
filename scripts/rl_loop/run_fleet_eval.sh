#!/bin/bash
# Generalized fleet canonical eval: wait for CS trains -> ship ckpts to Amarel -> ONE wave of
# 72-shard arrays for ALL models (submit cap ~500 tasks; 6x72=432 fits) -> poll -> pull back.
# Derived from run_arjuna_v2_eval.sh; differences: MODELS fully parameterized, single wave
# (two-wave split was conservatism; full QOS authorized [USER 2026-08-09]), OUT dir per fleet.
#
# THE WAIT GATES ON SLURM JOB STATE, NOT CHECKPOINT EXISTENCE (see run_arjuna_v2_eval.sh header).
#
# Usage:
#   AMAREL_ROOT=<remote home holding aquaman0/> AMAREL_REPO=<remote namo checkout> \
#   AMAREL_SAGE=<remote sage checkout> \
#   JOBS=205036,205037,205038,205039,205040,205041 \
#   MODELS="XB_s1 XB_s2 XB_s3 RP_s1 RP_s2 RP_s3" \
#   bash scripts/rl_loop/run_fleet_eval.sh
set -u
: "${AMAREL_ROOT:?}"; : "${AMAREL_REPO:?}"; : "${AMAREL_SAGE:?}"; : "${JOBS:?}"; : "${MODELS:?}"
R0="${NAMO_SCRATCH:?source env.<box>.sh first}/aquaman/round0"
CK=$AMAREL_ROOT/aquaman0/ckpts_bfix
OUT=$AMAREL_ROOT/aquaman0/eval_bfix
SUB=${CS_SUBMIT_HOST:-ilab2.cs.rutgers.edu}   # ilab1 stalls mid-key-exchange under load
NMOD=$(echo $MODELS | wc -w)

for i in $(seq 1 240); do
  st=$(ssh "$SUB" "sacct -X -n -j $JOBS --format=State" 2>/dev/null \
       | grep -oE 'COMPLETED|RUNNING|PENDING|FAILED|TIMEOUT|CANCELLED|OUT_OF_ME')
  live=$(echo "$st" | grep -cE 'RUNNING|PENDING')
  done_n=$(echo "$st" | grep -c 'COMPLETED')
  [ "$live" -eq 0 ] && [ -n "$st" ] && break
  sleep 60
done
echo "train wait over: COMPLETED=$done_n live=$live $(date)"
echo "$st" | sort | uniq -c
[ "$done_n" -eq "$NMOD" ] || { echo "ABORT: $done_n/$NMOD COMPLETED -- not evaluating a partial sweep"; exit 1; }

for m in $MODELS; do
  f=$(ls "$R0/models/$m/checkpoints/"epoch*.ckpt | head -1)
  echo "  $m <- $(basename "$f")"
  rsync -q "$f" "amarel:$CK/$m.ckpt" && echo "synced $m"
done

# ONE wave: every model's 72-shard array submitted together (432 tasks < ~500 cap)
ssh amarel "cd $AMAREL_REPO; for m in $MODELS; do \
  SAGE_REPO=$AMAREL_SAGE \
  CKPT=$CK/\$m.ckpt OUT=$OUT/\$m sbatch --array=0-71 --time=03:00:00 --job-name=fl_\$m \
  scripts/slurm/aquaman_eval_amarel.slurm; done" 2>&1 | tail -"$NMOD"
ssh amarel "for i in \$(seq 1 120); do q=\$(squeue -u \$USER -h -t R,PD -r | wc -l); \
  [ \"\$q\" -le 1 ] && break; sleep 60; done; echo queue_drained \$(date)"

GLOB=$(echo $MODELS | tr ' ' ',')
ssh amarel "n=\$(ls $OUT/{$GLOB}/{1push_hmax2,2push}/*.json 2>/dev/null | wc -l); echo \"eval shards=\$n/$((NMOD*72))\""
rsync -a --include='*/' --include='shard_*.jsonl' --include='shard_*.json' --exclude='*' \
  "amarel:$OUT/" "$R0/eval_bfix/"
echo "FLEET EVAL PULLED $(date)"
