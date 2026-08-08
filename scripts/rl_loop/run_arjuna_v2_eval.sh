#!/bin/bash
# Arjuna-0 v2 stage 2: wait for the trains, ship ckpts to Amarel, canonical eval, pull back.
# 6 models x 72 shards = 432 tasks -- at the submit cap, so it goes in two waves of 3.
#
# THE WAIT GATES ON SLURM JOB STATE, NOT ON CHECKPOINT EXISTENCE. Lightning writes an epoch000
# file within minutes of launch, so "6 ckpts exist" is true long before training is done; an
# earlier version of this script also ANDed a `squeue` check, but it runs on a box with no
# squeue, so that test was vacuously true and the pair would have shipped barely-trained models
# and produced a clean-looking, entirely false result table. Poll the scheduler, not the FS.
#
# Usage:
#   AMAREL_ROOT=<remote home>  AMAREL_REPO=<remote namo checkout>  AMAREL_SAGE=<remote sage> \
#   JOBS=203552,...  bash scripts/rl_loop/run_arjuna_v2_eval.sh
set -u
: "${AMAREL_ROOT:?set AMAREL_ROOT=<remote home holding aquaman0/> }"
: "${AMAREL_REPO:?set AMAREL_REPO=<remote namo checkout>}"
: "${AMAREL_SAGE:?set AMAREL_SAGE=<remote sage checkout -- must have action_motion_dim>}"
: "${JOBS:?set JOBS=<comma-separated SLURM job ids of the 6 trains>}"
R0="${NAMO_SCRATCH:?source env.<box>.sh first}/aquaman/round0"
CK=$AMAREL_ROOT/aquaman0/ckpts_bfix
OUT=$AMAREL_ROOT/aquaman0/eval_bfix
MODELS="${MODELS:-AJ2_s1 AJ2_s2 AJ2_s3 AJ2NR_s1 AJ2NR_s2 AJ2NR_s3}"
SUB=${CS_SUBMIT_HOST:-ilab2.cs.rutgers.edu}   # ilab1 stalls mid-key-exchange under load

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
[ "$done_n" -eq 6 ] || { echo "ABORT: $done_n/6 COMPLETED -- not evaluating a partial sweep"; exit 1; }

for m in $MODELS; do
  f=$(ls "$R0/models/$m/checkpoints/"epoch*.ckpt | head -1)
  echo "  $m <- $(basename "$f")"
  rsync -q "$f" "amarel:$CK/$m.ckpt" && echo "synced $m"
done

# wave 1: ranking-ON trio, wave 2: ranking-OFF trio (submit cap ~500 array tasks)
for wave in "AJ2_s1 AJ2_s2 AJ2_s3" "AJ2NR_s1 AJ2NR_s2 AJ2NR_s3"; do
  ssh amarel "cd $AMAREL_REPO; for m in $wave; do \
    SAGE_REPO=$AMAREL_SAGE \
    CKPT=$CK/\$m.ckpt OUT=$OUT/\$m sbatch --array=0-71 --time=03:00:00 --job-name=aq_\$m \
    scripts/slurm/aquaman_eval_amarel.slurm; done" 2>&1 | tail -3
  ssh amarel "for i in \$(seq 1 90); do q=\$(squeue -u \$USER -h -t R,PD -r | wc -l); \
    [ \"\$q\" -le 1 ] && break; sleep 60; done; echo wave_drained \$(date)"
done

ssh amarel "n=\$(ls $OUT/{AJ2_s1,AJ2_s2,AJ2_s3,AJ2NR_s1,AJ2NR_s2,AJ2NR_s3}/{1push_hmax2,2push}/*.json 2>/dev/null | wc -l); echo \"eval shards=\$n/432\""
rsync -a --include='*/' --include='shard_*.jsonl' --include='shard_*.json' --exclude='*' \
  "amarel:$OUT/" "$R0/eval_bfix/"
echo "ARJUNA V2 EVAL PULLED $(date)"
