#!/bin/bash
# Arjuna-0 v3 stage 2 -- STREAMING + RACED. Each model's eval is submitted the instant ITS OWN
# train exits, instead of waiting for the slowest of six.
#
# Why raced: the seed-1 pair landed on ilab1 (a4500, 6.7 min/epoch) while seed 2 got rlab2
# (a100, 1.5 min/epoch) -- 4.5x. Killing the ilab1 pair to move them would forfeit 45 min of
# real progress on the bet that a faster GPU is free, and rlab2's GPUs are not visible from its
# login shell. So instead duplicate seed-1 runs were launched on rlab2/rlab7 and whichever
# COMPLETES FIRST supplies the checkpoint; the losers are cancelled to free the shared cluster.
# Same H5, same seed, same epochs -- the duplicates are the same run, not a new condition.
#
# No wave split: Amarel QOS `main` allows 500 submitted jobs per user, 6 x 72 = 432 fits whole.
# Gates on SLURM JOB STATE, never on checkpoint existence -- Lightning writes an epoch000 file
# within minutes of launch, so "a ckpt exists" is a liveness signal, not a doneness signal.
set -u
# Amarel paths come from env, never baked in (portability guard):
#   AMAREL_ROOT  remote home holding aquaman0/   AMAREL_REPO  remote namo checkout
#   AMAREL_SAGE  remote sage checkout (must match the ckpt)
AMAREL_ROOT=${AMAREL_ROOT:?set AMAREL_ROOT=<remote home holding aquaman0/>}
AMAREL_REPO=${AMAREL_REPO:?set AMAREL_REPO=<remote namo checkout>}
AMAREL_SAGE=${AMAREL_SAGE:?set AMAREL_SAGE=<remote sage checkout>}
R0=/common/users/dm1487/scratch_namo/aquaman/round0
CK=${AMAREL_ROOT}/aquaman0/ckpts_bfix
OUT=${AMAREL_ROOT}/aquaman0/eval_v5
SUB=ilab2.cs.rutgers.edu

# model -> "jobid:ckptdir jobid:ckptdir ..."   (first COMPLETED wins)
declare -A CAND=(
  [AJ5_s1]="204312:AJ5_s1"
  [AJ5NR_s1]="204313:AJ5NR_s1"
  [AJ5_s2]="204314:AJ5_s2"
  [AJ5NR_s2]="204315:AJ5NR_s2"
  [AJ5_s3]="204316:AJ5_s3"
  [AJ5NR_s3]="204317:AJ5NR_s3"
)
ALL="AJ5_s1 AJ5_s2 AJ5_s3 AJ5NR_s1 AJ5NR_s2 AJ5NR_s3"
IDS=$(for m in $ALL; do for c in ${CAND[$m]}; do echo -n "${c%%:*},"; done; done | sed 's/,$//')
declare -A SENT=() DEAD=()
# RESTART SAFETY. SENT lives only in memory, so every restart of this script used to re-ship every
# already-finished model -- on 2026-08-08 that submitted AJ5_s1 twice, two arrays writing the same
# 72 filenames. Recover the real state from Amarel instead: a model with an output dir has already
# been submitted. Cheap, and makes the script idempotent across restarts.
for m in $(timeout 60 ssh amarel "ls $OUT 2>/dev/null"); do
  SENT[$m]=1
  echo "[resume] $m already submitted (output dir exists) -- not resubmitting"
done

submit_one() {                               # $1=model  $2=ckpt dir  $3=winning jobid
  local m=$1 dir=$2 win=$3 f
  f=$(ls "$R0/models/$dir/checkpoints/"epoch*.ckpt 2>/dev/null | head -1)
  [ -n "$f" ] || { echo "!! $m COMPLETED ($dir) but no ckpt -- skipping"; DEAD[$m]=1; return; }
  echo "[$(date +%H:%M:%S)] $m won by job $win ($dir) -> $(basename "$f")"
  for c in ${CAND[$m]}; do                   # free the shared cluster: cancel the losers
    [ "${c%%:*}" != "$win" ] && ssh "$SUB" "scancel ${c%%:*} 2>/dev/null" &
  done
  wait
  # Amarel QOS `main` caps SUBMITTED jobs at 500 per user. v3's 432 tasks are already in flight,
  # so wait for headroom before adding another 72 rather than getting a submit rejection.
  for i in $(seq 1 120); do
    q=$(ssh amarel "squeue -u dm1487 -h -t R,PD -r | wc -l" 2>/dev/null)
    [ -n "$q" ] && [ "$q" -lt 340 ] && break
    echo "  [throttle] amarel queue=$q, waiting for headroom"; sleep 60
  done
  rsync -q "$f" "amarel:$CK/$m.ckpt" || { echo "!! rsync failed for $m"; DEAD[$m]=1; return; }
  # NAMO_REPO and NAMO_BINDINGS are BOTH REQUIRED when submitting over ssh, and omitting either
  # fails SILENTLY -- 72/72 tasks dead in 1 s with 0-byte logs (2026-08-08, cost ~30 min):
  #   * NAMO_REPO -- the template derives REPO from ${BASH_SOURCE[0]}, but SLURM SPOOLS the batch
  #     script to /var/lib/slurm/slurmd/job<N>/slurm_script, so that resolves to /var/lib/slurm.
  #     It then cds there, `source env.amarel.sh` fails, and set -e kills the job with the error
  #     discarded by the >/dev/null 2>&1 on the source line.
  #   * NAMO_BINDINGS -- namo_bfix has no build_python of its own, and BIND is computed BEFORE
  #     env.amarel.sh is sourced, so nothing fills it in.
  # Earlier waves only worked because an interactive shell had both exported and sbatch
  # --export=ALL carried them in. Verified working before this rewrite (Amarel job 60298828).
  ssh amarel "cd ${AMAREL_REPO}; \
    NAMO_REPO=${AMAREL_REPO} \
    NAMO_BINDINGS=${AMAREL_REPO}/build_python \
    SAGE_REPO=${AMAREL_SAGE} \
    N1SH=80 N2SH=64 CKPT=$CK/$m.ckpt OUT=$OUT/$m sbatch --array=0-143 --time=03:00:00 --job-name=aq_$m \
    scripts/slurm/aquaman_eval_amarel.slurm" 2>&1 | tail -1
  SENT[$m]=1
}

for i in $(seq 1 400); do
  states=$(ssh "$SUB" "sacct -X -n -j $IDS --format=JobID,State" 2>/dev/null)
  for m in $ALL; do
    [ -n "${SENT[$m]:-}${DEAD[$m]:-}" ] && continue
    live=0
    for c in ${CAND[$m]}; do
      jid=${c%%:*}; dir=${c##*:}
      s=$(echo "$states" | grep -E "^\s*${jid}\s" | grep -oE 'COMPLETED|RUNNING|PENDING|FAILED|TIMEOUT|CANCELLED|OUT_OF_ME' | head -1)
      [ "$s" = COMPLETED ] && { submit_one "$m" "$dir" "$jid"; break; }
      case "$s" in RUNNING|PENDING|"") live=1 ;; esac
    done
    [ -z "${SENT[$m]:-}" ] && [ "$live" -eq 0 ] && { echo "!! $m: every candidate ended non-COMPLETED"; DEAD[$m]=1; }
  done
  [ $(( ${#SENT[@]} + ${#DEAD[@]} )) -eq 6 ] && break
  sleep 30
done
echo "all trains resolved: submitted=${#SENT[@]} dead=${#DEAD[@]} $(date)"
[ ${#SENT[@]} -eq 6 ] || echo "WARNING: only ${#SENT[@]}/6 models submitted (dead: ${!DEAD[*]:-none})"

ssh amarel "for i in \$(seq 1 120); do q=\$(squeue -u dm1487 -h -t R,PD -r -n aq_AJ5_s1,aq_AJ5_s2,aq_AJ5_s3,aq_AJ5NR_s1,aq_AJ5NR_s2,aq_AJ5NR_s3 | wc -l); \
  [ \"\$q\" -eq 0 ] && break; sleep 45; done; echo drained \$(date)"

ssh amarel "n=\$(ls $OUT/{AJ5_s1,AJ5_s2,AJ5_s3,AJ5NR_s1,AJ5NR_s2,AJ5NR_s3}/{1push_hmax2,2push}/*.json 2>/dev/null | wc -l); echo \"eval shards=\$n\""
rsync -a --include='*/' --include='shard_*.jsonl' --include='shard_*.json' --exclude='*' \
  "amarel:$OUT/" "$R0/eval_v5/"
echo "ARJUNA V5 EVAL PULLED $(date)"
