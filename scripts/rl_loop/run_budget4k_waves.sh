#!/bin/bash
# Deep-budget (4000-sim) 2push campaign: HY5U x3 seeds + uniform random x3 seeds, submitted in
# waves under Amarel's 500-job `main` QOS cap. [EXP-2026-08-09 § deep budget, USER 2026-08-13]
#
# Both arms run through the SAME script and settings (key, raw scale, agg/combine/discount), so
# this is a like-for-like comparison rather than model-vs-stale-baseline.
#
# Usage (from the CS side):
#   AMAREL_CKPTS=<remote ckpts> AMAREL_OUT=<remote out> bash scripts/rl_loop/run_budget4k_waves.sh
set -uo pipefail
AM=${AMAREL_HOST:-amarel}
REPO=${AMAREL_REPO:-/home/dm1487/projects/namo/namo_cpp}
SAGE=${AMAREL_SAGE:-/home/dm1487/projects/namo/sage_learning}
CK=${AMAREL_CKPTS:?set AMAREL_CKPTS=<remote ckpt dir>}
OUT=${AMAREL_OUT:?set AMAREL_OUT=<remote eval output root>}
BUDGET=${BUDGET:-4000}
N2SH=${N2SH:-250}                 # 1012 episodes / 250 = ~4 per task
WAVE_PAIRS=${WAVE_PAIRS:-2}       # 2 arms x 250 tasks = 500 = the cap

# arm spec: name|prior|ckpt|seed_base
ARMS=(
  "HY5U_s1|model|$CK/HY5U_s1.ckpt|7000"
  "HY5U_s2|model|$CK/HY5U_s2.ckpt|7000"
  "HY5U_s3|model|$CK/HY5U_s3.ckpt|7000"
  "rand_s7000|uniform|$CK/HY5U_s1.ckpt|7000"
  "rand_s8000|uniform|$CK/HY5U_s1.ckpt|8000"
  "rand_s9000|uniform|$CK/HY5U_s1.ckpt|9000"
)

submit() {   # $1=name $2=prior $3=ckpt $4=seed
  ssh "$AM" "cd $REPO && CKPT=$3 OUT=$OUT/$1 SIM_BUDGET=$BUDGET N2SH=$N2SH PRIOR=$2 \
    SEED_BASE=$4 NAMO_REPO=$REPO SAGE_REPO=$SAGE \
    sbatch --array=0-$((N2SH-1)) --time=6:00:00 --job-name=b4k_$1 \
    scripts/slurm/eval_budget_2push.slurm" 2>&1 | grep -oE '[0-9]+$'
}

drain() {    # block until this user's queue is (nearly) empty
  # ⛔ MUST NOT use `i` (or any caller variable): shell functions share global scope, and an
  # earlier version's `for i in ...` here clobbered the outer wave index, so waves 2-3 silently
  # never ran and the driver printed ALL WAVES DONE after wave 1.
  local _t
  for _t in $(seq 1 720); do
    q=$(ssh "$AM" 'squeue -u $USER -h -r | wc -l' 2>/dev/null)
    [ -n "$q" ] && [ "$q" -le 2 ] && return 0
    sleep 60
  done
}

i=0
while [ $i -lt ${#ARMS[@]} ]; do
  echo "=== wave starting at arm index $i  $(date)"
  for j in $(seq 0 $((WAVE_PAIRS-1))); do
    k=$((i+j)); [ $k -ge ${#ARMS[@]} ] && break
    IFS='|' read -r name prior ckpt seed <<< "${ARMS[$k]}"
    id=$(submit "$name" "$prior" "$ckpt" "$seed")
    echo "  submitted $name (prior=$prior seed=$seed) job=$id"
  done
  drain
  echo "=== wave drained $(date)"
  for j in $(seq 0 $((WAVE_PAIRS-1))); do
    k=$((i+j)); [ $k -ge ${#ARMS[@]} ] && break
    IFS='|' read -r name _ _ _ <<< "${ARMS[$k]}"
    n=$(ssh "$AM" "ls $OUT/$name/2push/shard_*.json 2>/dev/null | wc -l")
    echo "  $name: $n/$N2SH shards"
  done
  i=$((i+WAVE_PAIRS))
done
echo "ALL WAVES DONE $(date)"
