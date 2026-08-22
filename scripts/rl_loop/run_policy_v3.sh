#!/bin/bash
# POLICY MODE (zero search) campaign on the fixed-physics v3 population: HY5U x3 seeds + uniform
# random x3 seeds, one-push and two-push legs, K=10. [EXP-2026-08-22-policy-mode-hy5u]
#
# K=10 answers hmax=5 AND hmax=10 from ONE rollout per episode: each leaf records opened_at, so
# cumulative open@1..open@10 all come from the same trajectory. Do not launch a K=5 job.
#
# Both arms run through the same script and settings, so this is like-for-like rather than
# model-vs-stale-baseline. Uniform makes no model call; it still loads a ckpt to build the planner.
#
# Sizing (measured 2026-08-22, Amarel main, one middle shard per leg): ~60 s fixed startup per task
# plus ~1.6 s per 1push episode and ~2.8 s per 2push episode, worst-case episode ~3.5 s. At 16 shards
# a leg that is ~4 min/task, ~12 min worst case -- hence --time=02:00:00, generous on purpose.
# 6 arms x 32 tasks = 192 jobs, under Amarel's 500-job `main` QOS cap in a single wave.
#
# Usage (from the CS side). Every box path comes in by env -- the Amarel values live in the machine
# card, CLAUDE.amarel.md, not in this file:
#   AMAREL_REPO=... AMAREL_SAGE=... AMAREL_CKPTS=... AMAREL_OUT=... \
#     bash scripts/rl_loop/run_policy_v3.sh
set -uo pipefail
AM=${AMAREL_HOST:-amarel}
REPO=${AMAREL_REPO:?set AMAREL_REPO=<remote checkout>}
SAGE=${AMAREL_SAGE:?set AMAREL_SAGE=<remote sage checkout, matching the ckpt>}
CK=${AMAREL_CKPTS:?set AMAREL_CKPTS=<remote ckpt dir>}
OUT=${AMAREL_OUT:?set AMAREL_OUT=<remote eval output root>}
KMAX=${KMAX:-10}
N1SH=${N1SH:-16}
N2SH=${N2SH:-16}

# arm spec: name|prior|ckpt|seed
ARMS=(
  "HY5U_s1|q|$CK/HY5U_s1.ckpt|1"
  "HY5U_s2|q|$CK/HY5U_s2.ckpt|2"
  "HY5U_s3|q|$CK/HY5U_s3.ckpt|3"
  "rand_s7000|uniform|$CK/HY5U_s1.ckpt|7000"
  "rand_s8000|uniform|$CK/HY5U_s1.ckpt|8000"
  "rand_s9000|uniform|$CK/HY5U_s1.ckpt|9000"
)

submit() {   # $1=name $2=prior $3=ckpt $4=seed -- retries past the QOS submit cap
  # A rejected arm prints QOSMaxSubmitJobPerUserLimit and exits nonzero. Grepping only for a job id
  # loses that arm silently, to be discovered later by a shard count. Retry until it lands.
  local _try _out
  for _try in $(seq 1 60); do
    _out=$(ssh "$AM" "cd $REPO && CKPT=$3 OUT=$OUT/$1 PRIOR=$2 SEED=$4 KMAX=$KMAX \
      N1SH=$N1SH N2SH=$N2SH NAMO_REPO=$REPO SAGE_REPO=$SAGE \
      sbatch --array=0-$((N1SH+N2SH-1)) --time=02:00:00 --job-name=pol_$1 \
      scripts/slurm/policy_argmax_amarel.slurm" 2>&1)
    if echo "$_out" | grep -q "Submitted batch job"; then
      echo "$_out" | grep -oE '[0-9]+$'; return 0
    fi
    echo "  submit of $1 rejected (attempt $_try): $(echo "$_out" | tail -1)" >&2
    sleep 60
  done
  echo "FAILED_TO_SUBMIT_$1"; return 1
}

for spec in "${ARMS[@]}"; do
  IFS='|' read -r name prior ckpt seed <<< "$spec"
  id=$(submit "$name" "$prior" "$ckpt" "$seed")
  echo "submitted $name (prior=$prior seed=$seed) job=$id"
done
echo "all arms submitted $(date). expect $((N1SH+N2SH)) shards each under $OUT/<arm>/{1push_policy,2push_policy}/"
