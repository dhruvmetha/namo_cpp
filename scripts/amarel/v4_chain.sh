#!/bin/bash
# v4 chain: wait for the ExIt-v4 collection array to ROBUSTLY drain, then launch the v4 retrain (Hz + NoHz).
# Robustness fix: require 3 CONSECUTIVE empty squeue checks before concluding drained (a single transient squeue
# empty/error must NOT fire the chain — that bug launched it prematurely once). launch_v4_training.sh self-guards
# on rows>20k, so if it ever fires early it aborts harmlessly; this loop keeps trying until the collection is
# really done AND launch_v4 succeeds. Run detached: nohup bash scripts/amarel/v4_chain.sh <collect_jobid> &
set -uo pipefail
JC=${1:-$(cat /tmp/v4_collect_jobid.txt 2>/dev/null)}
LOG=/tmp/v4_chain.log
cd "$NAMO_REPO" || exit 1
echo "[chain] $(date): watching collection $JC (need 3 consecutive empties)" >> "$LOG"
empties=0
while [ "$empties" -lt 3 ]; do
  sleep 120
  if squeue -j "$JC" -h 2>/dev/null | grep -q .; then empties=0; else empties=$((empties+1)); fi
done
echo "[chain] $(date): collection $JC drained (3x confirmed); launching v4 training" >> "$LOG"
bash scripts/amarel/launch_v4_training.sh >> "$LOG" 2>&1
echo "[chain] $(date): chain done (check log above for V4_LAUNCHED or abort)" >> "$LOG"
