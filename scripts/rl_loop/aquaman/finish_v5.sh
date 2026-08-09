#!/bin/bash
# v5 endgame: wait on SHARD COUNTS (not log markers), pull, aggregate, build the comparison.
#
# Why not the agg chain: arjuna_v4_eval.log accumulated two stale "ARJUNA V4 EVAL PULLED" lines
# across streamer restarts, so anything grepping for that marker fires immediately on old text.
# Shard counts on Amarel are ground truth and cannot go stale.
#
# Expected counts differ per arm: the AJ5 trio was submitted at 72 shards, the AJ5NR trio at 144
# (sharding was doubled mid-flight to halve eval wall time). Both cover the same episodes.
set -u
# Amarel paths come from env, never baked in (portability guard):
#   AMAREL_ROOT  remote home holding aquaman0/   AMAREL_REPO  remote namo checkout
#   AMAREL_SAGE  remote sage checkout (must match the ckpt)
AMAREL_ROOT=${AMAREL_ROOT:?set AMAREL_ROOT=<remote home holding aquaman0/>}
AMAREL_REPO=${AMAREL_REPO:?set AMAREL_REPO=<remote namo checkout>}
AMAREL_SAGE=${AMAREL_SAGE:?set AMAREL_SAGE=<remote sage checkout>}
R0=/common/users/dm1487/scratch_namo/aquaman/round0
REM=${AMAREL_ROOT}/aquaman0/eval_v5
MAIN=/common/home/dm1487/robotics_research/ktamp/namo
PY=/common/users/dm1487/envs/mjxrl/bin/python

want() { echo 144; }

for i in $(seq 1 480); do          # up to 8 h
  ok=1
  for m in AJ5_s1 AJ5_s2 AJ5_s3 AJ5NR_s1 AJ5NR_s2 AJ5NR_s3; do
    n=$(ssh amarel "ls $REM/$m/{1push_hmax2,2push}/shard_*.json 2>/dev/null | wc -l" 2>/dev/null)
    [ -n "$n" ] && [ "$n" -ge "$(want $m)" ] || { ok=0; echo "[$(date +%H:%M)] waiting on $m: ${n:-0}/$(want $m)"; break; }
  done
  [ "$ok" -eq 1 ] && break
  sleep 120
done
[ "$ok" -eq 1 ] || { echo "TIMED OUT waiting for v5 shards"; exit 1; }
echo "ALL V4 SHARDS COMPLETE $(date)"

rsync -a --include='*/' --include='shard_*.jsonl' --include='shard_*.json' --exclude='*' \
  "amarel:$REM/" "$R0/eval_v5/"
echo "pulled $(ls $R0/eval_v5/*/{1push_hmax2,2push}/shard_*.jsonl 2>/dev/null | wc -l) jsonl"

cat > "$R0/arms_aj5_full.json" <<EOF
{"AJ5":   {"1push": ["$R0/eval_v5/AJ5_s1/1push_hmax2","$R0/eval_v5/AJ5_s2/1push_hmax2","$R0/eval_v5/AJ5_s3/1push_hmax2"],
           "2push": ["$R0/eval_v5/AJ5_s1/2push","$R0/eval_v5/AJ5_s2/2push","$R0/eval_v5/AJ5_s3/2push"]},
 "AJ5NR": {"1push": ["$R0/eval_v5/AJ5NR_s1/1push_hmax2","$R0/eval_v5/AJ5NR_s2/1push_hmax2","$R0/eval_v5/AJ5NR_s3/1push_hmax2"],
           "2push": ["$R0/eval_v5/AJ5NR_s1/2push","$R0/eval_v5/AJ5NR_s2/2push","$R0/eval_v5/AJ5NR_s3/2push"]}}
EOF
"$PY" "$MAIN/scripts/rl_loop/aquaman_agg.py" "$R0/arms_aj5_full.json" "$R0/gate_aj5.json"
"$PY" "$R0/build_final_comparison.py" > "$R0/final_comparison.out" 2>&1
echo "V5 COMPLETE AND AGGREGATED $(date)"
cat "$R0/final_comparison.out"
