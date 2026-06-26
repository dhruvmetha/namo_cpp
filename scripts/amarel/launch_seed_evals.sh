#!/bin/bash
# Submit the afterany eval-chains for every PENDING training seed (v3 s2/s3 + all v4 seeds). Each eval-chain job waits
# on its training task (afterany = COMPLETE or TIMEOUT), then resolves the best ckpt + runs reactive@2 + best-first@2(q).
# So the v3/v4 error bars compute the moment each seed converges overnight — ready by morning, no session needed.
#   bash scripts/amarel/launch_seed_evals.sh
set -uo pipefail
cd /cache/home/dm1487/projects/namo/namo_cpp
# LABEL | RUN_DIR | train job:task to wait on (afterany)
ROWS=(
  "Hz_v3_s2|qfull_v3_v4hq_s2|57014837_10"
  "Hz_v3_s3|qfull_v3_v4hq_s3|57014837_11"
  "NoHz_v3_s2|qfull_nohz_v3_v4hq_s2|57014838_10"
  "NoHz_v3_s3|qfull_nohz_v3_v4hq_s3|57014838_11"
  "Hz_v4_s1|qfull_v4_v4hq_s1|57001294_9"
  "NoHz_v4_s1|qfull_nohz_v4_v4hq_s1|57001295_9"
  "Hz_v4_s2|qfull_v4_v4hq_s2|57014839_10"
  "Hz_v4_s3|qfull_v4_v4hq_s3|57016500_11"
  "NoHz_v4_s2|qfull_nohz_v4_v4hq_s2|57016501_10"
  "NoHz_v4_s3|qfull_nohz_v4_v4hq_s3|57016501_11"
)
for row in "${ROWS[@]}"; do
  IFS='|' read -r LABEL RUN_DIR DEP <<<"$row"
  JID=$(sbatch --parsable --dependency=afterany:"$DEP" --kill-on-invalid-dep=yes \
    --export="ALL,RUN_DIR=$RUN_DIR,LABEL=$LABEL" scripts/amarel/eval_afterok.slurm)
  echo "  eval-chain $LABEL  (waits on $DEP) -> job $JID"
done
echo "SEED_EVALS_CHAINED: 4 v3 + 6 v4 = 10 afterany eval-chains submitted"
