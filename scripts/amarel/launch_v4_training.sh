#!/bin/bash
# v4 retrain: the FINISH-REBALANCE test. v4 mix = m2b + h2 + aug + EXIT-v4, where EXIT-v4 is the SCALED + dead-inclusive
# ExIt finish set (--setups both --topk-setups 15 on Hz-v3, v4_hq_exit_finish_v4) REPLACING v3's narrow 24k ExIt.
# ONE change vs v3: the finish data scaled ~24k -> ~100k with real dead-s1 coverage + more s1 diversity (the autopsy's
# v4 rebalance: finish was 3.6% of the mix + only 12% dead vs ~75% at deploy). Horizon AND NoHorizon ([USER] ablation).
# Default = 1-seed FEELER (array 9); re-run ARRAY=9-11 for 3 seeds once the gate moves.
#   ARRAY=9 bash scripts/amarel/launch_v4_training.sh
set -euo pipefail
SAGE="$SAGE_REPO"
H5="$NAMO_H5"
PY="$NAMO_PYTHON"
ARRAY=${ARRAY:-9}; WALL=${WALL:-14:00:00}

M2B=$H5/v4_hq_m2b_scorer/data.h5
H2=$H5/v4_hq_h2_scorer/data.h5
AUG=$H5/v4_hq_onepush_h2_aug/data.h5
# EXIT-v4 = the scaled broad-setup finish collection (both arms, topk=15) — diversity + dead coverage + test-difficulty
EXIT_SHARDS=$(ls $H5/v4_hq_exit_finish_v4/shard_*.h5 2>/dev/null | sort -V)
NEX=$(echo "$EXIT_SHARDS" | grep -c . || true)
echo "=== validating v4 mix ingredients (exit-v4 shards found: $NEX) ==="
[ "$NEX" -ge 1 ] || { echo "NO exit-v4 shards yet — collection not done; abort"; exit 1; }
for f in "$M2B" "$H2" "$AUG"; do [ -f "$f" ] || { echo "MISSING: $f"; exit 1; }; done
$PY - $EXIT_SHARDS <<'PYEOF'
import sys, h5py
need = {"ctx","f_grid","r_mask","contact_px","H","dead","object_center","xml","ratio"}
tot=0; dead=0
for p in sys.argv[1:]:
    f=h5py.File(p,"r"); n=int(f.attrs.get("n_samples", f["ctx"].shape[0]))
    miss=need-set(f.keys()); assert not miss, f"{p} missing {miss}"
    assert f["ctx"].shape[1:]==(5,64,64), f"{p} ctx {f['ctx'].shape}"
    tot+=n; dead += int(f["dead"][:].sum()) if n else 0; f.close()
print(f"  OK: {len(sys.argv)-1} exit-v4 shards, {tot} finish rows ({tot-dead} opener-bearing, {dead} dead = {100*dead/max(tot,1):.0f}% dead)")
assert tot>20000, f"only {tot} exit-v4 rows — collection looks incomplete, wait for more shards"
PYEOF

EXIT_JOINED=$(echo "$EXIT_SHARDS" | paste -sd ';' -)
DATA_DIR="$M2B;$H2;$AUG;$EXIT_JOINED"
echo "=== v4 DATA_DIR (v3-ExIt REPLACED by scaled exit-v4) = $((NEX+3)) H5s ==="

HZ_OV="+network.budget_cond=true +data.budget_h=true +model.head_mode=hl_gauss +network.value_bins=51"
NHZ_OV="+data.budget_h=false +model.head_mode=hl_gauss +network.value_bins=51"

cd "$SAGE"
echo "=== Horizon-v4 (qfull_v4_v4hq, array $ARRAY, time=$WALL) ==="
JH=$(sbatch --parsable --array="$ARRAY" --time="$WALL" \
  --export="ALL,RUN_PREFIX=qfull_v4_v4hq,DATA_DIR=$DATA_DIR,EXTRA_OVERRIDES=$HZ_OV" \
  scripts/train_h5_sampling.slurm)
echo "  Horizon-v4 job=$JH"
echo "=== NoHorizon-v4 (qfull_nohz_v4_v4hq, array $ARRAY, time=$WALL) ==="
JN=$(sbatch --parsable --array="$ARRAY" --time="$WALL" \
  --export="ALL,RUN_PREFIX=qfull_nohz_v4_v4hq,DATA_DIR=$DATA_DIR,EXTRA_OVERRIDES=$NHZ_OV" \
  scripts/train_h5_sampling.slurm)
echo "  NoHorizon-v4 job=$JN"
echo "V4_LAUNCHED Horizon=$JH NoHorizon=$JN  (feeler array=$ARRAY)"
