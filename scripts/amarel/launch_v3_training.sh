#!/bin/bash
# P2 of the core-fix plan: retrain the FINISH on diverse ExIt data (the generalization fix), Horizon AND NoHorizon
# (the [USER] ablation). v3 mix = m2b + h2 + aug + EXIT, with the narrow postpush REPLACED by the ExIt finish data
# (postpush is the data that failed to generalize). Same recipe as v2 otherwise. Default = 1-seed FEELER (array 9);
# once the GATE passes (novel-s1 finish sep 0.30 -> 0.6+) re-run with ARRAY=9-11 for the full 3 seeds.
#   ARRAY=9 sbatch-wrapper:  bash scripts/amarel/launch_v3_training.sh
set -euo pipefail
SAGE="$SAGE_REPO"
H5="$NAMO_H5"
PY="$NAMO_PYTHON"
ARRAY=${ARRAY:-9}; WALL=${WALL:-14:00:00}

M2B=$H5/v4_hq_m2b_scorer/data.h5
H2=$H5/v4_hq_h2_scorer/data.h5
AUG=$H5/v4_hq_onepush_h2_aug/data.h5
# EXIT = the diverse opener-rich finish data (--setups valid) + the earlier model-setup collection (deploy calibration)
EXIT_SHARDS=$(ls $H5/v4_hq_exit_finish_valid/shard_*.h5 $H5/v4_hq_exit_finish/shard_*.h5 2>/dev/null | sort -V)
NEX=$(echo "$EXIT_SHARDS" | grep -c . || true)
echo "=== validating v3 mix ingredients (exit shards found: $NEX) ==="
[ "$NEX" -ge 1 ] || { echo "NO exit shards yet — P1 collection not done; abort"; exit 1; }
for f in "$M2B" "$H2" "$AUG"; do [ -f "$f" ] || { echo "MISSING: $f"; exit 1; }; done
$PY - $EXIT_SHARDS <<'PYEOF'
import sys, h5py
need = {"ctx","f_grid","r_mask","contact_px","H","dead","object_center","xml","ratio"}
tot=0; op=0
for p in sys.argv[1:]:
    f=h5py.File(p,"r"); n=int(f.attrs.get("n_samples", f["ctx"].shape[0]))
    miss=need-set(f.keys()); assert not miss, f"{p} missing {miss}"
    assert f["ctx"].shape[1:]==(5,64,64), f"{p} ctx {f['ctx'].shape}"
    tot+=n; op += n-int(f["dead"][:].sum()) if n else 0; f.close()
print(f"  OK: {len(sys.argv)-1} exit shards, {tot} finish rows ({op} opener-bearing, {tot-op} dead)")
assert tot>2000, f"only {tot} exit rows — too few, wait for more shards"
PYEOF

EXIT_JOINED=$(echo "$EXIT_SHARDS" | paste -sd ';' -)
DATA_DIR="$M2B;$H2;$AUG;$EXIT_JOINED"
echo "=== v3 DATA_DIR (postpush REPLACED by exit) = $((NEX+3)) H5s ==="

HZ_OV="+network.budget_cond=true +data.budget_h=true +model.head_mode=hl_gauss +network.value_bins=51"
NHZ_OV="+data.budget_h=false +model.head_mode=hl_gauss +network.value_bins=51"

cd "$SAGE"
echo "=== Horizon-v3 (qfull_v3_v4hq, array $ARRAY, time=$WALL) ==="
JH=$(sbatch --parsable --array="$ARRAY" --time="$WALL" \
  --export="ALL,RUN_PREFIX=qfull_v3_v4hq,DATA_DIR=$DATA_DIR,EXTRA_OVERRIDES=$HZ_OV" \
  scripts/train_h5_sampling.slurm)
echo "  Horizon-v3 job=$JH"
echo "=== NoHorizon-v3 (qfull_nohz_v3_v4hq, array $ARRAY, time=$WALL) ==="
JN=$(sbatch --parsable --array="$ARRAY" --time="$WALL" \
  --export="ALL,RUN_PREFIX=qfull_nohz_v3_v4hq,DATA_DIR=$DATA_DIR,EXTRA_OVERRIDES=$NHZ_OV" \
  scripts/train_h5_sampling.slurm)
echo "  NoHorizon-v3 job=$JN"
echo "V3_LAUNCHED Horizon=$JH NoHorizon=$JN  (feeler array=$ARRAY)"
