#!/bin/bash
# Validate the v2 OOD mix ingredients, then launch Horizon-v2 + NoHorizon-v2 (the 2x2 matrix's v2 column).
# v2 mix = root (m2b H1 + h2 H1/H2) + 1-push@H2 aug (opener=1.0@H2) + post-push (OOD s1), ';'-joined data_dir.
# Mix proportions over-represent the two OOD failure modes via on-disk row counts + uniform sampling
# (ScorerDataModule has no weighted sampler; counts do the balancing for round 1).
# Same recipe as v1 (array 9-11 = B30 x seeds 1-3); v1 flags captured from the running qfull configs.
set -euo pipefail
SAGE=/cache/home/dm1487/projects/namo/sage_learning
H5=/scratch/dm1487/h5
PY=/scratch/dm1487/envs/namo/bin/python

M2B=$H5/v4_hq_m2b_scorer/data.h5
H2=$H5/v4_hq_h2_scorer/data.h5
AUG=$H5/v4_hq_onepush_h2_aug/data.h5
PP=$H5/v4_hq_postpush_v2
PP_SHARDS=$(ls "$PP"/shard_*.h5 2>/dev/null | sort -V)
NPP=$(echo "$PP_SHARDS" | grep -c . || true)
echo "=== validating v2 mix ingredients (postpush shards found: $NPP) ==="
[ "$NPP" -ge 1 ] || { echo "NO postpush shards in $PP — abort"; exit 1; }
for f in "$M2B" "$H2" "$AUG" $PP_SHARDS; do
  [ -f "$f" ] || { echo "MISSING: $f — abort"; exit 1; }
done
$PY - "$AUG" $PP_SHARDS <<'PYEOF'
import sys, h5py
need = {"ctx","f_grid","r_mask","contact_px","H","dead","object_center","xml","ratio"}
tot = 0
for p in sys.argv[1:]:
    f = h5py.File(p, "r")
    n = int(f.attrs.get("n_samples", f["ctx"].shape[0]))
    miss = need - set(f.keys())
    assert not miss, f"{p} missing {miss}"
    assert n > 0, f"{p} empty"
    assert f["ctx"].shape[1:] == (5,64,64), f"{p} ctx {f['ctx'].shape}"
    print(f"  OK {p.split('/')[-2]}/{p.split('/')[-1]}  rows={n}")
    tot += n
    f.close()
print(f"  aug+postpush rows = {tot}")
PYEOF

PP_JOINED=$(echo "$PP_SHARDS" | paste -sd ';' -)
DATA_DIR="$M2B;$H2;$AUG;$PP_JOINED"
echo "=== DATA_DIR ($((NPP+3)) H5s) = $DATA_DIR ==="

# Horizon-v2: budget_cond + budget_h true (H-conditioned), HL-Gauss 51-bin value head.
HZ_OV="+network.budget_cond=true +data.budget_h=true +model.head_mode=hl_gauss +model.value_bins=51"
# NoHorizon-v2: same head, NO horizon conditioning (budget_h=false, budget_cond default false).
NHZ_OV="+data.budget_h=false +model.head_mode=hl_gauss +model.value_bins=51"

cd "$SAGE"
echo "=== launching Horizon-v2 (qfull_v2_v4hq, array 9-11) ==="
JH=$(sbatch --parsable --array=9-11 \
  --export="ALL,RUN_PREFIX=qfull_v2_v4hq,DATA_DIR=$DATA_DIR,EXTRA_OVERRIDES=$HZ_OV" \
  scripts/train_h5_sampling.slurm)
echo "  Horizon-v2 job=$JH"
echo "=== launching NoHorizon-v2 (qfull_nohz_v2_v4hq, array 9-11) ==="
JN=$(sbatch --parsable --array=9-11 \
  --export="ALL,RUN_PREFIX=qfull_nohz_v2_v4hq,DATA_DIR=$DATA_DIR,EXTRA_OVERRIDES=$NHZ_OV" \
  scripts/train_h5_sampling.slurm)
echo "  NoHorizon-v2 job=$JN"
echo "V2_LAUNCHED Horizon=$JH NoHorizon=$JN"
