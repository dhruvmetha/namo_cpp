#!/bin/bash
# STAGE 1 — train the single-Q BOOTSTRAPPED value (Horizon DROPPED). One Q map, no budget conditioning.
# mix = m2b (1-push openers+dead, value=1.0/0) + ExIt-v4 finish (s1 openers) + bootstrap-SETUP (s0, target = gamma*V_GT(s1)).
# The bootstrap-SETUP H5 REPLACES the old h2 flat-0.9 setup labels with the GROUNDED recurrence target (cost-to-go).
# NoHorizon flags (budget_cond off = single Q = Horizon dropped). From scratch (grounded targets => stable, no online
# divergence; this is the Stage-1b "seeded" bootstrap done offline). VSUMMARY=density (Stage 1) | depth (Stage 3 control).
#   VSUMMARY=density bash scripts/amarel/launch_bootstrap.sh        # 1-seed feeler (array 9); ARRAY=9-11 for 3 seeds
set -euo pipefail
SAGE=/cache/home/dm1487/projects/namo/sage_learning; H5=/scratch/dm1487/h5; PY=/scratch/dm1487/envs/namo/bin/python
ARRAY=${ARRAY:-9}; WALL=${WALL:-14:00:00}; VSUMMARY=${VSUMMARY:-density}
M2B=$H5/v4_hq_m2b_scorer/data.h5
BOOT=$H5/v4_hq_boot_setup_${VSUMMARY}/data.h5
EXIT_SHARDS=$(ls $H5/v4_hq_exit_finish_v4/shard_*.h5 2>/dev/null | sort -V)
NEX=$(echo "$EXIT_SHARDS" | grep -c . || true)
[ -f "$M2B" ] && [ -f "$BOOT" ] && [ "$NEX" -ge 1 ] || { echo "MISSING (M2B=$M2B BOOT=$BOOT exit_shards=$NEX)"; exit 1; }
$PY - "$BOOT" <<'PYEOF'
import sys, h5py
f = h5py.File(sys.argv[1], "r"); n = int(f.attrs["n_samples"])
assert n > 1000, f"boot H5 only {n} rows"; assert f["ctx"].shape[1:] == (5, 64, 64), f["ctx"].shape
print(f"  boot-setup OK: {n} rows, ctx {f['ctx'].shape}")
PYEOF
EXIT_JOINED=$(echo "$EXIT_SHARDS" | paste -sd ';' -)
DATA_DIR="$M2B;$EXIT_JOINED;$BOOT"
OV="+data.budget_h=false +model.head_mode=hl_gauss +network.value_bins=51"   # NoHorizon = single Q
echo "=== STAGE 1 bootstrap: qboot_${VSUMMARY}, $((NEX+2)) H5s, array $ARRAY ==="
cd "$SAGE"
J=$(sbatch --parsable --array="$ARRAY" --time="$WALL" \
  --export="ALL,RUN_PREFIX=qboot_${VSUMMARY},DATA_DIR=$DATA_DIR,EXTRA_OVERRIDES=$OV" \
  scripts/train_h5_sampling.slurm)
echo "BOOT_LAUNCHED qboot_${VSUMMARY} job=$J  (anchor v2/v3/v4 untouched; new run-prefix)"
