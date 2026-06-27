#!/bin/bash
# STAGE 1 — train the single-Q BOOTSTRAPPED value (Horizon DROPPED). One Q map, no budget conditioning.
# mix = m2b (1-push openers+dead, value=1.0/0) + ExIt-v4 finish (s1 openers) + bootstrap-SETUP (s0, target = gamma*V_GT(s1)).
# The bootstrap-SETUP H5 REPLACES the old h2 flat-0.9 setup labels with the GROUNDED recurrence target (cost-to-go).
# NoHorizon flags (budget_cond off = single Q = Horizon dropped). From scratch (grounded targets => stable, no online
# divergence; this is the Stage-1b "seeded" bootstrap done offline). VSUMMARY=density (Stage 1) | depth (Stage 3 control).
#   VSUMMARY=density bash scripts/amarel/launch_bootstrap.sh        # 1-seed feeler (array 9); ARRAY=9-11 for 3 seeds
set -euo pipefail
SAGE="$SAGE_REPO"; H5="$NAMO_H5"; PY="$NAMO_PYTHON"
ARRAY=${ARRAY:-9}; WALL=${WALL:-14:00:00}; VSUMMARY=${VSUMMARY:-density}
M2B=$H5/v4_hq_m2b_scorer/data.h5
BOOT_SHARDS=$(ls $H5/v4_hq_boot_setup_${VSUMMARY}/shard_*.h5 2>/dev/null | sort -V)
NB=$(echo "$BOOT_SHARDS" | grep -c . || true)
EXIT_SHARDS=$(ls $H5/v4_hq_exit_finish_v4/shard_*.h5 2>/dev/null | sort -V)
NEX=$(echo "$EXIT_SHARDS" | grep -c . || true)
[ -f "$M2B" ] && [ "$NB" -ge 1 ] && [ "$NEX" -ge 1 ] || { echo "MISSING (M2B=$M2B boot_shards=$NB exit_shards=$NEX)"; exit 1; }
$PY - $BOOT_SHARDS <<'PYEOF'
import sys, h5py
tot = 0
for p in sys.argv[1:]:
    f = h5py.File(p, "r"); n = int(f.attrs["n_samples"]); tot += n
    assert f["ctx"].shape[1:] == (5, 64, 64), f["ctx"].shape
assert tot > 3000, f"boot shards only {tot} rows — build incomplete"
print(f"  boot-setup OK: {len(sys.argv)-1} shards, {tot} rows")
PYEOF
EXIT_JOINED=$(echo "$EXIT_SHARDS" | paste -sd ';' -)
BOOT_JOINED=$(echo "$BOOT_SHARDS" | paste -sd ';' -)
DATA_DIR="$M2B;$EXIT_JOINED;$BOOT_JOINED"
HEAD=${HEAD:-hl_gauss}    # hl_gauss = bootstrapped VALUE (Stage 1) | softmax_ce = setup RANKING (the contingency)
if [ "$HEAD" = "hl_gauss" ]; then OV="+data.budget_h=false +model.head_mode=hl_gauss +network.value_bins=51"
else OV="+data.budget_h=false +model.head_mode=$HEAD"; fi                     # NoHorizon (single Q) either way
TAG="qboot_${VSUMMARY}"; [ "$HEAD" = "hl_gauss" ] || TAG="qrank_${VSUMMARY}"
echo "=== STAGE 1 head=$HEAD: ${TAG}, $((NEX+2)) H5s, array $ARRAY ==="
cd "$SAGE"
J=$(sbatch --parsable --array="$ARRAY" --time="$WALL" --partition="${PART:-gpu-redhat}" \
  --export="ALL,RUN_PREFIX=${TAG},DATA_DIR=$DATA_DIR,EXTRA_OVERRIDES=$OV" \
  scripts/train_h5_sampling.slurm)
echo "${TAG}_LAUNCHED job=$J  (anchor v2/v3/v4 untouched; new run-prefix)"
