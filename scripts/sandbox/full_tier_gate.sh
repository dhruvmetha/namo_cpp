#!/bin/bash
# Definitive full-1018 per-tier gate: 4 models x 4 manifest slices = 16 parallel eval_bestfirst (combine=q).
# Merges per model, then per-tier breakdown vs pure2push_divisions.json. ~45 min wall, 16 cores, 4 GPUs.
set -u
cd /common/home/dm1487/robotics_research/ktamp/namo
source env.ilab.sh
EVAL=$NAMO_SCRATCH/eval/full_tier; mkdir -p "$EVAL"
DIV=$NAMO_SCRATCH/datasets/namo_testset_v1/labels/pure2push_divisions.json
PP="$PWD/build_python:$PWD/python:$SAGE_REPO"
O=$NAMO_OUTPUTS/scorer

declare -A CK
CK[qrank_density]=$O/qrank_density_s1/namo-classifier/z7ax3oj1/checkpoints/epoch010-val_loss3.6932.ckpt
CK[qrank_depth]=$O/qrank_depth_s1/namo-classifier/x5cujg88/checkpoints/epoch009-val_loss3.7055.ckpt
CK[qboot_density]=$O/qboot_density_s1/namo-classifier/v5x21lsi/checkpoints/epoch012-val_loss0.7152.ckpt
CK[qboot_depth]=$O/qboot_depth_s1/namo-classifier/xdbdc8vv/checkpoints/epoch014-val_loss0.7192.ckpt

SLICES=("0 246" "246 492" "492 738" "738 983")
GPUS=(1 2 3 4)
i=0
for M in qrank_density qrank_depth qboot_density qboot_depth; do
  s=0
  for SL in "${SLICES[@]}"; do
    set -- $SL; st=$1; en=$2
    G=${GPUS[$(( i % 4 ))]}
    CUDA_VISIBLE_DEVICES=$G PYTHONPATH="$PP" python scripts/sandbox/eval_bestfirst.py \
      --ckpt "${CK[$M]}" --combine q --hmax 2 --sim-budget 30 --start $st --end $en \
      --out "$EVAL/${M}_s${s}.json" --leaf-out "$EVAL/${M}_s${s}.jsonl" \
      > "$EVAL/${M}_s${s}.log" 2>&1 &
    i=$((i+1)); s=$((s+1))
  done
done
echo "launched $i eval procs $(date); waiting..."
wait
echo "=== all eval done $(date); merging ==="
for M in qrank_density qrank_depth qboot_density qboot_depth; do
  cat "$EVAL/${M}_s"*.jsonl > "$EVAL/${M}_full.jsonl" 2>/dev/null
  echo "$M: $(wc -l < "$EVAL/${M}_full.jsonl") episodes"
done
echo "=== PER-TIER BREAKDOWN (full 1018, combine=q, best-val ckpts) ==="
python3 scripts/sandbox/tier_breakdown.py --divisions "$DIV" \
  --leaf "$EVAL/qboot_density_full.jsonl" "$EVAL/qrank_density_full.jsonl" \
         "$EVAL/qboot_depth_full.jsonl" "$EVAL/qrank_depth_full.jsonl"
echo "=== FULL TIER GATE COMPLETE $(date) ==="
