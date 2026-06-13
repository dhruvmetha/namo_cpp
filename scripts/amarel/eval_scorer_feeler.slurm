#!/bin/bash
# H=1 SCORER feeler — run eval_scorer.py on one or more ckpts (auto-detects budget-Q, evals at H=1).
# Apples-to-apples with the champion panel (default episodes = namo_testset_v1/labels/onepush_episodes.json,
# 20% bar). Reusable for mid-training feelers (~ep8/ep11/ep15) AND the final per-seed verdict.
#
# Required env at submit:
#   CKPTS    semicolon-separated ckpt paths (one eval each)
#   OUT_DIR  where per-ckpt JSON lands
#SBATCH --job-name=eval-feeler
#SBATCH --partition=gpu,gpu-redhat
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=00:40:00
#SBATCH --output=/scratch/dm1487/logs/evalfeel-%j.out
#SBATCH --error=/scratch/dm1487/logs/evalfeel-%j.err
set -euo pipefail

cd /cache/home/dm1487/projects/namo/namo_cpp
export MJ_PATH=/scratch/dm1487/mujoco/mujoco-3.2.7
export LD_LIBRARY_PATH="$MJ_PATH/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$PWD/build_python:$PWD/python:$PWD/scripts:/cache/home/dm1487/projects/namo/sage_learning"

: "${CKPTS:?must set CKPTS (semicolon-separated)}"
: "${OUT_DIR:?must set OUT_DIR}"
mkdir -p "$OUT_DIR"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1 || true

IFS=';' read -ra arr <<< "$CKPTS"
for ck in "${arr[@]}"; do
    [ -z "$ck" ] && continue
    tag=$(echo "$ck" | sed -E 's#.*/scorer/([^/]+)/.*/(epoch[0-9]+).*#\1_\2#')
    out="$OUT_DIR/${tag}_h${EVAL_H:-1}.json"
    echo "=== eval $tag H=${EVAL_H:-1} ==="
    /scratch/dm1487/envs/namo/bin/python -u scripts/eval_scorer.py \
        --ckpt "$ck" --network edge_crossattn --num-depths 5 --h "${EVAL_H:-1}" --out "$out"
    /scratch/dm1487/envs/namo/bin/python - "$out" <<'PY'
import json, sys
r = json.load(open(sys.argv[1])); d = r["divisions"]
def sr(div, k): return d.get(div, {}).get("scorer_realistic", {}).get(k)
def we(div): return d.get(div, {}).get("wrong_edge_of_misses_pct")
print(f"  {sys.argv[1].split('/')[-1]}: "
      f"hard@1={sr('hard','@1')} @5={sr('hard','@5')} | med@1={sr('med','@1')} easy@1={sr('easy','@1')} | "
      f"hard wrong-edge/miss={we('hard')}% n={r.get('n_episodes')}")
PY
done
echo "FEELER DONE -> $OUT_DIR"
