#!/bin/bash
# Render one MP4 per 2-push success episode from car_2push_lenient_matched_results.
set -e

REPO=/common/home/dm1487/robotics_research/ktamp/namo
RESULTS=/common/home/dm1487/scratch_namo/car_2push_lenient_matched_results
OUT=/common/home/dm1487/scratch_namo/car_2push_replay_videos

mkdir -p "$OUT"
export MJ_PATH=/common/users/dm1487/ktamp/mujoco
export MUJOCO_GL=egl
export NAMO_FORCE_TELEPORT_NAV=1
export PYTHONPATH="$REPO/build_python_mjxrl_${HOSTNAME%%.*}:$REPO/python:$PYTHONPATH"
PY=/common/users/dm1487/envs/mjxrl/bin/python

# Emit lines of: pkl  xml  episode_idx_in_2push_subset  chosen_object_id
mapfile -t JOBS < <($PY -c "
import pickle, glob
for fp in sorted(glob.glob('$RESULTS/modular_data_*/*.pkl')):
    d = pickle.load(open(fp,'rb'))
    matched = -1
    for ep in d.get('episode_results', []):
        seq = ep.get('action_sequence') or []
        if not (ep.get('success') and len(seq)==2):
            continue
        matched += 1
        xml = ep['xml_file']
        obj = (ep.get('algorithm_stats') or {}).get('chosen_object_id','obj')
        print(fp, xml, matched, obj)
")

echo "Total 2-push episodes to render: ${#JOBS[@]}"

i=0
for line in "${JOBS[@]}"; do
    i=$((i+1))
    pkl=$(echo "$line" | awk '{print $1}')
    xml=$(echo "$line" | awk '{print $2}')
    idx=$(echo "$line" | awk '{print $3}')
    obj=$(echo "$line" | awk '{print $4}')
    base=$(basename "$pkl" .pkl)
    tag="${base}_ep${idx}_${obj}"
    qpos="$OUT/${tag}.qpos"
    mp4="$OUT/${tag}.mp4"

    echo ""
    echo "[$i/${#JOBS[@]}] === $tag ==="
    echo "  xml: $(basename "$xml")  episode-idx=$idx  object=$obj"

    $PY "$REPO/scripts/replay_solution.py" \
        --results-pkl "$pkl" \
        --xml "$xml" \
        --namo-config "$REPO/config/namo_config_car.yaml" \
        --qpos-out "$qpos" \
        --chain-length 2 \
        --success-only \
        --episode-idx "$idx" 2>&1 | tail -5

    if [[ ! -s "$qpos" ]]; then
        echo "  WARN: empty qpos, skipping render"
        continue
    fi

    $PY "$REPO/scripts/render_qpos_simple.py" \
        "$xml" "$qpos" "$mp4" \
        --cam-dist 1.5 --width 480 --height 480 --frame-skip 25 2>&1 | tail -3
done

echo ""
echo "Done. ${#JOBS[@]} 2-push videos in $OUT:"
ls -lh "$OUT"/*.mp4 2>/dev/null | tail -40
