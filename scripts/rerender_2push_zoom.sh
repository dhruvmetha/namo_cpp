#!/bin/bash
# Re-render existing qpos dumps at a larger camera distance.
set -e

REPO=/common/home/dm1487/robotics_research/ktamp/namo
RESULTS=/common/home/dm1487/scratch_namo/car_2push_lenient_matched_results
QPOS_DIR=/common/home/dm1487/scratch_namo/car_2push_replay_videos
OUT=/common/home/dm1487/robotics_research/ktamp/namo/test_xml/little-car-modeling-package/artifacts/push_replays_2push_zoomed
CAM_DIST=${1:-3.0}

mkdir -p "$OUT"
export MJ_PATH=/common/users/dm1487/ktamp/mujoco
export MUJOCO_GL=egl
PY=/common/users/dm1487/envs/mjxrl/bin/python

# Build xml-lookup table from pickles (qpos basename → xml path)
declare -A XMLS
while IFS= read -r line; do
    base=$(echo "$line" | awk '{print $1}')
    xml=$(echo "$line" | awk '{print $2}')
    XMLS["$base"]="$xml"
done < <($PY -c "
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
        base = fp.split('/')[-1].replace('.pkl','') + f'_ep{matched}_{obj}'
        print(base, xml)
")

cnt=0
total=$(ls "$QPOS_DIR"/*.qpos 2>/dev/null | wc -l)
for qpos in "$QPOS_DIR"/*.qpos; do
    base=$(basename "$qpos" .qpos)
    xml="${XMLS[$base]}"
    cnt=$((cnt+1))
    if [[ -z "$xml" ]]; then
        echo "[$cnt/$total] $base: no xml mapping, skipping"
        continue
    fi
    mp4="$OUT/${base}.mp4"
    echo "[$cnt/$total] $base (xml=$(basename "$xml"), cam_dist=$CAM_DIST)"
    $PY "$REPO/scripts/render_qpos_simple.py" \
        "$xml" "$qpos" "$mp4" \
        --cam-dist "$CAM_DIST" --width 480 --height 480 --frame-skip 25 2>&1 | tail -2
done

echo ""
echo "Done. Zoomed videos in $OUT (cam_dist=$CAM_DIST):"
ls -lh "$OUT"/*.mp4 2>/dev/null | wc -l
du -sh "$OUT"
