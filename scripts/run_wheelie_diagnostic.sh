#!/usr/bin/env bash
# Diagnostic: rerun the 2-region search with teleport nav + qpos dump,
# then plot chassis pitch vs time to check whether the runtime push
# physics actually causes the wheelie we saw in the rendered video.
#
# Usage:
#   bash scripts/run_wheelie_diagnostic.sh
#
# Outputs:
#   /common/home/dm1487/scratch_namo/wheelie_search.txt        — qpos dump
#   /common/home/dm1487/scratch_namo/wheelie_search.pitch.png  — pitch plot
set -euo pipefail

NAMO_DIR=/common/home/dm1487/robotics_research/ktamp/namo
SCRATCH=/common/home/dm1487/scratch_namo
ENV_XML="${SCRATCH}/diverse_car_envs/set2/sparse/benchmark_3/run_0012/env_0012_pair_000.xml"
DUMP="${SCRATCH}/wheelie_search.txt"
DIAG_DIR="${SCRATCH}/wheelie_diag"
DIAG_CONFIG=/tmp/wheelie_diag.yaml
MANIFEST=/tmp/manifest_2region.txt

mkdir -p "${DIAG_DIR}"
echo "${ENV_XML}" > "${MANIFEST}"

# Build a config based on region_opening_exhaustive_car.yaml but pointing
# the manifest at the single 2-region env.
cp "${NAMO_DIR}/python/namo/data_collection/region_opening_exhaustive_car.yaml" "${DIAG_CONFIG}"
sed -i "s|output_dir:.*|output_dir: ${DIAG_DIR}|" "${DIAG_CONFIG}"
sed -i "s|manifest:.*|manifest: ${MANIFEST}|" "${DIAG_CONFIG}"

rm -f "${DUMP}"
cd "${NAMO_DIR}"

NAMO_FORCE_TELEPORT_NAV=1 NAMO_QPOS_DUMP="${DUMP}" \
  python python/namo/data_collection/modular_parallel_collection.py \
    --config "${DIAG_CONFIG}" \
    --output-dir "${DIAG_DIR}" \
    --start-idx 0 --end-idx 1 \
    --manifest "${MANIFEST}"

echo
echo "=== Extracting pitch from ${DUMP} ==="
python scripts/pitch_from_qpos.py "${ENV_XML}" "${DUMP}"
