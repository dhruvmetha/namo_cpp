#!/bin/bash
# Build a fresh XML-discovery manifest as a SLURM job (keeps heavy recursive
# globs off the login node). Wraps scripts/generate_xml_manifest.py.
#
# Required env vars at submit:
#   INPUT_DIR   root dir to scan recursively for *.xml (excludes *_temp.xml)
#   OUTPUT      manifest path to write (one XML path per line, shuffled seed 42)
#
# Optional:
#   EXTRA_ARGS  passthrough to generate_xml_manifest.py (e.g. --no-shuffle)
#
# Example:
#   INPUT_DIR=$NAMO_DATASETS/car_envs/v3/aug9_car \
#   OUTPUT=$NAMO_MANIFESTS/v3_aug9_phase1.txt \
#     sbatch scripts/amarel/build_xml_manifest.slurm
#
#SBATCH --job-name=build-manifest
#SBATCH --partition=main
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=logs/build-manifest-%j.out
#SBATCH --error=logs/build-manifest-%j.err

set -euo pipefail
NAMO_REPO="${NAMO_REPO:?source env.<machine>.sh first}"
cd "$NAMO_REPO"
source scripts/amarel/activate.sh

: "${INPUT_DIR:?set INPUT_DIR}"
: "${OUTPUT:?set OUTPUT}"
mkdir -p "$(dirname "$OUTPUT")" "$NAMO_LOGS"

echo "host=$(hostname) input=$INPUT_DIR output=$OUTPUT"
echo "start=$(date -Is)"

python scripts/generate_xml_manifest.py \
    --input-dir "$INPUT_DIR" \
    --output "$OUTPUT" \
    ${EXTRA_ARGS:-}

echo "end=$(date -Is)"
echo "manifest lines: $(wc -l < "$OUTPUT")"
