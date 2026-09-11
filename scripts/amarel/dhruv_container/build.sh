#!/usr/bin/env bash
# Run under a bounded CPU SLURM allocation. The image is never overwritten.
set -euo pipefail
config=${1:?usage: build.sh /absolute/path/to/container.env}
source "$config"
recipes=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$recipes/common.sh"
refuse_existing "$CONTAINER_ROOT/dhruv-focal.sif" "$MJ_PATH" "$NAMO_REPO/build_python"
: "${BUILD_JOBS:?set BUILD_JOBS to the SLURM CPU allocation}"
export APPTAINER_CACHEDIR="$CONTAINER_ROOT/cache"
export APPTAINER_TMPDIR="$CONTAINER_ROOT/tmp"
mkdir -p "$APPTAINER_CACHEDIR" "$APPTAINER_TMPDIR" "$CONTAINER_ROOT/provenance" "$CONTAINER_ROOT/results"
cd "$CONTAINER_ROOT/artifacts"
sha256sum -c inputs.sha256
# Do not inject the RHEL host fakeroot binary into an older-glibc image.
env -u LD_LIBRARY_PATH apptainer build --ignore-fakeroot-command \
    --mksquashfs-args "-processors $BUILD_JOBS" \
    "$CONTAINER_ROOT/dhruv-focal.sif" "$recipes/ubuntu20.def"
sha256sum "$CONTAINER_ROOT/dhruv-focal.sif" > "$CONTAINER_ROOT/provenance/image.sha256"
env -u LD_LIBRARY_PATH apptainer exec --cleanenv --containall \
    --bind "$CONTAINER_ROOT:$CONTAINER_ROOT:rw,$FROZEN_ROOT:$FROZEN_ROOT:ro,$PYTHON_ENV:$PYTHON_ENV:ro,$config:$config:ro" \
    "$CONTAINER_ROOT/dhruv-focal.sif" \
    bash "$recipes/native-build.sh" "$config"
