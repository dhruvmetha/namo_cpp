#!/usr/bin/env bash
# Runs inside the pinned image, with only the new output root writable.
set -euo pipefail
source "${1:?usage: native-build.sh container.env}"
recipes=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$recipes/common.sh"
activate_native_runtime
: "${BUILD_JOBS:?set BUILD_JOBS to the SLURM CPU allocation}"
export BUILD_JOBS
refuse_existing "$MJ_PATH" "$NAMO_REPO/build_python" "$CONTAINER_ROOT/native-deps" "$CONTAINER_ROOT/mujoco-deps"
source_status=$(git -C "$NAMO_REPO" status --porcelain)
test -z "$source_status"
test "$(/usr/bin/c++ -dumpfullversion)" = 9.4.0
test "$(getconf GNU_LIBC_VERSION)" = 'glibc 2.31'
export CC=/usr/bin/cc CXX=/usr/bin/c++ PYTHON_BIN="$NAMO_PYTHON"
mkdir "$MJ_PATH" "$CONTAINER_ROOT/mujoco-deps" "$CONTAINER_ROOT/native-deps"
tar -xzf "$FROZEN_ROOT/mujoco-dbe18f57.tar.gz" -C "$MJ_PATH"
tar -xzf "$FROZEN_ROOT/mujoco-deps.tar.gz" -C "$CONTAINER_ROOT/mujoco-deps"
tar -xzf "$FROZEN_ROOT/namo-build-deps.tar.gz" -C "$CONTAINER_ROOT/native-deps"
deps=()
for source in "$CONTAINER_ROOT"/mujoco-deps/*-src; do
    name=$(basename "$source" -src)
    deps+=("-DFETCHCONTENT_SOURCE_DIR_${name^^}=$source")
done
# Dhruv's MuJoCo uses these defaults, including -mavx and LTO internally;
# unlike its NAMO binding, it was NOT built with -march=native.
cmake -S "$MJ_PATH" -B "$MJ_PATH/build" -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_C_FLAGS_RELEASE="-O3 -DNDEBUG" -DCMAKE_CXX_FLAGS_RELEASE="-O3 -DNDEBUG" \
    -DMUJOCO_BUILD_EXAMPLES=OFF -DMUJOCO_BUILD_SIMULATE=OFF \
    -DMUJOCO_BUILD_TESTS=OFF -DMUJOCO_TEST_PYTHON_UTIL=OFF "${deps[@]}"
cmake --build "$MJ_PATH/build" --target mujoco -j "$BUILD_JOBS"
cd "$NAMO_REPO"
# This file contains GCC9's resolved -march=native expansion ON DHRUV, not
# a fresh expansion on the heterogeneous Amarel CPU. CMake adds '-march='.
flags=$(<"$CONTAINER_ROOT/artifacts/dhruv-native.flags")
[[ "$flags" == -march=skylake* ]]
export NAMO_MARCH="${flags#-march=}"
cmake -S . -B build_python -DCMAKE_BUILD_TYPE=Release -DBUILD_PYTHON_BINDINGS=ON \
    -DNAMO_MARCH="$NAMO_MARCH" -DPython3_EXECUTABLE="$PYTHON_BIN" \
    -DPython3_SOABI="$("$PYTHON_BIN" -c 'import sysconfig; print(sysconfig.get_config_var("SOABI"))')" \
    -DNLOHMANN_JSON_INCLUDE_DIR="$CONTAINER_ROOT/native-deps/external/nlohmann_json/include" \
    -DFETCHCONTENT_SOURCE_DIR_PYBIND11="$CONTAINER_ROOT/native-deps/build_python/_deps/pybind11-src" \
    -DOpenCV_DIR=/usr/local/lib/cmake/opencv4
# Keep the canonical build/stamp code and bound its nproc parallelism.
cpus=$("$PYTHON_BIN" -c 'import os; print(",".join(map(str, sorted(os.sched_getaffinity(0))[:int(os.environ["BUILD_JOBS"])])))')
taskset -c "$cpus" bash ./build_python_bindings.sh
cp build_python/BUILD_INFO "$CONTAINER_ROOT/provenance/BUILD_INFO"
cp build_python/CMakeCache.txt "$CONTAINER_ROOT/provenance/namo-CMakeCache.txt"
cp "$MJ_PATH/build/CMakeCache.txt" "$CONTAINER_ROOT/provenance/mujoco-CMakeCache.txt"
cp /opt/installed-packages.tsv "$CONTAINER_ROOT/provenance/container-packages.tsv"
sha256sum build_python/namo_rl*.so "$MJ_PATH"/build/lib/libmujoco.so.3.2.0 > "$CONTAINER_ROOT/provenance/native.sha256"
touch "$CONTAINER_ROOT/provenance/build-ready"
