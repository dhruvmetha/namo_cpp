#!/bin/bash
# Canonical build script for NAMO Python bindings.
#
# Builds into ./build_python/ — the location every consumer in the repo
# expects (see python/namo/core/binding_loader.py and
# python/scripts/check_canonical_binding_policy.py).

set -euo pipefail

if [[ -z "${MJ_PATH:-}" ]]; then
    echo "Error: MJ_PATH environment variable is not set."
    echo "Set it to your MuJoCo installation directory, for example:"
    echo "  export MJ_PATH=/path/to/mujoco"
    exit 1
fi

BUILD_DIR="build_python"

# Architecture target. Defaults to "native" for local dev (fastest on the
# build host). Override to a portable baseline like "x86-64-v3" when the
# resulting .so must run on a heterogeneous fleet — e.g. Amarel shards
# scattered across skylake -> emeraldrapids:
#   NAMO_MARCH=x86-64-v3 ./build_python_bindings.sh
NAMO_MARCH="${NAMO_MARCH:-native}"

echo "Building namo_rl (Release) into ./${BUILD_DIR}"
echo "Using MuJoCo from: ${MJ_PATH}"
echo "Target arch (-march): ${NAMO_MARCH}"

# Resolve the active Python and its SOABI explicitly. CMake's FindPython3
# sometimes leaves Python3_SOABI empty on certain CMake/Python combinations,
# which produces a broken module file like "namo_rl..so". Pass them through.
PYTHON_BIN="${PYTHON_BIN:-$(command -v python3)}"
PYTHON_INCLUDE="$("$PYTHON_BIN" -c "import sysconfig; print(sysconfig.get_path('include'))")"
PYTHON_LIBDIR="$("$PYTHON_BIN" -c "import sysconfig; print(sysconfig.get_config_var('LIBDIR'))")"
PYTHON_LIBVER="$("$PYTHON_BIN" -c "import sysconfig; print(sysconfig.get_config_var('LDVERSION') or sysconfig.get_config_var('VERSION'))")"
PYTHON_SOABI="$("$PYTHON_BIN" -c "import sysconfig; print(sysconfig.get_config_var('SOABI'))")"
echo "Using Python: $($PYTHON_BIN --version) ($PYTHON_BIN), SOABI=$PYTHON_SOABI"

cmake -S . -B "$BUILD_DIR" -DCMAKE_BUILD_TYPE=Release -DBUILD_PYTHON_BINDINGS=ON \
    -DNAMO_MARCH="${NAMO_MARCH}" \
    -DPython3_EXECUTABLE="$PYTHON_BIN" \
    -DPython3_INCLUDE_DIR="$PYTHON_INCLUDE" \
    -DPython3_LIBRARY_RELEASE="$PYTHON_LIBDIR/libpython$PYTHON_LIBVER.so" \
    -DPython3_SOABI="$PYTHON_SOABI"
cmake --build "$BUILD_DIR" --target namo_rl -j"$(nproc)"

# Stamp what this .so was actually built from. Timestamps cannot answer "does this binary contain
# commit X" -- a build-then-commit workflow leaves the .so mtime EARLIER than the commit that
# describes it, and a stale checkout leaves it later than nothing at all. The C++ tree hash is exact:
# scripts/check_box_sync.sh compares it against HEAD's and refuses to launch a campaign on a mismatch.
# Motivated by 2026-08-08: an Amarel .so from July 2 predated d6088d0 (the sticky-collision fix to
# namo_push_controller.hpp), so a collection there would have run different push physics than CS with
# nothing in the output to reveal it.
# SLURM compute nodes may have no `git` on PATH (verified on Amarel main-redhat 2026-08-08), so the
# submitting host can pass these in: NAMO_BUILD_SHA / NAMO_BUILD_CPP_TREE / NAMO_BUILD_SRC_TREE /
# NAMO_BUILD_DIRTY. Local `git` is used when available and the env is unset.
{
  echo "git_sha=${NAMO_BUILD_SHA:-$(git rev-parse HEAD 2>/dev/null || echo unknown)}"
  echo "cpp_tree=${NAMO_BUILD_CPP_TREE:-$(git rev-parse HEAD:include 2>/dev/null || echo unknown)}"
  echo "src_tree=${NAMO_BUILD_SRC_TREE:-$(git rev-parse HEAD:src 2>/dev/null || echo unknown)}"
  echo "dirty_cpp=${NAMO_BUILD_DIRTY:-$(git status --porcelain -- include/ src/ cpp_bindings/ 2>/dev/null | wc -l)}"
  echo "built_at=$(date -Is)"
  echo "host=$(hostname)"
  echo "python=$($PYTHON_BIN --version 2>&1)"
  echo "march=${NAMO_MARCH}"
  echo "mj_path=${MJ_PATH:-unset}"
} > "$BUILD_DIR/BUILD_INFO"

echo
echo "Build completed successfully."
echo "Stamped $BUILD_DIR/BUILD_INFO:"; sed "s/^/  /" "$BUILD_DIR/BUILD_INFO"
echo "Canonical PYTHONPATH usage:"
echo "  export PYTHONPATH=$(pwd)/${BUILD_DIR}:\$PYTHONPATH"
