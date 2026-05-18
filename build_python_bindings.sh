#!/bin/bash
# Canonical build script for NAMO Python bindings.

set -euo pipefail

if [[ -z "${MJ_PATH:-}" ]]; then
    echo "Error: MJ_PATH environment variable is not set."
    echo "Set it to your MuJoCo installation directory, for example:"
    echo "  export MJ_PATH=/path/to/mujoco"
    exit 1
fi

echo "Building namo_rl (Release) into ./build_python"
echo "Using MuJoCo from: ${MJ_PATH}"

# Resolve the active Python and its SOABI explicitly. CMake's FindPython3
# sometimes leaves Python3_SOABI empty on certain CMake/Python combinations,
# which produces a broken module file like "namo_rl..so". Pass them through.
PYTHON_BIN="${PYTHON_BIN:-$(command -v python3)}"
PYTHON_INCLUDE="$("$PYTHON_BIN" -c "import sysconfig; print(sysconfig.get_path('include'))")"
PYTHON_LIBDIR="$("$PYTHON_BIN" -c "import sysconfig; print(sysconfig.get_config_var('LIBDIR'))")"
PYTHON_LIBVER="$("$PYTHON_BIN" -c "import sysconfig; print(sysconfig.get_config_var('LDVERSION') or sysconfig.get_config_var('VERSION'))")"
PYTHON_SOABI="$("$PYTHON_BIN" -c "import sysconfig; print(sysconfig.get_config_var('SOABI'))")"
echo "Using Python: $($PYTHON_BIN --version) ($PYTHON_BIN), SOABI=$PYTHON_SOABI"

cmake -S . -B build_python -DCMAKE_BUILD_TYPE=Release -DBUILD_PYTHON_BINDINGS=ON \
    -DPython3_EXECUTABLE="$PYTHON_BIN" \
    -DPython3_INCLUDE_DIR="$PYTHON_INCLUDE" \
    -DPython3_LIBRARY_RELEASE="$PYTHON_LIBDIR/libpython$PYTHON_LIBVER.so" \
    -DPython3_SOABI="$PYTHON_SOABI"
cmake --build build_python --target namo_rl -j"$(nproc)"

echo
echo "Build completed successfully."
echo "Canonical PYTHONPATH usage:"
echo "  export PYTHONPATH=$(pwd)/build_python:\$PYTHONPATH"
