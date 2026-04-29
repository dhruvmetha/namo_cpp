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

cmake -S . -B build_python -DCMAKE_BUILD_TYPE=Release -DBUILD_PYTHON_BINDINGS=ON
cmake --build build_python --target namo_rl -j"$(nproc)"

echo
echo "Build completed successfully."
echo "Canonical PYTHONPATH usage:"
echo "  export PYTHONPATH=$(pwd)/build_python:\$PYTHONPATH"
