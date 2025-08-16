#!/bin/bash
# Build script for NAMO Python bindings

set -e  # Exit on any error

echo "Building NAMO Python Bindings"
echo "============================="

# Check if MJ_PATH is set
if [[ -z "${MJ_PATH}" ]]; then
    echo "Error: MJ_PATH environment variable is not set."
    echo "Please set it to your MuJoCo installation directory:"
    echo "export MJ_PATH=/path/to/mujoco"
    exit 1
fi

echo "Using MuJoCo from: ${MJ_PATH}"

# Create build directory
BUILD_DIR="build_python"
if [[ -d "$BUILD_DIR" ]]; then
    echo "Removing existing build directory..."
    rm -rf "$BUILD_DIR"
fi

mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

echo "Configuring CMake..."
cmake .. -DCMAKE_BUILD_TYPE=Debug -DBUILD_PYTHON_BINDINGS=ON

echo "Building..."
make -j$(nproc) namo_rl

echo ""
echo "Build completed successfully!"
echo ""
echo "The Python module 'namo_rl' has been built in: $(pwd)"
echo ""
echo "To test the module:"
echo "1. Add the build directory to your Python path:"
echo "   export PYTHONPATH=$(pwd):\$PYTHONPATH"
echo "2. Run the test script:"
echo "   cd ../python"
echo "   python test_rl_env.py"
echo ""
echo "Or run the test directly:"
echo "   cd ../python && PYTHONPATH=$(pwd) python test_rl_env.py"
