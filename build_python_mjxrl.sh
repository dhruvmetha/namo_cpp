#!/bin/bash
# Build script for NAMO Python bindings using mjxrl environment

set -e  # Exit on any error

echo "Building NAMO Python Bindings for mjxrl environment"
echo "=================================================="

PYTHON_ENV="/common/users/dm1487/envs/mjxrl"
PYTHON_BIN="$PYTHON_ENV/bin/python"

# Check if the Python environment exists
if [[ ! -f "$PYTHON_BIN" ]]; then
    echo "Error: Python environment not found at $PYTHON_ENV"
    echo "Please check the path to your mjxrl environment."
    exit 1
fi

# Check Python version
PYTHON_VERSION=$($PYTHON_BIN --version)
echo "Using Python: $PYTHON_VERSION"
echo "Python executable: $PYTHON_BIN"

# Check if MJ_PATH is set
if [[ -z "${MJ_PATH}" ]]; then
    echo "Error: MJ_PATH environment variable is not set."
    echo "Please set it to your MuJoCo installation directory:"
    echo "export MJ_PATH=/path/to/mujoco"
    exit 1
fi

echo "Using MuJoCo from: ${MJ_PATH}"

# Create build directory
BUILD_DIR="build_python_mjxrl_${HOSTNAME%%.*}"
if [[ -d "$BUILD_DIR" ]]; then
    echo "Removing existing build directory..."
    rm -rf "$BUILD_DIR"
fi

mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

echo "Configuring CMake for mjxrl Python..."

# Get Python include and library paths
PYTHON_INCLUDE=$($PYTHON_BIN -c "import sysconfig; print(sysconfig.get_path('include'))")
PYTHON_LIBRARY_DIR=$($PYTHON_BIN -c "import sysconfig; print(sysconfig.get_config_var('LIBDIR'))")
PYTHON_VERSION=$($PYTHON_BIN -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")

echo "Python include path: $PYTHON_INCLUDE"
echo "Python library dir: $PYTHON_LIBRARY_DIR"
echo "Python version: $PYTHON_VERSION"

cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_PYTHON_BINDINGS=ON \
    -DPython3_EXECUTABLE="$PYTHON_BIN" \
    -DPython3_INCLUDE_DIR="$PYTHON_INCLUDE" \
    -DPython3_LIBRARY_RELEASE="$PYTHON_LIBRARY_DIR/libpython$PYTHON_VERSION.so"

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
echo "2. Activate your mjxrl environment and run:"
echo "   source $PYTHON_ENV/bin/activate"
echo "   cd ../python"
echo "   python demo_visualization.py"
echo ""
echo "Or run the test directly:"
echo "   cd ../python && PYTHONPATH=$(pwd) $PYTHON_BIN demo_visualization.py"
echo ""