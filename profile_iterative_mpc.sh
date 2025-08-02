#!/bin/bash

# Profile script for test_iterative_mpc
# Supports both gprof and perf profiling

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

# Default options
PROFILER="gprof"
BUILD_TYPE="Profile"
RUN_TEST=true
CLEAN_BUILD=false

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --profiler)
            PROFILER="$2"
            shift 2
            ;;
        --build-only)
            RUN_TEST=false
            shift
            ;;
        --clean)
            CLEAN_BUILD=true
            shift
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo "Options:"
            echo "  --profiler {gprof|perf}  Choose profiler (default: gprof)"
            echo "  --build-only             Only build, don't run test"
            echo "  --clean                  Clean build directory first"
            echo "  --help                   Show this help"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "=== NAMO Iterative MPC Profiling Setup ==="
echo "Profiler: $PROFILER"
echo "Build type: $BUILD_TYPE"
echo "Run test: $RUN_TEST"
echo ""

# Check if MuJoCo is available
if [[ -z "${MJ_PATH}" ]]; then
    echo "Error: MJ_PATH environment variable not set"
    echo "Please set it to your MuJoCo installation directory"
    exit 1
fi

echo "Using MuJoCo from: $MJ_PATH"

# Clean build if requested
if [[ "$CLEAN_BUILD" == "true" ]]; then
    echo "Cleaning build directory..."
    rm -rf build
fi

# Create build directory
mkdir -p build
cd build

# Configure with profiling
echo "Configuring build with profiling..."
if [[ "$PROFILER" == "perf" ]]; then
    # For perf, use debug symbols but no gprof flags
    cmake .. -DCMAKE_BUILD_TYPE=Debug -DCMAKE_CXX_FLAGS_DEBUG="-O2 -g -fno-omit-frame-pointer -DNDEBUG"
else
    # For gprof, use the Profile build type
    cmake .. -DCMAKE_BUILD_TYPE=Profile
fi

# Build the test
echo "Building test_iterative_mpc..."
cmake --build . --target test_iterative_mpc --parallel $(nproc)

if [[ "$RUN_TEST" == "false" ]]; then
    echo "Build complete. Executable: $(pwd)/test_iterative_mpc"
    exit 0
fi

# Check if required data files exist
REQUIRED_FILES=(
    "../data/nominal_primitive_scene.xml"
    "../data/motion_primitives.dat"
)

for file in "${REQUIRED_FILES[@]}"; do
    if [[ ! -f "$file" ]]; then
        echo "Warning: Required file not found: $file"
        echo "The test may fail to run."
    fi
done

echo ""
echo "=== Running Profiling Test ==="

if [[ "$PROFILER" == "perf" ]]; then
    echo "Using perf profiler..."
    
    # Check if perf is available
    if ! command -v perf &> /dev/null; then
        echo "Error: perf not found. Install with: sudo apt-get install linux-tools-generic"
        exit 1
    fi
    
    # Run with perf record
    echo "Running test with perf record..."
    perf record -g -o iterative_mpc.perf.data ./test_iterative_mpc
    
    echo ""
    echo "=== Profiling Results ==="
    echo "Perf data saved to: iterative_mpc.perf.data"
    echo ""
    echo "To analyze results:"
    echo "  perf report -i iterative_mpc.perf.data"
    echo "  perf annotate -i iterative_mpc.perf.data"
    echo "  perf script -i iterative_mpc.perf.data > profile.txt"
    
    # Generate a quick report
    echo ""
    echo "Quick performance summary:"
    perf report -i iterative_mpc.perf.data --stdio | head -20
    
elif [[ "$PROFILER" == "gprof" ]]; then
    echo "Using gprof profiler..."
    
    # Run the test to generate gmon.out
    echo "Running test to generate profile data..."
    ./test_iterative_mpc
    
    # Check if gmon.out was generated
    if [[ ! -f "gmon.out" ]]; then
        echo "Error: gmon.out not generated. Make sure the program ran to completion."
        exit 1
    fi
    
    echo ""
    echo "=== Profiling Results ==="
    echo "Profile data saved to: gmon.out"
    echo ""
    
    # Generate profile report
    echo "Generating gprof report..."
    gprof ./test_iterative_mpc gmon.out > iterative_mpc_profile.txt
    
    echo "Full profile report saved to: iterative_mpc_profile.txt"
    echo ""
    echo "Top 10 functions by time:"
    gprof ./test_iterative_mpc gmon.out | grep "^[[:space:]]*[0-9]" | head -10
    
else
    echo "Error: Unknown profiler: $PROFILER"
    exit 1
fi

echo ""
echo "=== Profiling Complete ==="
echo "Build directory: $(pwd)"
echo "Executable: $(pwd)/test_iterative_mpc"

if [[ "$PROFILER" == "gprof" ]]; then
    echo "Profile report: $(pwd)/iterative_mpc_profile.txt"
    echo "Raw data: $(pwd)/gmon.out"
else
    echo "Profile data: $(pwd)/iterative_mpc.perf.data"
fi