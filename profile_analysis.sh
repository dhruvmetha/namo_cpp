#!/bin/bash

# Quick profiling analysis script for test_iterative_mpc
# Focuses on identifying performance bottlenecks

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

echo "=== NAMO Iterative MPC Performance Analysis ==="
echo ""

# Check if build exists
if [[ ! -f "build/test_iterative_mpc" ]]; then
    echo "Building test_iterative_mpc with profiling..."
    ./profile_iterative_mpc.sh --build-only --profiler gprof
fi

cd build

echo "Running performance analysis..."
echo ""

# Method 1: Simple timing with time command
echo "1. Basic timing analysis:"
echo "========================"
echo "Running test with time command..."
time ./test_iterative_mpc 2>&1 | tail -10
echo ""

# Method 2: gprof analysis if available
if [[ -f "gmon.out" ]] || ./test_iterative_mpc &> /dev/null; then
    echo "2. gprof function profiling:"
    echo "============================"
    
    if [[ ! -f "gmon.out" ]]; then
        echo "Generating profile data..."
        timeout 60 ./test_iterative_mpc &> /dev/null || true
    fi
    
    if [[ -f "gmon.out" ]]; then
        echo "Top functions by cumulative time:"
        gprof ./test_iterative_mpc gmon.out 2>/dev/null | grep "^[[:space:]]*[0-9]" | head -15
        echo ""
        
        echo "Call graph summary (top functions):"
        gprof ./test_iterative_mpc gmon.out 2>/dev/null | grep -A 10 "Call graph" | grep "^[[:space:]]*[0-9]" | head -10
    else
        echo "No profile data generated (program may have exited early)"
    fi
else
    echo "2. Cannot run gprof analysis - executable issues"
fi

echo ""

# Method 3: Memory usage analysis
echo "3. Memory usage analysis:"
echo "========================="
echo "Checking for memory leaks and usage patterns..."

# Run with valgrind if available (brief run)
if command -v valgrind &> /dev/null; then
    echo "Running brief valgrind check..."
    timeout 30 valgrind --tool=massif --detailed-freq=1 --max-snapshots=10 ./test_iterative_mpc &> valgrind_output.txt || true
    
    if [[ -f "valgrind_output.txt" ]]; then
        echo "Memory usage summary:"
        grep -E "(ERROR|LEAK|heap usage)" valgrind_output.txt | head -5
    fi
else
    echo "Valgrind not available - skipping memory analysis"
fi

echo ""

# Method 4: System resource usage
echo "4. System resource monitoring:"
echo "============================="
echo "Running with system resource monitoring..."

# Create a background monitor
monitor_resources() {
    local pid=$1
    local output_file=$2
    
    echo "timestamp,cpu_percent,memory_mb,io_read_mb,io_write_mb" > "$output_file"
    
    while kill -0 "$pid" 2>/dev/null; do
        local timestamp=$(date +%s.%3N)
        local stats=$(ps -p "$pid" -o %cpu,rss --no-headers 2>/dev/null || echo "0 0")
        local cpu_percent=$(echo "$stats" | awk '{print $1}')
        local memory_mb=$(echo "$stats" | awk '{print $2/1024}')
        
        # Get I/O stats if available
        local io_stats="0 0"
        if [[ -f "/proc/$pid/io" ]]; then
            io_stats=$(awk '/read_bytes|write_bytes/ {sum+=$2} END {print sum/1024/1024 " " sum/1024/1024}' "/proc/$pid/io" 2>/dev/null || echo "0 0")
        fi
        
        echo "$timestamp,$cpu_percent,$memory_mb,$io_stats" >> "$output_file"
        sleep 0.1
    done
}

# Run test with resource monitoring
echo "Starting resource monitoring..."
./test_iterative_mpc &
TEST_PID=$!

monitor_resources $TEST_PID "resource_usage.csv" &
MONITOR_PID=$!

# Wait for test to complete (with timeout)
timeout 120 wait $TEST_PID 2>/dev/null || {
    echo "Test timed out after 120 seconds"
    kill $TEST_PID 2>/dev/null || true
}

# Stop monitoring
kill $MONITOR_PID 2>/dev/null || true
wait $MONITOR_PID 2>/dev/null || true

# Analyze resource usage
if [[ -f "resource_usage.csv" ]] && [[ $(wc -l < "resource_usage.csv") -gt 1 ]]; then
    echo ""
    echo "Resource usage summary:"
    echo "Max CPU usage: $(tail -n +2 resource_usage.csv | cut -d',' -f2 | sort -n | tail -1)%"
    echo "Max memory usage: $(tail -n +2 resource_usage.csv | cut -d',' -f3 | sort -n | tail -1) MB"
    echo "Average CPU usage: $(tail -n +2 resource_usage.csv | cut -d',' -f2 | awk '{sum+=$1; n++} END {if(n>0) print sum/n "%"; else print "0%"}')"
    echo "Average memory usage: $(tail -n +2 resource_usage.csv | cut -d',' -f3 | awk '{sum+=$1; n++} END {if(n>0) print sum/n " MB"; else print "0 MB"}')"
else
    echo "No resource usage data collected"
fi

echo ""
echo "=== Analysis Complete ==="
echo "Files generated:"
echo "  - resource_usage.csv (system resource usage)"
if [[ -f "gmon.out" ]]; then
    echo "  - gmon.out (gprof profile data)"
fi
if [[ -f "valgrind_output.txt" ]]; then
    echo "  - valgrind_output.txt (memory analysis)"
fi

echo ""
echo "Recommendations for optimization:"
echo "1. Check gprof output for functions consuming most CPU time"
echo "2. Look for memory allocation patterns in valgrind output"  
echo "3. Monitor resource_usage.csv for memory leaks or CPU spikes"
echo "4. Consider adding custom timing instrumentation to specific functions"