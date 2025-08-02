# Profiling Guide for test_iterative_mpc

This guide explains how to profile the `test_iterative_mpc` executable to identify performance bottlenecks and optimize the NAMO iterative MPC execution.

## Quick Start

### 1. Basic Profiling
```bash
# Build and run with gprof profiling
./profile_iterative_mpc.sh

# Run comprehensive analysis
./profile_analysis.sh
```

### 2. Advanced Options
```bash
# Use perf instead of gprof
./profile_iterative_mpc.sh --profiler perf

# Build only (don't run test)
./profile_iterative_mpc.sh --build-only

# Clean build
./profile_iterative_mpc.sh --clean
```

## Profiling Methods

### Method 1: gprof (Function-level profiling)
```bash
./profile_iterative_mpc.sh --profiler gprof
```

**Outputs:**
- `build/gmon.out` - Raw profile data
- `build/iterative_mpc_profile.txt` - Human-readable report

**Best for:** Finding which functions consume the most CPU time

### Method 2: perf (System-level profiling)
```bash
./profile_iterative_mpc.sh --profiler perf
```

**Outputs:**
- `build/iterative_mpc.perf.data` - Raw profile data

**Analysis commands:**
```bash
cd build
perf report -i iterative_mpc.perf.data          # Interactive report
perf annotate -i iterative_mpc.perf.data        # Source code annotation
perf script -i iterative_mpc.perf.data > profile.txt  # Export to text
```

**Best for:** Hardware-level analysis, cache misses, branch prediction

### Method 3: Comprehensive Analysis
```bash
./profile_analysis.sh
```

**Outputs:**
- `build/resource_usage.csv` - CPU/memory usage over time
- `build/valgrind_output.txt` - Memory leak analysis (if valgrind available)
- Timing analysis and performance summaries

**Best for:** Overall performance overview and resource usage patterns

## Build Configuration

The profiling setup adds a custom `Profile` build type to CMake:

```cmake
set(CMAKE_CXX_FLAGS_PROFILE "-O2 -g -pg -fno-omit-frame-pointer -DNDEBUG")
```

**Build manually:**
```bash
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Profile
cmake --build . --target test_iterative_mpc --parallel $(nproc)
```

## Key Performance Areas to Monitor

### 1. Wavefront Planning
- `IncrementalWavefrontPlanner::update_wavefront()`
- Grid computation and pathfinding algorithms
- Memory allocation in BFS queue

### 2. MuJoCo Integration
- `NAMOEnvironment::get_object_state()`
- `MuJoCoWrapper::step()`
- Physics simulation overhead

### 3. Push Controller
- `NAMOPushController::get_reachable_edge_indices()`
- Collision checking and reachability computation
- Motion primitive execution

### 4. Memory Management
- Fixed-size container performance
- Object pool allocation patterns
- Memory fragmentation

## Interpreting Results

### gprof Output
```
%   cumulative   self              self     total           
time   seconds   seconds    calls  ms/call  ms/call  name    
30.45      0.45     0.45       100     4.50     8.20  IncrementalWavefrontPlanner::update_wavefront()
25.32      0.82     0.37       50      7.40    12.30  NAMOPushController::get_reachable_edge_indices()
```

- **% time**: Percentage of total execution time
- **self seconds**: Time spent in this function only
- **calls**: Number of function calls
- **ms/call**: Average time per call

### Resource Usage Analysis
```csv
timestamp,cpu_percent,memory_mb,io_read_mb,io_write_mb
1659123456.123,85.5,234.2,0.1,0.0
```

- Monitor for CPU spikes during specific operations
- Check for memory leaks (steadily increasing memory usage)
- Look for I/O bottlenecks

## Optimization Strategies

### 1. Function-level Optimization
- Target functions with highest % time in gprof
- Look for unnecessary repeated computations
- Optimize hot loops and frequently called functions

### 2. Memory Optimization
- Check for excessive allocations in frequently called functions
- Optimize container usage (prefer fixed-size containers)
- Minimize memory fragmentation

### 3. Algorithm Optimization
- Improve wavefront planning algorithms
- Optimize collision checking
- Reduce redundant state updates

### 4. System-level Optimization
- Compiler optimization flags
- Memory pool tuning
- Parallel processing opportunities

## Common Performance Issues

### High CPU Usage in Wavefront Planning
```cpp
// Problem: Full recomputation every iteration
wavefront_planner.compute_full_wavefront();

// Solution: Use incremental updates
wavefront_planner.update_wavefront_incremental(changed_objects);
```

### Memory Allocation Hotspots
```cpp
// Problem: Dynamic allocation in hot loop
std::vector<int> temp_data(size);  // Called frequently

// Solution: Pre-allocate containers
static thread_local std::vector<int> temp_data;
temp_data.resize(size);
```

### Excessive MuJoCo Calls
```cpp
// Problem: Getting object state multiple times
auto state1 = env.get_object_state(name);
auto state2 = env.get_object_state(name);  // Redundant

// Solution: Cache state within iteration
auto cached_state = env.get_object_state(name);
```

## Integration with Development Workflow

### 1. Regular Performance Testing
```bash
# Add to CI/CD pipeline
./profile_analysis.sh > performance_report.txt
```

### 2. Performance Regression Detection
```bash
# Compare against baseline
./profile_iterative_mpc.sh --profiler gprof
gprof build/test_iterative_mpc build/gmon.out | grep "total runtime" > current_perf.txt
diff baseline_perf.txt current_perf.txt
```

### 3. Development Guidelines
- Profile after major algorithm changes
- Monitor memory usage during development
- Use profiling to validate optimization hypotheses

## Troubleshooting

### No Profile Data Generated
- Ensure program runs to completion (not killed early)
- Check that Profile build type is used for gprof
- Verify MuJoCo environment is properly configured

### perf Permission Issues
```bash
# If perf fails with permission errors
echo 1 | sudo tee /proc/sys/kernel/perf_event_paranoid
```

### Missing Dependencies
```bash
# Install profiling tools
sudo apt-get install linux-tools-generic linux-tools-common
sudo apt-get install valgrind
```

## Files and Outputs

| File | Description |
|------|-------------|
| `profile_iterative_mpc.sh` | Main profiling script |
| `profile_analysis.sh` | Comprehensive analysis script |
| `build/gmon.out` | gprof raw data |
| `build/iterative_mpc_profile.txt` | gprof report |
| `build/iterative_mpc.perf.data` | perf raw data |
| `build/resource_usage.csv` | System resource monitoring |
| `build/valgrind_output.txt` | Memory analysis |

## Next Steps

1. Run initial profiling to establish baseline performance
2. Identify top 3 performance bottlenecks
3. Implement targeted optimizations
4. Re-profile to validate improvements
5. Repeat until performance targets are met