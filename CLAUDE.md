# CLAUDE.md - NAMO Standalone Project

This file provides guidance to Claude Code (claude.ai/code) when working with the NAMO (Navigation Among Movable Obstacles) standalone codebase.

## Project Overview

This is a high-performance standalone implementation of NAMO planning, completely disconnected from the PRX library. The system focuses on incremental wavefront planning with zero-allocation runtime performance for robotic navigation among movable rectangular objects.

## Key Features

- **Incremental Wavefront Planning**: Avoids full recomputation by detecting changes in object positions/rotations
- **Zero-Allocation Runtime**: Pre-allocated memory pools and fixed-size containers
- **MuJoCo Integration**: Direct MuJoCo API without abstraction layers
- **High-Performance I/O**: Pre-allocated buffers for data collection and logging
- **Combined Motion Handling**: Supports both rotation and translation of rectangular objects

## Current Status

### Completed Components
- ✅ Project structure and CMake build system
- ✅ Core components (MuJoCo wrapper, parameter loader, memory manager)
- ✅ Incremental wavefront planner with change detection
- ✅ NAMO environment with object management
- ✅ Fixed-size container system (FixedVector template)
- ✅ Build system with automatic MuJoCo dependency handling

### Pending Tasks (High Priority)
- 🔧 **CRITICAL BUG**: Parameter conversion error in main.cpp:50 - `has_key` method incorrectly returns true for non-existent keys, causing boolean conversion to fail with "bad conversion" error
- 📁 Create minimal test scene XML file (data/test_scene.xml)
- 🧪 Test basic functionality with test scene

### Pending Tasks (Medium Priority)
- 🤖 Implement NAMO push controller and motion primitives
- 📊 Implement data collection and ZMQ communication features
- ✅ Add comprehensive testing and validation

### Pending Tasks (Low Priority)
- ⚡ Performance optimization and memory pool tuning

## Key Commands

### Build System
```bash
# Configure build (auto-detects MuJoCo)
cmake -B build -DCMAKE_BUILD_TYPE=Release

# Build project
cmake --build build --parallel 8

# Run basic test
./build/namo_standalone config/simple_test.yaml
```

### Environment Setup
```bash
# MuJoCo path (auto-detected)
export MJ_PATH=/path/to/mujoco

# Required packages (no sudo needed)
# - cmake, build-essential, libyaml-cpp-dev, libglfw3-dev, libgl1-mesa-dev
```

## Architecture Overview

### Core Components

**Memory Management (`src/core/memory_manager.cpp`)**
- Pre-allocated object pools for zero-allocation runtime
- RAII memory management with automatic cleanup
- Performance statistics and monitoring

**MuJoCo Wrapper (`src/core/mujoco_wrapper.cpp`)**
- Direct MuJoCo API integration (verified against official documentation)
- Visualization support with GLFW
- Body position/rotation access without abstraction layers

**Parameter Loader (`src/core/parameter_loader.cpp`)**
- YAML-cpp integration with fallback to simple parser
- Hierarchical key access (e.g., "wavefront_planner.resolution")
- **KNOWN BUG**: `has_key` method returns incorrect results

**Incremental Wavefront Planner (`src/planning/incremental_wavefront_planner.cpp`)**
- Change detection for rotating/translating objects
- Grid footprint tracking and differential updates
- Pre-allocated BFS queue (MAX_BFS_QUEUE = 100,000)

**NAMO Environment (`src/environment/namo_environment.cpp`)**
- Fixed-size object storage (MAX_STATIC_OBJECTS = 20, MAX_MOVABLE_OBJECTS = 10)
- Object state tracking and bounds calculation
- High-performance logging with pre-allocated buffers

### Data Structures

**Fixed-Size Containers (`include/core/types.hpp`)**
```cpp
template<size_t MAX_SIZE>
class FixedVector {
private:
    std::array<double, MAX_SIZE> data_;
    size_t size_ = 0;
public:
    void push_back(double val) { assert(size_ < MAX_SIZE); data_[size_++] = val; }
    // ... other methods for zero-allocation performance
};
```

**Object Representation**
```cpp
struct ObjectInfo {
    std::string name;
    std::array<double, 3> position;    // x, y, z
    std::array<double, 4> quaternion;  // w, x, y, z
    std::array<double, 3> size;        // width, height, depth
    bool is_movable;
};
```

## Configuration System

Uses YAML configuration with fallback to simple key=value parser:

**Main Config (`config/namo_config.yaml`)**
- Environment settings (XML path, visualization)
- Planning parameters (resolution, thresholds)
- Memory limits and performance tuning
- Data collection and ZMQ settings

**Simple Test Config (`config/simple_test.yaml`)**
- Minimal configuration for basic testing
- Currently missing visualize key (causes parameter bug)

## Critical Issues

### Parameter Loader Bug
**Location**: `src/core/parameter_loader.cpp:91` (`has_key` method)
**Symptom**: "bad conversion" error when loading boolean values
**Cause**: `has_key` returns true for non-existent keys, causing `get_bool` to fail
**Impact**: Prevents executable from running
**Priority**: CRITICAL - blocks all testing

### Missing Test Scene
**Location**: `data/test_scene.xml`
**Status**: File referenced in config but doesn't exist
**Impact**: Environment initialization will fail after parameter bug is fixed
**Priority**: HIGH - needed for basic testing

## Performance Targets

- **Zero Runtime Allocation**: All memory pre-allocated during initialization
- **Incremental Updates**: 10-100x speedup over full wavefront recomputation
- **Grid Resolution**: Configurable (default 0.05m for high precision)
- **Object Limits**: 20 static + 10 movable objects per scene

## Development Notes

- **MuJoCo API Compliance**: All MuJoCo function calls verified against official documentation
- **No PRX Dependencies**: Completely standalone implementation
- **Fixed-Size Containers**: Template-based for compile-time optimization
- **Change Detection**: Grid footprint comparison for efficient updates
- **RAII Memory Management**: Automatic cleanup with object pools

## File Structure

```
/common/home/dm1487/robotics_research/ktamp/namo/
├── CMakeLists.txt              # Main build configuration
├── include/
│   ├── core/
│   │   ├── types.hpp           # Fixed-size containers and core types
│   │   ├── memory_manager.hpp  # Zero-allocation memory pools
│   │   ├── mujoco_wrapper.hpp  # Direct MuJoCo API wrapper
│   │   └── parameter_loader.hpp # YAML configuration loader
│   ├── planning/
│   │   └── incremental_wavefront_planner.hpp
│   └── environment/
│       └── namo_environment.hpp
├── src/
│   ├── core/                   # Core components implementation
│   ├── planning/               # Incremental wavefront planning
│   ├── environment/            # NAMO environment management
│   └── main.cpp               # Main executable with testing
├── config/
│   ├── namo_config.yaml       # Full configuration
│   └── simple_test.yaml       # Minimal test config
└── build/                     # Build output directory
```

## Next Steps

1. **Fix Parameter Loader Bug**: Debug `has_key` method returning incorrect results
2. **Create Test Scene**: Generate minimal MuJoCo XML file for testing
3. **Basic Functionality Test**: Verify core components work together
4. **Implement Planning**: Add push controller and motion primitives
5. **Performance Optimization**: Tune memory pools and grid resolution