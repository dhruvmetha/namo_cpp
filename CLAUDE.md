# CLAUDE.md - NAMO Standalone Project

This file provides guidance to Claude Code (claude.ai/code) when working with the NAMO (Navigation Among Movable Obstacles) standalone codebase.

## Project Overview

This is a high-performance standalone implementation of NAMO planning, completely disconnected from the PRX library. The system focuses on wavefront planning with zero-allocation runtime performance for robotic navigation among movable rectangular objects.

## Key Features

- **Wavefront Planning**: Fast BFS-based reachability computation from robot position
- **Zero-Allocation Runtime**: Pre-allocated memory pools and fixed-size containers
- **MuJoCo Integration**: Direct MuJoCo API without abstraction layers
- **Region-Based Planning**: Multi-step global optimization with spatial reasoning
- **Skill System**: Universal interface for high-level planners

## Current Status

### Completed Components
- ✅ Core infrastructure (MuJoCo wrapper, parameter loader, memory manager)
- ✅ Wavefront planner with full recomputation approach
- ✅ Motion primitive system with universal displacement vectors
- ✅ MPC executor with two-stage planning architecture
- ✅ **NAMO Skill System**: Complete skill-based interface for high-level planners
- ✅ **Region-Based High-Level Planner**: Sophisticated spatial reasoning with global optimization
- ✅ Comprehensive testing suite (15+ test executables)

### Pending Tasks (High Priority)
- 🔧 **CRITICAL BUG**: Parameter conversion error in main.cpp:50 - `has_key` method incorrectly returns true for non-existent keys, causing boolean conversion to fail with "bad conversion" error
- 📁 Create minimal test scene XML file (data/test_scene.xml)
- 🧪 Test basic functionality with test scene

### Pending Tasks (Medium Priority)
- 📊 Implement data collection and ZMQ communication features
- 🧠 Add ML integration for object selection strategies
- 📈 Performance benchmarking against legacy PRX system

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

Detailed architecture documentation available in: [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md)

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
- **Fast Recomputation**: Efficient BFS algorithm for each wavefront update
- **Grid Resolution**: Configurable (default 0.05m for high precision)
- **Object Limits**: 20 static + 10 movable objects per scene

## Development Notes

- **MuJoCo API Compliance**: All MuJoCo function calls verified against official documentation
- **No PRX Dependencies**: Completely standalone implementation
- **Fixed-Size Containers**: Template-based for compile-time optimization
- **Full Wavefront Rebuild**: Simple and reliable approach for each update
- **RAII Memory Management**: Automatic cleanup with object pools

## Development Workflow

- Always build in debug mode

## NAMO Skill System

Complete skill system documentation available in: `docs/SKILL_USAGE_GUIDE.md`

### Key Features
- **Universal Interface**: Works with any high-level planner
- **Type Safety**: Compile-time parameter validation
- **Complete Lifecycle**: `is_applicable()` → `check_preconditions()` → `execute()`
- **Integration Patterns**: PDDL, Behavior Trees, RL Policies, Task Planning

### Usage Example
```cpp
#include "skills/namo_push_skill.hpp"

NAMOEnvironment env("scene.xml", false);
NAMOPushSkill skill(env);

std::map<std::string, SkillParameterValue> params = {
    {"object_name", std::string("box_1")},
    {"target_pose", SE2State(2.0, 1.5, 0.0)}
};

if (skill.is_applicable(params)) {
    auto result = skill.execute(params);
    if (result.success) {
        std::cout << "Success!" << std::endl;
    }
}
```

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
│   ├── environment/
│   │   └── namo_environment.hpp
│   └── skills/
│       ├── manipulation_skill.hpp # Abstract skill interface
│       └── namo_push_skill.hpp    # NAMO push skill implementation
├── src/
│   ├── core/                   # Core components implementation
│   ├── planning/               # Incremental wavefront planning
│   ├── environment/            # NAMO environment management
│   ├── skills/                 # Skill system implementation
│   └── main.cpp               # Main executable with testing
├── tests/
│   ├── test_namo_skill.cpp     # Skill system validation tests
│   └── test_simple_skill.cpp   # Basic skill interface tests
├── docs/
│   ├── SKILL_USAGE_GUIDE.md    # Complete skill usage documentation
│   └── SKILL_SYSTEM_SUMMARY.md # Implementation summary and achievements
├── examples/
│   └── skill_demo.cpp          # Working demonstration with integration examples
├── python/                     # Python package and bindings
│   ├── namo/                  # Main NAMO Python package
│   │   ├── core/              # Core interfaces (BasePlanner, xml_goal_parser)
│   │   ├── config/            # Configuration systems (MCTSConfig)
│   │   ├── strategies/        # Selection strategies (shared)
│   │   ├── planners/          # Planning algorithms
│   │   │   ├── idfs/         # Iterative Deepening algorithms
│   │   │   ├── mcts/         # Monte Carlo Tree Search
│   │   │   └── sampling/     # Sampling-based planners
│   │   ├── data_collection/   # Data collection workflows
│   │   ├── visualization/     # Image/mask processing
│   │   └── cpp_bindings/      # C++ interface files
│   ├── scripts/               # Standalone executables
│   └── README.md             # Python package documentation
├── config/
│   ├── namo_config.yaml       # Full configuration
│   └── simple_test.yaml       # Minimal test config
└── build/                     # Build output directory
```

## Documentation References

**Detailed Technical Documentation:**
- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) - Complete architecture overview and component details
- [`docs/LEGACY_ANALYSIS.md`](docs/LEGACY_ANALYSIS.md) - Analysis of PRX-based original implementation
- [`docs/IMPLEMENTATION_STATUS.md`](docs/IMPLEMENTATION_STATUS.md) - Current implementation status and capabilities
- [`docs/REGION_PLANNER.md`](docs/REGION_PLANNER.md) - Region-based high-level planner documentation

**Skill System Documentation:**
- `docs/SKILL_USAGE_GUIDE.md` - Complete skill usage guide with API reference
- `docs/SKILL_SYSTEM_SUMMARY.md` - Implementation summary and achievements

## Next Steps

1. **Fix Parameter Loader Bug**: Debug `has_key` method returning incorrect results (CRITICAL)
2. **Integration Testing**: Validate full system with complex multi-object scenes (including region-based planner)
3. **Performance Benchmarking**: Systematic comparison between region-based planner and legacy implementation
4. **ML Integration**: Implement ZMQ communication for distributed inference  
5. **Action Optimization**: Add exhaustive subset search for minimal sequences
6. **Region Planner Enhancement**: Add configuration section to ConfigManager for region planner parameters