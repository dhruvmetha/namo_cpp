# NAMO Tests - Quick Reference

## 🏃‍♂️ **Skill System Tests**
- `test_simple_skill.cpp` - Basic skill interface validation
- `test_namo_skill.cpp` - Comprehensive skill testing with preconditions
- `skill_demo.cpp` - Integration demonstration with PDDL/BT/RL examples + goal visualization
- `test_skill_benchmark.cpp` - Performance testing with benchmark environments

## 🔧 **Core Component Tests**
- `core/test_primitives_only.cpp` - Motion primitive loading and greedy planning
- `core/test_coordinate_transformations.cpp` - Coordinate system and geometry validation
- `core/test_search_limits.cpp` - Planning algorithm resource constraint testing
- `core/test_visual_markers.cpp` - MuJoCo visualization and debug markers

## 🎯 **Planning System Tests**
- `planning/test_planner_output.cpp` - Planning output validation and correctness
- `planning/test_mpc_executor.cpp` - Model Predictive Control component testing
- `planning/test_iterative_mpc.cpp` - Comparison of full-sequence vs iterative MPC approaches

## 🗺️ **Region-Based Planning Tests**
- `region/test_region_basic.cpp` - Region data structures and spatial representation
- `region/test_region_planner.cpp` - High-level region-based planner functionality
- `region/test_region_integration.cpp` - Region planner ↔ skill system integration
- `region/test_region_path.cpp` - Path planning algorithms within region framework
- `region/test_region_minimal.cpp` - Minimal region planner for quick validation

## 🔄 **Integration Tests**
- `integration/test_end_to_end.cpp` - Complete two-stage planning pipeline validation
- `integration/test_complete_planning.cpp` - Full system integration testing

## 📊 **Benchmarks & Analysis**
- `benchmarks/benchmark_region_planner.cpp` - Performance benchmarking and statistics

## 🚀 **Quick Commands**
```bash
# Build all tests
make -j8

# Run skill tests
./build/test_simple_skill && ./build/test_namo_skill && ./build/skill_demo

# Run core tests  
./build/test_primitives_only data/motion_primitives.dat
./build/test_coordinate_transformations

# Run planning tests
./build/test_end_to_end
./build/test_iterative_mpc

# Run region tests
./build/test_region_basic
./build/test_region_integration

# Performance analysis
./build/benchmark_region_planner
```