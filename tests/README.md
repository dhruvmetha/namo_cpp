# NAMO Test Suite Documentation

This directory contains all tests for the NAMO (Navigation Among Movable Obstacles) system. The tests are organized by functionality and provide comprehensive validation of the skill system.

## Test Categories

### 🎯 Skill System Tests

The skill system tests validate the NAMOPushSkill interface and functionality.

#### `test_simple_skill.cpp` - Basic Interface Validation
**Purpose:** Validates the fundamental skill interface without complex execution.

**What it tests:**
- ✅ Skill initialization and setup
- ✅ Basic metadata retrieval (`get_name()`, `get_description()`)
- ✅ Parameter schema validation
- ✅ Required parameter existence (`object_name`, `target_pose`)
- ✅ Environment loading without visualization

**When to use:** 
- Quick smoke test to ensure basic skill functionality works
- CI/CD pipeline validation
- Development sanity checks

**Dependencies:**
- `data/test_scene.xml` (test environment)
- NAMOEnvironment initialization
- NAMOPushSkill instantiation

---

#### `test_namo_skill.cpp` - Comprehensive Skill Testing
**Purpose:** Thorough validation of the complete skill system lifecycle.

**What it tests:**
- ✅ Full skill interface (metadata, schema, preconditions)
- ✅ Parameter applicability checks with real/invalid data
- ✅ Precondition validation system
- ✅ World state observation
- ✅ Duration estimation functionality
- ✅ Custom skill configuration
- ✅ Visualization enabled environment

**Key Features Tested:**
- Parameter schema structure and types
- Required vs optional parameter handling
- Precondition failure reporting
- World state introspection
- Performance estimation

**When to use:**
- Development testing during skill system changes
- Validation of skill logic and preconditions
- Debugging skill applicability issues

**Dependencies:**
- `data/test_scene.xml`
- Specific test objects (`obstacle_1_movable`)
- MuJoCo visualization capabilities

---

#### `skill_demo.cpp` - Integration Demonstration
**Purpose:** Demonstrates real-world integration patterns for high-level planners.

**What it demonstrates:**
- 🚀 Complete skill system workflow
- 🚀 High-level planner integration patterns
- 🚀 Error handling and graceful degradation
- 🚀 Multiple integration examples (PDDL, Behavior Trees, RL)

**Key Components:**
1. **Interface Demonstration**
   - Skill metadata and parameter inspection
   - World state monitoring
   - Configuration with custom parameters

2. **Planning Integration**
   - Applicability checking for action selection
   - Precondition validation for planning
   - Duration estimation for temporal planning
   - Actual skill execution with result handling

3. **Error Handling**
   - Invalid parameter handling
   - Boundary condition testing
   - Graceful degradation strategies
   - Parameter relaxation techniques

4. **Integration Examples**
   - **PDDLExecutor**: PDDL action execution interface
   - **PushObjectNode**: Behavior tree node implementation
   - **RLEnvironment**: Reinforcement learning environment wrapper

**When to use:**
- Understanding how to integrate skills into planners
- Learning best practices for error handling
- Testing complete execution workflows
- Demonstrating system capabilities

**Dependencies:**
- `data/test_scene.xml`
- Full MuJoCo environment
- Test objects for manipulation

---

#### `test_skill_benchmark.cpp` - Performance and Compatibility Testing
**Purpose:** Tests skill system with benchmark environments and validates performance.

**What it tests:**
- 🏁 Benchmark environment loading (`data/benchmark_env.xml`)
- 🏁 Skill initialization with larger/complex scenes
- 🏁 Object manipulation in realistic scenarios
- 🏁 Performance characteristics
- 🏁 Compatibility with different environment configurations

**Key Features:**
- Environment statistics reporting (static/movable objects)
- Real object manipulation testing
- Precondition validation in complex scenarios
- No-visualization mode for performance testing

**When to use:**
- Performance regression testing
- Validating complex environment compatibility
- Benchmarking skill execution times
- Testing with realistic object counts

**Dependencies:**
- `data/benchmark_env.xml` (complex test environment)
- Multiple movable objects
- Performance measurement capabilities

---

## Test Execution Guide

### Building Tests
```bash
# Build all skill tests
cmake --build build --target test_simple_skill test_namo_skill skill_demo test_skill_benchmark --parallel 8

# Build individual tests
cmake --build build --target test_simple_skill
cmake --build build --target test_namo_skill
cmake --build build --target skill_demo
cmake --build build --target test_skill_benchmark
```

### Running Tests

#### Quick Validation
```bash
# Fast smoke test (30 seconds)
./build/test_simple_skill
```

#### Comprehensive Testing
```bash
# Full skill system validation (2-3 minutes)
./build/test_namo_skill
```

#### Integration Learning
```bash
# Complete demonstration with examples (5 minutes)
./build/skill_demo
```

#### Performance Testing
```bash
# Benchmark and performance testing
./build/test_skill_benchmark
```

### Expected Outputs

#### Success Indicators
- ✅ "🎉 Simple skill test passed!" (test_simple_skill)
- ✅ "✅ NAMO Skill tests passed!" (test_namo_skill)
- ✅ "=== Demonstration Complete ===" (skill_demo)
- ✅ Environment statistics and skill validation (test_skill_benchmark)

#### Common Failure Modes
- ❌ Missing `data/test_scene.xml` or `data/benchmark_env.xml`
- ❌ MuJoCo initialization failures
- ❌ Missing test objects in environment
- ❌ Incorrect skill configuration
- ❌ Environment bounds violations

## Test Dependencies

### Required Files
```
data/
├── test_scene.xml          # Basic test environment
├── benchmark_env.xml       # Complex benchmark environment
└── motion_primitives.dat   # Motion primitive database
```

### Required Objects in Test Scenes
- `obstacle_1_movable` - Primary test object for manipulation
- Robot entity with proper collision geometry
- Static obstacles for navigation testing
- Proper bounds definition

### Environment Requirements
- MuJoCo library installation
- GLFW for visualization (optional for some tests)
- Proper environment variable setup (`MJ_PATH`)

## Development Guidelines

### Adding New Tests
1. Follow naming convention: `test_<component>_<purpose>.cpp`
2. Include comprehensive error handling
3. Provide clear success/failure indicators
4. Document test purpose and dependencies
5. Update this README with test description

### Debugging Test Failures
1. Check file dependencies first (`data/` files)
2. Verify MuJoCo environment setup
3. Validate test object names in scene files
4. Check skill configuration parameters
5. Review environment bounds and collision setup

### Best Practices
- Keep tests focused on specific functionality
- Provide clear output messages for debugging
- Handle exceptions gracefully with informative errors
- Test both success and failure scenarios
- Include performance considerations for benchmark tests

## Integration Patterns

The tests demonstrate several integration patterns for high-level planners:

### 1. Task Planning Integration
- Parameter validation before execution
- Precondition checking for plan validity
- Duration estimation for temporal constraints
- Success/failure reporting for replanning

### 2. Behavior Tree Integration
- Node-based execution model
- Status reporting (SUCCESS/FAILURE/RUNNING)
- Parameter passing through blackboard
- Graceful failure handling

### 3. Reinforcement Learning Integration
- Action space definition
- Reward computation based on execution results
- State observation through world state interface
- Episode termination on success/failure

These patterns provide templates for integrating the NAMO skill system into any high-level planning architecture.