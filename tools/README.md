# NAMO Tools Directory - Component Testing & Development Utilities

The `tools/` directory contains **component-level integration tests**, **debugging utilities**, and **development tools** for the NAMO system. Unlike the high-level API tests in `tests/`, these tools focus on **internal component validation**, **performance analysis**, and **debugging specific algorithms**.

## 🏗️ **Architecture: tools/ vs tests/**

### `tests/` - High-Level API Testing
- **Purpose:** User-facing skill system validation
- **Focus:** Interface correctness, integration patterns
- **Audience:** High-level planner developers, API users

### `tools/` - Component-Level Development Tools  
- **Purpose:** Internal component validation and debugging
- **Focus:** Algorithm correctness, performance measurement, debugging
- **Audience:** NAMO system developers, researchers

---

## 📂 **Tool Categories**

### 🧪 **Core Planning Pipeline Tests**
Component-level tests for the fundamental planning algorithms.

#### `test_primitives_only.cpp` - Motion Primitive System
**Purpose:** Validate primitive loading and basic greedy planning without full environment.

**What it tests:**
- Motion primitive binary file loading
- Primitive data structure integrity
- Basic greedy planner functionality
- Performance of primitive operations

**Usage:** `./build/test_primitives_only data/motion_primitives.dat`

**When to use:**
- Debugging primitive loading issues
- Validating motion primitive database integrity
- Testing primitive-based planning in isolation

---

#### `test_end_to_end.cpp` - Complete Planning Pipeline
**Purpose:** Full two-stage planning workflow validation.

**Components tested:**
- 🔄 Universal primitive loading
- 🎯 Abstract planning (GreedyPlanner) 
- 🚀 MPC execution with real physics
- 📊 Performance measurement and validation

**Test scenarios:**
- Multiple start/goal configurations
- Success/failure validation
- Timing analysis
- Different difficulty levels

**When to use:**
- Validating complete planning workflow
- Performance regression testing
- Integration testing after algorithm changes

---

#### `test_iterative_mpc.cpp` - MPC Algorithm Comparison
**Purpose:** Compare full-sequence vs iterative MPC execution approaches.

**Algorithms compared:**
- **Current:** Plan full sequence → execute all steps
- **Iterative:** Plan one step → execute → replan → repeat

**Metrics:**
- Execution accuracy
- Planning time comparison
- Error accumulation analysis
- Convergence behavior

**When to use:**
- Algorithm research and development
- Validating MPC improvements
- Performance optimization studies

---

#### `test_mpc_executor.cpp` - MPC Component Testing
**Purpose:** Isolated testing of MPC execution component.

**What it tests:**
- MPC controller accuracy
- Physics simulation integration
- Error handling in constrained environments
- Parameter sensitivity analysis

---

#### `test_coordinate_transformations.cpp` - Coordinate System Testing
**Purpose:** Validate coordinate transformations and geometric operations.

**What it tests:**
- SE2 state transformations
- Grid coordinate conversions
- Rotation matrix operations
- Numerical precision validation

---

### 🗺️ **Region-Based Planning Tests**
Tests for the advanced region-based high-level planner.

#### `test_region_basic.cpp` - Data Structure Testing
**Purpose:** Unit tests for region-based planning data structures.

**Components tested:**
- `GenericFixedVector` container
- `Region` spatial representation
- `NAMOState` state management
- Hash functions and equality operators

**When to use:**
- Debugging region data structure issues
- Validating spatial representation correctness
- Testing memory management

---

#### `test_region_planner.cpp` - Region Planner Integration
**Purpose:** Full region-based planner functionality testing.

**What it tests:**
- Region decomposition algorithms
- Goal proposal generation
- Tree search implementation
- Action sequence optimization

---

#### `test_region_integration.cpp` - Region System Integration
**Purpose:** Integration testing between region planner and skill system.

**Components tested:**
- Region planner ↔ NAMO skill integration
- End-to-end region-based planning
- Performance with complex scenarios
- Error handling and recovery

---

#### `test_region_path.cpp` - Path Planning in Regions
**Purpose:** Testing path planning algorithms within region framework.

**What it tests:**
- Intra-region path planning
- Inter-region transitions
- Path optimization
- Collision avoidance in regions

---

#### `test_region_minimal.cpp` - Minimal Region Test
**Purpose:** Lightweight region planner testing for quick validation.

**When to use:**
- Quick smoke tests for region functionality
- CI/CD pipeline validation
- Development sanity checks

---

### 📊 **Analysis & Debugging Tools**

#### `analyze_displacement_errors.cpp` - Accuracy Analysis
**Purpose:** Detailed analysis of planning accuracy and error sources.

**Metrics analyzed:**
- Position error (absolute and relative)
- Rotation error (radians and degrees)
- Error accumulation over time
- Statistical error distribution

**Output:**
- Detailed error reports
- Statistical summaries
- Performance recommendations

**When to use:**
- Validating planning accuracy requirements
- Debugging execution drift issues
- Performance optimization studies

---

#### `test_search_limits.cpp` - Search Algorithm Limits
**Purpose:** Testing planning algorithms under resource constraints.

**What it tests:**
- Memory usage limits
- Planning time constraints
- Search space explosion handling
- Graceful degradation under limits

---

#### `test_visual_markers.cpp` - Visualization Testing
**Purpose:** Testing visualization and debugging markers.

**What it tests:**
- MuJoCo visualization integration
- Debug marker rendering
- Visual feedback systems
- Real-time visualization performance

---

### 🔧 **Debugging & Development Utilities**

#### `debug_primitives.cpp` - Primitive Database Debugging
**Purpose:** Low-level debugging of motion primitive binary format.

**What it debugs:**
- Binary file structure validation
- Primitive data integrity
- File format correctness
- Data corruption detection

**When to use:**
- Debugging primitive loading failures
- Validating primitive database generation
- Investigating motion primitive issues

---

#### `debug_crash.cpp` - Crash Investigation Tool
**Purpose:** Debugging tool for investigating system crashes and failures.

**What it debugs:**
- Memory access issues
- Unhandled exceptions
- Resource cleanup problems
- State corruption detection

---

#### `debug_mpc_planning.cpp` - MPC Algorithm Debugging
**Purpose:** Detailed debugging of MPC planning algorithm internals.

**Debug features:**
- Step-by-step MPC execution
- Intermediate state logging
- Constraint violation detection
- Optimization trajectory analysis

---

#### `generate_motion_primitives_db.cpp` - Database Generation
**Purpose:** Tool for generating and updating motion primitive databases.

**Features:**
- Primitive generation from physics simulation
- Database optimization and compression
- Validation of generated primitives
- Performance benchmarking

---

#### `demo_region_visual.cpp` - Region Visualization Demo
**Purpose:** Visual demonstration of region-based planning algorithms.

**Features:**
- Interactive region visualization
- Real-time planning demonstration
- Algorithm step visualization
- Debug information overlay

---

#### `benchmark_region_planner.cpp` - Performance Benchmarking
**Purpose:** Comprehensive performance benchmarking of region-based planner.

**Metrics:**
- Planning time distribution
- Success rate analysis
- Scalability testing
- Memory usage profiling

---

## 🚀 **Usage Patterns**

### Development Workflow
```bash
# 1. Component development - test individual components
./build/test_primitives_only data/motion_primitives.dat
./build/test_region_basic

# 2. Integration testing - validate component interactions  
./build/test_end_to_end
./build/test_region_integration

# 3. Performance analysis - measure and optimize
./build/analyze_displacement_errors
./build/benchmark_region_planner

# 4. Debugging - investigate issues
./build/debug_primitives
./build/debug_mpc_planning
```

### Research & Development
```bash
# Algorithm comparison
./build/test_iterative_mpc

# Accuracy analysis
./build/analyze_displacement_errors

# Performance limits
./build/test_search_limits

# Visual debugging
./build/demo_region_visual
```

### Build System Integration
```bash
# All tools are built automatically with main project
cmake --build build --parallel 8

# Build specific categories
cmake --build build --target test_end_to_end test_region_integration
cmake --build build --target analyze_displacement_errors debug_primitives
```

---

## 🎯 **When to Use Each Category**

### **Component Tests** (`test_*.cpp`)
- **During development:** Validate individual algorithm changes
- **Before integration:** Ensure components work in isolation
- **After refactoring:** Verify component behavior unchanged
- **Performance tuning:** Measure component-level performance

### **Analysis Tools** (`analyze_*.cpp`)
- **Research validation:** Measure algorithm accuracy and performance
- **Problem diagnosis:** Understand system behavior and failure modes
- **Optimization:** Identify performance bottlenecks
- **Requirements validation:** Verify system meets accuracy requirements

### **Debug Tools** (`debug_*.cpp`)
- **Failure investigation:** Debug crashes and unexpected behavior
- **Data validation:** Verify data file integrity and format
- **Algorithm debugging:** Step through complex algorithm execution
- **Development troubleshooting:** Investigate development issues

### **Demo Tools** (`demo_*.cpp`)
- **Algorithm visualization:** Understand algorithm behavior visually
- **Presentation:** Demonstrate system capabilities
- **Educational:** Learn how algorithms work internally
- **Validation:** Visual verification of algorithm correctness

### **Benchmark Tools** (`benchmark_*.cpp`)
- **Performance regression:** Track performance over time
- **Scalability analysis:** Understand system limits
- **Comparative analysis:** Compare algorithm variants
- **Optimization guidance:** Identify optimization opportunities

---

## 🔍 **Key Insights: Why tools/ vs tests/**

The separation follows a **testing pyramid** approach:

```
tests/     ← High-level API testing (few, stable, user-focused)
   ↑
tools/     ← Component testing (many, detailed, developer-focused)
   ↑  
unit tests ← Function-level testing (extensive, fast, granular)
```

### **tools/** - "Testing the Engine"
- Tests **how** the system works internally
- Validates **algorithm correctness** and **performance**
- Helps **developers** understand and **debug** the system
- Changes frequently during **development** and **research**

### **tests/** - "Testing the Interface"  
- Tests **what** the system does from user perspective
- Validates **API contracts** and **integration patterns**
- Helps **users** understand how to **use** the system
- Remains stable as **public interface**

This organization supports both **rapid development** (tools/) and **stable integration** (tests/) simultaneously.