# Legacy Implementation Analysis (PRX-Based)

## Original NAMO System Architecture

The original implementation (`/common/home/dm1487/robotics_research/ktamp/ml4kp_ktamp/examples/namo/`) was a sophisticated research system built on the PRX robotics framework. This analysis provides the complete functional understanding needed for the standalone reimplementation.

### Core Entry Point: interface_namo.cpp

**Main Execution Flow:**
1. **Parameter Loading**: YAML configuration via `prx::param_loader`
2. **Environment Setup**: `NAMOEnvironment` with MuJoCo simulation
3. **Controller Initialization**: `PushController` with MPC capabilities
4. **Motion Primitive Generation**: Physics-based primitive computation
5. **Planning Loop**: `NAMOPlanner` with ML integration and optimization
6. **Data Collection**: Comprehensive state-action logging for ML training

### Complete Component Architecture

**NAMOEnvironment (environment.hpp)**
- **Role**: Central environment controller and MuJoCo interface
- **Key Features**:
  - Object categorization (static vs movable) via name prefixes
  - Real-time CSV logging of all object states
  - Environment bounds calculation for planning
  - Robot goal management and collision detection
- **Data Structures**:
  - `ObjectInfo`: Static properties (body_id, geom_id, position, size, quaternion, symmetry)
  - `ObjectState`: Dynamic states (position, quaternion, velocities)
- **Dependencies**: `prx::mujoco_simulator_t`, MuJoCo physics engine

**WavefrontPlanner (wavefront_planner.hpp)**
- **Algorithm**: Breadth-First Search on discretized grid
- **Optimization Strategy**:
  - Static obstacle grid pre-computed once
  - Dynamic full grid for movable objects
  - 8-connected neighborhood exploration
- **Performance**: Sub-millisecond incremental updates
- **Features**:
  - Goal reachability checking with tolerance
  - Robot inflation for collision-free planning
  - Piecewise wavefront for region-based planning

**MotionPrimitiveGenerator (motion_primitive_generator.hpp)**
- **Algorithm**: Physics-based simulation of pushing actions
- **Process**:
  1. Generate 12 edge points around object perimeter
  2. Simulate robot pushing from each edge point toward center
  3. Record resulting object trajectories
- **Data Structure**: `MotionPrimitive` (position, quaternion, edge_idx, push_steps)
- **MuJoCo Integration**: Custom XML scene compilation and simulation

[Full documentation continues with all remaining components...]

### Implications for Standalone Implementation

This comprehensive analysis reveals the sophisticated architecture that must be reimplemented in the standalone version. Key components requiring recreation:

1. **Core Infrastructure**: Environment management, object tracking, MuJoCo integration
2. **Planning Algorithms**: Incremental wavefront, motion primitives, MPC control
3. **Data Structures**: Fixed-size containers, object representations, action sequences
4. **Integration Systems**: Parameter loading, logging, communication protocols
5. **Optimization Features**: Action sequence minimization, performance profiling
6. **ML Infrastructure**: Data collection, ZMQ communication, online learning support

The standalone implementation should maintain this algorithmic sophistication while eliminating PRX dependencies and achieving zero-allocation performance goals.