# NAMO Reinforcement Learning Environment API Documentation

## Overview

The NAMO (Navigation Among Movable Obstacles) codebase provides a high-performance robotics simulation environment built on MuJoCo with Python bindings via pybind11. This documentation covers the complete RL environment API, skill system integration, and Python bindings.

## Table of Contents

1. [Core RL Environment API](#core-rl-environment-api)
2. [Skill System Integration](#skill-system-integration)
3. [Python Bindings Architecture](#python-bindings-architecture)
4. [Planning Algorithm Integration](#planning-algorithm-integration)
5. [Usage Examples](#usage-examples)
6. [Performance Considerations](#performance-considerations)

## Core RL Environment API

### RLEnvironment Class

The `RLEnvironment` class is the primary interface for interacting with the NAMO simulation environment. It provides a standard RL interface with specialized methods for MCTS planning and reachability queries.

#### Constructor

```cpp
RLEnvironment(const std::string& xml_path, const std::string& config_path, bool visualize = false);
```

**Parameters:**
- `xml_path`: Path to MuJoCo XML scene file
- `config_path`: Path to YAML configuration file
- `visualize`: Enable/disable visualization (default: false)

**Python Example:**
```python
import namo_rl

# Initialize environment
env = namo_rl.RLEnvironment(
    xml_path="data/scenes/simple_scene.xml",
    config_path="config/namo_config.yaml",
    visualize=False
)
```

#### Standard RL Interface

##### reset()
Resets the environment to its initial state.

```python
env.reset()
```

##### step(action) → StepResult
Executes an action and returns the result.

```python
# Create action
action = namo_rl.Action()
action.object_id = "box_1"
action.x = 2.0
action.y = 1.5
action.theta = 0.0

# Execute action
result = env.step(action)
print(f"Success: {result.done}, Reward: {result.reward}")
```

**StepResult Structure:**
```python
class StepResult:
    done: bool           # Whether episode is complete
    reward: float        # Sparse reward (+1 if goal reached, -1 otherwise)
    info: Dict[str, str] # Additional information
```

##### get_observation() → Dict[str, List[float]]
Returns current state as SE(2) poses for all objects.

```python
obs = env.get_observation()
for object_name, pose in obs.items():
    x, y, theta = pose
    print(f"{object_name}: position=({x}, {y}), orientation={theta}")
```

#### State Management for MCTS

The environment supports full state saving/restoring for Monte Carlo Tree Search algorithms.

##### get_full_state() → RLState
Captures complete simulation state including velocities.

```python
# Save current state
current_state = env.get_full_state()
print(f"State has {len(current_state.qpos)} position and {len(current_state.qvel)} velocity values")
```

##### set_full_state(state)
Restores simulation to a specific state.

```python
# Restore to saved state
env.set_full_state(current_state)
```

**RLState Structure:**
```python
class RLState:
    qpos: List[float]  # Joint positions
    qvel: List[float]  # Joint velocities (zeroed for consistency)
```

#### Reachability Queries

##### get_reachable_objects() → List[str]
Returns list of objects reachable by push actions.

```python
reachable = env.get_reachable_objects()
print(f"Can push: {reachable}")
```

##### is_object_reachable(object_name) → bool
Checks if specific object is reachable.

```python
if env.is_object_reachable("box_1"):
    print("Box 1 is reachable")
```

#### Object Information

##### get_object_info() → Dict[str, Dict[str, float]]
Returns cached object geometry and pose information.

```python
object_info = env.get_object_info()
for obj_name, properties in object_info.items():
    print(f"{obj_name}:")
    print(f"  Position: ({properties['x']}, {properties['y']})")
    print(f"  Size: {properties['width']} x {properties['height']}")
    print(f"  Orientation: {properties['theta']}")
```

##### get_world_bounds() → List[float]
Returns world boundaries [x_min, x_max, y_min, y_max].

```python
bounds = env.get_world_bounds()
x_min, x_max, y_min, y_max = bounds
print(f"World bounds: x=[{x_min}, {x_max}], y=[{y_min}, {y_max}]")
```

#### Robot Goal Management

##### set_robot_goal(x, y, theta=0.0)
Sets target robot position for planning.

```python
env.set_robot_goal(x=5.0, y=3.0, theta=0.0)
```

##### is_robot_goal_reachable() → bool
Checks if robot can reach its goal from current state.

```python
if env.is_robot_goal_reachable():
    print("Goal is reachable!")
```

##### get_robot_goal() → Tuple[float, float, float]
Returns current robot goal.

```python
goal_x, goal_y, goal_theta = env.get_robot_goal()
```

#### Action Constraints

##### get_action_constraints() → ActionConstraints
Returns constraints for action space sampling.

```python
constraints = env.get_action_constraints()
print(f"Distance range: [{constraints.min_distance}, {constraints.max_distance}]")
print(f"Theta range: [{constraints.theta_min}, {constraints.theta_max}]")
```

**ActionConstraints Structure:**
```python
class ActionConstraints:
    min_distance: float = 0.3    # Minimum distance from object
    max_distance: float = 1.0    # Maximum distance from object
    theta_min: float = -π        # Minimum orientation
    theta_max: float = π         # Maximum orientation
```

#### Visualization

##### render()
Renders current simulation state (requires visualization=True).

```python
env.render()
```

## Skill System Integration

### Architecture Overview

The skill system provides a universal interface for high-level planners to interact with the low-level control system. It bridges the gap between discrete planning actions and continuous robot control.

#### Skill Hierarchy

```
ManipulationSkill (Abstract Base)
    ↓
NAMOPushSkill (Concrete Implementation)
    ↓
Uses: GreedyPlanner + MPCExecutor
```

### NAMOPushSkill

The `NAMOPushSkill` class implements the manipulation skill interface for push actions.

#### Key Features

- **Universal Interface**: Compatible with any high-level planner (PDDL, MCTS, RL)
- **Type Safety**: Compile-time parameter validation using `std::variant`
- **Complete Lifecycle**: `is_applicable()` → `check_preconditions()` → `execute()`
- **Multi-Planner Support**: Automatically selects planner based on object shape

#### Skill Parameters

```cpp
std::map<std::string, SkillParameterValue> params = {
    {"object_name", std::string("box_1")},
    {"target_pose", SE2State(2.0, 1.5, 0.0)}
};
```

**Supported Parameter Types:**
- `std::string`: Object names and identifiers
- `double`/`int`/`bool`: Numerical parameters
- `SE2State`: 2D poses (x, y, theta)
- `std::array<double, 7>`: 3D poses
- `std::vector<double>`: Variable-length data

#### Skill Execution Flow

1. **Applicability Check**: Validates parameters and environment state
2. **Precondition Check**: Verifies action feasibility
3. **Planning**: Generates motion primitives using shape-specific planner
4. **Execution**: Uses MPC controller to follow trajectory
5. **Result**: Returns success/failure with detailed metrics

#### Python Usage Through RL Environment

The skill system is exposed through the RL environment's `step()` method:

```python
# The step() method internally:
# 1. Creates skill parameters from action
# 2. Checks skill applicability
# 3. Executes skill with MPC controller
# 4. Returns structured result

action = namo_rl.Action()
action.object_id = "box_1"
action.x = 2.0
action.y = 1.5
action.theta = 0.0

result = env.step(action)
```

#### Skill Result Structure

```cpp
struct SkillResult {
    bool success;                                      // Overall success flag
    std::string skill_name;                           // "namo_push_skill"
    std::map<std::string, SkillParameterValue> outputs; // Typed outputs
    std::string failure_reason;                       // Error description
    std::chrono::milliseconds execution_time;         // Performance metric
};
```

**Key Output Parameters:**
- `"robot_goal_reached"`: bool - Whether robot reached its goal
- `"steps_executed"`: int - Number of control steps taken
- `"final_object_pose"`: SE2State - Object's final position

## Python Bindings Architecture

### Pybind11 Integration

The Python bindings use pybind11 for efficient C++/Python interoperability with zero-copy semantics where possible.

#### Module Structure

```cpp
PYBIND11_MODULE(namo_rl, m) {
    m.doc() = "Python bindings for the NAMO RL environment";

    // Bind core data structures
    py::class_<namo::RLState>(m, "RLState")...
    py::class_<namo::RLEnvironment::Action>(m, "Action")...
    py::class_<namo::RLEnvironment::StepResult>(m, "StepResult")...
    py::class_<namo::RLEnvironment::ActionConstraints>(m, "ActionConstraints")...

    // Bind main environment class
    py::class_<namo::RLEnvironment>(m, "RLEnvironment")...
}
```

#### Memory Management

- **Zero-Copy Returns**: Object info and constraints returned by reference
- **Efficient State Copying**: State vectors copied only when necessary
- **Automatic Cleanup**: RAII ensures proper resource management
- **Exception Safety**: C++ exceptions converted to Python exceptions

#### Type Conversions

| C++ Type | Python Type | Notes |
|----------|-------------|--------|
| `std::string` | `str` | Automatic conversion |
| `std::vector<double>` | `List[float]` | Efficient copying |
| `std::map<std::string, T>` | `Dict[str, T]` | Preserves structure |
| `std::array<double, 3>` | `Tuple[float, float, float]` | Fixed-size arrays |
| `bool` | `bool` | Direct mapping |

### Build System Integration

#### CMake Configuration

```cmake
# Find pybind11
find_package(pybind11 REQUIRED)

# Create Python module
pybind11_add_module(namo_rl
    python/namo/cpp_bindings/bindings.cpp
    python/namo/cpp_bindings/rl_env.cpp
)

# Link dependencies
target_link_libraries(namo_rl PRIVATE
    namo_core
    namo_environment
    namo_skills
    mujoco
)
```

#### Python Package Structure

```
python/namo/
├── __init__.py                 # Package initialization
├── core/                       # Core interfaces
│   ├── base_planner.py        # Abstract planner interface
│   └── xml_goal_parser.py     # XML parsing utilities
├── planners/                   # Planning algorithms
│   ├── idfs/                  # Iterative deepening search
│   ├── mcts/                  # Monte Carlo tree search
│   └── sampling/              # Sampling-based methods
├── strategies/                 # Selection strategies
│   ├── object_selection_strategy.py
│   └── goal_selection_strategy.py
├── data_collection/           # Data collection workflows
├── visualization/             # Visualization tools
└── cpp_bindings/              # C++ binding source
    ├── rl_env.hpp            # RL environment header
    ├── rl_env.cpp            # RL environment implementation
    └── bindings.cpp          # Pybind11 bindings
```

## Planning Algorithm Integration

### Abstract Planner Interface

All planning algorithms implement the `BasePlanner` interface for interoperability.

#### BasePlanner API

```python
class BasePlanner(ABC):
    def __init__(self, env: namo_rl.RLEnvironment, config: PlannerConfig):
        """Initialize planner with environment and configuration."""

    @abstractmethod
    def search(self, robot_goal: Tuple[float, float, float]) -> PlannerResult:
        """Execute planning algorithm to find action sequence."""

    @abstractmethod
    def reset(self):
        """Reset internal algorithm state for new planning episode."""

    @property
    @abstractmethod
    def algorithm_name(self) -> str:
        """Return human-readable algorithm name."""
```

#### PlannerResult Structure

```python
@dataclass
class PlannerResult:
    # Core results
    success: bool
    solution_found: bool
    action_sequence: Optional[List[namo_rl.Action]] = None
    solution_depth: Optional[int] = None

    # State trajectories
    state_observations: Optional[List[Dict[str, List[float]]]] = None
    post_action_state_observations: Optional[List[Dict[str, List[float]]]] = None

    # Performance metrics
    search_time_ms: Optional[float] = None
    nodes_expanded: Optional[int] = None
    terminal_checks: Optional[int] = None
    max_depth_reached: Optional[int] = None

    # Algorithm-specific data
    algorithm_stats: Optional[Dict[str, Any]] = None
    error_message: str = ""
```

### Planner Factory System

#### Registration and Creation

```python
from namo.core import PlannerFactory, PlannerConfig

# Register planners
PlannerFactory.register_planner("idfs", StandardIterativeDeepeningDFS)
PlannerFactory.register_planner("tree_idfs", TreeIterativeDeepeningDFS)
PlannerFactory.register_planner("mcts", HierarchicalMCTS)

# Create planner instance
config = PlannerConfig(max_depth=5, random_seed=42)
planner = PlannerFactory.create_planner("mcts", env, config)
```

### Algorithm Examples

#### IDFS (Iterative Deepening First Search)

```python
from namo.planners.idfs import StandardIterativeDeepeningDFS
from namo.core import PlannerConfig

# Configure algorithm
config = PlannerConfig(
    max_depth=5,
    max_goals_per_object=5,
    max_terminal_checks=5000,
    random_seed=42
)

# Create and run planner
planner = StandardIterativeDeepeningDFS(env, config)
result = planner.search(robot_goal=(5.0, 3.0, 0.0))

if result.success:
    print(f"Found solution with {len(result.action_sequence)} actions")
    for i, action in enumerate(result.action_sequence):
        print(f"  {i}: Push {action.object_id} to ({action.x}, {action.y})")
```

#### MCTS (Monte Carlo Tree Search)

```python
from namo.planners.mcts import HierarchicalMCTS
from namo.config import MCTSConfig

# Configure MCTS
config = PlannerConfig(
    max_depth=10,
    algorithm_params={
        "num_simulations": 1000,
        "c_exploration": 1.414,
        "progressive_widening": True,
        "pw_alpha": 0.5,
        "pw_beta": 0.25
    }
)

# Create and run MCTS planner
planner = HierarchicalMCTS(env, config)
result = planner.search(robot_goal=(5.0, 3.0, 0.0))

if result.success:
    print(f"MCTS found solution in {result.search_time_ms}ms")
    print(f"Expanded {result.nodes_expanded} nodes")
```

## Usage Examples

### Basic Environment Setup

```python
import namo_rl
import numpy as np

# Initialize environment
env = namo_rl.RLEnvironment(
    xml_path="data/scenes/simple_scene.xml",
    config_path="config/namo_config.yaml",
    visualize=True  # Enable visualization
)

# Reset to initial state
env.reset()

# Set robot goal
env.set_robot_goal(x=5.0, y=3.0, theta=0.0)

# Check what objects are reachable
reachable_objects = env.get_reachable_objects()
print(f"Reachable objects: {reachable_objects}")
```

### Single Action Execution

```python
# Create push action
action = namo_rl.Action()
action.object_id = "box_1"
action.x = 2.5
action.y = 1.0
action.theta = 0.0

# Execute action
result = env.step(action)

print(f"Action successful: {result.done}")
print(f"Reward: {result.reward}")
print(f"Info: {result.info}")

# Check if goal is now reachable
if env.is_robot_goal_reachable():
    print("Robot can now reach the goal!")
```

### State Management for Search

```python
# Save initial state
initial_state = env.get_full_state()

# Try multiple actions
actions_to_try = [
    ("box_1", 2.0, 1.0),
    ("box_2", 3.0, 2.0),
]

best_action = None
best_reward = -float('inf')

for obj_id, x, y in actions_to_try:
    # Restore to initial state
    env.set_full_state(initial_state)

    # Create and execute action
    action = namo_rl.Action()
    action.object_id = obj_id
    action.x = x
    action.y = y
    action.theta = 0.0

    result = env.step(action)

    # Track best action
    if result.reward > best_reward:
        best_reward = result.reward
        best_action = action

print(f"Best action: push {best_action.object_id} to ({best_action.x}, {best_action.y})")
```

### Integration with Planning Algorithms

```python
from namo.core import PlannerFactory, PlannerConfig

# Create planner configuration
config = PlannerConfig(
    max_depth=5,
    max_goals_per_object=3,
    max_terminal_checks=1000,
    max_search_time_seconds=30.0,
    random_seed=42
)

# Create IDFS planner
planner = PlannerFactory.create_planner("idfs", env, config)

# Execute planning
env.reset()
env.set_robot_goal(5.0, 3.0, 0.0)

result = planner.search(robot_goal=(5.0, 3.0, 0.0))

if result.success:
    print(f"Planning succeeded in {result.search_time_ms:.2f}ms")
    print(f"Solution length: {result.solution_depth}")
    print(f"Nodes expanded: {result.nodes_expanded}")

    # Execute the solution
    for i, action in enumerate(result.action_sequence):
        print(f"Executing step {i+1}: Push {action.object_id}")
        step_result = env.step(action)

        if not step_result.done and i < len(result.action_sequence) - 1:
            print(f"  Intermediate step completed")
        elif step_result.done:
            print(f"  Goal reached!")
            break
```

### Data Collection Workflow

```python
from namo.data_collection import ModularParallelCollection
from namo.core import PlannerConfig

# Setup data collection
collection_config = {
    "xml_paths": ["data/scenes/scene_*.xml"],
    "output_dir": "./collected_data",
    "num_workers": 8,
    "episodes_per_scene": 10
}

# Create planner config
planner_config = PlannerConfig(
    max_depth=5,
    max_terminal_checks=5000,
    collect_stats=True
)

# Run parallel collection
collector = ModularParallelCollection(
    algorithm="tree_idfs",
    config=planner_config,
    **collection_config
)

results = collector.run_collection()
print(f"Collected {len(results)} episodes")
```

## Performance Considerations

### Memory Efficiency

- **Zero-Allocation Runtime**: Pre-allocated memory pools for real-time performance
- **Cached Object Info**: Object geometry cached at initialization
- **Efficient State Copying**: Minimal memory allocation during state save/restore
- **Smart Pointer Management**: Automatic memory cleanup with RAII

### Computational Performance

- **Wavefront Recomputation**: Fast BFS-based reachability computation
- **Shape-Specific Planners**: Optimized motion primitives for different object types
- **MPC Controller**: Two-stage planning architecture for efficient control
- **Parallel Data Collection**: Multi-process data generation with shared environments

### Scalability

- **Object Limits**: Supports 20 static + 10 movable objects per scene
- **Configurable Resolution**: Adjustable grid resolution (default 0.05m)
- **Timeout Management**: Configurable timeouts for search algorithms
- **Memory Bounds**: Fixed-size containers prevent memory leaks

### Best Practices

1. **Environment Reuse**: Create environment once, reset between episodes
2. **State Management**: Use `get_full_state()`/`set_full_state()` for search algorithms
3. **Batch Processing**: Use parallel data collection for large-scale experiments
4. **Configuration Tuning**: Adjust timeout and resolution parameters for your hardware
5. **Memory Monitoring**: Monitor memory usage in long-running processes

## Conclusion

The NAMO RL environment API provides a comprehensive interface for robotics planning and control research. Its modular design supports multiple planning algorithms while maintaining high performance through careful memory management and efficient C++/Python integration. The skill system abstracts low-level control details, allowing researchers to focus on high-level planning strategies.

For additional information, see:
- [Architecture Overview](ARCHITECTURE.md)
- [Skill Usage Guide](SKILL_USAGE_GUIDE.md)
- [Implementation Status](IMPLEMENTATION_STATUS.md)