# Region Opening Scripts

These helper scripts run the Region Opening algorithm from the Python tooling, either for quick visual inspection or for batch data collection. Both scripts forward any extra CLI arguments directly to the underlying Python entrypoints, so you can use all options those tools support.

## run_region_opening_visual.sh

- **Purpose**: Launch an interactive visualization for a single problem using the Region Opening algorithm.
- **Entrypoint**: `python/namo/visualization/visual_test_single.py`
- **Default config**: `python/namo/data_collection/region_opening_collection.yaml`

Usage:
```bash
./scripts/run_region_opening_visual.sh --xml-file /path/to/env.xml [extra-args]
```

Examples:
```bash
# Minimal (visualize a specific scene)
./scripts/run_region_opening_visual.sh --xml-file /absolute/path/to/env.xml

# With additional visualization options (forwarded to visual_test_single.py)
./scripts/run_region_opening_visual.sh \
  --xml-file /absolute/path/to/env.xml \
  --show-solution auto \
  --region-max-chain-depth 2 \
  --region-max-solutions-per-neighbor 5
```

Notes:
- `--config-yaml` is set by the script; override by passing your own `--config-yaml` after it if needed.
- All additional flags after `--xml-file` are passed through to `visual_test_single.py`.

## run_region_opening_collection.sh

- **Purpose**: Run batch data collection of Region Opening episodes (parallel-friendly).
- **Entrypoint**: `python/namo/data_collection/modular_parallel_collection.py`
- **Default config**: `python/namo/data_collection/region_opening_collection.yaml`

Usage:
```bash
./scripts/run_region_opening_collection.sh [extra-args]
```

Common examples:
```bash
# Collect a range of indices with 4 workers
./scripts/run_region_opening_collection.sh --start-idx 0 --end-idx 20 --workers 4

# Override defaults in the YAML (pass-through to modular_parallel_collection.py)
./scripts/run_region_opening_collection.sh \
  --output-dir /absolute/path/to/out \
  --workers 8 \
  --start-idx 100 --end-idx 200
```

Notes:
- The script fixes `--algorithm region_opening`. You can still override other parameters via CLI.
- For large runs, ensure your output directory has sufficient space and that your YAML points to valid scenes.

---

# NAMO Standalone

A C++ implementation of Navigation Among Movable Obstacles (NAMO) planning for robotic systems. This codebase provides path planning for robots that need to navigate environments containing objects they can push or move to reach their goals.

## What This Codebase Does

The NAMO planner solves the problem of robot navigation in cluttered environments where obstacles can be moved. Instead of treating all obstacles as static barriers, the system can plan sequences of actions that include both robot motion and object manipulation (pushing) to find paths to goal locations.

**Core Functionality:**
- **Wavefront Planning**: Computes reachable areas from the robot's current position using breadth-first search
- **Object Manipulation**: Plans pushing actions to move rectangular objects out of the way
- **Physics Simulation**: Uses MuJoCo physics engine for realistic object interactions
- **Motion Primitives**: Executes pre-computed push actions at different edge points and durations
- **Incremental Updates**: Efficiently recomputes reachability when objects move

**Key Components:**
- `NAMOEnvironment`: Manages the physics simulation and object states
- `WavefrontPlanner`: Computes reachable regions using grid-based BFS
- `NAMOPushController`: Plans and executes object pushing actions
- `Memory Manager`: Provides zero-allocation runtime using pre-allocated memory pools

## Installation

### Prerequisites

**Required:**
- Linux/Ubuntu system
- CMake 3.16+
- C++17 compatible compiler
- MuJoCo physics engine (set via `$MJ_PATH` environment variable)

**System packages:**
```bash
sudo apt-get install build-essential cmake pkg-config
```

### Optional Dependencies

The build system includes automatic fallbacks, but for full functionality:

**YAML configuration support:**
```bash
# Install yaml-cpp for advanced configuration parsing
# (falls back to simple key=value parser if not available)
```

**Visualization:**
```bash
# Install GLFW and OpenGL for 3D visualization
# (can run headless if not available)
```

### Build

```bash
# Set MuJoCo path
export MJ_PATH=/path/to/mujoco

# Configure and build
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel 8
```

### Quick Test

```bash
# Run basic test (requires data/test_scene.xml)
./build/namo_standalone config/simple_test.yaml

# Run test suite
./build/test_*
```

## Usage

### Basic Operation

The main executable takes a configuration file and runs the planning system:

```bash
./build/namo_standalone config/namo_config.yaml
```

**Configuration options (YAML or key=value format):**
```yaml
xml_path: "data/test_scene.xml"    # MuJoCo scene file
visualize: true                    # Enable 3D visualization
robot_goal: [2.0, 1.5]            # Target position [x, y]

wavefront_planner:
  resolution: 0.1                  # Grid resolution for planning

data_collection:
  enabled: false                   # Enable state logging
```

### Programming Interface

```cpp
#include "environment/namo_environment.hpp"
#include "wavefront/wavefront_planner.hpp"
#include "planning/namo_push_controller.hpp"

// Initialize environment with physics simulation
NAMOEnvironment env("scene.xml", true);  // with visualization

// Create wavefront planner for reachability computation
WavefrontPlanner planner(0.1, env, robot_size);

// Create push controller for object manipulation
NAMOPushController controller(env, planner, 10, 250, 1.0);

// Set goal and update reachability
std::array<double, 2> goal = {2.0, 1.5};
env.set_robot_goal(goal);

// Compute reachable regions
std::vector<double> robot_pos = {0.0, 0.0};
planner.update_wavefront(env, robot_pos);

// Check if goal is reachable
bool reachable = planner.is_goal_reachable(goal);

// If blocked, find objects to push
std::array<std::string, 20> pushable_objects;
size_t count;
controller.get_reachable_objects(pushable_objects, count);

// Execute push action
if (count > 0) {
    controller.execute_push_primitive(pushable_objects[0], 0, 2);
}
```

## Architecture

### System Overview

```
Core Systems:
├── Environment (src/environment/)
│   ├── NAMOEnvironment - MuJoCo physics integration
│   └── Object state management and simulation control
├── Planning (src/wavefront/, src/planning/)
│   ├── WavefrontPlanner - Grid-based reachability computation
│   └── NAMOPushController - Object manipulation planning
├── Skills (src/skills/)
│   ├── ManipulationSkill - Abstract skill interface
│   └── NAMOPushSkill - NAMO-specific push skill
└── Core (src/core/)
    ├── MemoryManager - Zero-allocation memory pools
    ├── MujocoWrapper - Physics engine interface
    └── ParameterLoader - Configuration management
```

### Python Planning Framework

The Python layer provides a sophisticated planning framework with interchangeable algorithms and strategies:

#### **Planning Algorithms** (`python/namo/planners/`)

**IDFS Family - Iterative Deepening Search:**
- **`StandardIterativeDeepeningDFS`**: Original algorithm that restarts from root for each depth
- **`TreeIDFS`**: Maintains search tree between iterations (more efficient)
- **`OptimalIDFS`**: Finds shortest solutions rather than first solution

**MCTS - Monte Carlo Tree Search:**
- **`HierarchicalMCTS`**: Two-level tree search with StateNodes (environment states) and ObjectNodes (decisions)

**Sampling-Based:**
- **`RandomSamplingPlanner`**: Baseline random action sampling

#### **Strategy System** (`python/namo/strategies/`)

The system uses a **strategy pattern** where algorithms can plug in different decision-making strategies:

**Object Selection Strategies** - Decide which objects to push first:
```python
# Available strategies:
NoHeuristicStrategy()           # Random order (original behavior)
NearestFirstStrategy()          # Push closest objects to robot first
GoalProximityStrategy()         # Push objects closest to robot goal first
FarthestFirstStrategy()         # Push farthest objects first
MLObjectSelectionStrategy()     # Use ML models to predict best objects
```

**Goal Selection Strategies** - Decide where to push selected objects:
```python
# Available strategies:
RandomGoalStrategy()            # Random polar coordinates (original)
GridGoalStrategy()              # Systematic grid pattern around object
AdaptiveGoalStrategy()          # Smart placement considering boundaries
```

#### **Strategy Integration Example**

```python
from namo.planners.idfs import StandardIterativeDeepeningDFS
from namo.strategies import NearestFirstStrategy, GridGoalStrategy

# Create planner with custom strategies
object_strategy = NearestFirstStrategy()           # Push nearest objects first
goal_strategy = GridGoalStrategy(num_angles=8)     # Try 8 directions systematically

planner = StandardIterativeDeepeningDFS(
    env, config,
    object_selection_strategy=object_strategy,
    goal_selection_strategy=goal_strategy
)

# Same algorithm, different behavior based on strategies
result = planner.search(robot_goal)
```

#### **Strategy Impact on Planning**

**Scenario**: Robot needs to reach goal, but Box A (near robot) and Box B (near goal) block the path.

- **`NearestFirstStrategy`**: Tries Box A first → may create more obstacles → longer search
- **`GoalProximityStrategy`**: Tries Box B first → clears goal path directly → faster solution
- **`MLObjectSelectionStrategy`**: Uses trained models → discovers learned optimal strategies

#### **MCTS Hierarchical Structure**

```python
# Two-level node hierarchy:
StateNode (complete environment state)
    └── ObjectNode (decision: which object to push)
        └── StateNode (result after push action)
            └── ObjectNode (next decision)
                └── StateNode (...)

# MCTS phases use strategies:
def mcts_iteration():
    # 1. Selection: Navigate tree using UCB1
    node = select_best_child()

    # 2. Expansion: Use strategies to decide objects/goals
    reachable_objects = node.get_reachable_objects()
    ordered_objects = object_strategy.select_objects(reachable_objects, state, env)
    goals = goal_strategy.generate_goals(selected_object, state, env)

    # 3. Simulation: Random rollout to terminal state
    reward = simulate_random_policy()

    # 4. Backpropagation: Update ancestor statistics
    update_node_values(reward)
```

#### **Algorithm Comparison Framework**

```python
from namo.core import PlannerFactory, compare_planners

# Configure different algorithm variants
configs = {
    "idfs_nearest": PlannerConfig(
        algorithm_params={
            "object_selection_strategy": "nearest_first",
            "goal_selection_strategy": "random"
        }
    ),
    "mcts_goal_directed": PlannerConfig(
        algorithm_params={
            "mcts_budget": 1000,
            "object_selection_strategy": "goal_proximity"
        }
    )
}

# Systematic comparison on same problems
results = compare_planners(env, robot_goal, configs, num_trials=10)
```

#### **ML Integration**

```python
# ML strategies integrate trained models
class MLObjectSelectionStrategy:
    def select_objects(self, reachable_objects, state, env):
        # Convert environment to visual input
        image = env.render_to_image()

        # Use trained diffusion model for predictions
        object_scores = self.ml_model.predict_object_preferences(
            image, reachable_objects, robot_goal
        )

        # Return objects ranked by ML predictions
        return sorted(reachable_objects, key=lambda obj: object_scores[obj], reverse=True)
```

This modular design enables systematic research: any algorithm can use any strategy combination, allowing researchers to isolate the impact of different decision-making approaches on planning performance.

## File Structure

```
namo/
├── src/                    # Source code
│   ├── main.cpp           # Main executable
│   ├── core/              # Core utilities and memory management
│   ├── environment/       # Physics simulation and state management
│   ├── wavefront/         # Grid-based planning algorithms
│   ├── planning/          # Push motion planning
│   └── skills/            # High-level skill interfaces
├── include/               # Header files (mirrors src/ structure)
├── tests/                 # Test executables
│   ├── core/              # Core component tests
│   ├── planning/          # Planning algorithm tests
│   ├── region/            # Region-based planner tests
│   └── integration/       # End-to-end integration tests
├── config/                # Configuration files
├── data/                  # Scene files and test environments
├── python/                # Python bindings and utilities
└── docs/                  # Documentation
```

## Performance Characteristics

- **Memory**: Zero runtime allocations during planning using pre-allocated pools
- **Planning Speed**: Sub-second wavefront computation for typical grid sizes
- **Physics**: Real-time MuJoCo simulation with object dynamics
- **Scalability**: Handles environments with 10+ movable objects efficiently

## Configuration

The system supports both YAML and simple key=value configuration formats:

**Full YAML (config/namo_config.yaml):**
```yaml
xml_path: "data/test_scene.xml"
visualize: true
robot_goal: [2.0, 1.5]

wavefront_planner:
  resolution: 0.05

memory_limits:
  max_static_objects: 20
  max_movable_objects: 10
  grid_max_size: 2000

motion_primitives:
  control_steps: 10
  push_steps: 50
  control_scale: 1.0
```

**Simple format (config/simple_test.yaml):**
```
xml_path=data/test_scene.xml
visualize=true
robot_goal=[0.0, 0.0]
```

## Testing

The codebase includes comprehensive tests:

```bash
# Core functionality tests
./build/test_primitives_only
./build/test_coordinate_transformations

# Planning algorithm tests
./build/test_mpc_executor
./build/test_planner_output

# Integration tests
./build/test_complete_planning
./build/test_end_to_end

# Skill system tests
./build/test_namo_skill
./build/test_simple_skill
```

## Known Issues

- Parameter loader has a bug with boolean value parsing (`has_key` method)
- Missing test scene file: `data/test_scene.xml` referenced in configs but not present
- Some executables require specific scene files that may not be included

## Development

**Build Types:**
- `Release`: Optimized for performance (`-O3 -march=native`)
- `Debug`: Full debugging information (`-O0 -g`)
- `Profile`: Profiling enabled (`-O2 -g -pg`)

**Memory Management:**
- Uses fixed-size containers and memory pools for zero-allocation runtime
- All dynamic memory allocated at initialization
- RAII-based cleanup and error handling

**Dependencies:**
- Core: MuJoCo physics engine (required)
- Optional: YAML-cpp, GLFW, OpenGL, ZMQ
- Header-only: nlohmann/json (auto-downloaded)