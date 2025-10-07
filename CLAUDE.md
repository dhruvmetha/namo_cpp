# CLAUDE.md - NAMO Project Guide

Context for Claude Code when working with NAMO (Navigation Among Movable Obstacles) codebase.

## Core Architecture

### C++ Backend (High-Performance Physics & Planning)
- **WavefrontPlanner** ([wavefront_planner.hpp:1](include/wavefront/wavefront_planner.hpp#L1)): BFS-based reachability computation, rebuilds from scratch each update
- **NAMOPushSkill** ([namo_push_skill.hpp:1](include/skills/namo_push_skill.hpp#L1)): Push skill with shape-based planner selection (square/wide/tall)
- **RLEnvironment** ([rl_env.cpp:1](python/namo/cpp_bindings/rl_env.cpp#L1)): Python bindings exposing C++ environment to planners

### Python Planning Layer (Search Algorithms)
- **StandardIterativeDeepeningDFS** ([standard_idfs.py:1](python/namo/planners/idfs/standard_idfs.py#L1)): Restart-based IDFS with pluggable object/goal strategies
- **ReachabilityExpandingIDFS** ([reachability_expanding_idfs.py:1](python/namo/planners/idfs/reachability_expanding_idfs.py#L1)): Succeeds when new objects become reachable (exploration-focused)
- **ModularParallelCollection** ([modular_parallel_collection.py:1](python/namo/data_collection/modular_parallel_collection.py#L1)): Multi-worker data collection with smoothing/refinement
- **VisualTestSingle** ([visual_test_single.py:1](python/namo/visualization/visual_test_single.py#L1)): Single-run planner testing with visualization

## Key Design Patterns

### Two-Tier Success Conditions
- **Standard**: Robot goal reachable (WavefrontPlanner checks via BFS)
- **Reachability-Expanding**: Robot goal reachable OR new objects reachable (line 229 in [reachability_expanding_idfs.py:229](python/namo/planners/idfs/reachability_expanding_idfs.py#L229))

### Robot Goal Management
- Set via `skill.set_robot_goal(x, y, theta)` (line 92 in [namo_push_skill.hpp:92](include/skills/namo_push_skill.hpp#L92))
- Checked via `skill.is_robot_goal_reachable()` (line 93 in [namo_push_skill.hpp:93](include/skills/namo_push_skill.hpp#L93))
- Leverages cached wavefront from last skill execution

### Planner-Skill Integration
```python
# Python planner → C++ skill execution
env.set_robot_goal(x, y, theta)  # Set target
result = skill.execute(params)    # Push object
reached = env.is_robot_goal_reachable()  # Check success
```

### Shape-Based Planner Selection
NAMOPushSkill uses object size ratio (5% tolerance) to select specialized planners:
- `x/y < 1.05` → square planner
- `x > y` → wide planner
- `y > x` → tall planner
(lines 55-63 in [namo_push_skill.hpp:55-63](include/skills/namo_push_skill.hpp#L55-63))

## Common Workflows

### Running Data Collection
```bash
python python/namo/data_collection/modular_parallel_collection.py \
  --algorithm reachability_expanding_idfs \
  --output-dir ./data --start-idx 0 --end-idx 100
```

### Visual Testing
```bash
python python/namo/visualization/visual_test_single.py \
  --xml-file path/to/env.xml \
  --algorithm standard_idfs \
  --visualize-search --show-solution auto
```

### Build C++ Components
```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel 8
```

## Critical Implementation Details

### Wavefront Updates
- Full grid rebuild on each `update_wavefront()` call (line 39 in [wavefront_planner.hpp:39](include/wavefront/wavefront_planner.hpp#L39))
- BFS queue pre-allocated: 4M elements for 1410x2210 grids (line 142 in [wavefront_planner.hpp:142](include/wavefront/wavefront_planner.hpp#L142))
- 8-connected grid with obstacle inflation

### Terminal State Checks
- Standard IDFS: `env.is_robot_goal_reachable()` (line 417 in [standard_idfs.py:417](python/namo/planners/idfs/standard_idfs.py#L417))
- Reachability-expanding: Two conditions checked (lines 218-239 in [reachability_expanding_idfs.py:218-239](python/namo/planners/idfs/reachability_expanding_idfs.py#L218-239))
- Respects `max_terminal_checks` limit (line 411 in [standard_idfs.py:411](python/namo/planners/idfs/standard_idfs.py#L411))

### State Management
- `get_full_state()` / `set_full_state()` for search backtracking (lines 78-119 in [rl_env.cpp:78-119](python/namo/cpp_bindings/rl_env.cpp#L78-119))
- qvel always zeroed for physics consistency (line 114 in [rl_env.cpp:114](python/namo/cpp_bindings/rl_env.cpp#L114))
- SE(2) observations cached before/after actions (lines 502-509 in [standard_idfs.py:502-509](python/namo/planners/idfs/standard_idfs.py#L502-509))

## File Organization

```
namo/
├── include/
│   ├── skills/namo_push_skill.hpp        # Shape-based skill execution
│   └── wavefront/wavefront_planner.hpp   # BFS reachability computation
├── python/namo/
│   ├── cpp_bindings/rl_env.cpp           # C++ ↔ Python interface
│   ├── planners/idfs/
│   │   ├── standard_idfs.py              # Restart-based IDFS
│   │   └── reachability_expanding_idfs.py # Exploration-focused IDFS
│   ├── data_collection/modular_parallel_collection.py
│   └── visualization/visual_test_single.py
└── config/
    ├── namo_config_complete.yaml         # Full config
    └── headless_test.yaml                # Testing config
```

## API Reference

### C++ NAMOPushSkill ([namo_push_skill.hpp](include/skills/namo_push_skill.hpp))
```cpp
// Reachability queries (uses cached wavefront from last skill execution)
std::vector<std::string> get_reachable_objects() const;
bool is_object_reachable(const std::string& object_name) const;

// Robot goal management (for planners to check termination)
void set_robot_goal(double x, double y, double theta = 0.0);
bool is_robot_goal_reachable() const;  // Uses cached wavefront - zero cost!
std::array<double, 3> get_robot_goal() const;
void clear_robot_goal();

// Skill execution (ManipulationSkill interface)
SkillResult execute(const std::map<std::string, SkillParameterValue>& params);
bool is_applicable(const std::map<std::string, SkillParameterValue>& params) const;
```

### C++ WavefrontPlanner ([wavefront_planner.hpp](include/wavefront/wavefront_planner.hpp))
```cpp
// Update wavefront (rebuilds from scratch via BFS)
bool update_wavefront(NAMOEnvironment& env, const std::vector<double>& start_pos);

// Reachability queries
bool is_goal_reachable(const std::array<double, 2>& goal_pos, double goal_size = 0.05) const;

// Grid access
const std::vector<std::vector<int>>& get_grid() const;  // -2=obstacle, 0=unreachable, 1=reachable
int get_grid_width() const;
int get_grid_height() const;
double get_resolution() const;
```

### Python RLEnvironment ([rl_env.cpp](python/namo/cpp_bindings/rl_env.cpp))
```python
# State management (for search backtracking)
state = env.get_full_state()          # Returns RLState with qpos/qvel
env.set_full_state(state)              # Restores state (qvel always zeroed)

# Action execution
result = env.step(action)              # Returns StepResult(done, reward, info)

# Observations
obs = env.get_observation()            # Dict[str, List[float]] - SE(2) poses
reachable = env.get_reachable_objects()  # List[str] - object names
is_reach = env.is_object_reachable(name) # bool

# Robot goal (for termination checks)
env.set_robot_goal(x, y, theta)
reached = env.is_robot_goal_reachable()  # bool - uses cached wavefront

# Environment info
bounds = env.get_world_bounds()        # [xmin, xmax, ymin, ymax]
obj_info = env.get_object_info()       # Dict[str, Dict[str, float]] - cached geometry
```

### Python BasePlanner Interface ([standard_idfs.py](python/namo/planners/idfs/standard_idfs.py))
```python
# Main planner methods
result = planner.search(robot_goal)    # Returns PlannerResult

# Common workflow pattern in DFS implementations:
state = env.get_full_state()           # Save state
env.set_full_state(state)              # Restore state for queries
reachable = env.get_reachable_objects() # Check reachability
is_done = env.is_robot_goal_reachable() # Check termination
result = env.step(action)               # Execute action
new_state = env.get_full_state()       # Capture result
```

### Common Usage Patterns

**Standard Terminal Check (StandardIDFS):**
```python
def _is_terminal_state(self, state):
    self.env.set_full_state(state)
    return self.env.is_robot_goal_reachable()
```

**Reachability-Expanding Terminal Check (ReachabilityExpandingIDFS):**
```python
def _is_terminal_state(self, state):
    self.env.set_full_state(state)
    # Two conditions: goal reached OR new objects reachable
    return (self.env.is_robot_goal_reachable() or
            self._has_reachability_expanded(state))
```

**Action Execution with State Capture:**
```python
def _execute_action(self, state, action):
    self.env.set_full_state(state)
    self.env.step(action.to_namo_action())
    return self.env.get_full_state()
```

## Coding Guidelines

- **No Defensive Programming**: Trust design patterns (e.g., self-registration)
- **Single Responsibility**: Avoid redundant validation layers
- **Prefer Editing**: Always edit existing files over creating new ones
- **No Unsolicited Docs**: Only create documentation when explicitly requested