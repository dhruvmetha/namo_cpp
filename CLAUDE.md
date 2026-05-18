# CLAUDE.md - NAMO Project Guide

Context for Claude Code when working with NAMO (Navigation Among Movable Obstacles) codebase.

**→ For data collection instructions, see [DATA_COLLECTION_GUIDE.md](DATA_COLLECTION_GUIDE.md)**

## Python Environment

- Use `/common/home/tdn39/.virtualenvs/mujoco/bin/python` for Python commands in this repo.
- Do not default to system `python` or `python3`.
- When commands need the compiled bindings or in-repo Python package, prefer `PYTHONPATH="$PWD/build_python:$PWD/python"` with that interpreter.
- After changing files under `src/`, `include/`, or `python/namo/cpp_bindings/`, rebuild the canonical module with `MJ_PATH=/common/home/tdn39/mujoco/mujoco-3.3.6 ./build_python_bindings.sh` before running Python validation.

## Core Architecture

### C++ Backend (High-Performance Physics & Planning)
- **WavefrontPlanner** ([wavefront_planner.hpp:1](include/wavefront/wavefront_planner.hpp#L1)): BFS-based reachability computation, rebuilds from scratch each update
- **NAMOPushSkill** ([namo_push_skill.hpp:1](include/skills/namo_push_skill.hpp#L1)): Push skill with shape-based planner selection (square/wide/tall)
- **RLEnvironment** ([rl_env.cpp:1](python/namo/cpp_bindings/rl_env.cpp#L1)): Python bindings exposing C++ environment to planners

### Python Planning Layer (Search Algorithms)
- **RegionOpeningPlanner** ([region_opening.py:178](python/namo/planners/opening/region_opening.py#L178)): Region-by-region opening via push primitives — the active data-collection planner
- **FullNAMOPlanner** ([full_namo_planner.py:46](python/namo/planners/full_namo/full_namo_planner.py#L46)): Multi-region full NAMO solver
- **RandomSamplingPlanner** ([random_sampling.py](python/namo/planners/sampling/random_sampling.py)): Baseline sampling planner
- **ModularParallelCollection** ([modular_parallel_collection.py:1](python/namo/data_collection/modular_parallel_collection.py#L1)): Multi-worker data collection with optional smoothing
- **VisualTestSingle** ([visual_test_single.py:1](python/namo/visualization/visual_test_single.py#L1)): Single-run planner testing with visualization

Registered planners (`PlannerFactory.list_available_planners()`): `region_opening`, `full_namo`, `random_sampling`.

## Key Design Patterns

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
  --algorithm region_opening \
  --output-dir ./data --start-idx 0 --end-idx 100
```

### Visual Testing
```bash
python python/namo/visualization/visual_test_single.py \
  --xml-file path/to/env.xml \
  --algorithm region_opening \
  --visualize-search --show-solution auto
```

### Build C++ Components
**Always use the build script** (not cmake directly):
```bash
./build_python_bindings.sh
```
This script handles all CMake configuration, environment setup, and builds the `namo_rl` Python module.

## Critical Implementation Details

### Wavefront Updates
- Full grid rebuild on each `update_wavefront()` call (line 39 in [wavefront_planner.hpp:39](include/wavefront/wavefront_planner.hpp#L39))
- BFS queue pre-allocated: 4M elements for 1410x2210 grids (line 142 in [wavefront_planner.hpp:142](include/wavefront/wavefront_planner.hpp#L142))
- 8-connected grid with obstacle inflation

### Terminal State Checks
- Goal reachability: `env.is_robot_goal_reachable()` — uses the wavefront cached by the last skill execution
- Region opening succeeds when a target region becomes reachable from the robot's current region

### State Management
- `get_full_state()` / `set_full_state()` for search backtracking (lines 78-119 in [rl_env.cpp:78-119](python/namo/cpp_bindings/rl_env.cpp#L78-119))
- qvel always zeroed for physics consistency (line 114 in [rl_env.cpp:114](python/namo/cpp_bindings/rl_env.cpp#L114))

## File Organization

```
namo/
├── include/
│   ├── skills/namo_push_skill.hpp                  # Shape-based skill execution
│   └── wavefront/wavefront_planner.hpp             # BFS reachability computation
├── python/namo/
│   ├── cpp_bindings/rl_env.cpp                     # C++ ↔ Python interface
│   ├── planners/
│   │   ├── opening/region_opening.py               # Active: region-opening planner
│   │   ├── opening/ml_driven_search.py             # ML-guided opening variant
│   │   ├── full_namo/full_namo_planner.py         # Full NAMO solver
│   │   ├── sampling/random_sampling.py             # Baseline sampler
│   │   ├── mcts/hierarchical_mcts.py               # MCTS (not in active rotation)
│   │   └── utils/                                  # solution_smoother, failure_codes
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

### Python BasePlanner Interface ([base_planner.py](python/namo/core/base_planner.py))
```python
# Every registered planner exposes:
result = planner.search(robot_goal)    # Returns PlannerResult

# State-management workflow common to all search implementations:
state = env.get_full_state()           # Save state
env.set_full_state(state)              # Restore state for queries
reachable = env.get_reachable_objects() # Check reachability
is_done = env.is_robot_goal_reachable() # Check termination
result = env.step(action)               # Execute action
new_state = env.get_full_state()       # Capture result
```

## Robots

Two robots share the C++ backend via `RobotAdapter` ([include/robot/robot_adapter.hpp](include/robot/robot_adapter.hpp)):
- **HolonomicAdapter** — original 30 cm point robot (2-DOF slide joints, teleport-style control)
- **DiffDriveAdapter** — 7 cm diff-drive car (freejoint chassis + two wheel hinges, velocity actuators on wheels)

Pick the robot via `config/namo_config.yaml` (point) vs `config/namo_config_car.yaml` (car). The skip-body list and pose source come from the adapter — code outside should not branch on robot type.

### Diff-drive nav (`DiffDriveNavigation`)
State machine: rotate-in-place → drive-straight → ... → final rotate. Heading changes >`sharp_turn_threshold` split the wavefront path into segments. Each phase ends with a passive coast (`wait_steps` × 10 ms) so wheel/caster momentum dissipates before the next phase. Tunables in [diff_drive_navigation.hpp:32](include/navigation/diff_drive_navigation.hpp#L32).

Wheel actuators are MuJoCo `<velocity>` (kv=0.75, forcerange ±0.3 Nm). A motor + custom PI experiment was tried and reverted — explicit PI saturated against MuJoCo's implicit velocity solver and produced worse startup slip. The motor+PI architecture lives in commit `9e7f1c5` if we ever need it for sim-to-real.

### Car XML generation
- `test_xml/little-car-modeling-package/scripts/make_empty_env.py` — minimal 4-wall + 1-obstacle test env
- `test_xml/little-car-modeling-package/scripts/scale_environment.py` — scales any point-robot env (SCALE=0.233) and swaps the robot body for a car

### Trajectory recording for videos
`NAMO_QPOS_DUMP=path NAMO_NAV_LOG=1` makes a run dump per-tick qpos + emit `[NAV_PATH]/[NAV_POSE]` to stderr. The shared dumper ([navigation/qpos_dump.hpp](include/navigation/qpos_dump.hpp)) is wired into both nav (phases 0/1/2) and the push primitive (phase 3) so a single run captures the full nav+push trajectory. Render with `test_xml/little-car-modeling-package/scripts/render_nav_video.py` (needs GPU/EGL — `srun -w rlab2 --gres=gpu:1`).

## Coding Guidelines

- **No Defensive Programming**: Trust design patterns (e.g., self-registration)
- **Single Responsibility**: Avoid redundant validation layers
- **Prefer Editing**: Always edit existing files over creating new ones
- **No Unsolicited Docs**: Only create documentation when explicitly requested
