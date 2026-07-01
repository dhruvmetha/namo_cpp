# CLAUDE.md - NAMO Project Guide

Context for Claude Code when working with NAMO (Navigation Among Movable Obstacles) codebase.

**→ For data collection instructions, see [DATA_COLLECTION_GUIDE.md](DATA_COLLECTION_GUIDE.md)**

**→ GOTCHA: one room (`xml`) has MANY episodes (different object/goal each).** Never key analyses,
difficulty buckets, dedup, or train/val splits on `xml` alone — the unit is **(pushed object, goal
region)**. Match samples to episodes by `object_center` (~0 mm), bin difficulty per episode, hold out
by room. See [docs/pipeline/multi_episode_rooms.md](docs/pipeline/multi_episode_rooms.md).

**→ Skills & tools.** Data / eval / training-data / manifest / split work auto-triggers the
**`namo-data-pipeline`** skill (`.claude/skills/`) — reuse-an-existing-script-first + the per-episode
invariant gate above. Amarel GPU helpers live in `~/bin` (global, on PATH): **`getgpu`** (interactive
node, reuse without re-queue), **`gpufree`** (idle GPUs now), **`gpueta`** (job ETAs, flags >1h).
GPU/SLURM policy: submit `gpu,gpu-redhat`, never Camden, never wait >1h (relax/resubmit).

**→ PORTABILITY (DONE — env-native, multi-box).** Detect the box (`hostname` / repo path `/cache/home/dm1487`=Amarel,
`/common/users/dm1487`=ilab), read its machine card (**[CLAUDE.amarel.md](CLAUDE.amarel.md)** / **[CLAUDE.ilab.md](CLAUDE.ilab.md)**),
then `source env.<machine>.sh`. Code reads every path from the env (`namo.paths` / `$NAMO_*`) — no path-rewriting; label-JSON
keys remapped at load by `namo.paths.resolve()`. New box / full runbook: **[docs/PORTABILITY.md](docs/PORTABILITY.md)**. A guard
(`scripts/portability/check_no_hardcoded_paths.sh`) blocks new hardcoded paths; per-checkout tweaks → `CLAUDE.local.md`.
Project skills (`.claude/skills/`) travel with `git clone`; user skills (`~/.claude/skills/`) don't (machine-specific).
- **Compute — where to run + how to switch** (CS-iLab direct GPUs / iLab SLURM `ilab1` / Amarel HPC; auth + filesystem map + fallback order): **[docs/COMPUTE_RESOURCES.md](docs/COMPUTE_RESOURCES.md)** (standalone copy at `/common/home/dm1487/COMPUTE_RESOURCES.md`). Key: CS-iLab = Kerberos (`ssh ilab1`, concrete host, shared FS ⇒ no copy); Amarel = SSH key + push/pull/rebuild.

**→ ⛔ FOUNDATIONAL CONSTRAINT — NO EXHAUSTIVE GROUND TRUTH [USER, do NOT re-assume or re-derive, EVER]:** We will **never**
have exhaustive enumeration of (setup × finish) outcomes at scale / deployment — it is **infeasible**. The exhaustive labels
that *do* exist (the small car test set: `pure2push` 1-push + depth-2) are an **evaluation luxury ONLY**, never the operating
assumption. Consequences you must honor: (1) the method must learn value/ranking from **limited, sampled, model-guided**
experience — NOT enumerated truth; (2) **NEVER** argue "supervised-on-the-oracle beats RL because we can enumerate" — we
*cannot*; the **search / bootstrap / ExIt** machinery exists *precisely to avoid enumeration*; (3) do not propose any solution
that assumes we can label/sim every push. If a plan needs the full oracle, it's wrong by construction.

**→ HORIZON-Q (the active project). READ THESE FIRST, every session / after any compaction — do not work from memory or glob:**
0. **▶ ACTIVE WORK [2026-06-25/26]: staged bootstrapped-value redesign. ⏩ QUICK RESUME (esp. on ilab / a fresh machine): [docs/experiments/ILAB_RESUME.md](docs/experiments/ILAB_RESUME.md) — one page: you-are-here + the ONE thing to train + the gate.** Branch **`feat/horizon-q-redesign`** (anchor `feat/horizon-q` @ `3d65375` is FROZEN — never overwrite). Full **EXECUTION journal = [docs/experiments/horizon_q_redesign_execution.md](docs/experiments/horizon_q_redesign_execution.md)** — Stage 0 (instrument, DONE: setup is the bottleneck) → Stage 1 (drop Horizon, bootstrapped Q — GPU-blocked on Amarel, training on ilab). **RESUME from ILAB_RESUME.md, then the journal's EXECUTION LOG (bottom).** Self-contained brief: [horizon_q_HANDOFF.md](docs/experiments/horizon_q_HANDOFF.md). **✅ RENDER SPEEDUP DONE [2026-06-29..07-01]: model-input render 2019→101ms (20×); 3 changes (BFS→ndimage.label, circle-bbox window, `fast_scorer` skip) all BIT-IDENTICAL — gate 158/158 diff=0 (`scripts/sandbox/test_render_equiv.py` + `scripts/amarel/render_equiv.slurm`) ⇒ NO retrain. Committed: sage `feat/render-speedup`, namo_cpp here.** ✅ TIMING + stratified success curves DONE (reactive + best-first, sim & wall-time, easy/med/hard, 1-push & 2-push, 3-seed): `scripts/sandbox/{time_bestfirst,plot_curves,plot_seeded}.py`, dirs `/scratch/dm1487/eval/timebench/` — render WAS the deploy bottleneck, post-fix the SIM is. **🆕 HYPOTHESIS (research, to test) [2026-07-01]: policy+value decoupled search — [docs/experiments/policy_value_search_hypothesis.md](docs/experiments/policy_value_search_hypothesis.md) — keep V PURE/grounded (NOT bootstrapped, NOT findability/density); at H=2 it re-derives {finish, setup-ranking, horizon}; the falsifiable claim = 2 heads (calibrated V + recall π) vs `combine=q` on sims-to-solve.**
1. **🔀 SEARCH-FIRST REDESIGN journal (the thesis/decision ledger behind the redesign) — [docs/experiments/horizon_q_search_redesign_journal.md](docs/experiments/horizon_q_search_redesign_journal.md)** — the 2026-06-23 pivot: the model is a SIMS-MINIMIZING SEARCH HEURISTIC (a ranker); cost-to-go in SIMS not depth; D2 finish-ranker / D3 recurrence. Read §0–§3.
2. **Build journal (the empirical record) — [docs/experiments/horizon_q_build_journal.md](docs/experiments/horizon_q_build_journal.md)** — state, decisions, hypothesis ledger, v2/v3/v4 numbers. Resume from §9. Design spec: [docs/experiments/multipush_horizonQ_journal.md](docs/experiments/multipush_horizonQ_journal.md).
3. **Model registry — [docs/experiments/horizon_q_model_registry.md](docs/experiments/horizon_q_model_registry.md)** — every ckpt path / headline number / eval dir. **Read it for paths; NEVER reconstruct ckpt paths by glob** (wandb-hash dirs are unrecoverable that way). Every trained model goes here the moment it trains.
4. **🆕 POLICY+VALUE decoupled-search HYPOTHESIS (research, to test) — [docs/experiments/policy_value_search_hypothesis.md](docs/experiments/policy_value_search_hypothesis.md)** — the 2026-07-01 research hypothesis (NOT a committed design): split the search into **π (action proposal/ranking)** + **V (state selection / frontier ordering)**. Has the exact train + inference loops, the **"keep V PURE/grounded"** decision (NOT bootstrapped, NOT findability/density — avoids the moving-target instability), and the critical read: at **H=2 it re-derives {finish, setup-ranking, horizon}**, so the one honest testable claim = **2 heads (calibrated V + recall π) vs `combine=q`** on sims-to-solve. Read before proposing any value/policy redesign.

These files survive compaction; the conversation does not. When picking up Horizon-Q work, read the EXECUTION journal (current Stage + log) + registry before launching anything; resume the staged plan where the log left off.

## How to talk to me

Default to plain English. Short, sharp sentences. No jargon unless I'm already using it back at you. If you have to use a technical term, give the one-sentence intuition the first time. Walls of text are a failure mode — prefer a 3-line answer with a "want more?" hook over a 30-line essay I have to skim. Code snippets and numbers belong in the answer when they're load-bearing, not as decoration.

**⛔ STRICT — NEVER HAND-WAVE [USER].** Do not present an unverified guess as a conclusion. "Almost certainly X", "probably because Y", "likely due to Z" used to *explain* something you haven't checked are BANNED. Either **verify it against the code/data first** (read the path, sample the data, check the job state) and then state it — or **explicitly label it "UNVERIFIED HYPOTHESIS — haven't checked."** When numbers look off, check job state / file completeness / the actual values *before* inventing a cause. Saying "I'll check" and taking a minute beats shipping a confident wrong answer. (Reinforced after a real miss: I blamed differing eval counts on "shard wall-timeouts" — verification showed jobs completed fine and it was a premature-file-read artifact.)

## Python Environment

- Use `/scratch/dm1487/envs/namo/bin/python` (Python 3.11) for Python commands in this repo. It's already on `PATH` here, so plain `python` resolves to it — but reference the absolute path when writing scripts or docs.
- Do not default to the system `python3` (different env).
- When commands need the compiled bindings or in-repo Python package, prefer `PYTHONPATH="$PWD/build_python:$PWD/python"` with that interpreter.
- After changing files under `src/`, `include/`, or `python/namo/cpp_bindings/`, rebuild the canonical module with `./build_python_bindings.sh` before running Python validation. The script expects `MJ_PATH` to be set in your shell — currently `/scratch/dm1487/mujoco/mujoco-3.2.7` (MuJoCo 3.2.7 prebuilt; `LD_LIBRARY_PATH` already includes its `lib/`).

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

### Navigation

Both robots use teleport navigation — set the chassis pose to the target SE(2) via `env.set_robot_se2(x, y, theta)`, zero velocities, settle for `kSettleSteps` physics ticks (default 100, override via `NAMOPushController::set_settle_steps`). The push that follows tracks a pure-pursuit + CTE-PD path; see [push_path_follower.hpp](include/navigation/push_path_follower.hpp).

Wheel actuators on the car are MuJoCo `<velocity>` (kv=0.75, forcerange ±0.3 Nm). A motor + custom PI experiment was tried and reverted — explicit PI saturated against MuJoCo's implicit velocity solver and produced worse startup slip. The motor+PI architecture lives in commit `9e7f1c5` if we ever need it for sim-to-real.

### Car XML generation
- `test_xml/little-car-modeling-package/scripts/make_empty_env.py` — minimal 4-wall + 1-obstacle test env
- `test_xml/little-car-modeling-package/scripts/scale_environment.py` — scales any point-robot env (SCALE=0.233) and swaps the robot body for a car

### Trajectory recording for videos
`NAMO_QPOS_DUMP=path` makes a run dump per-tick qpos to that file. `NAMO_NAV_LOG=1` adds per-tick `[PUSH_PATH]` and `[PUSH_CTRL]` lines to stderr during the push. The shared dumper ([navigation/qpos_dump.hpp](include/navigation/qpos_dump.hpp)) is wired into the push primitive (phase 3) — pre/post-push settle ticks are included.

## Coding Guidelines

- **No Defensive Programming**: Trust design patterns (e.g., self-registration)
- **Single Responsibility**: Avoid redundant validation layers
- **Prefer Editing**: Always edit existing files over creating new ones
- **No Unsolicited Docs**: Only create documentation when explicitly requested
