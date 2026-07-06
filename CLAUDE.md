# CLAUDE.md — NAMO Project Guide

NAMO (Navigation Among Movable Obstacles): C++ physics/planning backend + Python search/learning layer. The robot reaches a goal by pushing movable obstacles aside.

**⛔ READ FIRST — [docs/problem_and_approach.md](docs/problem_and_approach.md):** the plain-English north-star (problem → search-is-expensive → learn-a-ranker → beat-random). Read it before reasoning about the method.

**Live framing:** the model is a single **ranker / search heuristic**, NOT horizon/budget-conditioned — no `Q(s,a,H)`, no per-horizon heads. The `horizon_q_*` docs are historical; the current framing lives in [docs/problem_and_approach.md](docs/problem_and_approach.md).

**Doc map:** everything is indexed in [docs/INDEX.md](docs/INDEX.md); research lives in `docs/experiments/`, read on demand. Keep THIS file lean — durable every-session facts only; anything dated or "currently" belongs in a journal.

## How to talk to me [USER — top priority]

- **⛔ PLAIN ENGLISH, first try.** Lead with the point in sentence one; define any term the instant you use it (or don't use it); everyday analogy before jargon. If I'd have to re-read a sentence to decode it, it failed. A 10-second answer beats a precise one I need 10 minutes to parse. Every reply.
- **Short and sharp — walls of text are a failure.** Prefer a 3-line answer + a "want more?" hook. Numbers/code in the answer when load-bearing.
- **⛔ NEVER HAND-WAVE.** No unverified guess as a conclusion; "probably/almost certainly because" (to explain something unchecked) is BANNED. Verify against code/data/job-state first, or label it "UNVERIFIED HYPOTHESIS." When numbers look off, check job/file state before inventing a cause.

## Experiments

- **Loop:** stub-note → run → results-sheet ([WORKFLOW.md](docs/experiments/WORKFLOW.md)). User writes idea-note Hypotheses; Claude writes Plan/Run/Result, appends [RESULTS.md](docs/experiments/RESULTS.md), updates the [model registry](docs/experiments/horizon_q_model_registry.md). **Commit before every run.**
- **Orchestration (parallel experiments):** one experiment = one forked subagent; **file-partition** — disjoint files per agent, agents NEVER commit (orchestrator owns commits), never fork two agents writing the same files. Tier: `scout` for recon/mechanical, `experiment-runner` (opus/xhigh) for reasoning. Full mechanics (worktrees, merge-back, status→DASHBOARD, tiering) → [ORCHESTRATION.md](docs/experiments/ORCHESTRATION.md).
- **Reporting splits [USER]:** ALWAYS by difficulty (easy/med/hard) AND horizon (1push/2push), never aggregate-only. Conventions (regime framing, depth) + binning → [WORKFLOW.md](docs/experiments/WORKFLOW.md).
- **Timing [USER]:** NEVER put wall-times from different boxes on one axis — sims are the only cross-box substrate; wall-time compares only on identical HW. Protocol → [WORKFLOW.md](docs/experiments/WORKFLOW.md).

## Environment & build

- **Per-box first:** detect the box → read its machine card ([CLAUDE.amarel.md](CLAUDE.amarel.md) / [CLAUDE.ilab.md](CLAUDE.ilab.md) — the latter covers the whole CS estate: ilab/rlab/arrakis/westeros) → activate its env (`source env.<machine>.sh`). **The python interpreter, `MJ_PATH`, data roots, and box GPU helpers all come from that env** (`namo.paths`/`$NAMO_*`) — box-specific, so never hardcode them here (guard: `check_no_hardcoded_paths.sh`). Runbook [PORTABILITY.md](docs/PORTABILITY.md); per-checkout tweaks → `CLAUDE.local.md`.
- **Bindings:** `PYTHONPATH="$PWD/build_python:$PWD/python"` (repo-relative). Rebuild after editing `src/`/`include/`/`cpp_bindings/`: `./build_python_bindings.sh` (needs `MJ_PATH` from the box env).
- **Compute:** `compute-resources` skill. SLURM policy: submit `gpu,gpu-redhat`; never Camden; never wait >1h.

## Data

- **`namo-data-pipeline` skill** for any data/eval/manifest/split work — reuse a script before writing one. Collection: [DATA_COLLECTION_GUIDE.md](DATA_COLLECTION_GUIDE.md).
- **INVARIANT:** one room (`xml`) = MANY episodes. The unit is (pushed object, goal region), never `xml` alone. Match samples by `object_center` (~0 mm), bin difficulty per episode, hold out by room. See [multi_episode_rooms.md](docs/pipeline/multi_episode_rooms.md).
- **⛔ FOUNDATIONAL [USER — don't re-derive]: NO exhaustive ground truth at scale.** We learn value/ranking from sampled, model-guided experience, not enumerated truth (`pure2push` test set is an eval luxury only). Never propose a plan that labels/sims every push — the search/bootstrap machinery exists precisely to avoid it.

## Architecture

**C++ backend:**
- `WavefrontPlanner` ([hpp](include/wavefront/wavefront_planner.hpp)) — BFS reachability, rebuilt each `update_wavefront()`; grid `-2` obstacle / `0` unreachable / `1` reachable.
- `NAMOPushSkill` ([hpp](include/skills/namo_push_skill.hpp)) — shape-based planner pick (square if `x/y<1.05`, else wide if `x>y`, else tall); robot-goal reachability reuses the cached wavefront.
- `RLEnvironment` ([cpp](python/namo/cpp_bindings/rl_env.cpp)) — Python bindings; read the headers for signatures.

**Python planners** (registered: `region_opening`, `full_namo`, `random_sampling`):
- `region_opening` ([py](python/namo/planners/opening/region_opening.py)) — active data-collection planner; one object pushed per step.
- Backtracking pattern (all planners): `s=get_full_state()` → query (`get_reachable_objects`, `is_robot_goal_reachable`) → `step(action)` → `set_full_state(s)`. `set_full_state` always zeroes qvel.

**Robots** share the backend via `RobotAdapter` ([hpp](include/robot/robot_adapter.hpp)) — code outside must not branch on robot type. point (30cm holonomic, `config/namo_config.yaml`) / car (7cm diff-drive, `config/namo_config_car.yaml`). Both use teleport nav (`set_robot_se2()` → settle `kSettleSteps` → pure-pursuit+CTE-PD push); video `NAMO_QPOS_DUMP=path`, per-tick nav log `NAMO_NAV_LOG=1`.

## Coding

- No defensive programming (trust self-registration). Single responsibility (no redundant validation layers). Prefer editing over new files. No unsolicited docs.
- **Markdown prose = one line per paragraph [USER]** — never hard-wrap sentences across source lines (Obsidian soft-wraps; hard breaks read as mid-sentence breaks). Tables/code exempt. Applies to all docs, cards, RESULTS.md — and sub-agents.
