# CLAUDE.md — NAMO Project Guide

NAMO (Navigation Among Movable Obstacles): C++ physics/planning backend + Python search/learning layer. The robot reaches a goal by pushing movable obstacles aside.

**⛔ READ FIRST — [docs/problem_and_approach.md](docs/problem_and_approach.md):** the plain-English north-star (problem → search-is-expensive → learn-a-ranker → beat-random). Read it before reasoning about the method.

**Current framing (2026-07): horizon/budget-conditioning is DROPPED.** Live model = a single value/ranker over pushes ("NoHz") whose job is ranking the FIRST push (the "setup"). No budget input, no `Q(s,a,H)`, no per-horizon heads. The `horizon_q_*` docs are historical → [RESULTS.md §4](docs/experiments/RESULTS.md), [policy_value_search_hypothesis.md](docs/experiments/policy_value_search_hypothesis.md).

**Doc map:** everything is indexed in [docs/INDEX.md](docs/INDEX.md); research lives in `docs/experiments/`, read on demand. Keep THIS file lean — durable every-session facts only; anything dated or "currently" belongs in a journal.

## How to talk to me [USER — top priority]

- **⛔ PLAIN ENGLISH, first try.** Lead with the point in sentence one; define any term the instant you use it (or don't use it); everyday analogy before jargon. If I'd have to re-read a sentence to decode it, it failed. A 10-second answer beats a precise one I need 10 minutes to parse. Every reply.
- **Short and sharp — walls of text are a failure.** Prefer a 3-line answer + a "want more?" hook. Numbers/code in the answer when load-bearing.
- **⛔ NEVER HAND-WAVE.** No unverified guess as a conclusion; "probably/almost certainly because" (to explain something unchecked) is BANNED. Verify against code/data/job-state first, or label it "UNVERIFIED HYPOTHESIS." When numbers look off, check job/file state before inventing a cause.

## Experiments

- **Loop:** stub-note → run → results-sheet ([WORKFLOW.md](docs/experiments/WORKFLOW.md)). User writes idea-note Hypotheses; Claude writes Plan/Run/Result, appends [RESULTS.md](docs/experiments/RESULTS.md), updates the [model registry](docs/experiments/horizon_q_model_registry.md). **Commit before every run.**
- **Orchestration:** one experiment = one forked subagent. A freshly-forked experiment-runner gets an isolated worktree (`.claude/worktrees/agent-<id>/`, off HEAD-at-fork); a resumed agent runs in the shared checkout. **Rule — file-partition:** disjoint files per agent, agents NEVER commit (orchestrator owns commits), never fork two agents writing the same files. Merge-back: copy worktree agents' OWNED files only (not the whole tree), commit, then `git worktree remove`. Gate expensive steps (retrains) on the user. Status = each card's `status` frontmatter (`idea→live→done`) → drives [DASHBOARD.md](docs/experiments/DASHBOARD.md) (single source of truth); SLURM/job state lives in the card's Run section.
- **Tiering (cost):** `scout` (sonnet) for recon/mechanical fan-outs; `experiment-runner` (opus/xhigh) for real experiment reasoning; inline Bash for trivial one-liners. Both the orchestrator and an experiment-runner may spawn scouts.
- **Reporting splits [USER]:** ALWAYS by difficulty (easy/med/hard) AND horizon (1push/2push), never aggregate-only. Binning: 2push = `pure2push_divisions.json`; 1push = `onepush_episodes.json` solve_rate tertiles; table shape = `_reactive_search.md` / `agg_react_search.py`.
- **Reporting depth [USER]:** card (`_*.md`) = full verbose detail (all tables/plots/diagnostics/caveats); RESULTS.md = curated paper-style (main table + figure + tight key-finding).
- **Timing [USER]:** sims/episode-counts are machine-independent (compare freely); wall-time only on IDENTICAL hardware — `time_bestfirst.py` interleaved, `--exclusive`, CPU-microarch-pinned; re-time a baseline as an anchor. NEVER put wall-times from different boxes on one time axis.

## Environment & build

- **Python:** `/scratch/dm1487/envs/namo/bin/python` (3.11) — use the absolute path; never system `python3`.
- **Bindings:** `PYTHONPATH="$PWD/build_python:$PWD/python"`. Rebuild after editing `src/`/`include/`/`cpp_bindings/`: `./build_python_bindings.sh` (needs `MJ_PATH`, currently `/scratch/dm1487/mujoco/mujoco-3.2.7`).
- **Multi-box:** detect the box → read its card ([CLAUDE.amarel.md](CLAUDE.amarel.md)/[CLAUDE.ilab.md](CLAUDE.ilab.md)) → `source env.<machine>.sh`. Paths come from the env (`namo.paths`/`$NAMO_*`) — never hardcode (guard: `check_no_hardcoded_paths.sh`). Runbook [PORTABILITY.md](docs/PORTABILITY.md); per-checkout tweaks → `CLAUDE.local.md`.
- **Compute:** `compute-resources` skill. Amarel helpers on PATH (`getgpu`/`gpufree`/`gpueta`). SLURM: submit `gpu,gpu-redhat`; never Camden; never wait >1h.

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
