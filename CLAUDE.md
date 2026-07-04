# CLAUDE.md — NAMO Project Guide

NAMO (Navigation Among Movable Obstacles): a C++ physics/planning backend with a Python search + learning
layer. The robot reaches a goal by pushing movable obstacles out of the way.

**Doc map — every guide, journal, experiment, and model registry is indexed in [docs/INDEX.md](docs/INDEX.md).**
Ongoing research lives under `docs/experiments/` — read it *on demand* from there, not from here. Keep this
file lean: durable, every-session facts only. Anything with a date or a "currently" belongs in a journal.

**Experiment loop:** we run experiments via a stub-note → run → results-sheet loop — protocol in
[docs/experiments/WORKFLOW.md](docs/experiments/WORKFLOW.md). Role split: the **user** writes idea-note
Hypotheses; **Claude** writes Plan/Run/Result, appends [RESULTS.md](docs/experiments/RESULTS.md), and updates
the [model registry](docs/experiments/horizon_q_model_registry.md). **Commit before every run.**

**Orchestration:** one experiment = one **forked subagent that MUST run in its own git worktree**
(`Agent(isolation: "worktree")`) — never fork two experiment agents onto the shared checkout (they clobber
each other's files, and the Obsidian git plugin commits from that same tree). The orchestrator (main loop)
pulls, forks each active experiment (Opus, xhigh) in a worktree, tracks status in `_experiments_board.md`
(done/running/pending), gates expensive steps (retrains) on the user, then merges each agent's card/plot edits
back and owns the shared files (RESULTS.md, board, commits) so parallel agents don't race.
**Tiering (cost):** delegate recon/mechanical fan-outs (node/queue checks, reuse-scans, rsyncs, aggregation) to
`scout` (sonnet/medium); reserve `experiment-runner` (opus/xhigh) for real experiment reasoning. Don't fork an
agent for a trivial one-liner — inline Bash. Both the orchestrator and an experiment-runner may spawn scouts.

**Measuring time (wall-clock) — MUST be consistent.** `t_wall` is hardware-dependent; **sims / episode counts
are machine-independent** (compare those across any box freely). To compare TIME across methods or experiments,
measure on IDENTICAL hardware the same way: `time_bestfirst.py` **interleaved** (every method hits the same
episode back-to-back on one node), `--exclusive`, CPU-microarch-**pinned** (`--constraint=emeraldrapids`/`icelake`)
so times pool. Re-time a shared baseline (e.g. `random`) as an **anchor** to prove a new run pools with prior
ones. **NEVER put wall-times from different boxes (arrakis vs Amarel vs westeros) on the same success-vs-time
axis** — re-time on the baselines' exact setup instead.

**Reporting splits [USER] — ALWAYS.** Every experiment's results are broken down by **difficulty
(easy/med/hard)** AND by **horizon (1push/2push)**, never aggregate-only. 2push difficulty = per-episode
`division` in `pure2push_divisions.json`; 1push = `onepush_episodes.json` `solve_rate` tertiles. Canonical
table shape = `_reactive_search.md`; reuse `agg_react_search.py`'s binning. If only one horizon ran,
run/aggregate the other — don't ship aggregate-only.

## How to talk to me

Default to plain English. Short, sharp sentences. No jargon unless I'm already using it back at you — give
the one-sentence intuition the first time you use a technical term. Walls of text are a failure mode: prefer a
3-line answer with a "want more?" hook. Code and numbers belong in the answer when they're load-bearing.

**⛔ NEVER HAND-WAVE.** Do not present an unverified guess as a conclusion. "Almost certainly / probably
because / likely due to", used to *explain* something you haven't checked, are BANNED. Either verify against
the code / data / job-state first and then state it — or label it "UNVERIFIED HYPOTHESIS." When numbers look
off, check job state and file completeness *before* inventing a cause. "I'll check" + a minute beats a
confident wrong answer.

## Environment & build

- **Python:** `/scratch/dm1487/envs/namo/bin/python` (3.11) — plain `python` resolves to it here, but use the
  absolute path in scripts/docs. Never the system `python3`.
- **In-repo package / compiled bindings:** `PYTHONPATH="$PWD/build_python:$PWD/python"`.
- **Rebuild after editing `src/`, `include/`, or `python/namo/cpp_bindings/`:** `./build_python_bindings.sh`
  (always the script, not cmake directly). Needs `MJ_PATH` set — currently `/scratch/dm1487/mujoco/mujoco-3.2.7`.
- **Multi-box:** detect the box (`hostname` / repo path), read its card
  (**[CLAUDE.amarel.md](CLAUDE.amarel.md)** / **[CLAUDE.ilab.md](CLAUDE.ilab.md)**), then `source env.<machine>.sh`.
  All paths come from the env (`namo.paths` / `$NAMO_*`) — never hardcode; a guard
  (`scripts/portability/check_no_hardcoded_paths.sh`) blocks new ones. Runbook: [docs/PORTABILITY.md](docs/PORTABILITY.md).
  Per-checkout tweaks → `CLAUDE.local.md`.
- **Compute (which machine / GPU / SLURM):** the `compute-resources` skill. Amarel GPU helpers are on PATH:
  `getgpu`, `gpufree`, `gpueta`. SLURM policy: submit `gpu,gpu-redhat`; never Camden; never wait >1h (relax/resubmit).

## Data

- The **`namo-data-pipeline`** skill fires for any data / eval / manifest / split work — reuse an existing
  script before writing a new one. Collection command + details: [DATA_COLLECTION_GUIDE.md](DATA_COLLECTION_GUIDE.md).
- **INVARIANT — one room (`xml`) holds MANY episodes** (different object/goal each). The unit is
  **(pushed object, goal region)**, never `xml` alone. Match samples to episodes by `object_center` (~0 mm),
  bin difficulty per episode, hold out by room. See [docs/pipeline/multi_episode_rooms.md](docs/pipeline/multi_episode_rooms.md).
- **⛔ FOUNDATIONAL CONSTRAINT [USER — do NOT re-derive]: NO exhaustive ground truth at scale.** The method
  learns value/ranking from limited, sampled, model-guided experience — not enumerated truth. The small
  exhaustive test set (`pure2push`) is an eval luxury only. Never propose a plan that assumes we can label/sim
  every push; the search / bootstrap / ExIt machinery exists precisely to avoid enumeration.

## Architecture

**C++ backend** (physics + planning):
- `WavefrontPlanner` ([wavefront_planner.hpp](include/wavefront/wavefront_planner.hpp)) — BFS reachability,
  rebuilt from scratch each `update_wavefront()`; grid values `-2` obstacle / `0` unreachable / `1` reachable.
- `NAMOPushSkill` ([namo_push_skill.hpp](include/skills/namo_push_skill.hpp)) — push skill; shape-based planner
  pick (square if `x/y < 1.05`, else wide if `x > y`, else tall). Robot-goal reachability reuses the cached
  wavefront (zero cost).
- `RLEnvironment` ([rl_env.cpp](python/namo/cpp_bindings/rl_env.cpp)) — Python bindings. **Read the headers for
  exact signatures; don't reproduce them here.**

**Python planners** (registered: `region_opening`, `full_namo`, `random_sampling`):
- `region_opening` ([region_opening.py](python/namo/planners/opening/region_opening.py)) — the active
  data-collection planner; one object pushed per step.
- Search-backtracking pattern (all planners): `s = env.get_full_state()` → query
  (`get_reachable_objects`, `is_robot_goal_reachable`) → `env.step(action)` → `env.set_full_state(s)` to restore.
  **`set_full_state` always zeroes qvel** for physics consistency.

**Robots** share the backend via `RobotAdapter` ([robot_adapter.hpp](include/robot/robot_adapter.hpp)) — code
outside must not branch on robot type:
- point (30 cm holonomic) via `config/namo_config.yaml`; car (7 cm diff-drive) via `config/namo_config_car.yaml`.
- Both use teleport nav: set SE(2) via `env.set_robot_se2()`, settle `kSettleSteps` ticks, then a
  pure-pursuit + CTE-PD push. Video dumps: `NAMO_QPOS_DUMP=path`, per-tick nav log `NAMO_NAV_LOG=1`.

## Coding guidelines

- **No defensive programming** — trust design patterns (e.g. self-registration).
- **Single responsibility** — avoid redundant validation layers.
- **Prefer editing** existing files over creating new ones.
- **No unsolicited docs** — only create documentation when explicitly asked.
