# CLAUDE.md — NAMO Project Guide

NAMO (Navigation Among Movable Obstacles): C++ physics/planning backend + Python search/learning layer. The robot reaches a goal by pushing movable obstacles aside.

**⛔ READ FIRST — the ONE mandatory first read for ANYTHING in the region-opening problem [USER — reminded repeatedly, stop making me]: [docs/problem_and_approach.md](docs/problem_and_approach.md).** Read it BEFORE reasoning about the model, "1-push"/opener/setup, labels, difficulty, or results — so the detail lands in the right frame. Hold this frame or you WILL muddle it: the model is a **RANKER / search heuristic** that orders which pushes the search tries first, so search solves in a handful of simulator calls instead of hundreds; the simulator is a **perfect, free verifier** (so we need the right ORDER, not calibrated "does-this-open-it" probabilities); success = **beat the random ranker** — far fewer sim calls to a solution, on every difficulty tier. 1-push vs 2-push is just how deep the *search* goes; the ranker's job is the ordering at each state. If you catch yourself treating the model as a standalone "does this push open the goal" classifier — STOP and re-read.

**Live framing:** the model is a single **ranker / search heuristic**, NOT horizon/budget-conditioned — no `Q(s,a,H)`, no per-horizon heads. The `horizon_q_*` docs are historical; the current framing lives in [docs/problem_and_approach.md](docs/problem_and_approach.md).

**⛔ The problem is ONE region-opening [USER — stop muddling this]:** an episode = **robot region + goal region + the one blocking object between them**; the job is the push(es) on that object that **MERGE robot+goal**. **1-push** = one push merges them; **2-push** = two pushes merge the SAME pair — a *setup* push (doesn't merge yet) then a *finish* push (merges). This is NOT multi-hop, NOT "goal is N regions away, open one region then the next." A goal not adjacent to the robot region (more than that one object between them) is **out of scope**, not a 2-push.

**Doc map:** everything is indexed in [docs/INDEX.md](docs/INDEX.md); research lives in `docs/experiments/`, read on demand. Keep THIS file lean — durable every-session facts only; anything dated or "currently" belongs in a journal.

## Answer from the code, never from recall [USER — top priority]

- **⛔ NEVER HAND-WAVE — for a NEW question, READ THE CODE FIRST** [USER — a day of guessing cost us; stop it]. Answer from the actual code/config/data/job-state, NOT from assumption, recall, or "how it probably works." Skip the code-check ONLY when the fact is EXPLICITLY in memory — and even then re-verify anything that names a file/flag/number (memory is a cache, not truth). No unverified guess as a conclusion; "probably/almost certainly because" (to explain something unchecked) is BANNED — verify, or label it "UNVERIFIED HYPOTHESIS." When numbers look off, check job/file state before inventing a cause.

## Token-efficient Codex delegation

- Main orchestrator: use Sol/high for decisions, synthesis, and final validation.
- Do not delegate work that one focused search, file read, or command can resolve.
- Start with at most one read-only `code-scout` (Luna/low) for a narrowly bounded code search; use `code-tracer` (Terra/medium) only when the scout reports unresolved cross-file behavior.
- Run at most two subagents concurrently, only on independent, non-overlapping, read-heavy scopes; avoid parallel write-heavy work.
- Give every worker one exact question and an explicit search boundary. The worker stops when answered and returns at most eight bullets with `path:line` evidence, without unrelated summaries or implementation proposals.
- The main orchestrator validates worker evidence and owns edits and conclusions.

## Experiments

- **Loop:** stub-note → run → results-sheet ([WORKFLOW.md](docs/experiments/WORKFLOW.md)). User writes idea-note Hypotheses; Claude writes Plan/Run/Result, appends [RESULTS.md](docs/experiments/RESULTS.md), updates the [model registry](docs/experiments/horizon_q_model_registry.md). **Commit before every run.**
- **Evaluated-artifact registry [USER]:** BEFORE launching or repeating an evaluation, read [`horizon_q_model_registry.md` § Canonical evaluated-model artifacts](docs/experiments/horizon_q_model_registry.md#canonical-evaluated-model-artifacts--read-this-before-launching-an-eval). It is the durable lookup for every evaluated model and random baseline: exact checkpoint/ranker, canonical population, complete search policy, aggregate files, raw per-episode files, and status. Reuse an entry only when the full protocol matches, and register every completed full canonical evaluation before closing its card. Canonical manifests and baseline definitions live in [eval_set_registry.md](docs/experiments/eval_set_registry.md).
- **Orchestration (parallel experiments):** one experiment = one forked subagent; **file-partition** — disjoint files per agent, agents NEVER commit (orchestrator owns commits), never fork two agents writing the same files. Tier: `scout` for recon/mechanical, `experiment-runner` (opus/xhigh) for reasoning. Full mechanics (worktrees, merge-back, status→DASHBOARD, tiering) → [ORCHESTRATION.md](docs/experiments/ORCHESTRATION.md).
- **Reporting splits [USER]:** ALWAYS by difficulty (easy/med/hard) AND horizon (1push/2push), never aggregate-only. Conventions (regime framing, depth) + binning → [WORKFLOW.md](docs/experiments/WORKFLOW.md).
- **Timing [USER]:** NEVER put wall-times from different boxes on one axis — sims are the only cross-box substrate; wall-time compares only on identical HW. Protocol → [WORKFLOW.md](docs/experiments/WORKFLOW.md).

## Environment & build

- **Per-box first:** detect the box → read its machine card ([CLAUDE.amarel.md](CLAUDE.amarel.md) / [CLAUDE.ilab.md](CLAUDE.ilab.md) — the latter covers the whole CS estate: ilab/rlab/arrakis/westeros) → activate its env (`source env.<machine>.sh`). **The python interpreter, `MJ_PATH`, data roots, and box GPU helpers all come from that env** (`namo.paths`/`$NAMO_*`) — box-specific, so never hardcode them here (guard: `check_no_hardcoded_paths.sh`). Runbook [PORTABILITY.md](docs/PORTABILITY.md); per-checkout tweaks → `CLAUDE.local.md`.
- **Bindings:** `PYTHONPATH="$PWD/build_python:$PWD/python"` (repo-relative). Rebuild after editing `src/`/`include/`/`cpp_bindings/`: `./build_python_bindings.sh` (needs `MJ_PATH` from the box env).
- **Compute:** `compute-resources` skill. SLURM policy: submit `gpu`; never Camden; never wait >1h.
- **Consult gpt-5.5 (2nd-opinion model):** `codex` CLI is on PATH (config default model = `gpt-5.5`). Non-interactive at max reasoning: `codex exec -m gpt-5.5 -c model_reasoning_effort="xhigh" -s read-only --skip-git-repo-check - < prompt.txt`. Pattern [USER]: spawn an Opus subagent to interface — it writes the prompt, runs codex, relays the answer. xhigh takes minutes → long Bash timeout; startup stale-temp-dir warning is benign.
- **Cross-box physics = IDENTICAL (verified 2026-07-14):** car-push sim is bit-identical across the CS estate (arrakis) and Amarel — a 48-push cross-box replay gave **0.000 mm / 0.000° delta** (tighter than each box's own ~1 mm warmstart jitter), *despite* different `libmujoco` versions (CS **3.2.8** vs Amarel **3.2.7** — the gap is physics-inert for our scenes). The physics is the C++ `libmujoco` linked into the `namo_rl` bindings (`MJ_PATH`), NOT the pip `mujoco`. → **safe to mix Amarel-collected/labeled data with CS eval; no train/eval physics mismatch.** Also verified: model_2 vs rank01 hard@1 delta = +6.8 on BOTH boxes (every @1 solve@k bit-identical).

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
