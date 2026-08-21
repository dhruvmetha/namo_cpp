# PORTABILITY — run NAMO on any machine (Amarel ↔ ilab ↔ …)

> Goal: a repeatable recipe to stand up the **full** pipeline (train + eval + data-gen) on a fresh box.
> The blocker was never the model — it's **paths**. The repo is now **env-var native**: code reads every
> machine-specific root from the environment, so the **same committed files run on every box** and stay
> clean under `git pull`/`push`. The only per-machine thing is `env.<machine>.sh`.

## 0. TL;DR (do these, in order)
1. Clone `namo_cpp` + `sage_learning`.
2. Python env + **MuJoCo 3.2.7** + **build the C++ bindings** (`./build_python_bindings.sh`).
3. **Set the env vars**: from the repo root, `source env.<machine>.sh` (template/contract in §1–§2).
4. **Move the data** (§4).
5. Smoke-test (§6). Then train / eval normally. **No path-rewriting step anymore** (see §3).

## 1. The env-var contract — the single source of truth
Machine-specific roots live in the **environment**, read in exactly one place each:
- **Python:** `python/namo/paths.py` (`from namo.paths import DATASETS, OUTPUTS, H5, resolve, …`). No script hardcodes a path; they all import from here. It is **fail-loud**: if `NAMO_SCRATCH` is unset it raises with a clear message instead of silently defaulting to `/scratch/dm1487` (a silent default is how a wrong-box run hides). There is **no** `python-dotenv` auto-load — you must `source`/export the vars (or pass via SLURM `--export`, which is the default).
- **Shell / SLURM:** scripts use `$NAMO_*` / `$SAGE_REPO` / `$NAMO_PYTHON` / `$NAMO_REPO` directly (bare — with `set -u` an unset var fails fast, mirroring the Python module).

| var | default (Amarel) | meaning |
|---|---|---|
| `NAMO_SCRATCH` | `/scratch/dm1487` | **base**; the data roots derive from it unless individually set |
| `NAMO_DATASETS` | `$NAMO_SCRATCH/datasets` | test sets, car_envs |
| `NAMO_H5` | `$NAMO_SCRATCH/h5` | training H5s |
| `NAMO_MANIFESTS` | `$NAMO_SCRATCH/manifests` | scene-list manifests |
| `NAMO_OUTPUTS` | `$NAMO_SCRATCH/outputs` | misc outputs |
| `NAMO_LOGS` | `$NAMO_SCRATCH/logs` | SLURM/job logs |
| `NAMO_REPO` | `$PWD` (repo root) | this repo (`namo_cpp`); scripts `cd` here |
| `SAGE_REPO` | `/cache/home/dm1487/projects/namo/sage_learning` | training repo (eval imports from it) |
| `MJ_PATH` | `/scratch/dm1487/mujoco/mujoco-3.2.7` | MuJoCo install (bindings + sim) |
| `NAMO_PYTHON` | `$NAMO_SCRATCH/envs/namo/bin/python` | interpreter slurm/sh scripts invoke |
| `NAMO_GLOBAL_SEED` | `42` | global seed |

So **all** code ports by setting `NAMO_SCRATCH` + `SAGE_REPO` + `MJ_PATH` (+ `NAMO_PYTHON` for the interpreter); the data roots and `NAMO_REPO` derive automatically. Set them once per machine via `env.<machine>.sh`.

> Two equivalent entry points export this contract: **`env.<machine>.sh`** (login + general use, no conda) and
> **`scripts/amarel/activate.sh`** (Amarel compute nodes — also loads modules + activates conda). Both export the
> same canonical names, so converted code works under either.

## 2. `env.<machine>.sh` template (ilab example)
```bash
# env.ilab.sh  — `source env.ilab.sh` from the repo root (it exports)
export NAMO_REPO="$PWD"
export NAMO_SCRATCH=/common/users/dm1487/scratch_namo
export SAGE_REPO=/common/users/dm1487/fresh_start/projects/namo/sage_learning
export MJ_PATH=/common/users/dm1487/fresh_start/mujoco/mujoco-3.2.7      # wherever you put MuJoCo
export NAMO_PYTHON=python                                                # or /path/to/conda/envs/namo/bin/python
export NAMO_GLOBAL_SEED=42
# derived NAMO_DATASETS/NAMO_H5/NAMO_MANIFESTS/NAMO_OUTPUTS/NAMO_LOGS auto-resolve from NAMO_SCRATCH
# runtime:
export PYTHONPATH="$PWD/build_python:$PWD/python:$PWD/scripts:$PWD/scripts/sandbox:$PWD/scripts/pipeline:$SAGE_REPO"
export LD_LIBRARY_PATH="$MJ_PATH/lib:${LD_LIBRARY_PATH:-}"
```
`env.amarel.sh` + `env.ilab.sh` are committed (one per box). Keep `.env` out of git if it ever holds secrets.

## 3. ⚠ No more path-rewriting — what changed
The old `rewrite_paths.sh` (sed over tracked files) is **gone** — it was incompatible with git portability (rewriting tracked files makes every box's checkout dirty → merge conflicts on `git pull`). Both cases it handled are now solved without touching tracked files:

- **(a) ~100 scripts that hardcoded `/scratch/dm1487`** → all converted to read the env (Python via `namo.paths`, shell via `$NAMO_*`). Nothing to rewrite. A guard keeps it that way:
  ```bash
  bash scripts/portability/check_no_hardcoded_paths.sh     # fails if a box path re-enters code
  git config core.hooksPath scripts/githooks               # (once per clone) run it as a pre-commit hook
  ```
- **(b) Label JSONs that bake absolute XML paths** (`namo_testset_v1/labels/pure2push.json` keys) → resolved **at load time** by `namo.paths.resolve()`, which maps a legacy `/scratch/dm1487/...` key onto the current box's `NAMO_SCRATCH`. The JSON stays as-is; the eval opens the right file. No rewrite.

(YAML configs use `${NAMO_DATASETS}/...` and are `expandvars`-ed by the data-collection loader.)

## 4. Data manifest — what to move (rsync; sizes from Amarel)
| what | path (under `$NAMO_SCRATCH`) | size | needed for |
|---|---|---|---|
| test labels | `datasets/namo_testset_v1` | **2.0 G** | eval (the gate) |
| test scene XMLs | `datasets/car_envs/v3/test` | **130 M** | eval — the labels point at these |
| manifests | `manifests/` | **477 M** | eval (scene lists) |
| oracle pairmap | `eval/exhaustive_pairmap_pure2.pkl` | ~42 M | Stage-0 rank analysis (optional) |
| gate baseline ckpt | `sage_outputs/scorer/qfull_nohz_v3_v4hq_s1` | ~53 M | the NoHz-v3 40.7/37.8 compare |
| MuJoCo | `mujoco/mujoco-3.2.7` | 4.5 M | bindings + sim |
| re-run H5s | `h5/{v4_hq_h2_scorer 1.6G, v4_hq_m2b_scorer 1.3G, v4_hq_onepush_h2_aug 417M, v4_hq_exit_finish_valid 92M, v4_hq_exit_finish 55M}` | ~3.5 G | the clean re-run (v3 mix) |
**One committed script does all the rsyncs** (run ON the new box — it ssh's out to Amarel; one rsync per dir):
```bash
bash scripts/portability/pull_from_amarel.sh eval     # ~2.7G — gate data
bash scripts/portability/pull_from_amarel.sh train    # ~3.5G — re-run training data
bash scripts/portability/pull_from_amarel.sh          # both
```

## 5. System setup (one-time per machine)
1. `git clone` both repos (namo_cpp `feat/horizon-q-redesign`, sage_learning `feat/horizon-q`).
2. Python env: `torch pytorch-lightning h5py hydra-core wandb numpy opencv-python pyyaml mujoco==3.2.7`.
3. MuJoCo 3.2.7: copy `mujoco/mujoco-3.2.7` (4.5 M) or use the pip wheel; `export MJ_PATH`.
4. **Build the C++ bindings:** `MJ_PATH=… ./build_python_bindings.sh` → `build_python/namo_rl*.so` (everything physics-y imports `namo_rl`). The build is path-clean but has **4 prerequisites that bite a fresh box**:
   - **CMake ≥3.16 + g++ C++17** (Amarel used cmake 3.26 / gcc 12.3).
   - **`python3-dev` headers** — CMake does `find_package(Python3 COMPONENTS Development REQUIRED)`.
   - **OpenCV** — `find_package(OpenCV REQUIRED)`; install `libopencv-dev` (or conda `opencv`) so CMake finds it.
   - **Internet on the BUILD node** — pybind11 is pulled via `FetchContent` at configure time. Compute nodes often have no internet → build on a **login/internet node**, or pre-vendor pybind11.
   - **`-march=native` trap:** default targets the *build* CPU. If you build and run on different CPUs you'll get illegal-instruction crashes → build on the run-node CPU, or `NAMO_MARCH=x86-64-v3 ./build_python_bindings.sh`.
5. `source env.<machine>.sh` (from the repo root).
6. (optional) `git config core.hooksPath scripts/githooks` to enable the no-hardcoded-paths pre-commit guard.

## 6. Verify it ported (2-min smoke)
```bash
# loads a ckpt + renders + simulates 2 episodes — proves bindings + paths + MuJoCo all line up:
python scripts/sandbox/eval_reactive_argmax.py --ckpt <any ckpt> --start 0 --end 2 --out /tmp/smoke.json
```
If it prints `reactive_argmax@2` without a path/import error, the box is good. (If `NAMO_SCRATCH` isn't set you'll get a clear `RuntimeError` from `namo.paths` telling you to source the env — that's the fail-loud guard working.) Then `eval_afterok` / training run normally.

## 7. The general rule (any system)
**Set `NAMO_SCRATCH`+`SAGE_REPO`+`MJ_PATH`(+`NAMO_PYTHON`) via `env.<machine>.sh` → move the §4 data → build bindings → smoke.** That's it — no path rewrites. If you ever need to reference a new machine-specific root, add it to `python/namo/paths.py` + `env.<machine>.sh`; never hardcode it in a script (the §3 guard will catch you).

## 8. Porting the Claude skills + machine cards
Two kinds of skill, ported differently:
- **Project skills** (`.claude/skills/`, repo knowledge) — **committed** (`.gitignore` un-ignores `.claude/skills/`), so they **travel with `git clone`** (e.g. `namo-data-pipeline`). Put any new *shared* skill here.
- **User skills** (`~/.claude/skills/`, home dir) — do **NOT** travel; they're per-machine. `amarel-gpu` is Amarel-specific (leans on `~/bin/{getgpu,gpufree,gpueta}` + the `gpu` partitions) → **don't copy it to ilab.** The **machine cards** carry each box's compute guidance; write an `ilab-gpu` user skill once ilab's scheduler is known.
- **Machine cards** (`CLAUDE.<machine>.md`) — committed, so they travel. The main `CLAUDE.md` detects the box and routes to the right card. `CLAUDE.local.md` (gitignored, auto-loaded) is for uncommitted per-checkout overrides.
