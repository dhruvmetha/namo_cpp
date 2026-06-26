# PORTABILITY — run NAMO on any machine (Amarel ↔ ilab ↔ …)

> Goal: a repeatable recipe to stand up the **full** pipeline (train + eval + data-gen) on a fresh box. The blocker is
> never the model — it's **paths**. There are two layers: (1) a clean **env-var contract** the core code honors, and
> (2) **portability debt** (hardcoded paths + baked-in absolute paths in data) that must be rewritten. Both are below.

## 0. TL;DR (do these, in order)
1. Clone `namo_cpp` + `sage_learning`.
2. Python env + **MuJoCo 3.2.7** + **build the C++ bindings** (`./build_python_bindings.sh`).
3. **Set the env vars** (source an `env.<machine>.sh` — template in §2).
4. **Move the data** (§4) — and **rewrite the two sets of baked-in `/scratch/dm1487` paths** (§3): the hardcoded scripts and the label-JSON XML keys.
5. Smoke-test (§6). Then train / eval normally.

## 1. The env-var contract — "the dotenv" (`namo_cpp/python/namo/.../paths`, central config)
The core code reads these from the **environment** (there is **no** `python-dotenv` auto-load — you must `source`/export them, or pass via SLURM `--export`):

| var | default (Amarel) | meaning |
|---|---|---|
| `NAMO_SCRATCH` | `/scratch/dm1487` | **base**; the others derive from it unless individually set |
| `NAMO_DATASETS` | `$NAMO_SCRATCH/datasets` | test sets, car_envs |
| `NAMO_H5` | `$NAMO_SCRATCH/h5` | training H5s |
| `NAMO_MANIFESTS` | `$NAMO_SCRATCH/manifests` | scene-list manifests |
| `NAMO_OUTPUTS` | `$NAMO_SCRATCH/outputs` | misc outputs |
| `SAGE_REPO` | `/cache/home/dm1487/projects/namo/sage_learning` | training repo (eval imports from it) |
| `MJ_PATH` | `/scratch/dm1487/mujoco/mujoco-3.2.7` | MuJoCo install (bindings + sim) |
| `NAMO_GLOBAL_SEED` | `42` | global seed |

So **most** code ports by setting `NAMO_SCRATCH` + `SAGE_REPO` + `MJ_PATH`. Set them once per machine.

## 2. `env.<machine>.sh` template (ilab example)
```bash
# env.ilab.sh  — `set -a; source env.ilab.sh; set +a`  (or just run it; it exports)
export NAMO_SCRATCH=/common/users/dm1487/scratch_namo
export SAGE_REPO=/common/users/dm1487/fresh_start/projects/namo/sage_learning
export MJ_PATH=/common/users/dm1487/fresh_start/mujoco/mujoco-3.2.7      # wherever you put MuJoCo
export NAMO_GLOBAL_SEED=42
# derived NAMO_DATASETS/NAMO_H5/NAMO_MANIFESTS/NAMO_OUTPUTS auto-resolve from NAMO_SCRATCH (override only if your layout differs)
# runtime:
export PYTHONPATH="$PWD/build_python:$PWD/python:$PWD/scripts:$PWD/scripts/sandbox:$PWD/scripts/pipeline:$SAGE_REPO"
export LD_LIBRARY_PATH="$MJ_PATH/lib:${LD_LIBRARY_PATH:-}"
```
(Commit `env.amarel.sh` + `env.ilab.sh` so each machine has its own; keep `.env` out of git if it holds anything secret.)

## 3. ⚠ Portability DEBT — must rewrite on a new box (the env-var contract does NOT cover these)
Two kinds of baked-in `/scratch/dm1487`:

**(a) 57 `.py` scripts hardcode `/scratch/dm1487`** — and crucially the **eval critical-path scripts**
(`eval_reactive_argmax`, `scorer_beam`, `eval_m3`, `live_scorer`) hardcode it in **arg-defaults + module constants**
(`scorer_beam`'s manifests, champion ckpt). **They do NOT read `NAMO_SCRATCH`** → setting the env var does nothing for
the eval → **this rewrite is MANDATORY for eval on a new box, not optional.**

**(b) Label JSONs bake in absolute XML paths.** `namo_testset_v1/labels/pure2push.json` keys are full paths like
`/scratch/dm1487/datasets/car_envs/v3/test/.../env_XXXX.xml`; the eval opens those → they must resolve.

**Both are handled by one committed script** (run after `source env.<machine>.sh` + after the data lands; idempotent):
```bash
bash scripts/portability/rewrite_paths.sh        # rewrites (a) scripts + (b) label JSONs to $NAMO_SCRATCH
```
(Proper long-term fix = make these read `NAMO_SCRATCH` like the core. Tracked as debt. Symlinking `/scratch/dm1487 → base`
also fixes both at once, but `/scratch` on ilab isn't user-creatable, so rewrite.)

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
**One committed script does all the rsyncs** (run ON ilab — it ssh's out to Amarel; one rsync per dir):
```bash
bash scripts/portability/pull_from_amarel.sh eval     # ~2.7G — gate data
bash scripts/portability/pull_from_amarel.sh train    # ~3.5G — re-run training data
bash scripts/portability/pull_from_amarel.sh          # both
```

## 5. System setup (one-time per machine)
1. `git clone` both repos (namo_cpp `feat/horizon-q-redesign`, sage_learning `feat/horizon-q`).
2. Python env: `torch pytorch-lightning h5py hydra-core wandb numpy opencv-python pyyaml mujoco==3.2.7`.
3. MuJoCo 3.2.7: copy `mujoco/mujoco-3.2.7` (4.5 M) or use the pip wheel; `export MJ_PATH`.
4. **Build the C++ bindings:** `MJ_PATH=… ./build_python_bindings.sh` → `build_python/namo_rl*.so`. (Needs CMake + g++.) This is the one real compile step; everything physics-y imports `namo_rl`.
5. `source env.<machine>.sh`.
6. Run the two rewrites in §3 (scripts + label JSONs).

## 6. Verify it ported (2-min smoke)
```bash
# loads a ckpt + renders + simulates 2 episodes — proves bindings + paths + MuJoCo all line up:
python scripts/sandbox/eval_reactive_argmax.py --ckpt <any ckpt> --start 0 --end 2 --out /tmp/smoke.json
```
If it prints `reactive_argmax@2` without a path/import error, the box is good. Then `eval_afterok` / training run normally.

## 7. The general rule (any system)
**Set `NAMO_SCRATCH`+`SAGE_REPO`+`MJ_PATH` → move the §4 data → run the two §3 rewrites → build bindings → smoke.**
The recurring gotchas are always (a) hardcoded `/scratch/dm1487` in scripts and (b) absolute XML paths inside the label
JSONs. Fix those two and the rest follows from the env-var contract.
