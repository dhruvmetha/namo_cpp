# CLAUDE.ilab.md — machine card: CS iLab estate (arrakis · westeros · ilab1-4 · rlab1-7)

> Read this **+ the main CLAUDE.md** at session start on any CS box. One card covers the whole estate: every box shares `/common/home` + `/common/users` (one checkout, one env, one `.so`) — only the compute attached to each box differs. First-time setup: [docs/PORTABILITY.md](docs/PORTABILITY.md) §0 · project pickup: [docs/experiments/ILAB_RESUME.md](docs/experiments/ILAB_RESUME.md).

## Am I on this estate? Which box?
- `hostname` → `*.cs.rutgers.edu` (`arrakis`, `westeros`, `ilab*`, `rlab*`), **or** the repo path is under `/common/home/dm1487/...` (Amarel uses `/cache/home/dm1487`).
- After env: `echo $NAMO_SCRATCH` → `/common/users/dm1487/scratch_namo`.
- Per-box differences (everything else — paths, env, bindings — is identical):
  - **arrakis** — 5× RTX 6000 Ada, direct (no scheduler). **Oldest glibc (2.35) → this is the LCD-build box** (see setup step 2).
  - **westeros** — 8 GPUs, direct (no scheduler).
  - **ilab1-4 / rlab1-7** — SLURM submit hosts (partition `unlimited`); also direct-ssh login nodes. `ilab1` often stalls mid-ssh — try the next node, never conclude the estate is down from one host.

## Layout
- Repos: `/common/home/dm1487/robotics_research/ktamp/{namo, sage_learning}` (= `NAMO_PARENT`; repo dir is `namo` here, `namo_cpp` on Amarel; `sage_learning` is its sibling). `…/fresh_start/projects/namo/` holds only the `h5` data, **not** the repos.
- Data / h5 / outputs: under `/common/users/dm1487/scratch_namo` (= `NAMO_SCRATCH`)
- **Env:** `source env.ilab.sh` (MJ_PATH is baked in: `/common/users/dm1487/ktamp/mujoco`, MuJoCo 3.2.7 source build).
- ⚠ **Ignore the parent `../.env`** — dead legacy config; nothing reads it (its `scripts/env/activate.sh` consumer was deleted) and its `NAMO_PYTHON` points at the wrong env.

## First-time setup (once per fresh checkout)
1. `source env.ilab.sh`
2. **Build bindings:** `NAMO_MARCH=x86-64-v2 ./build_python_bindings.sh` — needs `python3-dev`, **OpenCV**, **internet on the build node** (pybind11 is FetchContent-downloaded). **Build on arrakis** — it has the oldest glibc (2.35; ilab2/rlab ≥2.38), so the single shared `build_python/*.so` built there loads on every box (glibc-backward-compatible, no AVX2). [PORTABILITY §5]
   - **Symptom → fix:** `import namo_rl` → `ImportError: … GLIBC_2.38 not found` means the `.so` was built on a newer-glibc node than the one you're on. Rebuild on arrakis; don't just rebuild in place.
3. **Pull data:** `bash scripts/portability/pull_from_amarel.sh eval` (~2.7G) — add `train` (~3.5G) for the re-run.
4. **Smoke:** `python scripts/sandbox/eval_reactive_argmax.py --ckpt <any ckpt> --start 0 --end 2 --out /tmp/smoke.json`

(No path-rewrite step — code reads `NAMO_SCRATCH` etc. from the env via `namo.paths`/`$NAMO_*`; label-JSON keys are remapped at load by `namo.paths.resolve()`. See [PORTABILITY §3](docs/PORTABILITY.md). If you see a `RuntimeError: $NAMO_SCRATCH is not set`, you forgot `source env.ilab.sh`.)

## Networking (important)
- CS boxes can ssh **OUT** to Amarel → `pull_from_amarel.sh` works (pull data FROM Amarel here).
- Amarel **cannot** reach the CS estate → to send ckpts back, **push FROM here** (`rsync … dm1487@amarel.rutgers.edu:…`).

## Compute
- Where-to-run, free-GPU checks, SLURM templates, Kerberos auth, Amarel fallback → **`compute-resources` skill** (single source of truth; authored on this estate at `~/.claude/skills/compute-resources/`).
- Training runs here (GPU). Eval needs bindings+MuJoCo → if not yet built here, rsync the ckpt to Amarel and run the gate there (eval is cheap CPU).
