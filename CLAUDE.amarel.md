# CLAUDE.amarel.md — machine card: Amarel (Rutgers Amarel cluster)

> Read this **+ the main CLAUDE.md** at session start when on Amarel. This is the ORIGINAL box — its paths are the `env.amarel.sh` defaults. Code is env-native (reads `NAMO_SCRATCH` etc.), so nothing is rewritten per-box. Full portability runbook: [docs/PORTABILITY.md](docs/PORTABILITY.md).

## Am I on this box?
- `hostname` → `amarel*` / `*.amarel.rutgers.edu`, **or** the repo path is under `/cache/home/dm1487/...`.
- After env: `echo $NAMO_SCRATCH` → `/scratch/dm1487`.
- **Login (2026-07 RHEL9 migration):** `ssh amarel` now → **`amarel-new.hpc.rutgers.edu`** (RHEL9 login, amarel3/4). The old CentOS-7 `amarel.rutgers.edu` (amarel1) is aliased `ssh amarel-old` and is being retired — submitting from it to RHEL9 compute nodes breaks `module` (lmod `posix` error).

## Layout
- Repos: `/cache/home/dm1487/projects/namo/{namo_cpp, sage_learning}`
- Data / h5 / outputs: under `/scratch/dm1487` (= `NAMO_SCRATCH`)
- **Env:** `source env.amarel.sh` · **Python:** `/scratch/dm1487/envs/namo/bin/python` (3.11; plain `python` resolves to it)
- **Bindings:** `build_python/namo_rl*.so` (gitignored, per-machine). **No path rewrite** — code reads roots from the env. Rebuild recipe ↓.

## Rebuild the bindings (RHEL9 — MODULE-FREE)
The `.so` is per-machine; rebuild after any C++ change. Post-migration the build is **module-free**: RHEL9 compute nodes have system **g++ 11.5** (C++17-ready) so no compiler module is needed, and `module load` is broken from the old login (lmod `posix`); the legacy community tree moved to `/projects/community-old`. `scripts/amarel/activate.sh` now sets this up (conda env + `cmake` pip-installed into it + `MJ_PATH`, NO modules). Build on a **compute node** (`main-redhat`):
```bash
ssh amarel && cd /cache/home/dm1487/projects/namo/namo_cpp
srun --partition=main-redhat --cpus-per-task=8 --mem=16G --time=00:30:00 bash -c \
  'cd /cache/home/dm1487/projects/namo/namo_cpp && source scripts/amarel/activate.sh && rm -rf build_python && NAMO_MARCH=x86-64-v3 ./build_python_bindings.sh'
```
`NAMO_MARCH=x86-64-v3` = portable baseline (heterogeneous fleet; a `native` `.so` can SIGILL on another node). `rm -rf build_python` clears a stale CMake cache pinned to a removed compiler.

## Compute (SLURM) — Amarel invariants only; where-to-run guidance lives in the `compute-resources` skill
- Partitions: CPU default **`main-redhat`** (huge capacity, often 150+ idle; `main` is small and usually full) · GPU `gpu,gpu-redhat` · **NEVER Camden (`cgpu-*`)** (no `/scratch` mount). Never wait >1h — relax/resubmit.
- Heavy work → `sbatch`; login node = light orchestration only. Helpers on PATH (`~/bin`): `getgpu` (interactive node, reuse without re-queue), `gpufree` (idle GPUs), `gpueta` (job ETAs).
- Skills are machine-local (they don't travel with git): the `amarel-gpu` user skill lives only here; `compute-resources` is **authored on the CS estate** and mirrored here — re-sync after edits, run **from a CS box** (Amarel can't reach them): `rsync -avz ~/.claude/skills/compute-resources/ amarel:.claude/skills/compute-resources/`

## What lives here
- The full eval toolchain + test set + MuJoCo bindings (the physics sim). Eval is cheap CPU → **run the gate here**.
- ilab can ssh **OUT** to here (rsync pulls); Amarel **cannot** reach ilab.
