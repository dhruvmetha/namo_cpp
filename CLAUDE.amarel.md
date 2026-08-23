# CLAUDE.amarel.md — machine card: Amarel (Rutgers Amarel cluster)

> Read this **+ the main CLAUDE.md** at session start when on Amarel. This is the ORIGINAL box — its paths are the `env.amarel.sh` defaults. Code is env-native (reads `NAMO_SCRATCH` etc.), so nothing is rewritten per-box. Full portability runbook: [docs/PORTABILITY.md](docs/PORTABILITY.md).

## Am I on this box?
- `hostname` → `amarel*` / `*.amarel.rutgers.edu`, **or** the repo path is under `/cache/home/dm1487/...`.
- After env: `echo $NAMO_SCRATCH` → `/scratch/dm1487`.
- **Login:** `ssh amarel` → **`amarel-new.hpc.rutgers.edu`** → **amarel3 or amarel4 only**, both RHEL 9.6 (verified 2026-08-21: 8 connections, no other host). Every login DNS name (`amarel.rutgers.edu`, `amarel.hpc.rutgers.edu`, `amarel-new.hpc.rutgers.edu`) resolves to the same two IPs, so `ssh amarel-old` lands in the same place and the old cross-OS `module` breakage (lmod `posix` error) is unreachable. **amarel1 and amarel2 were retired 2026-08-21** — amarel1 answers ssh only to print a "use amarel.hpc.rutgers.edu" notice and hang up, amarel2 times out. Use the `amarel` alias, not the bare hostname (bare `ssh amarel.hpc.rutgers.edu` fails host key verification here).

## Layout
- Repos: `/cache/home/dm1487/projects/namo/{namo_cpp, sage_learning}`
- Data / h5 / outputs: under `/scratch/dm1487` (= `NAMO_SCRATCH`)
- **Env:** `source env.amarel.sh` · **Python:** `/scratch/dm1487/envs/namo/bin/python` (3.11; plain `python` resolves to it)
- **Bindings:** `build_python/namo_rl*.so` (gitignored, per-machine). **No path rewrite** — code reads roots from the env. Rebuild recipe ↓.

## Rebuild the bindings (RHEL9 — MODULE-FREE)
The `.so` is per-machine; rebuild after any C++ change. Post-migration the build is **module-free**: RHEL9 compute nodes have system **g++ 11.5** (C++17-ready) so no compiler module is needed, and `module load` is broken from the old login (lmod `posix`); the legacy community tree moved to `/projects/community-old`. `scripts/amarel/activate.sh` now sets this up (conda env + `cmake` pip-installed into it + `MJ_PATH`, NO modules). Build on a **compute node** (`main`):
```bash
ssh amarel && cd /cache/home/dm1487/projects/namo/namo_cpp
srun --partition=main --cpus-per-task=8 --mem=16G --time=00:30:00 bash -c \
  'cd /cache/home/dm1487/projects/namo/namo_cpp && source scripts/amarel/activate.sh && rm -rf build_python && NAMO_MARCH=x86-64-v3 ./build_python_bindings.sh'
```
`NAMO_MARCH=x86-64-v3` = portable baseline (heterogeneous fleet; a `native` `.so` can SIGILL on another node). `rm -rf build_python` clears a stale CMake cache pinned to a removed compiler.

## Compute (SLURM) — Amarel invariants only; where-to-run guidance lives in the `compute-resources` skill
- Partitions (the Aug-2026 maintenance dropped every `-redhat` suffix; `main-redhat` and `gpu-redhat` no longer exist, verified 2026-08-20): CPU **`main`** (508 nodes, 274 idle when checked) · GPU **`gpu`** · **NEVER Camden (`cgpu`, `cmain`, `cmem`)** (no `/scratch` mount). Never wait >1h — relax/resubmit.
- Heavy work → `sbatch`; login node = light orchestration only. Helpers on PATH (`~/bin`): `getgpu` (interactive node, reuse without re-queue), `gpufree` (idle GPUs), `gpueta` (job ETAs).
- Skills are machine-local (they don't travel with git): the `amarel-gpu` user skill lives only here; `compute-resources` is **authored on the CS estate** and mirrored here — re-sync after edits, run **from a CS box** (Amarel can't reach them): `rsync -avz ~/.claude/skills/compute-resources/ amarel:.claude/skills/compute-resources/`

## What lives here
- The full eval toolchain + test set + MuJoCo bindings (the physics sim). Eval is cheap CPU → **run the gate here**.
- ilab can ssh **OUT** to here (rsync pulls); Amarel **cannot** reach ilab.
