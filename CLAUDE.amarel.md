# CLAUDE.amarel.md — machine card: Amarel (Rutgers Amarel cluster)

> Read this **+ the main CLAUDE.md** at session start when on Amarel. This is the ORIGINAL box — paths here == the
> hardcoded defaults, so no rewrite needed. Full portability runbook: [docs/PORTABILITY.md](docs/PORTABILITY.md).

## Am I on this box?
- `hostname` → `amarel*` / `*.amarel.rutgers.edu`, **or** the repo path is under `/cache/home/dm1487/...`.
- After env: `echo $NAMO_SCRATCH` → `/scratch/dm1487`.

## Layout
- Repos: `/cache/home/dm1487/projects/namo/{namo_cpp, sage_learning}`
- Data / h5 / outputs: under `/scratch/dm1487` (= `NAMO_SCRATCH`)
- **Env:** `source env.amarel.sh` · **Python:** `/scratch/dm1487/envs/namo/bin/python` (3.11; plain `python` resolves to it)
- **Bindings:** already built (`build_python/namo_rl*.so`). **No path rewrite** — this box is the hardcoded default.

## Compute (SLURM)
- GPU: submit `gpu,gpu-redhat`; **NEVER Camden (`cgpu-*`)** (no `/scratch` mount). Never wait >1h — relax/resubmit.
- Helpers on PATH (`~/bin`): `getgpu` (interactive node, reuse without re-queue), `gpufree` (idle GPUs), `gpueta` (job ETAs).
- Heavy work → `sbatch`; login node = light orchestration only.
- (This is the `amarel-gpu` user-skill's home; the skill is machine-local and stays here — see main CLAUDE.md "skills" note.)

## What lives here
- The full eval toolchain + test set + MuJoCo bindings (the physics sim). Eval is cheap CPU → **run the gate here**.
- ilab can ssh **OUT** to here (rsync pulls); Amarel **cannot** reach ilab.
