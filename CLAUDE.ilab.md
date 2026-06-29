# CLAUDE.ilab.md — machine card: Rutgers CS ilab

> Read this **+ the main CLAUDE.md** at session start when working on ilab. We moved here because Amarel's GPUs were
> backlogged. Full first-time setup: [docs/PORTABILITY.md](docs/PORTABILITY.md) §0. Project pickup: [docs/experiments/ILAB_RESUME.md](docs/experiments/ILAB_RESUME.md).

## Am I on this box?
- `hostname` → `ilab*` / `*.cs.rutgers.edu`, **or** the repo path is under `/common/users/dm1487/...` (Amarel uses `/cache/home/dm1487`).
- After env: `echo $NAMO_SCRATCH` → `/common/users/dm1487/scratch_namo`.

## Layout
- Repos: `/common/users/dm1487/fresh_start/projects/namo/{namo_cpp, sage_learning}`
- Data / h5 / outputs: under `/common/users/dm1487/scratch_namo` (= `NAMO_SCRATCH`)
- **Env:** `source env.ilab.sh` — **edit `MJ_PATH`** to where MuJoCo 3.2.7 actually lives.

## First-time setup (once per fresh checkout)
1. `source env.ilab.sh`
2. **Build bindings:** `./build_python_bindings.sh` — needs `python3-dev`, **OpenCV**, **internet on the build node**
   (pybind11 is FetchContent-downloaded → use a login node). **Build LCD (lowest-common-denominator):** the nodes share
   one filesystem but run different OSes (arrakis glibc 2.35 … ilab2/rlab ≥2.38) — so build on the **oldest-glibc node**
   with a conservative arch: `NAMO_MARCH=x86-64-v2 ./build_python_bindings.sh`. The single shared `build_python/*.so`
   then loads on every node (glibc-backward-compatible, no AVX2). [PORTABILITY §5]
   - **Symptom → fix:** `import namo_rl` → `ImportError: … GLIBC_2.38 not found` means the `.so` was built on a
     newer-glibc node than the one you're on. Rebuild LCD (above) on the oldest-glibc box; don't just rebuild in place.
3. **Pull data:** `bash scripts/portability/pull_from_amarel.sh eval` (~2.7G) — add `train` (~3.5G) for the re-run.
4. **Smoke:** `python scripts/sandbox/eval_reactive_argmax.py --ckpt <any ckpt> --start 0 --end 2 --out /tmp/smoke.json`

(No path-rewrite step — code reads `NAMO_SCRATCH` etc. from the env via `namo.paths`/`$NAMO_*`; label-JSON keys
are remapped at load by `namo.paths.resolve()`. See [PORTABILITY §3](docs/PORTABILITY.md). If you see a
`RuntimeError: $NAMO_SCRATCH is not set`, you forgot `source env.ilab.sh`.)

## Networking (important)
- ilab can ssh **OUT** to Amarel → `pull_from_amarel.sh` works (pull data FROM Amarel here).
- Amarel **cannot** reach ilab → to send ckpts back, **push FROM ilab** (`rsync … dm1487@amarel.rutgers.edu:…`).

## Compute
- This is where **training runs now** (GPU). Eval needs the bindings+MuJoCo (heavy) → if not yet built here, rsync the
  ckpt back to Amarel and run the gate there (eval is cheap CPU). Fill in ilab's scheduler specifics once known.
