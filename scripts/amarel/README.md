# Amarel SLURM Quick Reference for NAMO

Practical guide for requesting CPUs, threads, and GPUs on Rutgers Amarel.
Everything below is verified against the live cluster (`sinfo`, `scontrol`), not just docs.

## TL;DR — picking compute for your task

| You need… | Submit to… | Why |
|---|---|---|
| Data collection (multiprocessing.Pool, CPU only) | `main-redhat` | Biggest CPU pool, RHEL 9, new hardware |
| Quick CPU job, no special hardware | `main` or `main-redhat` | Either works; `-redhat` has more nodes |
| GPU training (PyTorch / JAX) | `gpu-redhat` + `--gres=gpu:1` | ~90 % of GPU inventory is here |
| Specific L40S (48 GB VRAM) | `gpu-redhat` + `--constraint=adalovelace` | Newest cards, biggest pool |
| Specific A100 | `gpu-redhat` + `--constraint=ampere` | ~15 nodes, 2–4 cards each |
| > 256 GB RAM | `mem-redhat` | 512 GB / 1 TB / 2 TB nodes |
| Camden hardware | `cmain-redhat` / `cgpu-redhat` | Smaller pool, often less queued |

**Default rule of thumb: use the `-redhat` variant of whatever partition you want.** Reason explained in §1.5 below.

## Files in this directory

| File | Purpose |
|---|---|
| `activate.sh` | The canonical env. Sourcing it sets modules, conda, MJ_PATH, PYTHONPATH, BLAS pins, and every `NAMO_*` path. All sbatch scripts source it too. |
| `run_amarel_collect.slurm` | Single-shard sbatch. Requires `MANIFEST=<path>` at submit. Optional: `START_IDX`, `END_IDX`, `OUTPUT_DIR`, `CONFIG_YAML`. |
| `run_amarel_collect_array.slurm` | Array sbatch (default `--array=0-29`). Same overrides + `SHARD_SIZE` (default 1000), `GLOBAL_START` (default 0), `EXTRA_ARGS`. |
| `run_full_namo_solvability_unfiltered.slurm` | CPU-only exact-`n` full-pipeline solvability sweep. Auto-discovers path lengths in an XML dir and writes one output subdir per `n`. |
| `README.md` | This file. |

Companion configs (sibling, not in this dir):
- `python/namo/data_collection/region_opening_amarel_car.yaml` — diff-drive car defaults for `region_opening`.
- `config/namo_config_car.yaml` — car robot adapter config.

## Workspace layout & env vars

`activate.sh` exports the values below. Each uses `${VAR:-default}`, so any
of them can be overridden by exporting before sourcing — no need to edit
the script.

| Variable | Default | Purpose |
|---|---|---|
| `NAMO_PARENT` | `/cache/home/dm1487/projects/namo` | Parent dir holding all NAMO repos |
| `NAMO_REPO` | `$NAMO_PARENT/namo_cpp` | This repo |
| `NAMO_SAGE` | `$NAMO_PARENT/sage_learning` | Diffusion / flow-matching training |
| `NAMO_ENV_CREATOR` | `$NAMO_PARENT/mujoco_env_creator` | XML env generator |
| `NAMO_DATA_ROOT` | `/scratch/dm1487` | Big-data root |
| `NAMO_DATASETS` | `$NAMO_DATA_ROOT/datasets` | Env XML pools (`car_envs/v1/`, …) |
| `NAMO_MANIFESTS` | `$NAMO_DATA_ROOT/manifests` | Per-version manifest files |
| `NAMO_OUTPUTS` | `$NAMO_DATA_ROOT/outputs` | Collection PKLs |
| `NAMO_LOGS` | `$NAMO_DATA_ROOT/logs` | Slurm stdout/stderr |
| `NAMO_CONDA_ENV` | `/scratch/dm1487/envs/namo` | Conda environment |
| `MJ_PATH` | `/scratch/dm1487/mujoco/mujoco-3.2.7` | MuJoCo install |

**Filesystem rule:** code on `/cache/home` (NFS, persistent, no purge). Datasets,
outputs, conda envs, MuJoCo on `/scratch` (large, fast, **not backed up**,
90-day inactive purge — touch files or stash a tar on `/home` monthly).

## First-time setup (one-time per Amarel account)

Two steps: sibling repos → canonical `.so`. (Cross-cluster data transfer is
not a setup step — see the "Staging envs from ilab" section below.)

### 1. Sibling repos

Clone alongside this repo so generator + training paths resolve:

```bash
cd "${NAMO_PARENT:-/cache/home/dm1487/projects/namo}"
[ -d mujoco_env_creator ] || git clone https://github.com/dhruvmetha/mujoco_env_creator.git
[ -d sage_learning ]      || git clone https://github.com/dhruvmetha/sage_learning.git
```

### 2. Build the canonical `.so` (portable across CPU generations)

```bash
unset SLURM_JOB_ID
srun --partition=main --cpus-per-task=8 --mem=16G --time=1:00:00 --pty bash
source "$NAMO_REPO/scripts/amarel/activate.sh"
NAMO_MARCH=x86-64-v3 ./build_python_bindings.sh
exit
```

Result: `$NAMO_REPO/build_python/namo_rl*.so` — every shard loads this same
binary at startup.

| Knob | Why |
|---|---|
| `NAMO_MARCH=x86-64-v3` | Default `-march=native` bakes in build-node ISA → shards on older CPUs SIGILL. `v3` (Haswell+: AVX2/FMA/BMI2) runs everywhere on Amarel at ~few-% cost vs. native, lost in MuJoCo noise. |
| Build OS = run OS | el7 build for `--partition=main`, el9 for `--partition=main-redhat`. libstdc++/glibc ABIs don't cross. |

The slurm scripts hard-fail if `build_python/namo_rl*.so` is missing, so a
stale or absent binding can never silently launch a 30-shard array.

## Staging envs from ilab → Amarel

ilab.cs.rutgers.edu's sshd does not accept SSH public-key auth (server
policy: only `password`, `keyboard-interactive`, `gssapi-keyex`,
`gssapi-with-mic`). That makes Amarel→ilab `rsync` impractical to script.

**Solution: invert the direction. Push from ilab → Amarel.** Amarel's sshd
accepts public-key auth normally, so you set up ilab→Amarel key auth once
and `rsync` runs unattended afterwards.

### One-time setup (from your laptop)

```bash
ssh dm1487@ilab.cs.rutgers.edu                                  # password to ilab
[ -f ~/.ssh/id_ed25519 ] || ssh-keygen -t ed25519 -N "" -f ~/.ssh/id_ed25519
ssh-copy-id dm1487@amarel.rutgers.edu                           # ONE Amarel password
ssh dm1487@amarel.rutgers.edu hostname                          # verify, no prompt
exit
```

### Stage a version

From your laptop, in one line — runs the rsync on ilab, lands on Amarel:

```bash
ssh dm1487@ilab.cs.rutgers.edu \
  'rsync -av --info=progress2 --no-compress --inplace \
    /common/users/dm1487/corl2026/namo/envs/v1/ \
    dm1487@amarel.rutgers.edu:/scratch/dm1487/datasets/car_envs/v1/'
```

For large transfers, wrap it in tmux on ilab so it survives your laptop sleeping:
```bash
ssh dm1487@ilab.cs.rutgers.edu
tmux new -s namo_rsync
rsync -av --info=progress2 --no-compress --inplace \
    /common/users/dm1487/corl2026/namo/envs/v1/ \
    dm1487@amarel.rutgers.edu:/scratch/dm1487/datasets/car_envs/v1/
# Ctrl-b d to detach; reattach later with: tmux attach -t namo_rsync
```

### Build the manifest (on Amarel, after rsync finishes)

```bash
# Inside an srun + after source scripts/amarel/activate.sh
python scripts/generate_xml_manifest.py \
    --input-dir $NAMO_DATASETS/car_envs/v1 \
    --output    $NAMO_MANIFESTS/car_envs_v1.txt
```

## Quickstart — submitting shards

Once v1 is staged and its manifest is built:

```bash
# Grab a compute node, source env
unset SLURM_JOB_ID
srun --partition=main --cpus-per-task=4 --mem=8G --time=2:00:00 --pty bash
source /cache/home/dm1487/projects/namo/namo_cpp/scripts/amarel/activate.sh

# Smoke test (2 envs, foreground, ~minutes)
python python/namo/data_collection/modular_parallel_collection.py \
    --config-yaml python/namo/data_collection/region_opening_amarel_car.yaml \
    --workers 2 --start-idx 0 --end-idx 2 \
    --manifest "$NAMO_MANIFESTS/car_envs_v1.txt" \
    --output-dir "$NAMO_OUTPUTS/smoke" --run-name smoke

exit   # leave the interactive srun

# Submit one shard (32 cores × 1000 envs × ≤24h)
MANIFEST=$NAMO_MANIFESTS/car_envs_v1.txt \
OUTPUT_DIR=$NAMO_OUTPUTS/car_v1 \
sbatch $NAMO_REPO/scripts/amarel/run_amarel_collect.slurm

# Or fire the full array (30 shards × 1000 envs, scheduled independently)
MANIFEST=$NAMO_MANIFESTS/car_envs_v1.txt \
OUTPUT_DIR=$NAMO_OUTPUTS/car_v1 \
sbatch $NAMO_REPO/scripts/amarel/run_amarel_collect_array.slurm

# Override planner params per-run via EXTRA_ARGS
EXTRA_ARGS="--region-max-chain-depth 1 --search-timeout 120" \
MANIFEST=$NAMO_MANIFESTS/car_envs_v1.txt \
OUTPUT_DIR=$NAMO_OUTPUTS/car_v1_fast \
sbatch $NAMO_REPO/scripts/amarel/run_amarel_collect_array.slurm
```

`MANIFEST=` is required at submit (no default). Each version of the dataset
gets its own manifest, and forcing the path at submit makes the run record
self-explanatory in slurm logs.

---

## 1. Cluster layout (what you're actually asking for)

### Login nodes vs compute nodes

- **Login nodes** (`amarel1`/`amarel2`): SSH endpoints only. **Never run compute, builds, or long file walks here.** OARC explicitly forbids it.
- **Compute nodes** (`hal*`, `gpu*`, `mem*`, etc.): reached via `sbatch` (queued) or `srun` (interactive/streamed). All your code runs here.
- Filesystems visible from both: `/home/dm1487` (100 GB, backed up), `/scratch/dm1487` (1 TB soft / 2 TB hard, **not** backed up, 90-day inactive purge), `/cache/home/dm1487` (NFS), `/projects/<group>` (if granted).

### Partitions (live snapshot)

Every workload type exists in two flavours: a plain partition (CentOS 7 image) and a `-redhat` partition (RHEL 9.6 image). See §1.5 for the OS split.

| Partition | OS | Time | RAM/node | Cores/node | GPUs/node | Notes |
|---|---|---|---|---|---|---|
| `main`* | CentOS 7 | 3 d | 256 GB | 64 | none | Default. ~60 nodes. Piscataway. |
| `main-redhat` | RHEL 9.6 | 3 d | 192–512 GB | 32–64 | none | **The big one.** ~415 nodes, all Intel generations including Emerald Rapids. |
| `gpu` | CentOS 7 | 3 d | 250 GB | 32 | 4 | Only ~10 nodes. Use only if you need el7 specifically. |
| `gpu-redhat` | RHEL 9.6 | 3 d | 190 GB–515 GB | 24–64 | 2–4 | **The big GPU pool.** A100, L40S, V100. |
| `mem` | CentOS 7 | 3 d | 2 TB | 64 | none | Single node. |
| `mem-redhat` | RHEL 9.6 | 3 d | 1–2 TB | 40–64 | none | 12 high-memory nodes. |
| `nonpre` | CentOS 7 | 3 d | 192 GB | 32 | none | Non-preemptible (16 nodes). |
| `graphical` | CentOS 7 | 1 d | 256 GB | 64 | none | OnDemand desktops. |
| `cmain` | CentOS 7 | 3 d | 192–256 GB | 32–64 | none | Camden, small. |
| `cmain-redhat` | RHEL 9.6 | 3 d | 192–256 GB | 32–64 | none | Camden, RHEL 9. |
| `cgpu-redhat` | RHEL 9.6 | 3 d | 192 GB–1.5 TB | 40–52 | 2–4 | Camden GPUs (A100, V100). |

\*default partition

### 1.5 The OS split — CentOS 7 vs RHEL 9 (what `-redhat` actually means)

Amarel is in the middle of migrating off CentOS 7 (EOL June 2024). The result is a **two-OS cluster** with hardware split across two parallel partition trees:

- **Plain partitions** (`main`, `gpu`, `mem`, `cmain`) → nodes running **CentOS 7** (kernel `3.10.0-…el7.x86_64`).
- **`-redhat` partitions** (`main-redhat`, `gpu-redhat`, etc.) → nodes running **RHEL 9.6** (kernel `5.14.0-…el9_6.x86_64`).

**Why you should default to `-redhat`:**
1. ~80 % of total cores live on `-redhat` nodes.
2. The newest hardware (Emerald Rapids CPUs, NDR InfiniBand, L40S GPUs — the "Clark" upgrade) is RHEL-9 only.
3. CentOS 7 is end-of-life; the el7 partitions will shrink and eventually vanish.

**Why you'd ever pick a plain partition:**
- You have legacy binaries built against el7 glibc that don't run on RHEL 9.
- Your queue is shorter (less contention) on the plain partitions some days.

**Login-node trap:** SSH into `amarel.rutgers.edu` round-robins you across the four login nodes (`amarel1`–`amarel4`). Some are CentOS 7, some are RHEL 9. Always check what you've got:
```bash
cat /etc/os-release          # which OS?
hostname                      # which login node?
```
If you compile native code on a CentOS 7 login node, the binary may not run on RHEL 9 compute nodes (different glibc, different CUDA driver). Three safe paths:
- **Python + conda env** — conda packages are mostly OS-agnostic. This is what `scripts/amarel/activate.sh` does. Safe on either OS.
- **Build inside a compute allocation** — `srun --partition=main-redhat …` then build there. Then sbatch into `*-redhat` partitions only.
- **Containers** — Apptainer (formerly Singularity) is the cluster standard. Build once, run anywhere.

### Hardware features (`--constraint=...`)

CPU generations available — pick if you need a specific ISA:

- `skylake` (oldest)
- `cascadelake`
- `icelake`
- `sapphirerapids`
- `emeraldrapids` (Clark, newest — preferred for raw throughput)

GPU types (use BOTH `--partition=gpu-redhat` AND `--constraint=<type>`):

- `volta` — NVIDIA V100, 16 GB. Only 3 nodes × 2 GPUs, on `gpu-redhat`. Older but reliable.
- `ampere` — NVIDIA A100, 40 or 80 GB. ~15 nodes, 2–4 GPUs each. Best for memory-heavy training.
- `adalovelace` — NVIDIA L40S, 48 GB VRAM. ~30 nodes, 3–4 GPUs each. **Biggest pool, newest hardware.**

GPU inventory at a glance (live `sinfo` snapshot, May 2026):

| Architecture | Partition | Nodes | GPUs/node | Total GPUs | VRAM |
|---|---|---|---|---|---|
| L40S (`adalovelace`) | `gpu-redhat` | ~30 | 3–4 | ~90 | 48 GB |
| A100 (`ampere`) | `gpu-redhat` | ~15 | 2–4 | ~40 | 40 / 80 GB |
| V100 (`volta`) | `gpu-redhat` | 3 | 2 | 6 | 16 GB |
| L40S (`adalovelace`) | `gpu` (el7) | 6 | 4 | 24 | 48 GB |
| A100 (`ampere`) | `gpu` (el7) | 4 | 4 | 16 | 40 GB |

Some nodes carry the `oarc` feature — those are condo nodes owned by OARC and may preempt your job if an OARC user submits. Look at the `AvailableFeatures` column from `scontrol show node <name>`.

Interconnect (rarely useful unless doing multi-node MPI): `edr` (100 Gbps), `hdr` (200 Gbps), `ndr` (400 Gbps).

### Important constant: NO HYPERTHREADING

Every Amarel node has `ThreadsPerCore=1`. Each "CPU" in SLURM's view is **one physical core**. `--cpus-per-task=32` = 32 real cores. Don't double-count or compensate.

---

## 2. CPU / thread model

SLURM separates three concepts. Pick the right combination for your code.

| Concept | Flag | Meaning |
|---|---|---|
| Nodes | `--nodes=N` | How many physical machines |
| Tasks | `--ntasks=K` or `--ntasks-per-node=K` | How many independent **processes** (MPI ranks) |
| CPUs per task | `--cpus-per-task=C` | Cores given to each task (use for threads / subprocesses) |

### Four common shapes for our work

**A. Single-node, multi-process (`multiprocessing.Pool`)** — what `modular_parallel_collection.py` is.
```bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32     # 32 worker pool
#SBATCH --mem=64G
export OMP_NUM_THREADS=1       # pin BLAS so it doesn't fight the pool
python collect.py --workers $SLURM_CPUS_PER_TASK
```
Why `ntasks=1` + big `cpus-per-task`: SLURM only spawns the python *parent* once; the parent forks N pool workers itself. SLURM just reserves the cores.

**B. Single-node, multi-thread (OpenMP / BLAS / pthreads)** — most numerical C++.
```bash
#SBATCH --nodes=1 --ntasks=1
#SBATCH --cpus-per-task=32
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
./my_program
```

**C. Multi-node MPI** — only if your code is actually MPI.
```bash
#SBATCH --nodes=4 --ntasks-per-node=64
#SBATCH --cpus-per-task=1
srun ./mpi_program
```

**D. Job array (independent shards)** — replicate the same recipe across many shards.
```bash
#SBATCH --array=0-29           # 30 array tasks
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=32
START=$((SLURM_ARRAY_TASK_ID * 1000))
python collect.py --start-idx $START --end-idx $((START+1000))
```
Each array task gets its own 32-core allocation; SLURM schedules them independently. Logs separate via `--output=name-%A_%a.out`.

### Memory rules

- `--mem=64G` — **total** memory for the entire job step.
- `--mem-per-cpu=4G` — per allocated CPU; total = `cpus-per-task * mem-per-cpu`.
- **Use one, not both.** Prefer `--mem` unless you genuinely scale RAM with cores.
- Default if you omit: ~3 GB/core. Almost always too little; set it.

### Sharing vs exclusive

- Default: nodes are **shared** with other users' jobs. You only get the resources you asked for, but the rest of the node is contended (cache, memory bandwidth).
- `--exclusive` locks the whole node. Use only if you can use all of it (otherwise you waste).

### Anti-pattern checklist

- ❌ `--cpus-per-task=32 --ntasks=32` → asks for 32×32 = 1024 cores. Almost never what you want.
- ❌ `--workers=64` with `--cpus-per-task=32` → pool oversubscribes the allocation.
- ❌ Forgetting `OMP_NUM_THREADS=1` when using `multiprocessing.Pool` with NumPy/SciPy → each pool worker spawns 32 BLAS threads on top, mass contention.
- ❌ Submitting from a stale shell with `SLURM_JOB_ID` still set → `srun` will reuse the (dead) old job ID. Run `unset SLURM_JOB_ID` first or open a fresh shell.

---

## 3. GPU model

GPU jobs **always** need both flags:

```bash
#SBATCH --partition=gpu-redhat       # almost always what you want; not plain "gpu"
#SBATCH --gres=gpu:1
```

`--gres=gpu:1` = "any 1 GPU." To target a specific architecture:

```bash
#SBATCH --partition=gpu-redhat
#SBATCH --gres=gpu:1
#SBATCH --constraint=adalovelace     # L40S, biggest pool
# or constraint=ampere|adalovelace   # either A100 or L40S — schedules faster
```

To request 2 GPUs on the same node:
```bash
#SBATCH --gres=gpu:2
```

### Checking what GPUs are free right now

Run these on a login node (they're cheap, read-only):

```bash
# Per-partition node state — idle (all GPUs free) / mix (some free) / alloc (full)
sinfo -p gpu-redhat -o "%20N %10t %25f %15G"

# Just the nodes that have slots open
sinfo -p gpu,gpu-redhat -t idle,mix -o "%N %t %G %f"

# Who's hogging what
squeue -p gpu-redhat -o "%.10i %.8u %.8T %.10M %.6D %R %b"

# How many GPUs are actually free on a specific node
scontrol show node gpuk006 | grep -E "Gres|AllocTRES|State"
```

Reading the output: `mix` means some GPUs on the node are still free — your single-GPU job will land there. `alloc` is full. Asking for `--gres=gpu:4` will wait far longer than `--gres=gpu:1` because few nodes have all 4 cards free at once.

### GPU training recipe (PyTorch / JAX / generic ML)

```bash
#!/bin/bash
#SBATCH --job-name=ml-train
#SBATCH --partition=gpu-redhat
#SBATCH --gres=gpu:1
#SBATCH --constraint=ampere|adalovelace   # either A100 or L40S — schedules sooner
#SBATCH --nodes=1 --ntasks=1
#SBATCH --cpus-per-task=8                 # DataLoader workers / preprocessing
#SBATCH --mem=64G                         # system RAM (NOT GPU VRAM)
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/dm1487/logs/train-%j.out
#SBATCH --error=/scratch/dm1487/logs/train-%j.err

set -euo pipefail
mkdir -p /scratch/dm1487/logs

module use /projects/community/modulefiles
# Optional: load a CUDA module. Most conda envs ship their own CUDA toolkit.
# module load cuda/12.4 cudnn/8.9

source /cache/home/dm1487/miniforge3/etc/profile.d/conda.sh
conda activate /scratch/dm1487/envs/sage   # or whatever env

cd /cache/home/dm1487/projects/sage-learning
python train.py --config configs/default.yaml
```

Multi-GPU on a single node (DDP / `torchrun`):
```bash
#SBATCH --partition=gpu-redhat
#SBATCH --gres=gpu:2                      # 2 GPUs on same node
#SBATCH --constraint=adalovelace
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G

torchrun --standalone --nproc_per_node=2 train.py
```

### GPU caveats

- `nvidia-smi` inside a job only sees the GPUs SLURM gave you. If it returns nothing, your `--gres` request silently failed and you landed on a CPU-only node.
- VRAM is separate from `--mem`. `--mem=64G` is system RAM. VRAM is fixed by the GPU model (40/80 GB on A100, 48 GB on L40S, 16 GB on V100).
- The GPU queue is the most contended on the cluster. Smaller asks (`--gres=gpu:1`) almost always schedule within minutes; 4-GPU asks can wait hours.
- Module names are case sensitive; `module avail cuda` shows what's installed.
- Conda envs are recommended over system CUDA modules — the env's bundled CUDA runtime is OS-agnostic and works on both el7 and el9 nodes.

---

## 4. NAMO-specific recipes

The first-time setup and Quickstart at the top of this file cover the
common path. Below are situational variations.

### Fatter pool per node (>32 workers)

Edit the sbatch directives:
```bash
#SBATCH --cpus-per-task=64           # one whole node
#SBATCH --exclusive                  # lock it down
#SBATCH --mem=200G
```
Throughput scales roughly linearly with workers — `region_opening` physics
steps are independent across workers, no shared global state.

### Switching to RHEL 9 (`--partition=main-redhat`)

Two things:
1. Rebuild `$NAMO_REPO/build_python/` on a `main-redhat` srun — el7 and el9
   libstdc++/glibc ABIs are incompatible.
2. Change one line in the sbatch script: `#SBATCH --partition=main-redhat`.

Verify the binding loads on the target partition:
```bash
srun --partition=main-redhat --pty bash -c \
    "source $NAMO_REPO/scripts/amarel/activate.sh && \
     python -c 'import namo_rl; print(namo_rl.__file__)'"
```

### GPU for ML goal strategy (region_opening with `goal_strategy=ml*`)

The diffusion model wants CUDA. Switch partition + add gres; CPU count
can stay modest since the heavy lifting moves to the GPU.
```bash
#SBATCH --partition=gpu-redhat
#SBATCH --gres=gpu:1
#SBATCH --constraint=ampere|adalovelace
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
# add to EXTRA_ARGS: --goal-strategy ml --ml-goal-model <hydra-dir>
```

### Staging a new env version (e.g. `v2`)

Same flow as v1 above — change the version in the path. From your laptop:
```bash
ssh dm1487@ilab.cs.rutgers.edu \
  'rsync -av --info=progress2 --no-compress --inplace \
    /common/users/dm1487/corl2026/namo/envs/v2/ \
    dm1487@amarel.rutgers.edu:/scratch/dm1487/datasets/car_envs/v2/'
```
Then on Amarel (inside an srun + sourced env):
```bash
python scripts/generate_xml_manifest.py \
    --input-dir $NAMO_DATASETS/car_envs/v2 \
    --output    $NAMO_MANIFESTS/car_envs_v2.txt
```
Versions coexist — submit shards against whichever you want via `MANIFEST=` at submit time.

### Keeping `/scratch` from being purged

OARC garbage-collects `/scratch` files inactive for 90 days (per-file `atime`).
For datasets you don't touch weekly:
```bash
# Touch a sentinel monthly from anywhere on Amarel
touch $NAMO_DATASETS/car_envs/v1/.keep $NAMO_CONDA_ENV/.keep $MJ_PATH/.keep

# Or stash a compressed copy of the manifest + a sample env on /home as insurance
tar c $NAMO_MANIFESTS | zstd > ~/backups/manifests-$(date +%Y%m%d).tar.zst
```

---

## 5. Day-to-day commands

```bash
# What's queued/running for you
squeue -u dm1487

# Detailed status of one job
scontrol show job <jobid>

# Why is my job pending?
squeue -u dm1487 --format="%.18i %.9P %.8j %.8T %.10M %.6D %.20R"
# The last column "R" lists the reason (Resources, Priority, QOSGrpNodeLimit, etc.)

# Cancel a job (or all your jobs)
scancel <jobid>
scancel -u dm1487

# Real-time interactive shell on a compute node
unset SLURM_JOB_ID
srun --partition=main-redhat --cpus-per-task=4 --mem=8G --time=2:00:00 --pty bash

# Attach to a *running* job (debug from another shell)
srun --jobid=<id> --overlap bash -c 'ps -ef --forest -u $USER'

# Historic info — finished/failed jobs
sacct -X -j <jobid> --format=JobID,State,Elapsed,MaxRSS,ReqMem,NCPUS,Partition,NodeList
sacct -X -u dm1487 -S today

# What partitions exist (with time limits and idle counts)
sinfo -s

# What features each node advertises
sinfo -o "%P %D %c %m %G %f" | sort -u

# Which GPUs have slots free *right now*
sinfo -p gpu,gpu-redhat -t idle,mix -o "%N %t %G %f"

# What OS is this login/compute node on?
cat /etc/os-release | grep PRETTY_NAME      # CentOS 7  vs  Red Hat 9.x
uname -r                                     # el7  vs  el9_6

# Quota on /scratch (Amarel uses GPFS)
mmlsquota --block-size=auto cache:scratch
```

---

## 6. Cost / etiquette

- **Pick the smallest box that fits.** A 256-core, 4-GPU request schedules slowly and wastes others' time.
- **Time limits are caps, not goals.** Asking 24h for a 1h job is fine; the scheduler uses the cap to plan. But asking 3 days for a 6-min job blocks shorter slots from filling around you.
- **Avoid the login node** for compute (build, tar of big trees, du of big trees). Login nodes get killed by OARC if they spike.
- **One job per shard** beats one giant job. Arrays + fail-isolated shards survive single-node hiccups; a single 256-core job loses everything on one OOM.

---

## 7. References

- Amarel Cluster User Guide: <https://sites.google.com/view/cluster-user-guide/amarel/welcome>
- OARC Amarel overview: <https://oarc.rutgers.edu/resources/amarel/>
- Rutgers Amarel Cluster Tutorial (community): <https://github.com/LMC4S/Rutgers-Amarel-Cluster-Tutorial>
- OARC support: help@oarc.rutgers.edu
