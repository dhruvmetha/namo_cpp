# Amarel SLURM Quick Reference for NAMO

Practical guide for requesting CPUs, threads, and GPUs on Rutgers Amarel.
Everything below is verified against the live cluster (`sinfo`, `scontrol`), not just docs.

## Files in this directory

| File | Purpose |
|---|---|
| `activate.sh` | Source inside an `srun --pty bash` shell to set MJ_PATH, conda env, PYTHONPATH, BLAS thread caps. |
| `run_amarel_collect.slurm` | Single-job sbatch — one shard. Override via env vars (`START_IDX`, `END_IDX`, `OUTPUT_DIR`, `CONFIG_YAML`, `MANIFEST`, `EXTRA_ARGS`). |
| `run_amarel_collect_array.slurm` | Array sbatch — 30 shards × 1000 envs by default. Override `--array=`, `SHARD_SIZE`, etc. |
| `README.md` | This file. |

Companion configuration file (kept with sibling YAMLs, not here):
- `python/namo/data_collection/region_opening_amarel_car.yaml` — Amarel + diff-drive car defaults for `region_opening`.

## Quickstart

```bash
# 1. Grab a compute node and source the env
unset SLURM_JOB_ID
srun --partition=main --cpus-per-task=4 --mem=8G --time=2:00:00 --pty bash
source scripts/amarel/activate.sh

# 2. Submit a single shard (defaults: envs 0..1000, 32 workers, 24h)
sbatch scripts/amarel/run_amarel_collect.slurm

# 3. Or submit the full array (30 × 1000 envs)
sbatch scripts/amarel/run_amarel_collect_array.slurm

# 4. Override anything via env vars at submit time
EXTRA_ARGS="--region-max-chain-depth 1 --search-timeout 120" \
OUTPUT_DIR=/scratch/dm1487/outputs_fast \
sbatch scripts/amarel/run_amarel_collect_array.slurm
```

---

## 1. Cluster layout (what you're actually asking for)

### Login nodes vs compute nodes

- **Login nodes** (`amarel1`/`amarel2`): SSH endpoints only. **Never run compute, builds, or long file walks here.** OARC explicitly forbids it.
- **Compute nodes** (`hal*`, `gpu*`, `mem*`, etc.): reached via `sbatch` (queued) or `srun` (interactive/streamed). All your code runs here.
- Filesystems visible from both: `/home/dm1487` (100 GB, backed up), `/scratch/dm1487` (1 TB soft / 2 TB hard, **not** backed up, 90-day inactive purge), `/cache/home/dm1487` (NFS), `/projects/<group>` (if granted).

### Partitions (live snapshot)

| Partition | Time | RAM/node | Cores/node | GPUs/node | Notes |
|---|---|---|---|---|---|
| `main`* | 3 d | 192–256 GB | up to 128 | none | Default. Piscataway, shared. |
| `gpu` | 3 d | 250 GB–1.5 TB | 64–128 | 2–4 | Piscataway. GPU jobs only. |
| `mem` | 3 d | 512 GB–2 TB | 64 | none | High-memory. Rare. |
| `nonpre` | 3 d | 256 GB | 64 | none | Non-preemptible (small pool, ~16 nodes). |
| `graphical` | 1 d | 256 GB | 64 | none | OnDemand desktops. |
| `cmain`/`cgpu` | 3 d | 192–256 GB | 64 | 2–4 (cgpu only) | Camden Amarel resources. |
| `*-redhat` | same | same | same | same | RHEL-8 variants of each; identical hardware, newer OS. |

\*default partition

### Hardware features (`--constraint=...`)

CPU generations available — pick if you need a specific ISA:

- `skylake` (oldest)
- `cascadelake`
- `icelake`
- `sapphirerapids`
- `emeraldrapids` (Clark, newest — preferred for raw throughput)

GPU types (use BOTH `--partition=gpu` AND `--constraint=<type>`):

- `volta` — NVIDIA V100 (oldest)
- `ampere` — NVIDIA A100 (most common)
- `adalovelace` — NVIDIA L40S (Clark, newest, 48 GB VRAM)

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
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
```

`--gres=gpu:1` = "any 1 GPU." To target a specific architecture:

```bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=adalovelace     # L40S
# or constraint=ampere|adalovelace   # either A100 or L40S
```

To request 2 GPUs on the same node:
```bash
#SBATCH --gres=gpu:2
```

To request a specific GPU type via gres (alternative syntax):
```bash
#SBATCH --gres=gpu:a100:1            # works if SLURM knows the name; --constraint is safer
```

### GPU resource recipe

```bash
#!/bin/bash
#SBATCH --job-name=ml-train
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=adalovelace
#SBATCH --nodes=1 --ntasks=1
#SBATCH --cpus-per-task=8            # DataLoader workers, etc.
#SBATCH --mem=64G                    # system RAM, separate from GPU VRAM
#SBATCH --time=24:00:00

module use /projects/community/modulefiles
module load cuda/12.4 cudnn/8.9
source ~/miniforge3/bin/activate myenv
python train.py
```

### GPU caveats

- L40S (`adalovelace`) has 48 GB VRAM, no NVLink, ECC. A100 (`ampere`) has 40 or 80 GB depending on node.
- `nvidia-smi` only sees the GPUs SLURM gave you; if it shows nothing, your `--gres` failed and you're on a non-GPU node.
- The GPU queue is contended. Expect wait time. `nonpre` doesn't exist for GPU — owners get priority.
- Module names are case sensitive; `module avail cuda` shows what's installed.

---

## 4. NAMO-specific recipes

All three live at `scripts/amarel/`.

### Smoke test (interactive, validate one shard)
```bash
unset SLURM_JOB_ID
srun --partition=main --cpus-per-task=2 --mem=8G --time=00:30:00 --pty bash
# Inside the shell:
module use /projects/community/modulefiles
module load gcc/14.2.0-cermak cmake/3.31.8-rdp135
source /cache/home/dm1487/miniforge3/etc/profile.d/conda.sh
conda activate /scratch/dm1487/envs/namo
export MJ_PATH=/scratch/dm1487/mujoco/mujoco-3.2.7
export LD_LIBRARY_PATH=$MJ_PATH/lib:$LD_LIBRARY_PATH
export PYTHONPATH=/cache/home/dm1487/projects/namo/namo_cpp/build_python_mjxrl_amarel2:/cache/home/dm1487/projects/namo/namo_cpp/python:$PYTHONPATH
cd /cache/home/dm1487/projects/namo/namo_cpp
python python/namo/data_collection/modular_parallel_collection.py \
    --config-yaml python/namo/data_collection/region_opening_amarel_car.yaml \
    --workers 2 --start-idx 0 --end-idx 2 \
    --manifest /scratch/dm1487/manifests/car_envs.txt \
    --output-dir /scratch/dm1487/outputs --run-name smoke
```

### Single 32-core production job (24h)
```bash
sbatch scripts/amarel/run_amarel_collect.slurm
# override defaults via env vars at submit:
START_IDX=0 END_IDX=1000 sbatch scripts/amarel/run_amarel_collect.slurm
```

### Job array — full 30k envs in 30 shards × 32 cores each
```bash
sbatch scripts/amarel/run_amarel_collect_array.slurm
# Each array task: 32 cores, 1000 envs. 30 tasks scheduled independently.
```

### If you want a fatter pool per node
Edit the sbatch directives:
```bash
#SBATCH --cpus-per-task=64           # one whole node
#SBATCH --exclusive                  # lock it down
#SBATCH --mem=200G
```
With 64 workers, expect ~2× the per-shard throughput of 32-worker runs (assuming the workload scales — region_opening's mujoco physics step doesn't share global state, so it scales well).

### GPU for ML goal strategy (region_opening with `goal_strategy=ml*`)
The diffusion model wants CUDA. Switch the partition + add gres; CPU count can stay modest because the heavy lifting moves to the GPU.
```bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=adalovelace
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
# pass --goal-strategy ml + --ml-goal-model <hydra-dir> in the python command
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
srun --partition=main --cpus-per-task=4 --mem=8G --time=2:00:00 --pty bash

# Attach to a *running* job (debug from another shell)
srun --jobid=<id> --overlap bash -c 'ps -ef --forest -u $USER'

# Historic info — finished/failed jobs
sacct -X -j <jobid> --format=JobID,State,Elapsed,MaxRSS,ReqMem,NCPUS,Partition,NodeList
sacct -X -u dm1487 -S today

# What partitions exist (with time limits and idle counts)
sinfo -s

# What features each node advertises
sinfo -o "%P %D %c %m %G %f" | sort -u

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
