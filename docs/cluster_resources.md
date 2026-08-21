# Amarel cluster resources

Grounded in the Rutgers Amarel user guide (<https://sites.google.com/view/cluster-user-guide>) **and** local `sinfo` / `scontrol` probing on this cluster (2026-06-04). Re-verify with `sinfo -N -o "%N %P %G %f"` if node features change.

## Partitions we use

| Use | Partition(s) | Notes |
|---|---|---|
| **CPU** — collection, NPZ-gen, manifests, analysis scans | `main` | 429 nodes. All heavy ops go here (see [[DATA_COLLECTION_GUIDE]] + [[feedback_slurm_first]]). |
| **GPU** — training | `gpu`, `gpu` | Submit to **both** (`--partition=gpu`) for the fastest start. |

## ⛔ Never Camden

Camden-hosted partitions are **`c`-prefixed**: `cgpu`, `cmain`, `cmem`. **Never submit to `cgpu`.** Every `gpu` node has the `piscataway` feature; Camden nodes don't — so staying on `gpu` already avoids Camden.

## GPU types (from node features)

| GPU | `--constraint` feature | Nodes (2026-06-04) |
|---|---|---|
| **A100** | `ampere` | gpu015–gpu028 |
| **L40S** | `adalovelace` | gpu029–gpu044+ |

Request **either** with the OR-constraint (the guide's own AlphaFold example uses this exact form):
```
--constraint=ampere|adalovelace
```
Most GPU nodes expose `gpu:2`, `gpu:3`, or `gpu:4`, so **2-GPU jobs are widely available.** (2026-06-06: A6000 / A4500-Ada are **not** on Amarel — those are iLab cards. Amarel GPU = L40S + A100 + a few V100.)

## ⚡ Finding a free GPU fast — CPUs are the real bottleneck, not GPUs

**Observation (2026-06-06).** A `--gres=gpu:1 --cpus-per-task=24 --mem=48G` job sat `PENDING(Resources)` for ~20 min while L40S/A100 GPUs sat **idle** — because the idle-GPU nodes already had most CPUs taken by other jobs (e.g. `gpu031`: 3 GPUs free but only 16 of 32 CPUs free → a 24-CPU job can't fit). **Outcome.** Dropping to `--cpus-per-task=8 --mem=32G` (same `--gres=gpu:1`) **landed instantly on gpu018.**

**Rule: to always find a GPU, lean the CPU + mem ask, not just the GPU count — GPU nodes are CPU-contended.** Default lean job that backfills almost anywhere:
```bash
sbatch --partition=gpu --gres=gpu:1 --cpus-per-task=8 --mem=32G ...   # RHEL9 partition; gpu is draining
```
- **Diagnose a stuck job:** `scontrol show job <id>` (`Reason=Resources`) + `scontrol show node <n>` (`CPUTot`/`CPUAlloc`, `AllocMem`). If a node's free CPUs < your `--cpus-per-task`, that's the blocker.
- **Find real free capacity:** `gpufree`, or `sinfo -N -o "%N %t %C %G %f" -p gpu` (`%C` = Alloc/Idle/Other/Total CPUs).
- **Our scorers are tiny (≤~6 M params)** → `gpu:1` + 8 CPUs is plenty. Don't request 24 CPUs / 2 GPUs / 120 G — that's what queues you next to idle GPUs.

## Training GPU policy (standing preference)

> ⚠️ **Reality check (2026-06-06):** the "prefer 2× GPU + 16 cpu / 120 G" recipe below is what *causes*
> long `PD` waits. For our small models, **lead with the lean single-GPU `gpu` job above** and only
> scale up if a model genuinely needs it. Lean-first is the reliable way to "always find a GPU."

1. **Prefer multi-GPU (2×) on A100 or L40S:**
   ```bash
   sbatch --partition=gpu --constraint=ampere|adalovelace \
          --gres=gpu:2 --cpus-per-task=16 --mem=120G --time=08:00:00 <train.slurm>
   ```
2. **Fall back to single GPU** (still A100/L40S) if 2-GPU won't start:
   ```bash
   sbatch --partition=gpu --constraint=ampere|adalovelace \
          --gres=gpu:1 --cpus-per-task=8 --mem=64G --time=09:00:00 <train.slurm>
   ```
3. **Get one quickly / keep looking:** submit to both partitions, add no narrower constraint; if the multi-GPU job stays `PD` (pending) too long, fall back to single GPU — **don't wait indefinitely.**
4. **Never Camden** (`cgpu-*`).

The trainer (`sage_learning/src/train_generative.py trainer=multi_gpu`, `devices=auto`) uses all visible GPUs, so the **same script runs on 1 or 2 GPUs** — only the sbatch `--gres` changes.
