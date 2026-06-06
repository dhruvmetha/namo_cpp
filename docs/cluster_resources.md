# Amarel cluster resources

Grounded in the Rutgers Amarel user guide (<https://sites.google.com/view/cluster-user-guide>)
**and** local `sinfo` / `scontrol` probing on this cluster (2026-06-04). Re-verify with
`sinfo -N -o "%N %P %G %f"` if node features change.

## Partitions we use

| Use | Partition(s) | Notes |
|---|---|---|
| **CPU** — collection, NPZ-gen, manifests, analysis scans | `main-redhat` | 429 nodes. All heavy ops go here (see [[DATA_COLLECTION_GUIDE]] + [[feedback_slurm_first]]). |
| **GPU** — training | `gpu`, `gpu-redhat` | Submit to **both** (`--partition=gpu,gpu-redhat`) for the fastest start. |

## ⛔ Never Camden

Camden-hosted partitions are **`c`-prefixed**: `cgpu-redhat`, `cmain-redhat`, `cmem-redhat`, `cmain`.
**Never submit to `cgpu-redhat`.** Every `gpu`/`gpu-redhat` node has the `piscataway` feature;
Camden nodes don't — so staying on `gpu,gpu-redhat` already avoids Camden.

## GPU types (from node features)

| GPU | `--constraint` feature | Nodes (2026-06-04) |
|---|---|---|
| **A100** | `ampere` | gpu015–gpu028 |
| **L40S** | `adalovelace` | gpu029–gpu044+ |

Request **either** with the OR-constraint (the guide's own AlphaFold example uses this exact form):
```
--constraint=ampere|adalovelace
```
Most GPU nodes expose `gpu:2`, `gpu:3`, or `gpu:4`, so **2-GPU jobs are widely available.**

## Training GPU policy (standing preference)

1. **Prefer multi-GPU (2×) on A100 or L40S:**
   ```bash
   sbatch --partition=gpu,gpu-redhat --constraint=ampere|adalovelace \
          --gres=gpu:2 --cpus-per-task=16 --mem=120G --time=08:00:00 <train.slurm>
   ```
2. **Fall back to single GPU** (still A100/L40S) if 2-GPU won't start:
   ```bash
   sbatch --partition=gpu,gpu-redhat --constraint=ampere|adalovelace \
          --gres=gpu:1 --cpus-per-task=8 --mem=64G --time=09:00:00 <train.slurm>
   ```
3. **Get one quickly / keep looking:** submit to both partitions, add no narrower constraint; if the
   multi-GPU job stays `PD` (pending) too long, fall back to single GPU — **don't wait indefinitely.**
4. **Never Camden** (`cgpu-*`).

The trainer (`sage_learning/src/train_generative.py trainer=multi_gpu`, `devices=auto`) uses all
visible GPUs, so the **same script runs on 1 or 2 GPUs** — only the sbatch `--gres` changes.
