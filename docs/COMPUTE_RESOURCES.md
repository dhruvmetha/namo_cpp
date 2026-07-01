# COMPUTE_RESOURCES.md — where to run NAMO compute (and how to switch)

> Every machine/cluster we can run on, how to reach each, how to submit + monitor jobs, and how to fall
> back when one is busy. **Kept in two places:** this repo (`docs/COMPUTE_RESOURCES.md`) **and**
> `/common/home/dm1487/COMPUTE_RESOURCES.md` (visible from any CS iLab box even without the repo — keep
> them in sync; the repo copy is canonical). Verified 2026-07-01 from arrakis.

## TL;DR — pick by availability

| Need | Use | Sync cost |
|---|---|---|
| A few GPUs now, small/interactive | **arrakis (here), direct GPUs** — `CUDA_VISIBLE_DEVICES=N` | none (local) |
| More GPUs / arrakis busy | **CS iLab SLURM `ilab1`** — `ssh ilab1 … sbatch` on the SAME repo | **none — shared FS** ⭐ |
| A specific idle CS box | **westeros**, direct GPUs (shared FS) | none |
| CS iLab full / want HPC scale | **Amarel SLURM** | push → pull → rebuild (separate FS) |

**Golden rule:** all CS iLab boxes (arrakis, westeros, ilab1, rlab, …) **share `/common/home` + `/common/users`**,
so the repo + data are identical everywhere — moving work between them needs **zero copying**. **Amarel is a
separate world** (own filesystem, own auth) — it needs a git sync + a C++ rebuild.

---

## 0. The two worlds

**World 1 — Rutgers CS iLab** (where we are). Shared NFS: `/common/home/dm1487` (home + repo),
`/common/users/dm1487` (scratch). Auth = **Kerberos**.
- `arrakis`, `westeros`, … = **direct-GPU boxes, NO scheduler** — ssh in, run on their GPUs.
- `ilab1`, `rlab`, … = **SLURM submit hosts** — ssh in, `sbatch` to the iLab GPU cluster.

**World 2 — Amarel** (Rutgers OARC HPC). Separate NFS: `/home/dm1487` (= `/cache/home/dm1487`, cached on
compute nodes), `/scratch/dm1487`. Auth = **SSH key**. SLURM.

---

## 1. Auth cheat-sheet

### CS iLab — **Kerberos / GSSAPI (NOT ssh keys)**
- The gateway has **publickey auth DISABLED**. Do **not** try to add SSH keys (that's the Amarel method) —
  it always fails: `Permission denied (gssapi-keyex,gssapi-with-mic,password,keyboard-interactive)`.
- It uses **Kerberos**. Your NetID login grants a ticket → intra-iLab SSH is then passwordless.
  Check: `klist` → want `Default principal: dm1487@CS.RUTGERS.EDU`, unexpired.
- **Use a CONCRETE hostname:** `ssh ilab1.cs.rutgers.edu` ✅ — **not** the round-robin alias
  `ssh ilab.cs.rutgers.edu` ❌ (GSSAPI needs the exact host's service principal; the alias breaks it).
- No / expired ticket? `kinit dm1487@CS.RUTGERS.EDU` (needs your password — interactive). Tickets last ~a day.

### Amarel — **SSH key**
- `ssh amarel` (in `~/.ssh/config` → `Host amarel` → `id_ed25519_amarel`, key authorized on Amarel). Passwordless.

---

## 2. Filesystem map

| | CS iLab (shared) | Amarel (separate) |
|---|---|---|
| home | `/common/home/dm1487` | `/home/dm1487` (login) = `/cache/home/dm1487` (compute cache) |
| **repo (namo)** | `/common/home/dm1487/robotics_research/ktamp/namo` | `/home/dm1487/projects/namo/namo_cpp` |
| sage_learning | `…/ktamp/sage_learning` | `/home/dm1487/projects/namo/sage_learning` |
| scratch | `/common/users/dm1487/scratch_namo` (`$NAMO_SCRATCH`) | `/scratch/dm1487` |
| env | `source env.ilab.sh` | Amarel env (see `CLAUDE.amarel.md`) |
| git version | modern | **old 1.8.3.1** — avoid `--show-current` / `-C`; use `git rev-parse --abbrev-ref HEAD` |

---

## 3. Option A — Direct GPU on a CS box (arrakis / westeros) — no scheduler

Simplest; you own the GPUs you grab. **Courtesy: ≤ half the cores** on shared boxes.

```bash
# on arrakis (here) — or: ssh westeros.cs.rutgers.edu  (same repo, same FS, no copy)
cd /common/home/dm1487/robotics_research/ktamp/namo && source env.ilab.sh
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader   # find free GPUs
CUDA_VISIBLE_DEVICES=2 python <train/eval …>                                     # pin to a free one
```
- No job to monitor — it's a plain process (background long runs).
- arrakis today: **5× RTX 6000 Ada (48 GB)**. westeros: check `nvidia-smi` (same code, shared FS).

---

## 4. Option B — CS iLab SLURM (ilab1 / rlab) — shared FS → **NO sync** ⭐

The easiest scale-out. Because the FS is shared, you `sbatch` a job that runs on **this same checkout** —
nothing to copy; output lands in `/common/…` and is readable from arrakis instantly.

```bash
ssh ilab1.cs.rutgers.edu            # Kerberos, passwordless (concrete host!). Same /common/home.
sinfo -o "%.14P %.5a %G" | head     # partitions / GPU types
squeue -u dm1487                    # your jobs
```
Submit script (`job.sbatch`) — runs directly on the shared repo:
```bash
#!/bin/bash
#SBATCH --job-name=namo
#SBATCH --partition=unlimited     # default iLab partition (infinite time). 'guest' = borrow idle group nodes.
#SBATCH --gres=gpu:1              # or a type: --gres=gpu:a5000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=/common/users/dm1487/scratch_namo/slurm/%x_%j.out
cd /common/home/dm1487/robotics_research/ktamp/namo && source env.ilab.sh
python <your command>
```
GPU types seen on the `unlimited` partition: `a100, a4000, a4500, a5000, a6000, 4500_ada, 5000_Blackwell,
6000_Blackwell, H200`. (Bindings built on arrakis run on these — same shared `.so`.)
```bash
mkdir -p /common/users/dm1487/scratch_namo/slurm
sbatch job.sbatch                                     # -> Submitted batch job NNNN
squeue -u dm1487 -j NNNN                               # watch
sacct -j NNNN --format=JobID,State,Elapsed,ExitCode   # final status
# output is already on /common/... -> read it straight from arrakis. No copy.
```
**Drive it from arrakis without leaving your session:**
```bash
ssh ilab1.cs.rutgers.edu 'cd /common/home/dm1487/robotics_research/ktamp/namo && sbatch job.sbatch'
ssh ilab1.cs.rutgers.edu 'squeue -u dm1487'
```

---

## 5. Option C — Amarel SLURM — separate FS → sync + rebuild

Use when CS iLab is saturated. Costs a git round-trip + a C++ rebuild, and you must **not clobber the
parallel Amarel session's checkout** — use a **dedicated clone**.

```bash
# 1) on CS iLab: commit + push
git push origin <branch>

# 2) on Amarel: pull into a DEDICATED clone (create once: git clone <origin> namo_run)
ssh amarel 'git -C /home/dm1487/projects/namo/namo_run pull'

# 3) rebuild C++ ONLY if src/include/bindings changed (pull does NOT rebuild the .so)
ssh amarel 'cd /home/dm1487/projects/namo/namo_run && source <amarel env> && ./build_python_bindings.sh'

# 4) submit to a GPU partition (policy: gpu / gpu-redhat; never Camden; don't wait >1h — relax/resubmit)
ssh amarel 'cd /home/dm1487/projects/namo/namo_run && sbatch scripts/amarel/<job>.slurm'
ssh amarel 'squeue -u dm1487'

# 5) pull results back to CS iLab
rsync -avhP amarel:/scratch/dm1487/<results>/ /common/users/dm1487/scratch_namo/<results>/
```
Amarel GPU partitions: `gpu-redhat` (many nodes), `legacy-gpu`, `cgpu-redhat`. CPU-only test: `main`.
Data pull helper: `scripts/portability/pull_from_amarel.sh {eval|train}`.

---

## 6. Switching / fallback logic

1. **Start on arrakis** — 5× RTX 6000 Ada, zero setup, direct.
2. arrakis busy / need more GPUs → **`ilab1` SLURM** (same files, just `sbatch`). First fallback, ~zero cost. ⭐
3. Want a specific idle box → **westeros** (direct, shared FS).
4. All of CS iLab saturated / want HPC scale → **Amarel** (accept push/pull/rebuild; use the dedicated clone; mind the GPU backlog that sent us to ilab in the first place).

## 7. Gotchas (all bit us on 2026-07-01 — don't re-learn)

- **CS iLab = Kerberos, not keys.** Adding SSH keys there NEVER works (publickey disabled). Need a `klist`
  ticket; SSH to **concrete** hostnames (`ilab1`, not `ilab`).
- **Amarel = keys + separate FS.** Old git (1.8.3.1); rebuild bindings after code changes; coordinate with
  the parallel Amarel Claude session (its checkout is `/home/dm1487/projects/namo/namo_cpp` on
  `feat/horizon-q-redesign` — use your **own** clone); GPU queue can backlog.
- **CPU courtesy** on shared direct boxes (arrakis/westeros): **≤ half the cores**.
- **Kerberos tickets expire (~daily)** — `kinit` to renew if `ssh ilab1` suddenly asks for a password.
- **Amarel data ≠ CS iLab data** — the test set / eval keys live on Amarel; pull with
  `scripts/portability/pull_from_amarel.sh` (see it for exact paths).
