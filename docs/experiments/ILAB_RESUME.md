---
status: frozen
tags: [experiment]
updated: 2026-06-26
---

# SESSION RESUME — pick up here (ilab / fresh chat)

> **⚠ HISTORICAL / SUPERSEDED (2026-07-06).** This "pick up here" note targeted the budget-conditioned Horizon-Q / qboot line, which was **dropped** (horizon-conditioned ≈ no-horizon, **NoHz** ahead — 40.7 vs 34.1). It is superseded by the board + the front door [../problem_and_approach.md](../problem_and_approach.md). Kept as history; do not resume from here.

> Started on Amarel, moved to ilab because Amarel's GPUs were backlogged. The CHAT doesn't carry over — this repo does.
> **Read order:** this page → [horizon_q_redesign_execution.md](horizon_q_redesign_execution.md) (the staged plan + full
> log) → [horizon_q_HANDOFF.md](horizon_q_HANDOFF.md) (arch + algorithm + v2/v3/v4 results) → [horizon_q_model_registry.md](horizon_q_model_registry.md) (ckpt paths).
> A fresh Claude on ilab: just `cd` into this repo and it reads CLAUDE.md, which points here.

## YOU ARE HERE
Project = **search-first redesign of Horizon-Q**: the model is a **sims-minimizing search RANKER**, not a value-classifier; the objective is **cost-to-go in SIMS** (our old γ-labels discounted DEPTH, which is ~constant at H=2 — the wrong cost).
- **✅ Stage 0 (instrumentation) — DONE.** The realized-rank measurement REORDERED the plan: the **SETUP value/ranking is the bottleneck** (setup hard top-1 ~20%, median rank 5); the **FINISH is near-oracle** (hard rank 0 — already solved); **collection is well-covered** ⇒ the finish-ranker AND guided-collection (Stage 2) are NOT the levers. The setup is.
- **▶ Stage 1 (bootstrapped single-Q value, drop Horizon) — BUILT, needs a GPU.** This is the ONE active change.

## THE ONE THING TO DO ON ILAB — train qboot
Single-Q value (Horizon DROPPED), from scratch, on the moved data. Two arms (the density-vs-depth = Stage-3 ablation, free):
```
H5=<ilab>/h5     # = /common/users/dm1487/fresh_start/projects/namo/h5
DATA="$H5/v4_hq_m2b_scorer/data.h5;$(ls $H5/v4_hq_exit_finish_v4/shard_*.h5|paste -sd';' -);$(ls $H5/v4_hq_boot_setup_density/shard_*.h5|paste -sd';' -)"
python <entrypoint, = sage_learning/scripts/train_h5_sampling.slurm lines 56-62 minus SLURM> \
  --config-name=train_scorer_edge name=qboot_density data_dir="$DATA" +seed=1 +data.sample_seed=1 \
  +data.budget_h=false +model.head_mode=hl_gauss +network.value_bins=51        # NoHorizon = single Q
```
Repeat with `v4_hq_boot_setup_depth` + `name=qboot_depth`. **What the data is:** the bootstrap-setup H5 (`build_bootstrap_setup.py`) relabels each setup at s0 with target **γ·V_GT(s1)** read from the exhaustive labels' `frac_first_push` (no re-sim); `--vsummary density` = γ·(finish density = findability), `--vsummary depth` = γ·(solvable?1:0 = existence). m2b gives 1-push openers (value 1.0); ExIt-v4 gives the finish (s1) value. So the single Q learns a cost-to-go: open-now 1.0 > good-setup γ·V > dead 0.

## THE GATE (did it work?)
reactive@2 + best-first@2 (combine=q) vs **NoHorizon-v3: reactive 40.7 / best-first 37.8 @2** (region criterion, n=1018, object-constrained). Eval is cheap CPU + needs MuJoCo/bindings → easiest to **rsync the ckpt back to Amarel** and run `scripts/amarel/eval_afterok.slurm` (RUN_DIR=qboot_density_s1, LABEL=boot_density, MINEP=8).
- **PRE-REGISTERED prediction [CLAUDE]:** the bootstrap likely **MATCHES-not-beats** NoHz, because the finish is near-oracle ⇒ V(s1) ≈ existence ≈ the status-quo 0.9 setup label ⇒ the relabel ≈ what we already train. And **predict depth ≥ density** (density wrongly penalizes a 1-needle-finish setup the model can finish cheaply). The gate decides; don't rationalize.

## PARKED — DO NOT DO NOW
- **Setup ranking loss = LAST RESORT** [USER: "one change at a time, don't change too many things at once"]. Only if the bootstrap gate lands AND simpler levers are exhausted AND a setup mirror-measurement confirms setup-top1 is FIXABLE (not aliased — Stage 0's "top pick is a plausible-DEAD setup" pattern raises real aliasing risk). It's a reactive-only lever (search dissolves it). The `softmax_ce` head is the ready impl (eval-compatible). `HEAD=softmax_ce bash launch_bootstrap.sh`.
- **Stage 2 guided collection** — deprioritized (Stage 0(e): collection well-covered, doesn't starve).

## GIT STATE
- **Anchor (frozen working v2/v3/v4 line):** branch `feat/horizon-q` @ `3d65375` — NEVER overwrite.
- **Active redesign:** branch `feat/horizon-q-redesign` (all Stage 0/1 code + journals). Repos: github.com/dhruvmetha/{namo_cpp, sage_learning}. (sage_learning training code = branch `feat/horizon-q`.)
- New scripts: `scripts/sandbox/{stage0_instrument,build_bootstrap_setup,relabel_bootstrap_setup,agg_seed_table}.py`, `scripts/amarel/{build_bootstrap_setup.slurm,launch_bootstrap.sh,eval_afterok.slurm,reactive_argmax.slurm}`.
