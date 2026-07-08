---
type: experiment
status: done
created: 2026-07-06
commit: 9bdb7be
metric: "budget PLATEAUS, doesn't close the search gap — 2push open@2→@5 40.7→56.5 (flat after push 3), still ~39pp below best-first 95.9; hard 2push caps at 47.3. Reactive mistakes are largely irreversible → search is load-bearing."
tags: [experiment]
thread: scorer-search
---
# Reactive MPC to depth 5 — how much of the test set falls to zero-search control?

## Hypothesis
_(you, via chat 2026-07-06)_ Why don't we do reactive up to 5 levels deep? How much of the test set can then be solved for both 1push and 2push — instead of search, we do MPC.

_(Claude framing, 2026-07-06)_ The reactive-vs-search gap on 2push is huge (reactive@2 42.1 vs best-first ~95 solve@900). This experiment asks **which part of that gap is the push budget vs the lookahead**: MPC can't undo a bad push, but it CAN keep pushing. If open@5 recovers most of the gap → greedy mistakes are self-correctable by acting more, and deploy barely needs search. If open@k plateaus by k=3 → greedy mistakes are largely *irreversible*, which is the cleanest evidence yet that search (simulate-and-undo) is load-bearing. Either outcome is signal; the anytime curve open@1..5 is the figure. Caveat stated up front: this is NOT compute-matched to search (search spends sims, MPC spends real pushes) — it answers the deploy question "how much falls to zero-simulation control," not "which is cheaper."

## Plan
_(Claude, 2026-07-06)_ **CAR only**, testset `namo_testset_v1` (2push key n≈1018 · onepush key n≈1323), region-open criterion, protocol = the `_reactive_search.md` forced-dive setup extended to a depth-k loop.

- **Script:** extend `scripts/sandbox/eval_reactive_argmax.py` with `--max-pushes` (default 2 → backward compatible). Loop: at state s_i, rank candidates (`restrict_obj` = labeled object, matching the existing protocol), take argmax (or random for the floor), `env.step`, check `goal_open_pts`, stop early on open or empty pool. Leaf record gains `opened_at` (push index 1..5, or 0 = never) so open@k for all k≤5 comes from ONE run.
- **Object restriction stays ON** (all ≤5 pushes on the labeled object) — keeps difficulty bins + episode semantics comparable to reactive@2 and the random floor. Free-object MPC is a follow-up variant, not this card.
- **Arms:** NoHz-v3 (3 seeds, registry ckpts — reuse, no retrain) and random floor (10 seeds, `--prior uniform`, model-free), both horizons.
- **Compute:** SLURM CPU per feedback_slurm_first — random floors anywhere; model arm on Amarel (iLab `sage_learning` had `fast_scorer` skew per `_reactive_search.md` Run notes; re-verify before assuming). Shard as in the prior run.
- **Aggregate:** reuse `agg_react_search.py` binning (2push = `pure2push_divisions.json` divisions; 1push = solve_rate tertiles). Tables = open@1..5 × difficulty × horizon, mean±std across seeds. Figure = anytime curve (open@k vs k) per difficulty, model vs random. Anchor check: open@1/@2 must reproduce the `_reactive_search.md` numbers (82.3/42.1 model, 37.0/4.7 random) within seed noise.

## Run
_(Claude, 2026-07-06)_ commit `9bdb7be` on `feat/horizon-q-redesign`. Script `scripts/sandbox/eval_reactive_argmax.py` extended with `--max-pushes` (default 2 = backward-compatible; depth-k loop: rank→argmax/random→step→open? early-stop on open or empty pool; leaf gains `opened_at`∈0..k). Backward-compat verified: max-pushes=5 random seed=100 first shard reproduced the reactive@2 leaf EXACTLY (n=82, open@1=0, open@2=4).
- **Compute:** iLab `unlimited` SLURM, shared FS, CPU (`OMP_NUM_THREADS=2`). Launcher `/common/users/dm1487/scratch_namo/eval/react_mpc_d5/react_mpc_d5.slurm` (worktree-edited script + MAIN `build_python`). 13 shards/run (SHARD=80 2push / 102 1push), MAXPUSH=5.
- **Model path now runs off-Amarel:** the `fast_scorer` skew that forced the predecessor's model arm to Amarel is RESOLVED on the shared `sage_learning` — model (`prior=q`) eval runs clean on arrakis/iLab (5-xml smoke scored fine, no TypeError). So the whole campaign is iLab-native, no Amarel sync.
- **NoHz-v3 ckpts (best-val ep012, shared FS):** s1 `qfull_nohz_v3_v4hq_s1/.../wl8k6iyv/checkpoints/epoch012-val_loss0.6896.ckpt` · s2 `.../s2/.../kzph0acr/.../epoch012-val_loss0.6922.ckpt` · s3 `.../s3/.../dlopoael/.../epoch012-val_loss0.6874.ckpt` (root `/common/users/dm1487/scratch_namo/sage_outputs/scorer/`).
- **Jobs (submitted 2026-07-06, iLab `unlimited`, array 0-12 each):** model 2push s1/s2/s3 = 171832/171845/171858 · model 1push s1/s2/s3 = 171871/171872/171873 · random 2push s100-109 = 171874-172027 · random 1push s100-109 = 172028-172109. Out dirs `/common/users/dm1487/scratch_namo/eval/react_mpc_d5/{nohz,rand}{1,2}push_*`.
- Model = 3 training-seed ckpts (deterministic argmax → rng-seed irrelevant); random = 10 rng seeds 100-109. `--h` is inert here (NoHz ignores horizon; random pool is H-independent) — verified h1≡h2 on the predecessor 1push (82.24/89.72 both).

## Result + Verdict
_(Claude, 2026-07-06 — CAR, region criterion, cumulative open@k, mean ± std across seeds. NoHz-v3 = 3 ckpt-seeds s1/s2/s3 best-val ep012; random = 10 rng-seeds 100-109. Aggregator `scripts/sandbox/agg_react_mpc_d5.py`, plot `scripts/sandbox/plot_react_mpc_d5.py`.)_

**Anchor check — PASS (all within seed noise).** open@1/@2 reproduce `_reactive_search.md`:
- Model 1push open@1 = **82.3 ± 0.2** (anchor 82.3, EXACT); 2push open@2 = **40.7 ± 0.2** (registry headline "NoHz 40.7"; predecessor's 42.1 was inflated by a non-best-val ep011 s3 ckpt — see note below).
- Random 1push open@1 = **37.5 ± 1.1** (anchor 37.0); 2push open@2 = **4.5 ± 0.5** (anchor 4.7).
- Per-difficulty open@1/@2 also match tightly (e.g. model 1push hard@1 54.3 vs 54.3; random 1push hard@1 6.4 vs 6.2). My s1 (ep012) reproduces the predecessor's s1 leaf byte-for-byte (40.77 vs 40.67, sim jitter). Backward-compat verified independently: max-pushes=5 random seed=100 first shard = old reactive@2 leaf exactly (n=82, open@1=0, open@2=4).

_Provenance note:_ the predecessor's reused-Amarel 2push s2/s3 leaves used **ep011** (`s2 epoch011-0.6938`, `s3 epoch011-0.6897`), NOT the registry's best-val **ep012**. I ran best-val ep012 for all 3 seeds (`s1 0.6896 / s2 0.6922 / s3 0.6874`). Same ep012 s1 matches exactly; the ep011 s3 (44.5) is why the old 3-seed mean read 42.1 vs my clean 40.7 ± 0.2. Mine is the registry-consistent number.

### 2push (pure2push key, n=1018) — cumulative open@k

**NoHz-v3 (argmax MPC, 3 ckpt-seeds)**

| difficulty | open@1 | open@2 | open@3 | open@4 | open@5 |
|---|---|---|---|---|---|
| easy   | 0.0 | 59.8 ± 3.6 | 66.8 ± 3.0 | 67.4 ± 3.6 | 67.9 ± 3.8 |
| medium | 0.0 | 42.5 ± 1.6 | 55.5 ± 2.5 | 57.6 ± 2.0 | 58.1 ± 2.0 |
| hard   | 0.0 | 26.3 ± 0.7 | 43.3 ± 2.4 | 46.3 ± 1.6 | 47.3 ± 1.4 |
| **all**    | 0.0 | **40.7 ± 0.2** | **53.7 ± 0.9** | **55.8 ± 0.5** | **56.5 ± 0.4** |

**random floor (10 rng-seeds)**

| difficulty | open@1 | open@2 | open@3 | open@4 | open@5 |
|---|---|---|---|---|---|
| easy   | 0.0 |  8.0 ± 1.4 | 23.0 ± 2.6 | 35.5 ± 3.8 | 46.2 ± 2.9 |
| medium | 0.0 |  4.5 ± 0.8 | 12.6 ± 1.9 | 21.6 ± 1.9 | 31.6 ± 2.4 |
| hard   | 0.0 |  2.1 ± 0.5 |  7.0 ± 1.1 | 13.3 ± 2.1 | 21.3 ± 2.4 |
| **all**    | 0.0 | **4.5 ± 0.5** | **13.0 ± 1.0** | **21.8 ± 1.6** | **31.3 ± 1.7** |

Reference: best-first search s@900 (NoHz-v3, region, n=1018) = **95.9** (registry). open@1 = 0 by construction (pure-2push scenes never open on a single push).

### 1push (onepush key, n=1323) — cumulative open@k

**NoHz-v3 (argmax MPC, 3 ckpt-seeds)**

| difficulty | open@1 | open@2 | open@3 | open@4 | open@5 |
|---|---|---|---|---|---|
| easy   | 98.7 ± 0.4 | 99.4 ± 0.3 | 99.4 ± 0.3 | 99.4 ± 0.3 | 99.4 ± 0.3 |
| medium | 93.9 ± 0.5 | 96.8 ± 0.2 | 96.9 ± 0.1 | 96.9 ± 0.1 | 97.0 ± 0.2 |
| hard   | 54.3 ± 0.4 | 73.0 ± 1.5 | 76.6 ± 1.7 | 77.4 ± 1.1 | 77.6 ± 1.1 |
| **all**    | **82.3 ± 0.2** | **89.7 ± 0.5** | **91.0 ± 0.5** | **91.3 ± 0.3** | **91.3 ± 0.4** |

**random floor (10 rng-seeds)**

| difficulty | open@1 | open@2 | open@3 | open@4 | open@5 |
|---|---|---|---|---|---|
| easy   | 72.6 ± 1.9 | 91.3 ± 0.9 | 96.3 ± 0.6 | 97.8 ± 0.6 | 98.4 ± 0.5 |
| medium | 33.0 ± 2.2 | 60.4 ± 2.1 | 74.8 ± 2.5 | 82.9 ± 2.2 | 87.2 ± 1.8 |
| hard   |  6.4 ± 1.4 | 21.5 ± 1.8 | 35.3 ± 1.7 | 47.5 ± 1.6 | 56.7 ± 1.7 |
| **all**    | **37.5 ± 1.1** | **57.9 ± 1.0** | **68.9 ± 1.1** | **76.1 ± 0.9** | **80.8 ± 0.7** |

![[react_mpc_d5.png]] _(anytime curves: cumulative open@k vs push budget k, NoHz-v3 vs random, per horizon × difficulty; bands = ±std across seeds; dotted = best-first search ceiling 95.9 on 2push·all. Source PNG lives at `assets/react_mpc_d5.png`; regenerate via `scripts/sandbox/plot_react_mpc_d5.py`.)_

**Verdict [on numbers]: the extra push budget does NOT close the reactive-vs-search gap — the model PLATEAUS by push 3-4.** On 2push·all the model goes 40.7 (open@2) → 53.7 → 55.8 → 56.5 (open@5): marginal gains collapse to +13.0 / +2.1 / +0.7 per push and it is essentially flat after push 3. Best-first search reaches 95.9, so even a 5-deep greedy dive leaves a **~39pp gap** to search. Budget decomposition: extra budget closes only 15.8 of the 55.2pp reactive-vs-search gap (**~29%**); the residual ~70% (39.4pp) needs search's simulate-and-undo. The plateau is the cleanest evidence yet that **greedy reactive mistakes are largely irreversible** — you cannot "keep pushing" your way out of a bad first push; you have to be able to take it back (search). This holds hardest on the hardest scenes: hard 2push caps at **47.3%** even at 5 pushes (>half of hard scenes un-openable reactively), vs search ~96. On 1push the model saturates by push 3 at 91.3 (open@1 82.3 → open@5 91.3, +9.0) — even 1-push-solvable scenes have a ~9% reactive-irrecoverable tail. Secondary finding: **random keeps climbing (no plateau by push 5)** and eats into the model's lift (2push·all lift shrinks +36.2 → +25.2 from open@2 to open@5) — with enough blind budget random slowly accumulates opens the model's early greedy commitment forgoes, though it stays ~25pp behind and far from search. Caveat (as pre-registered): NOT compute-matched — search spends sims, MPC spends real pushes; this answers "how much falls to zero-simulation control," not "which is cheaper."

## Next
_(Claude, 2026-07-06)_ The plateau localizes the reactive-vs-search gap to **lookahead (simulate-and-undo), not budget** — reinforces that search is load-bearing for deploy. Natural follow-ups: (1) ~~**free-object MPC**~~ — **VETOED [USER 2026-07-06]**, do not run; (2) **compute-matched** comparison (cap search sims at the MPC's real-push count) to answer "which is cheaper," not just "what falls to zero-sim control"; (3) a **shallow-search** middle ground (k=1 undo / beam-2) to price how little lookahead recovers most of the 39pp residual. (2)/(3) not gated — orchestrator's call.

## Discussion
_(you ↔ Claude — ask here; I answer inline, dated. Newest at the bottom.)_
