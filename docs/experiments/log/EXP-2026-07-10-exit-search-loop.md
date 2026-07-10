---
status: planning
thread: rl_loop
created: 2026-07-10
---

# EXP-2026-07-10 — Expert-Iteration depth-ladder ranker (single-object RO)

## TL;DR (plain English)

We teach one network to **rank the pushes of a single blocking object** so a best-first search opens the way in as few simulator calls as possible.
It teaches itself by climbing a **depth ladder**: first learn "which one shove opens it" (from *random* tries), then use that to make a two-shove search cheap, and let the search's own wins and dead-ends label the next, sharper ranker.
The one idea that separates this from our earlier rollout loop: we collect by **branching search**, so we get **negatives** — pushes we searched and *didn't* find a win through — which is what lets the ranker put the real winners first.

## Hypothesis (USER to confirm/edit)

> A ranker trained by a **depth-laddered, search-collected** loop — where every searched-but-unproven push is labeled a soft `0` *below* the observed winners (`γ^k`) — **dominates a random-order search on sims-to-solve**, reaching any given solve-rate at a fraction of random's cost, **across every difficulty × horizon tier**, with **no budget/horizon conditioning**. (Reference: random hard-tier median ≈ 118 sims; the learned ranker should be a large multiple cheaper.)

---

## The problem (single-object RO — pinned)

- Episode = **(scene, target object, goal region)**. The blocking object is **given**. The job: open the way by pushing **that one object**, once or twice.
- **Success** = the wavefront says the goal region is reachable.
- **Action grid** = **(contact point 0–59, depth 0–4) = 300** shoves of that object (60 contact points = 4 edges × 15 points; 5 depths). The wavefront trims this to **~50–150 reachable**.
- **Depth is 1 or 2.** One shove clears it, or a first shove **repositions the same object** so a second shove clears it. No second object anywhere.

## What we learn

A scalar **value field `Q(s, ·)` over the 60×5 grid** — one number per shove — that orders which shoves the search tries first. `EdgeCrossAttn`, head shape **(60, 5)**. Not a softmax policy.
Optional: a **separate feasibility head** predicting the reachable region (the `−1` band), so the model internalizes reachability (useful for the continuous case); or fold it into the value field as a `−1` band. A/B this.

## The label scheme (the crux) — on the 60×5 grid, per state

| cell | target | how it's earned | note |
|---|---|---|---|
| **unreachable** | **−1** | wavefront — *free, no sim* | dense feasibility signal |
| tried, on an **observed** solution | **γ^k** (opener 1, setup γ, …) | search found the path | certain positive |
| tried, **searched to the depth/budget, nothing found** | **0** | had to search its subtree | **deliberate soft negative** |
| **never tried** (reachable but not sampled/searched) | **MASK** | — | genuinely no signal |

The `0` is the engine: it **suppresses** searched-but-unproven shoves *below* the shoves we **observed** reaching the goal — accepting that a few `0`s are false (a solution existed just past our budget). It is a **ranking** signal, not a ban: search still falls back to low-ranked shoves, so a false `0` costs *extra sims*, never a lost room. Never a permanent zero — see re-stamping.

---

## Where the environments come from — REGENERATE on CS (not Amarel fishing)

We generate a **fresh, clean car-room pool on the CS box** with `mujoco_env_creator`, rather than fishing Amarel's tangled pool (`v3/test` is ~18k *pair-inflated* files that dedup to only ~1877 unique geometries — too messy to hold out cleanly).

| step | detail |
|---|---|
| **generator** | `mujoco_env_creator/generate_envs.py` (CS-native; injects the diff-drive car body when the config says `robot_type: diff_drive`) |
| **recipe** | reuse `mujoco_env_creator/scripts/generate_v3_aug9_joblist.py`, fixing its two Amarel paths: `NAMO_CONFIG` → `…/namo/config/namo_config_complete_skill15_car_1x.yaml`, `PYTHON` → the CS env |
| **templates** | `templates/aug9_car_v3/` (10) + `templates/feb_car/`; object side 6–16 cm; seed-strided (no geometry collisions) |
| **family mix [USER]** | **60% aug9_car / 40% feb_car** — set at generation (which templates). No post-filter, so this is the generated-pool ratio. |
| **hard tilt** | bias generation toward hard layouts (denser / larger obstacles; `placement_strategies.py` adjacency bias) so a healthy *fraction* need 2 shoves — a **distribution tilt, NOT per-room certification** |
| **NO pre-filter [USER]** | we **cannot** know a room is 2-push without solving it (≈ the exhaustive depth-1 check — banned). So we don't sort rooms up front. **Rung-1 sorts them for free:** rooms it opens → 1-push data; rooms it doesn't → the rung-2 workload. |
| **disjointness** | seed ranges outside testset_v1's, then geometry-check vs `namo_testset_v1/manifests/canonical_scenes.txt` → 0 leaks |
| **target size** | go **BIG** — a broad pool (tens of thousands). Rung-1 sorts it, so more raw material is strictly better; no over-generate-then-filter math needed. |
| **robot / config** | CAR only — `config/namo_config_complete_skill15_car_1x.yaml`, primitives `1x_car_d5_*` |

Why regenerate instead of reuse: the Amarel pool is heavily duplicated and split under confusing `test/` naming, so a clean, seed-controlled, geometry-disjoint pool is both easier to trust and easier to hold out than fishing + deduping 18k inflated files.

---

## The training flow — climb the depth ladder

**Step 0 (every state): wavefront.** Is the goal already open? Which shoves are reachable? → the feasible set (for search) **and** the `−1` cells (for labels). This is the first operation everywhere, in collection and inference.

### Rung 1 — depth ≤ 1, base policy = **uniform random** (ε = 1) — **ONE seed round**
No ranker exists yet, so the policy *is* random. This is a **single seed pass, not a loop** — a random policy doesn't self-improve, so iterating depth-1 only adds samples, not quality.
- **Collect:** per episode over the **whole generated pool**, sample ~m (≈25) reachable shoves, execute **one shove each**, re-run the wavefront: opened?
- **Label (60×5):** opener → **1** · searched-&-didn't-open → **0** · unreachable → **−1** · never-sampled → **MASK**.
- **Train → Q1** ("does one shove open it now?"). Crude, but far better than random for ordering.
- **Free byproduct — the sort:** rooms this pass opened = 1-push rooms; rooms it didn't = **the rung-2 workload**. We never certify "2-push" up front; this pass *is* the sorter.

### Rung 2 — depth ≤ 2, **Q1 steers**, ε ≈ 0.15 — **the flywheel, ~3–6 rounds**
This is where the iteration lives: `Q1 → Q2 → Q3 …`, each round a better ranker → cheaper search → more/shorter solutions + more buried setups re-stamped. **Stop when hard-tier sims-to-solve plateaus.** ("Upgrade to depth-2" is not a discrete switch — once Q1 exists it *is* the pruner; you just point the depth-2 search at the rung-2 workload.)
- **Collect (best-first search, budget B):** Q_g scores first shoves by lookahead `score(a) = max_b Q_g(s'=after-a, b)`; **exploit** the top-k, **explore** an ε-fraction of Q_g-*low* shoves. For each first shove, search its second shoves (Q_g-ranked) until open or budget spent. Record the **whole tree**.
- **Label the tree:** 2-shove win → **γ^k** (setup γ, opener 1) · searched-to-budget-nothing → **0** · unreachable → **−1** · untried → **MASK**.
- **Re-stamp:** any Rung-1 `0` this search now solves *through* → flip to **γ** (it was a buried setup — often found *because* of an ε-random try).
- **New 1-push finishes come free:** every winning second shove is a value-**1** label at a **post-shove state `s'`** Q1 never saw → Q2 learns "opens-now" *deeper in the tree*, which **patches the seed→lookahead bridge on-distribution**.
- **Train (rung-1 + all rung-2 rounds pooled) → Q_{g+1}**, then repeat the round with the sharper ranker. First round's Q1→Q2 is the biggest jump; later rounds mostly shorten solutions and un-bury setups.

### Rung 3+ — same pattern with Q2 as the steering wheel. **RO stops at Rung 2** (rooms need ≤ 2 shoves). Rung 3+ is for a deeper domain (bin-picking), not RO.

**Why randomness, and its schedule.** ε is the **error-correction channel**: Q ranks genuine setups low (a setup opens nothing immediately), so without exploration we'd never try them, never learn, stay wrong. The ε-random tries surface them → new `γ` positives → re-stamp → Q corrects.

| stage | ranker | ε | why |
|---|---|---|---|
| rung 1 | none | **1.0** | random *is* the policy |
| rung 2+ | Q1, Q2… | **~0.15**, decaying | trust Q, but keep catching its mistakes |
| converged | good Q | **→ small** | little left to correct |

Everything is **grounded** (targets from real sims, MC returns, **no TD bootstrapping**) and **pooled** (one network over all depths — the curriculum lives in *collection*, not the SGD order).

---

## The inference flow (final Q2 — no learning, no labels)

1. **Wavefront** — goal already open? → **done, 0 pushes.** Else get the feasible shoves + `−1`s.
2. **Score** the feasible shoves with Q2 (one forward pass).
3. **Best-first search** — try highest-Q2 first → sim → re-run wavefront ("open now?"); if not, expand its second shoves (Q2-ranked); keep popping the best node until open.
4. **rank-not-ban** — low-ranked shoves still get tried if the good ones dead-end. Report **sims-to-solve**.

Report **two regimes**: **with search** (best-first, backtracks — the real deploy; a wrong `0` costs sims, not the room) and **reactive** (greedy top-pick, no backtrack — stress-test / lower bound, where a wrong `0` *can* lose the room).

---

## The data pipeline — real scripts (grounded)

| stage | what | script | status |
|---|---|---|---|
| **generate rooms** | fresh CAR pool, 60/40 aug9/feb, hard-tilted, disjoint seeds, go big | `mujoco_env_creator/generate_envs.py` + fixed aug9 joblist | ✅ reuse (2 path fixes) |
| **sort by rung-1** | depth-1 pass splits the pool: opened → 1-push data, unopened → rung-2 workload (**no pre-filter**) | falls out of the rung-1 collection | ✅ free byproduct |
| collect (sampled) | region_opening, `region_sample_k=m`, CAR | `modular_parallel_collection.py` + new sampled car yaml | ✅ config only |
| rung-1 label / seed | opener/−1/0/mask on 60×5 | `build_train_h5.py` (+ `−1` band, `r_mask`) | ⚠ extend labels |
| **rung-2 search + tree-log** | best-first to depth 2, ε-greedy, **record the tree** | `time_bestfirst.py::timed_bf()` | ⛔ **NEW** — today it keeps only the winning path |
| **tree-log → rows** | γ^k / 0 / −1 / mask + re-stamp | new row-source in `build_train_h5.py` | ⛔ **NEW** |
| train | `EdgeCrossAttn`, MC value head (HL-Gauss), pooled | `train_gen.py`, `rl_dataset.py` | ✅ reuse (verify V-head fix 9191960) |
| eval | sims-to-solve + reactive, both tiers | `time_bestfirst.py` (search) + `eval_testset.sh` (reactive) | ✅ reuse |

**The whole new surface is small:** keep the search's **tree** (Stage rung-2), wire it into the label builder that already knows `γ^k`/`0`, and add the `−1` band + `MASK` handling. Everything else — sampled collection, trainer, eval, generation reweighting (`buffer.py`) — exists.

---

## The experiment, staged (fail-fast)

### Stage A — bridge check (offline, NO collection, ~1 day)
Train Q1 on a sampled rung-1 seed. Take known 2-shove solutions; sim the first (setup) shove to `s'`; check whether `max_b Q1(s', b)` ranks the *true* finisher high — i.e. does a good setup produce a **visible opener** for Q1?
**Gate:** holds → Stage B. Fails → redesign the seed before spending collection-sims. (Note: even if weak here, Rung-2's post-shove finishes are designed to repair it — but we want to know the starting point.)

### Stage B — seed + first flywheel rounds (rung-1 → 1–2× rung-2)
**1 seed round** (random depth-1 over the whole pool) → Q1; the pass sorts the pool (opened = 1-push, unopened = rung-2 workload). Then **1–2 depth-2 rounds** (ε≈0.15) on the rung-2 workload → Q2 (→Q3). Evaluate **vs random** (sims-to-solve, all tiers). Success = Q2's sims curve **dominates random's**, and the first climb pushes it further below random.

### Stage C — run the flywheel to plateau + ε/budget tuning (only if B turns)
Keep looping depth-2 (~3–6 rounds total) until hard-tier sims-to-solve flattens; tune ε and budget B; turn on heavier wide-collect only if coverage stalls.

## Metrics (pre-registered, numbers-only verdict)

| metric | why | target |
|---|---|---|
| **sims-to-solve vs random (search)** | the headline (north-star bar) | dominate random's solve-rate-vs-sims curve, every tier; hard-tier median ≪ random's 118 |
| setup-shove rank | is the setup un-buried? | setups tried early, not last |
| solve-rate @ fixed sim budget | the curve, by tier | above random at every budget × tier |

All by **difficulty (easy/med/hard) × horizon (1push/2push)**, both regimes. Sims-to-solve primary; wall-time only on identical HW.

## Baseline — RANDOM ONLY [USER]
The **only** comparison is a **random-order search** (`--models random` in `time_bestfirst.py`; also = the Rung-1 base policy) — the project's canonical success bar (problem_and_approach.md §3): dominate random's solve-rate-vs-sims curve across all tiers. We do **not** compare against NoHz (budget-conditioned → apples-to-oranges) or RL-π (different pool → confound).

## Compute
- Collection + search → **SLURM CPU** (sim-bound), sharded.
- Training → **GPU** (arrakis / gpu-redhat).

## Risks / kill signals
- **Bridge weak (Stage A)** → note it; Rung-2 finishes should repair it, but a *very* weak Q1 means slow start.
- **Flat climb** (Q2 ≈ Q1 on sims-to-solve) → the negatives aren't landing on setups; check the `0`s and re-stamping.
- **Coverage collapse** — only ~2 rungs for RO so mild; watch new-positives-per-sim and stale-`0` flip rate.
- **Wrong pool** — a 1-shove-dominated pool repeats the last wall. Mitigated by regenerating + probe-filtering to genuine-2-push; verify the F=∅ composition and 60/40 mix before Stage B.
- **Seed/geometry leak** — regenerated rooms must be geometry-disjoint from testset_v1; check against `canonical_scenes.txt` (0 leaks) before training.
- **V-head hang** regression → the spawn-context fix (9191960) must hold.
