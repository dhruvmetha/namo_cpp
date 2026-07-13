---
status: running
thread: rl_loop
robot: car
updated: 2026-07-12
supersedes: EXP-2026-07-11-curriculum-ladder (findings kept; this is the clean plan)
---

# EXP-2026-07-12 — Self-improving opener via an automated hard-curriculum loop

## The one sentence

Learn ONE model that, shown a scene, finds the push that opens the goal — and make it strong on HARD scenes with an **automated loop** that keeps generating fresh scenes, keeps the ones the model *fails* on, labels them, and retrains — round after round, no manual steps.

## What we are actually optimizing (the objective function)

The robot reaches its goal by pushing obstacles aside. Trying a push = one physics sim = expensive. So we learn a **ranker**: given a scene, score every possible push so the *opening* push sits near the top. Then the robot solves in a **handful** of sims instead of the dozens random needs.

**Success (the bar):** in every difficulty bucket (easy / medium / hard), solve **≥95%** of scenes in **far fewer sims than random** — measured both *with search* (top-k) and *reactively* (top-1, one shot).

We score the model by **solve@k** = "is a real opener inside the model's top-k picks."

## The model — input, output, loss

**INPUT (what the model sees):**
1. A **picture of the scene**, 5 channels × 64×64 pixels. The channels are: walls (static), all movable objects, **the target object** (the one we're pushing), where the robot can currently reach, and the goal region.
2. The **60 contact points'** pixel positions (60×2) — the spots around the object where the robot could push from.

**OUTPUT (what it predicts):**
- A score for each of **300 candidate pushes = 60 contact points × 5 push-distances**. Each score ∈ [0,1] = **"probability this single push opens the goal now."** So the output is a 60×5 grid of open-probabilities.

**NETWORK:** `EdgeCrossAttn` — attention over the 60 contact points, emits the 60×5 grid. (Same net we've been using.)

**LOSS:** per-cell **binary cross-entropy**. Target = **1** if the push opens the goal, **0** if it was tried and didn't. Only pushes that are **reachable AND tried** contribute (unreachable / untried cells carry no gradient).
- **No ranking loss. No loss-reweighting.** The *difficulty* is carried by the **data** (we feed it hard scenes), not by a loss trick. [USER veto stands.]

**Why only "opens now"?** We deliberately do NOT learn a "setup value." *Finishing* a 2-push problem is just this same opener model run on a *post-shove* scene. *Setup value* is computed by **search** at deploy time (push → new scene → ask "opener here?"). So the model's whole job, forever, is one thing: **"does this push open the goal now."** That is what sidesteps the way Q2 failed.

## The algorithm — the automated loop, step by step

```
model_0  ← bootstrap (round 0):  generate a broad batch of fresh scenes → label them all → train model_0

Round r  (repeat, r = 1, 2, 3, …):
  1. GENERATE   fresh batch of scenes
  2. SCREEN     run model_{r-1} + search on each scene → record sims-to-solve (cheap: search stops at the solution)
                → KEEP the hard ones (many sims, or unsolved)          ← this is where "harder and harder" comes from
  3. LABEL      exhaustively try every push on the KEPT hard scenes → the 60×5 opener/dead grid
  4. TRAIN      accumulate ALL labeled scenes; retrain from scratch on a ~75% hard / ~25% not-hard(easy+med) SAMPLE → model_r
  5. EVAL       solve@1 / solve@k by bucket on the held-out testset
  6. LOOP/STOP  improved & not plateaued → round r+1 ;  plateaued or out of budget → stop
```

**The hardening is emergent** — we never set a difficulty dial. Each round, a *better* model means the scenes that survive the screen are *harder*. The loop chases its own frontier. (This is Expert-Iteration / DAgger, applied to the opener.)

**Everything runs automatically:** one **orchestrator** drives the whole loop — it submits the generate / screen / label jobs to SLURM, runs training on the GPU, evals, decides, and fires the next round. No manual steps.

## The pieces (and what's new)

| piece | job | status |
|---|---|---|
| **generator** | emit fresh car scenes, matching the testset's distribution | ⚠ confirm it runs headless |
| **labeler** | exhaustive 1-push (`region_sample_k:0`) → opener/dead grid | ✅ built |
| **screener** | best-first search → sims-to-solve (hardness signal) | ✅ exists, wrap as a filter |
| **trainer** | EdgeCrossAttn + per-cell BCE | ✅ built, adapt |
| **orchestrator** | chain all of it, loop rounds, handle failures, hands-off | 🔨 **the new build** |

## Data & eval — fully fresh, fair

- **No old data.** exit_pool and NoHz's set are set aside entirely. The model starts from random weights; the pool is 100% freshly generated + labeled.
- **Fair eval:** we generate scenes from the **same distribution as `namo_testset_v1`** (the car aug9/feb generator), on rooms **held out** from the test split — so testset eval is apples-to-apples (this is the fix for the earlier off-lineage mistake).
- **Held-out testset = `namo_testset_v1`** (car_envs/v3 test), eval only, never trained on.

## Decisions locked [USER]

1. **Fully fresh** — no old data, no old model, even for the seed.
2. **One opener model** — "does this push open now"; setups handled by search, not learned.
3. **Hardness = model+search sims-to-solve** (model-hard), not absolute solve-rate.
4. **Loss = per-cell BCE**, no reweighting, no ranking loss.
5. **Data = accumulate + weighted sample** — keep ALL labeled scenes (expensive GT); each round retrain **from scratch** on a **~75% hard / ~25% not-hard (easy+medium)** sample. Never train on the newest-hard alone (catastrophic forgetting); never dump the pool flat (dilutes hard).
6. **Automated** — one orchestrator, no manual steps between rounds.
7. **Parallelize hard** [USER] — (a) **shard generation across nodes** (like the label campaign), NOT single-node (round-0 wasted ~1h running gen on one node while 5-6 sat idle); (b) **pipeline across rounds** — overlap generate(N+1) with train(N), label with the prior eval, etc. Each stage that's embarrassingly parallel (gen, label, eval-sims) fans across the idle rlab/ilab cores (rlab7 alone ≈250).

## Orchestrator DAG + speedups (2026-07-12 deep-dive)

Steady-state round pipeline (the controller encodes this):
`GEN(r)→FILTER(r)` run **one round ahead** (model-free → off the critical path) + **sharded across nodes** (round-0's single-node gen 1h19m → ~14m sharded) → `SCREEN(r)` [needs model_{r-1}; ∥ EVAL(r-1); **fanned across nodes — launcher DOESN'T EXIST YET, must build** (current growth code is single-node)] → `LABEL(r, HARD-ONLY ~30-40%)` [not all scenes → ~2.5-3× less than round-0's label-everything] → `BUILD(r, streamed per-node as label shards land)` → merge into pool → `TRAIN(r)` → `EVAL(r)`.
Cross-cutting: **shuffle the manifest before every cross-node split** (round-0's difficulty-sorted contiguous split → ~15-min straggler tail) + right-size workers to node cores. Render is already fast (~101ms, `fast_scorer` wired) — build is NOT the bottleneck. **Net steady-state round: ~half-day → ~1h overlapped.** Full analysis: the speed deep-dive report (2026-07-12).

## Related work (a synthesis, not a new algorithm)

**DAgger** (Ross et al. 2011 — aggregate data on the model's own failures, relabel with an oracle) · **Expert-Iteration / AlphaZero** (Anthony 2017, Silver 2018 — search + learn, chase your frontier) · **hard-example mining / active learning** · **self-paced / automatic curriculum** (Bengio 2009). The principled twist: **oracle supervision** (the sim gives *exact* opener labels + BCE) instead of RL **value-bootstrapping** — DAgger-appropriate precisely because the oracle is cheap to query (unlike AlphaZero, which bootstraps because it only has sparse reward). Upgrade path if random-generate-then-filter is too wasteful: **Unsupervised Environment Design** (POET, PAIRED — adversarially generate hard-but-solvable scenes).

## Hypotheses [USER to confirm]

- **H1** — the automated hard-curriculum lifts **hard @1** round over round (and hard-sims-to-95% drops).
- **H2** — the loop beats a fixed same-size random-scene training set (the emergent hardening is worth it).
- **H3** — one opener model + search covers 1-push now, and (Phase 2) finish + setups later, with no new learning objective.

## Phases

- **Phase 1 — 1-push opener loop** on original scenes. ← **START HERE.**
- **Phase 2 — 2-push / finish** [USER: **trigger = Phase-1 hard @1 plateaus**; no rush]. Add post-shove scenes → the *same* opener model learns to finish. **Collect finishes with MODEL-GUIDED best-first search, NOT exhaustive** — the finisher is a rare needle (~8% solve rate), so exhaustive finish is far too slow (testset-only finish GT already cost ~430k sims); the model finds it in a few guided tries. Tune K high enough to catch the finisher (finish @1 only ~53% on hard → do NOT take top-1) so we don't re-inject false-negative labels. This is the efficient "exhaustive-first-ply + guided-finish" collection, model-accelerated.
- **Phase 3 — deploy with search** → setups fall out; full 2-push eval vs the bar.

## Run log

**Round 0 (2026-07-12) — DONE.** Fresh pipeline end-to-end: 35,814 scenes (37/63 aug9/feb, 0 geometry-leaks) → exhaustive label (67,321 episodes, opener rate 34.1%) → stuck→dead relabel → train model_0 (BCE**+Dice** — note: code adds Dice; card said pure BCE; USER loss call pending) → eval. model_0 ckpt `curriculum2/round0/model_0/checkpoints/epoch029-val_loss0.8219.ckpt` (early-stop ep44, held-out opener AUC 0.883, top-1 hit 75.5% vs 34.5% base). Val plateaued ~0.82 (round-0 data wrung out — the curriculum breaks the plateau).

**Round-0 baseline — 1push solve@1 by tertile bucket (model_0 / NoHz-ref / random / old off-lineage depth1_v1):**
| bucket | model_0 | NoHz | random | depth1_v1 | gap→NoHz was→now |
|---|--:|--:|--:|--:|---|
| easy | 93.1 | 98.4 | 74.9 | 88.8 | 9.6→5.3 |
| med | 78.6 | 94.7 | 34.7 | 73.6 | 21.1→16.1 |
| hard | 35.1 | 54.0 | 7.9 | 32.4 | 21.3→18.9 |
| all | 69.0 | 82.4 | 39.4 | 65.0 | — |
(NoHz re-run 98.4/94.7/54.0 ≈ card's 53.7 → harness validated.)

**Read:** fresh on-lineage data closed a REAL chunk of the depth1_v1→NoHz gap on every bucket (the wrong-rooms confound was genuine, ~+3-5 @1/bucket) — but **model_0 still trails NoHz** (~19 @1 hard). Residual ≈ **data scale** (67k vs NoHz's ~253k eps, 3.75×; the loop closes this) + minor **head** diff (model_0 BCE vs NoHz/depth1_v1 hl_gauss — an isolating A/B later). model_0 dominates random everywhere → strong screener seed. **Loop test:** does the curriculum push hard @1 from 35 toward/past 54?

**Ops notes for the orchestrator:** arrakis `&`-backgrounded jobs die at tool-call boundaries → run eval on SLURM (or `setsid`); model-scoring works on arrakis+iLab GPUs (no Amarel for eval — that note is stale). Promote to `scripts/pipeline/`: `filter_geom_disjoint.py`, `relabel_stuck0.py`, `agg_onepush_tertile.py`, `round0_gen_joblist.py`.

_(appended as rounds complete)_
