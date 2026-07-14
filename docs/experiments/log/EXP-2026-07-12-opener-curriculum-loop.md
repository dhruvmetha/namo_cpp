---
status: running
thread: rl_loop
robot: car
updated: 2026-07-14
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

**LOSS: `hl_gauss`** [USER 2026-07-12 — switched from BCE to match NoHz's head, removing the head confound so residual gap to NoHz is cleanly *data*]. The value head (51 bins, [0,1], same as NoHz) regresses the target (opener=**1** / dead=**0** — binary for Phase-1; no 0.9 setups, those are search's job) via **masked cross-entropy to the Gaussian-smoothed target** ("Stop Regressing" / HL-Gauss). Masked to **reachable AND tried** cells. NO ranking loss, NO reweighting — difficulty is carried by the DATA, not the loss. Whole line is hl_gauss from model_0 onward (model_0 retrained with it). (Trainer = `train_q2.py`; NoHz/Q2/cal-Q2 all trained on hl_gauss stably.)

**Why only "opens now"?** We deliberately do NOT learn a "setup value." *Finishing* a 2-push problem is just this same opener model run on a *post-shove* scene. *Setup value* is computed by **search** at deploy time (push → new scene → ask "opener here?"). So the model's whole job, forever, is one thing: **"does this push open the goal now."** That is what sidesteps the way Q2 failed.

## The algorithm — the automated loop, step by step

```
model_0  ← bootstrap (round 0):  generate a broad batch of fresh scenes → label them all → train model_0

Round r  (repeat, r = 1, 2, 3, …):
  1. GENERATE   fresh batch of scenes
  2. SCORE      run model_{r-1} on each scene — ONE forward pass, NO search, NO budget → predicted opener grid
                → DROP the scenes it confidently NAILS (an opener at/near the top of its ranking; verify the top pick with 1 sim) — model already solves these, no new signal. Keep everything else for labeling.
  3. LABEL      exhaustively try every reachable push on EVERY remaining scene → the 60×5 opener/dead grid.
                The LABEL (ground truth) draws the 1-push/2-push line — NOT a search budget:
                     ≥1 opener exists   → KEEP  (model-hard: an opener EXISTS but the model buried it — even at rank 200 — = its mistake, THE signal)
                     0 openers anywhere → BANK  (truly dead → a 2-push scene → phase2_bank/, NOT trained in Phase-1)
                ← self-hardening is emergent: a better model NAILS (DROPs) more scenes, so the KEEP — the hard-solvable ones it still gets wrong — shrinks + hardens each round (yield→0 = plateau = stop signal).
  4. TRAIN      accumulate ALL labeled scenes; retrain from scratch on a ~75% hard / ~25% not-hard(easy+med) SAMPLE → model_r
  5. EVAL       solve@1 / solve@k by bucket on the held-out testset
  6. LOOP/STOP  improved & not plateaued → round r+1 ;  plateaued or out of budget → stop
```

**The hardening is emergent** — we never set a difficulty dial. Each round, a *better* model NAILS (DROPs) more scenes, so the scenes it still gets wrong are *harder*. The loop chases its own frontier. (This is Expert-Iteration / DAgger, applied to the opener.)

**⛔ DESIGN CHANGE 2026-07-14 [USER] — the SCREEN (budget-search) is retired; SCORE (forward pass) + let the LABEL reject.**
Old loop: SCREEN ran best-first search with a budget (T=4, budget 60) and drew the 1-push/2-push line by "did model_{r-1} solve it within 60 tries." Two flaws surfaced in round-3.
(1) The budget (60) is BELOW the candidate pool (~75+ reachable pushes), so "unsolved@budget" banked scenes that DO have an opener — just ranked past 60. Round-3's screen banked **90%** as "2-push" while round-2's *exhaustive* label put true-dead at ~58%: we were throwing away the scenes where the model is MOST wrong — exactly the DAgger training signal we want to KEEP.
(2) The budget-search is REDUNDANT — every KEEP scene got searched once by the screen, then re-searched from scratch by the exhaustive label.
New loop: model_{r-1} SCORES every scene in one forward pass (zero sims) → DROP only the scenes it confidently nails → exhaustively LABEL the rest, and the LABEL (ground truth) draws the line: opener exists → KEEP (even if the model buried it), zero openers → truly dead → BANK. The forward-pass "screen" costs 0 sims, so **nothing is searched twice** — the expensive sims happen exactly once, at the label. Correct AND non-redundant.
Cost: labels more scenes than the budget-screen (all non-DROP, including the dead we now confirm) — but Amarel makes exhaustive labeling cheap (~15 min for 270k), so the budget-screen's "cheapness" was false economy that cost us correctness.
Diagnostic gating the full switch: exhaustive sample-label of ~4k *budget-hit* banked scenes → measures the TRUE dead-rate (is the 90% real?) + whether recovered openers are learnable or unlearnable needles. [result pending 2026-07-14]

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
3. **Hardness = the model gets it WRONG, decided by GROUND TRUTH not a search budget** [REVISED 2026-07-14 — see DESIGN CHANGE above]. model_{r-1} scores each scene (forward pass) → DROP the ones it confidently nails → exhaustively LABEL the rest → KEEP = an opener exists but the model buried it (its mistake), BANK = zero openers = truly dead. Self-hardens (better model → DROPs more → KEEP shrinks + hardens; yield→0 = plateau). The old budget-search screen (T=4, budget 60) is retired — it banked solvable-but-deep scenes as false 2-push and re-searched every KEEP redundantly.
4. **Loss = hl_gauss** (NoHz's 51-bin value head; switched from per-cell BCE on 2026-07-12 to remove the head confound — residual gap to NoHz is now cleanly *data*). No reweighting, no ranking loss.
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

**Clean hl_gauss baseline (2026-07-12) — model_0 retrained on the SAME round-0 data with hl_gauss (head confound removed).** ckpt `curriculum2/round0/model_0_hlgauss/checkpoints/epoch022-val_loss0.8989.ckpt` (two-reload identical). 1push solve@1 by tertile bucket:
| bucket | model_0 hlgauss | model_0 BCE | NoHz | random |
|---|--:|--:|--:|--:|
| easy | 97.3 | 93.1 | 98.4 | 74.9 |
| med | 80.7 | 78.6 | 94.5 | 34.7 |
| hard | 34.7 | 35.1 | 54.0 | 7.9 |
| all | 71.0 | 69.0 | 82.4 | 39.4 |
**Read:** head swap lifted easy/med (+2-4 @1) but hard **FLAT** (34.7 vs BCE 35.1) → the residual **19.3 @1 hard gap to NoHz is now pure DATA scale** (67k vs NoHz's ~253k eps), objective ruled out. This is the line the curriculum must climb. NoHz reproduced exactly (98.4/94.5/54.0) → harness validated. This hlgauss model_0 is the round-1 screener.

**Round 1 (2026-07-12) — in progress.** GEN 154,364 fresh scenes (74,506 rooms, seeds 2e9/3e9, sharded 6 nodes) → FILTER geometry-disjoint vs testset (0 leaks, 0 unparseable) → SCREEN (model_0_hlgauss + best-first, budget 60, 7 nodes rlab1-4/6/7+ilab3 @ ~79.5 scenes/s). Screen had a one-time `spawn`→`fork` fix (semaphore-reattach storm), then clean. Threshold LOCKED **sims≥4** (`--sims-min 4`), yield-ladder printed for defensibility, bank unsolved→phase2_bank/. **Worker cap `nproc-24` from LABEL onward** [USER — courtesy headroom on shared boxes]. Awaiting SCREEN_COMPLETE → select(T=4) → label(hard-only, capped) → build/merge → train model_1 → eval (does hard@1 climb from 34.7?).

**CORRECTED baseline — FIXED CUTS (2026-07-13) [USER: 1push difficulty = `eval_common.bin_of` fixed solve_rate cuts, NEVER tertiles].** The two baseline tables above are TERTILE-binned (via `agg_onepush_tertile.py`) → **SUPERSEDED for the difficulty axis.** Canonical fixed cuts (hard solve_rate<0.05 n=204, med<0.30 n=421, easy n=698). Provenance: `eval_scorer.py` can't run on CS (its `v3_test_*_lzf_tight` H5s are Amarel-only), so numbers are the **search-solve@k harness** (`time_bestfirst`) re-binned on fixed cuts — a first-class WITH-SEARCH metric; **@1 = reactive/no-search, @k = with-search (both regimes in one table)**; an `eval_scorer` reactive cross-check would be ~redundant (best-first tries in model-score order), available on Amarel if ever wanted.

| bucket | model_0_hlgauss @1/5/10/20 | NoHz @1/5/10/20 | random@1 | n |
|---|---|---|---|---|
| easy | 93.3/98.7/99.9/99.9 | 97.7/99.3/99.9/99.9 | 62.6 | 698 |
| medium | 59.9/83.6/90.7/95.2 | 81.5/92.2/96.4/98.3 | 19.2 | 421 |
| hard | **17.6**/41.7/58.8/71.1 | **31.4**/54.9/68.6/82.8 | 1.5 | 204 |
| all | 71.0/85.1/90.6/94.0 | 82.3/90.2/94.0/96.7 | 39.4 | 1323 |

**Corrected line: model_0_hlgauss hard@1 = 17.6 (NoHz 31.4, gap 13.8). model_1 is measured against THIS, not the tertile 34.7.** Flag for the analyzer: on hard the NoHz gap barely shrinks with k (13.8@1 → 11.7@20) — top-20 doesn't close it, so it's NOT pure ranking-recall; being diagnosed.

**model_0 analysis (2026-07-13, analyzer subagent) — the NoHz gap is 100% RECALL, and hard@1 is the WRONG success metric.**
- **Pure recall/ranking, NOT coverage:** model_0, NoHz, AND random all hit the SAME solve-within-budget ceiling (easy 99.9 / med 100 / hard 98.5) → the opener is always in the candidate set; model_0 just ranks it deeper. The NoHz gap collapses monotonically to 0 as k grows (med +21.6@1 → 0@100; hard +13.8@1 → 0@300). One number: **median sims-to-solve on hard = 8 (model_0) / 4 (NoHz) / 28 (random).**
- **Biggest @1 deficit is MEDIUM (−21.6), not hard (−13.8)** — tertile binning folded true-medium (sr 0.05–0.17) into "hard" and hid this. True-hard (sr<0.05, n=204) is a needle slice where even NoHz gets only 31.4@1 (opener ~1-in-30 → @1 is near-noise for ANY ranker).
- **⇒ SUCCESS METRIC REFRAMED:** hard@1 will NEVER hit the 95% bar by ranking alone (NoHz caps 31.4); the ceiling (98.5%) is reachable at budget → the loop's real deliverable is an EFFICIENCY win. **Watch: medium@1 (biggest gap), hard @{5,10,20}, and hard median sims-to-solve (8→4) — NOT hard@1.**
- **Hard-mining sims≥4 = CONFIRMED correct lever.** Documented 1-push bottleneck (`scorer_hacman_journal.md`, `EXP-2026-07-12-depth-geometric-grounding.md`) = edge-selection precision (opener median rank ~7/75; 88–96% of top-1 errors are wrong-EDGE, depth is solved). "sims≥4" mines exactly "opener outside top-3" = wrong-edge-ranking failures, self-targeting the weak buckets (~68% hard / ~24% med / ~3% easy). Same-arch NoHz (EdgeCrossAttn) proves better ranking is learnable from more/harder data → capacity ruled out.
- **"data scale" caveats:** mostly right (67k eps vs NoHz's 253k H=1 rows, 3.75×) but (a) it's DISTRIBUTION not just volume (NoHz = curated v4_hq + 311k 2-push rows; curriculum's bet H2 = targeted-hard > raw-size), (b) part of edge-difficulty is INHERENT (wrong-edge% plateaued ~88% across arch fixes; task sparsity ~2 valid/63). No label-noise evidence.

**Finish-axis FEELER (2026-07-13, subagent — gauge only, no decision) — the opener ALREADY transfers to 2-push finish (H3 validated).** GT `eval/finish_full/finish_gt.json` (6120 pure2push states, `is_opener` labels; `tier` = fixed cut on #valid-setups: hard 1-2 / med 3-8 / easy ≥9 — a DIFFERENT fixed-cut axis than eval_common.bin_of, but apples-to-apples across models). model_0 trained on 1-push openers ONLY (never a finish label), yet ranks finishes 3-4× better than random:

| bucket | model_0 finish@1/5/10/20 | NoHz-v3 @1/5/10/20 | random@1 |
|---|---|---|---|
| easy (3597) | 55.1/75.8/83.5/88.8 | 70.3/85.3/90.2/94.5 | 18.2 |
| med (1983) | 45.0/68.0/78.6/87.9 | 61.2/79.4/87.3/93.9 | 14.1 |
| hard (540) | 36.9/64.3/75.7/82.6 | 50.1/72.1/81.2/88.8 | 9.1 |
| all (6120) | 50.2/72.3/81.2/87.9 | 65.5/82.2/88.5/93.8 | 16.0 |

model_0→NoHz gap @1 ≈ 13-15pp (hard −13.2) = SAME data-scale gap as 1-push (a touch smaller) → finish isn't uniquely broken, it tracks general opener quality. **Transfer baseline: model_0 hard finish@1 = 36.9 → if model_1 climbs toward NoHz 50.1, the 1-push curriculum lifted finish too.** Reusable harness `scripts/sandbox/finish_score_ckpt.py` (~24 min/GPU, no sim, reads is_opener) — RE-RUN on model_1 for the transfer check.

**ROUND-1 RESULT (2026-07-13) — model_1 (33,128 mined model-hard episodes, 75/25 mix). WEAK POSITIVE, NOT the win.** Eval: search harness, fixed cuts, judged on sims + med@1 + hard@k (not hard@1).
| metric | model_0_hlg | model_1 | NoHz |
|---|--:|--:|--:|
| medium@1 | 59.9 | **64.4 (+4.5)** | 81.5 |
| hard median sims | 8 | **8 (FLAT)** | 4 |
| hard@1/5/10/20 | 17.6/41.7/58.8/71.1 | 16.7/42.6/56.9/74.5 | 31.4/54.9/68.6/82.8 |
| easy@1 | 93.3 | 91.0 (−2.3) | 97.9 |
| all@1 | 71.0 | 71.1 | 82.4 |

**Read:** medium@1 +4.5 (the one robust gain, n=421) but the hard efficiency headline is FLAT (median sims 8→8; only tail nudged, hard mean 15.8→14.4, p75 22→19), mild easy regression, all@1 flat. Leading hypothesis: the sims≥4 mine skews MEDIUM (medium scenes far outnumber true-hard sr<0.05) → hard under-represented in training → doesn't move. Per USER (flat → DIAGNOSE, don't hold): diagnosis agent running → recommend continue-round-2 / tweak (hard-weight the mine?) / fix. **Infra hiccups en route:** build-stall (~25min, dormant watcher) + trainer NFS-tempdir teardown hang (~5min; FIX = TMPDIR=/tmp, baked into round-2) — both root-caused + fixed.

**ROUND-1 DIAGNOSIS (2026-07-13, subagent) — the "75% hard" mix was actually 5% hard-solvable (STRUCTURAL mix bug). Verdict: CONTINUE, fix mix first (free).**
- **Bug:** `build_train_mix.py` picks the "hard" bucket by COUNT-tertiles of round-0 solve_rate; round-0 is 37.8% dead → the bottom tertile is 100% dead (sr=0). So "hard 75%" = 22,440 pure-DEAD + a mine only 8.6% hard-solvable → **genuine hard-solvable = 3,720/74,091 = 5.0%.** Round-1 never trained hard-heavy — the real experiment hasn't been run.
- **Eval moved where the data moved:** medium +9,841 fresh eps (+30% vol) → med@1 +4.5; hard-solvable +1,529 abs, drowned in 41% dead + 34% easy → hard flat (starvation, not underfit — tail did nudge); easy −2.3 = its share fell as dead rose (not forgetting; anchor fine).
- **Mine is medium-skewed** (hardest object per kept scene: hard 20% / med 53% / easy 26%; sims≥4 = weak hardness proxy). Do NOT raise the sims threshold (≥4→≥20 DROPS absolute hard-solvable 2,842→1,237).
- **SURPRISE — NoHz's edge is VOLUME, not hard-fraction:** its H=1 data is 2.4% hard-solvable / 51% dead (LOWER hard-fraction, MORE dead than ours), but 2.8× solvable rows + 311k 2-push (~7.6× total). **Hard is VOLUME-bound, not ceiling-bound** — correctly-composed data should move it.
- **FIX (free, existing labels): redefine "hard" by fixed cuts on true sr** (pool ~5,033 hard needles across r0+r1, approaching NoHz's 6,171; cap dead at ~15-20%) → hard-solvable share 5%→~14%. First honest test of H2 (targeted-hard > raw-size).
- **ACTION: building corrected mix + retraining model_1b (~40 min, free, no new sims, agent ac3ea0c6) as the honest confirmation BEFORE any full round-2.** If model_1b's hard median sims move (8→ toward 4) → scale to round-2 with the corrected mix; if not → deeper rethink. Even fixed, hard is volume-bound → expect several corrected rounds.

**=== SESSION SNAPSHOT 2026-07-13 (RS + volume ablation; BEST = model_1a_rs) ===**
All numbers = MY runs: `eval_one_local.sh` (best-first hmax=1 budget 300) → `agg_onepush_tertile.py --binning fixed --model-filter Hz` + `pipeline/sims_by_bin.py`, on namo_testset_v1 onepush (1323 eps). Apples-to-apples.

| model | data | dead | RS | easy@1 | med@1 | hard@1 | all@1 | hard@5 | hard@20 | hard sims |
|---|---|--:|--|--:|--:|--:|--:|--:|--:|--:|
| model_0 | r0 | 41% | – | 93.3 | 59.9 | 17.6 | 71.0 | 41.7 | 71.6 | 8 |
| model_1 | r0+r1 | 41% | – | 91.0 | 64.4 | 16.7 | 71.1 | 42.6 | 74.5 | 8 |
| model_1a | r0+r1 solv | 0% | – | 93.1 | 67.9 | 17.6 | 73.5 | 45.6 | 76.0 | 7 |
| **model_1a_rs** | **r0+r1 solv** | 0% | ✓ | 94.3 | 65.8 | **20.6** | **73.8** | 45.6 | 75.0 | **6** ← BEST |
| model_0a_rs | r0 solv | 0% | ✓ | 93.8 | 64.1 | 19.6 | 72.9 | 40.2 | 68.6 | 10 |
| model_1b | subset | 17% | – | 81.7 | 55.6 | 15.2 | 63.1 | 41.7 | 73.5 | 7 |
| model_1c | subset | 0% | – | 85.5 | 58.2 | 14.2 | 65.8 | 48.5 | 77.0 | 6 |
| NoHz-v3 | 253k+ | 51% | mixed | 97.9 | 81.7 | 30.9 | 82.4 | 54.9 | 82.4 | 4 |

**FINDINGS (verified):**
1. **RS (reachability supervision) is the win.** Fold unreachable `−1 → target 0`, `r_mask=1` (`build_model1a_rs.py`). hard@1 17.6→20.6, sims 7→6. `pred_vs_gt.py` proves it kills the OPTIMISM LEAK: non-RS predicts **0.29 mean on unreachable** (19% >0.5, hallucinated); RS predicts **0.020**. Openers unchanged (~0.70). Order-preserving, sharper hard.
2. **Full-volume + no-dead (model_1a) > model_1.** The 1b/1c easy/med crashes were **SHRINKAGE** (cut to 34k), NOT dead-removal. Drop dead + KEEP volume = good.
3. **Round-1 mine helped hard-WITH-SEARCH** (0a_rs r0-only vs 1a_rs r0+r1): hard@5 +5.4, @20 +6.4, sims 10→6. Flat reactive@1. +26k rows doubled hard supply.
4. **CORRECTED — solvable-1push gap to NoHz is ~1.8–3× (NOT 5–10×).** NoHz M2B solvable=**123,269** (49% of 252,805, rest dead) +AUG 80k = 203k. Ours 68,044. NoHz TOTAL=668,730 incl **311k H2 (2-push)** + 24.6k finish → the 5-10× wrongly counted dead+2push. NoHz is a MIXED 1+2push model.
5. **Biggest reactive gap = MEDIUM@1 (~16), not hard.** With-search we're close (all@20 within 2).
6. Hard supply ceiling **~5,169 needles** (r0 2,246 + r1 2,923) → can't exceed ~15% hard without oversampling; more needs generation.
7. Dead helps as EXTRA negatives (M2b +3.5, documented — suppresses reachable-junk impostors) but only ADDED not REPLACING solvable. We dropped ours; adding back untested.

**REGROUP DIRECTION:** scale to **~150k solvable** (2.2× current, ≈ NoHz M2B) = the decisive data-vs-composition test (does MEDIUM close?). Weight MEDIUM (biggest gap) + all HARD (rare); keep easy (natural ~61%, don't starve it). Favor BROAD volume over aggressive sims≥4 mining. Keep RS on; consider dead-as-negatives test.

**⭐ RANKING-AUX RESULT (2026-07-13) — DECISIVE WIN, biggest single lever so far.** `L = hl_gauss(value) + λ·softmax_CE(value/T over TRIED-REACHABLE cells, opener=+)`. Subclass `scripts/rl_loop/train_q2_rankaux.py` (monkeypatches `train_q2.build_module`; NO edit to shared `weighted_module.py`; val monitor stays PURE hl_gauss → apples-to-apples ckpt pick). Aux = listwise margin over `HLGauss.value(logits)` (differentiable E[bin], bounded [0,1]), softmax T=0.15. Bracketed λ∈{0.1, 0.5} on model_1a_rs data (68,044 rows), 40 ep, eval through the SAME `eval_one_local.sh` harness (1323 eps, 6 shards). `agg_fixed.py` (validated: reproduces model_1a_rs exactly).

| HARD solve@k | 1a_rs base | **λ=0.1** | λ=0.5 | NoHz (~10× data) |
|---|---|---|---|---|
| @1 | 20.6 | **27.0** | 24.5 | 30.9 |
| @2 | 28.4 | **40.2** | 38.2 | — |
| @5 | 45.6 | **57.4** | 55.4 | 54.9 |
| @10 | 62.3 | 71.6 | **74.5** | 68.1 |
| @20 | 75.0 | 85.8 | **86.8** | 82.4 |

med@1 65.8→**71.7**(λ.1)/65.6(λ.5)/81.7 NoHz · all@1 73.8→**76.1**/73.5/82.4 · easy@1 94.3→93.1/92.7/97.9. **λ=0.1 is the winner** (best hard@1, best medium, best all@1, calibration untouched val_loss 0.640 ≈ base 0.6376 — it just REORDERS). λ=0.5 overshoots (buys @10/@20, loses @1+medium+calibration). **On HARD we now MATCH-to-BEAT NoHz at every budget except @1** (hard@5 57.4>54.9; @10/@20 ahead) despite ~10× less data; hard@1 gap to NoHz **10.8→3.9**. Effect ≫ eval jitter (±3-4) AND consistent across both λ AND monotone in k → real, not noise. Sanity ("80% openers already top-1") UNDER-sold it: that was aggregate-with-easy; the HARD test bin openers ARE buried and the aux surfaces them. **New best model = `model_1a_rs_rank01` (λ=0.1)**: `curriculum2/round1/model_1a_rs_rank01/checkpoints/` + `eval_model_1a_rs_rank01/agg_fixed.json`. **IMPLICATION: this crosses the LOCKED "no ranking objective" constraint — the numbers say fold λ=0.1 aux into the loop; USER call to lift the lock.** This is also the tool the 2-push RARE FINISH will need (finish=γ·V, 8% needle).

**OPS (durable):**
- **hl_gauss trainer HANGS on exit EVERY run** (TMPDIR=/tmp does NOT reliably fix): completes (best ckpt + reload-check + `[eval_scorer-load check]` in log) then 97% CPU / 33GB GPU forever → detect the `[eval_scorer-load check]` marker, `pkill -9 -f "train_q2.py.*<name>"` (ckpt safe). ⚠ pkill pattern must not match its own shell (self-kill bug bit me).
- Train: `source env.ilab.sh && export TMPDIR=/tmp && CUDA_VISIBLE_DEVICES=<free> python -u scripts/rl_loop/train_q2.py --h5 <mix> --out-dir <dir> --epochs 60 --batch-size 256 --lr 3e-4 --num-workers 16 --warmup-steps 200 --patience 15 --seed 0`. `torch.load(..., weights_only=False)` (PyTorch 2.6).
- Difficulty/solve_rate = openers/**r_mask** (reachable), NEVER value_mask (≡1). Fixed cuts hard<0.05/med<0.30.
- ckpts: model_1a `epoch019`, model_1a_rs `epoch021-val_loss0.6376`, model_0a_rs `epoch021`, model_1c `epoch024`, model_0_hlgauss round0 `epoch022`. NoHz `outputs/scorer/qfull_nohz_v3_v4hq_s1/.../wl8k6iyv/epoch012`. Scripts + agg jsons + solve_curves plot all in `curriculum2/round1/`.

_(appended as rounds complete)_
