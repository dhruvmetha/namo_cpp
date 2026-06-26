# Horizon-Q — BOOTSTRAPPED-VALUE + GUIDED-COLLECTION — Staged Execution Journal

> **Started 2026-06-25 [USER staged plan].** The EXECUTION journal for the redesign. Reads on top of:
> - **Thesis + decision ledger:** [horizon_q_search_redesign_journal.md](horizon_q_search_redesign_journal.md)
>   (model = sims-minimizing ranker; cost-to-go in SIMS not depth; D2 finish-ranker / D3 recurrence).
> - **Empirical record (v2/v3/v4 numbers):** [horizon_q_build_journal.md](horizon_q_build_journal.md) §9.
> - **Self-contained brief:** [horizon_q_HANDOFF.md](horizon_q_HANDOFF.md) (arch + algorithm + data collection).
> - **Ckpt paths (NEVER glob):** [horizon_q_model_registry.md](horizon_q_model_registry.md).
>
> **Branching (anchor safety):** `feat/horizon-q` = FROZEN anchor (the working v2/v3/v4 line, commit `3d65375`). All
> radical changes happen on **`feat/horizon-q-redesign`** (this branch) + a matching branch in `sage_learning`. NEVER
> overwrite registered ckpts; every new run uses a NEW run-prefix. Old models stay as comparison anchors.
>
> **Each stage = Goal / Build (grounded in our code) / GATE (the measured condition to proceed) / Guardrail (what
> autonomous exec must NOT do) / Status.** Gates let us STOP early and not build stage N+1 if stage N says it's pointless.

## THE CORE INSIGHT (why this plan)
Once the value is **bootstrapped** — `Q(s,a) = [a opens] ∨ γ·V(T(s,a))`, `V(s')=top-k mean Q(s',·)` — the **Horizon
becomes redundant** (it was a crutch for the non-bootstrapped supervised setup, and the data showed Hz≈NoHz + a
structural dive tax). The bootstrap encodes cost-to-go directly. So: drop Horizon, bootstrap one Q, then fix the
**collection** (the likely real bottleneck) with value-guided + exploration sampling. v2→v3→v4 saturation (finish data
is tapped out) is the empirical green-light to switch axes from "more data" to "bootstrap + guided collection."

---

## STAGE 0 — Instrumentation & baselines (read-only; the unlock)
**Goal:** make the bottleneck measurable before changing anything; produce every downstream gate.
**Build:** `scripts/sandbox/stage0_instrument.py` over the test set (`pure2push.json` + the exhaustive
`exhaustive_pairmap_pure2.pkl` as GT), tiered by difficulty (`pure2push_divisions.json`):
- **(a) realized rank of the true SETUP at s0** — score s0 with the current best setup model (Hz-v4 H=2 head, registry),
  rank the GT valid setups (pairmap a1 with ≥1 opener). Via `scorer_beam`/`live_scorer` H=2.
- **(b) realized rank of the true FINISH at observed s1** — for each GT setup's saved s1, score finishes (H=1), rank the
  GT openers. (Pairmap gives openers; ExIt H5 / trees give the s1 crop.)
- **(c) rollouts-to-solve, tiered** — reuse the `bfq_*` best-first leaf jsonl (avg-sims already computed).
- **(d) viable-finish-count distribution at s1** — openers-per-reachable at each solvable s1 (the findability-HAS-signal
  check). Partly known: median 6.9% density, 88% of s1 dead → **gates Stage 3** (skip if bimodal/degenerate).
- **(e) push/state COVERAGE of the random-K collection** — which reachable (edge,depth) cells & s1 states the collection
  sampled vs the exhaustive-reachable set (H5 `r_mask` vs pairmap reachable). → **gates/justifies Stage 2.**
- **(f) [CLAUDE addition] train-vs-test rank separation** — does the model rank openers well on SEEN states? (aliasing
  vs under-prioritization: if seen-state rank is sharp but test is not → signal present, ranking/collection fixable.)
**GATE:** none — this stage *creates* the gates. (Decision rules it sets: rank gaps → is the model or the data the
bottleneck; (d) → run Stage 3?; (e) → run Stage 2?; (f) → is the signal even there?)
**Guardrail:** READ-ONLY on training. Touch only eval/logging. Use registered ckpts; never glob.
**Status:** ✅ COMPLETE [2026-06-25 ~22:20 ET]. **CONVERGED VERDICT — the bottleneck is the SETUP value/ranking:**
- (a) SETUP rank @s0 (HARD): median **5**, top1 **~20%** = THE bottleneck.
- (b) FINISH rank @s1 (HARD): median **0**, top1 **~58%** = near-ORACLE, already solved.
- (d) viable-finish-count VARIES (median 5, p10 1→p90 12) ⇒ **Stage 3 gate PASSES** (depth-vs-density testable).
- (e) collection WELL-COVERED (setups exhaustive=median 45/ep; finishes near-exhaustive=median 44 a2 tried/~45 reachable)
  ⇒ random-K does NOT starve ⇒ **Stage 2 (guided collection) DEPRIORITIZED — low headroom.**
- (f) finish signal present (train-sep 0.75 vs test 0.30 — prior); Hz≈NoHz on all ranks ⇒ dropping Horizon is free.
⇒ **PLAN REORDER: the lever is the SETUP value/ranking (Stage 1 + maybe a setup ranking loss); NOT the finish (D2) and
NOT collection (Stage 2).** The Stage-1 density-bootstrap already tests a graded (ranking-ish) setup signal via the value
head (eval-compatible, no new loss); escalate to an explicit setup ranking loss only if density/depth match-not-beat NoHz.

## STAGE 1 — Single bootstrapped value (drop Horizon)
**Goal:** remove the dive-tax machinery; establish the clean bootstrapped baseline.
**Build (sage_learning, new branch):** one Q map, `budget_cond=False`. Bootstrapped targets from existing transitions
`(s, a, s', opens)` — the 2-push trees already store s0→a1→**s1** and s1's finish labels give `V(s1)=top-k mean`. Add a
**target network** (periodic sync) + **replay buffer** (mandatory for bootstrap stability). Warm-start from a supervised
model (NoHz-v3/v4) so Q starts non-random. New run-prefix `qboot_*`. THE HARD PART = building the `(s,a,s')` transition
dataset (join h2 setup rows → their ExIt/postpush s1 value).
**GATE:** reactive AND best-first ≥ current NoHorizon (38.2 / 34.9 @2 region) AND **dive tax → 0**. If bootstrap
DIVERGES (loss explodes / value collapses to constant) → STOP, report, do Stage 1b. Do NOT proceed past divergence.
**Guardrail:** if unstable > ~2k steps → HALT, dump diagnostics, don't burn compute. NEVER overwrite anchor models.
**Status:** ⬜ PENDING (gated on Stage 0 read).

> **⏳ STAGE 1 STATUS [2026-06-26 ~00:25 ET] — BUILT, QUEUED, GPU-BLOCKED (post-maintenance backlog).** Bootstrap-setup
> H5s built (density+depth, 20 shards each, 5169 rows). qboot training launched after 2 fixes: (1) sharded the build
> (render ~1.8s/ep blew the 2h wall), (2) **cluster dropped the `gpu` partition post-maintenance → override to
> `gpu-redhat`** (`PART` env in launch_bootstrap.sh; relaxed to `gpu-redhat,legacy-gpu`). Jobs: **qboot_density
> `57177529_9`, qboot_depth `57177530_9`** (1-seed feelers, run dirs `qboot_{density,depth}_s1`). **0 idle GPUs;
> SLURM ETA ~25h (pessimistic; backfill may start sooner).** **Eval auto-chained (afterany):** `57177596` (boot_density),
> `57177597` (boot_depth) → reactive@2 + best-first@2(q) the moment each converges. **GATE vs NoHz-v3 (reactive 40.7 /
> best-first 37.8 @2).** RESUME: when the GPU frees, training runs → eval-chains fire → aggregate `reactarg_boot_*` +
> `bfq_boot_*`, compare to NoHz. (Stray: my own CPU interactive node `57157833` on halk0057, ~7h idle — flagged to USER.)

## STAGE 1b — Stability insurance (ONLY if Stage 1 gate fails on divergence)
**Goal:** get the bootstrap off the ground from a cold start.
**Build:** seed the replay buffer with grounded MC labels (our existing γ-labels — even partial) so Q is non-random
before bootstrapping takes over; slower target sync, larger buffer.
**GATE:** bootstrap now trains stably. If still diverging → fall back to the current MC-label model, report that
bootstrapping needs more work. Do NOT silently ship a broken model.
**Status:** ⬜ CONTINGENT.

## STAGE 2 — Guided collection (the real bottleneck fix; gated on Stage 0(e) + Stage 1)
**Goal:** replace uniform random width-K with value-guided collection so hard instances aren't data-starved.
**Build:** a collection loop (`modular_parallel_collection` + `region_opening`, model in the loop) that uses the current
Q to pick which K branches to expand per node (vs random K) — ExIt extended from finish (`exit_collect.py`, already
on-policy) to the WHOLE tree incl setups. Retrain on the augmented buffer.
**GATE:** on HARD instances, does guided catch good setups more often than random-K (the ~5% H=2-setup-success rises)?
AND does the retrained model improve hard-tier reactive / rollout-count? If hard metrics don't move → collection wasn't
the bottleneck; report and STOP.
**Guardrail:** MUST run with Stage 2-exploration (guided-only collapses coverage via feedback loops). Do NOT run guided
collection without exploration.
**Status:** ⬜ PENDING.

## STAGE 2-exploration — coverage/bias control (runs WITH Stage 2)
**Goal:** stop guided collection from going blind on pushes the current Q dislikes.
**Build:** ε-greedy (mostly follow Q, sometimes random branch) or value-uncertainty exploration; track push/state
coverage as a first-class metric. **Experiment:** sweep ε ∈ {0, 0.1, 0.3, pure-random}; output a coverage-vs-performance
curve (the paper ablation). Too little → Q blind spots → wrong rankings on unsampled pushes; too much → random-K
starvation.
**GATE:** find the ε that maximizes hard-tier rollout reduction; report the curve regardless.
**Guardrail:** coverage metric must not collapse below threshold; auto-flag if it does.
**Status:** ⬜ PENDING.

## STAGE 3 — Findability vs depth (label-semantics ablation; gated on Stage 0(d))
**Goal:** does density-sensitive value (findability) beat path-existence value (depth)?
**Build:** two variants identical except the `V(s')` summary — **max-like (depth)** vs **top-k-mean / soft-count
(findability)**. Same data, same everything else.
**GATE:** only run if Stage 0(d) shows viable-finish-count actually VARIES (skip if bimodal — findability can't help).
Then: does findability reduce rollouts on the SETUP decision specifically? Yes → paper result; no → report the null,
keep the simpler depth value.
**Guardrail:** change ONLY the `V(s')` summary; any other diff confounds the ablation.
**Status:** ⬜ PENDING (gated on Stage 0(d)).

## STAGE 4 — Depth scaling H=3 (optional; gated on committing to depth)
**Goal:** confirm bootstrapped guided-collection extends past depth 2.
**Build:** run the Stage-2 pipeline at H=3; measure whether it finds depth-3 solutions at all (random-K won't).
**GATE:** entirely gated on whether H>2 is a real target for this paper. If H=2 is the endgame → SKIP.
**Guardrail:** depth-3 collection is expensive; cap compute, report partial.
**Status:** ⬜ DEFERRED (H=2 is the current endgame).

---

## AUTONOMOUS EXECUTION LOG (append-only; Slack U07N1DR8S94 at each milestone)
- **2026-06-25 ~17:30 ET** — [USER] handed the staged plan, AFK ≥5h. Anchor committed (`feat/horizon-q` `3d65375`),
  branched `feat/horizon-q-redesign`. This journal created. Starting Stage 0 (instrumentation, read-only). Slack cadence:
  update at Stage-0 gates, Stage-1 build, stability feeler.
- **2026-06-25 ~18:30 ET — ✅ STAGE 0 GATES (the measurement REORDERS the plan).** Realized ranks (current Hz-v3 /
  NoHz-v3 vs the exhaustive GT pairmap, n=1577, tiered; `stage0_instrument.py` + `.slurm`, outputs `stage0_{hz,nohz}_v3/`):
  | metric (HARD tier) | Hz-v3 | NoHz-v3 | read |
  |---|---|---|---|
  | **FINISH rank @s1** (med / top1 / top5) | 0 / 58% / 85% | 1 / 50% / 72% | **near-ORACLE — finish already solved** |
  | **SETUP rank @s0** (med / top1 / top5) | 5 / 19% / 48% | 5 / 22% / 49% | **the BOTTLENECK** |
  - **THE FINDING:** the pairmap "finish needle = 18 sims/hard" was vs RANDOM order; the **MODEL already crushes random
    on the finish** (hard median rank 0). So **a finish-ranker (D2 in the search-redesign journal) is NOT the realizable
    lever** — its oracle ceiling is real but already captured. The **SETUP** is weak (hard top1 ~20%, median rank 5):
    reactive ≈ setup-top1(0.20) × finish-top1(0.58) on hard ⇒ the setup factor is the limiter. ⇒ **Stage 1
    (bootstrapped SETUP value, drop Horizon) is CONFIRMED as the right lever; D2 deprioritized.**
  - (d) viable-finish-count VARIES smoothly (median 5, p10 1→p90 12; 29% ≤2, 32% ≥8) ⇒ **Stage 3 gate PASSES** (testable).
  - **Hz ≈ NoHz on both ranks** ⇒ dropping the Horizon (Stage 1) loses nothing. (e) coverage still TODO.
  - **De-risk for Stage 1:** finish VALUES are mushy (sep 0.30) but finish RANK is good ⇒ seed the bootstrap with
    **GT V(s1)** from the true finish labels (`relabel_bootstrap_setup.py`), not the mushy model value → stable iter-0.
  - **BUILT (committed):** `collect_transitions.py` ((s0,a1,s1) transitions — kept for the later model-V bootstrap), 
    `relabel_bootstrap_setup.py`. **CHEAPER PATH FOUND:** the training labels (`labels_exhaustive_pure2push.json`)
    already store `frac_first_push=[[e,d,n_open,n_tried],...]` per setup ⇒ V_GT(s1) is FREE; only render s0.
    `build_bootstrap_setup.py` (+`.slurm`) does this (no re-sim). Smoke OK (30 ep: 156 solvable / 1660 dead setup
    cells; density targets graded 0.007→0.48 = the findability signal). Full builds launched (density `57171507`,
    depth `57171508`). Launcher `launch_bootstrap.sh` (single-Q, drop Horizon, mix m2b+ExIt-finish+boot-setup, NoHz
    flags, from scratch — grounded targets ⇒ stable, no online divergence = the Stage-1b seed done offline).
- **2026-06-25 ~22:10 ET — ⚠ HYPOTHESIS from Stage 0 [CLAUDE, pre-registered, the gate decides]:** since the FINISH is
  **near-oracle** (model finishes any solvable s1 in ~1 sim), the correct `V(s1)` ≈ **existence (DEPTH)**, NOT density —
  density would wrongly penalize a 1-needle-finish setup the model can actually finish cheaply. ⇒ **predict depth ≥
  density** (the Stage-3 ablation, run together). DEEPER: depth-bootstrap-with-GT ≈ the status-quo 0.9 setup label ⇒
  Stage 1 may just MATCH NoHz (pass the gate, not improve); if so, the setup bottleneck (top1 20%) is a **discrimination
  problem (solvable vs plausible-DEAD setups)** not a value-target one → the real fix would be a **setup RANKING loss**
  (the setup analog of the finish ranker). Running Stage 1 (both summaries) to TEST this; if it matches-not-beats NoHz,
  pivot to the setup ranking loss. [Surfaced to USER on Slack.]
- **⛔ [USER DIRECTIVE 2026-06-26] ONE-CHANGE-AT-A-TIME — setup ranking loss DEPRIORITIZED TO THE VERY END.** "Don't
  change too many things at once." The bootstrap (drop-Horizon + grounded setup value) is the SINGLE active change.
  Do NOT build/launch the setup ranking loss until: bootstrap gate landed + simpler levers exhausted + a setup
  mirror-measurement confirms setup-top1 is FIXABLE (not aliased). It's a reactive-only lever (search dissolves it).
- **2026-06-26 ~00:10 ET — build hiccup + fix (autonomous).** First bootstrap-setup builds (single-task) TIMED OUT at
  the 2h wall — the s0 render is **~1.8s/episode** (the wavefront BFS rebuild dominates), so 5076 ep exceeded the wall
  and the end-written H5 was lost. FIX: **sharded** the build (array 0-19, 260 ep/shard, incremental per-shard H5s =
  wall-safe). `build_bootstrap_setup.slurm` now an array; `launch_bootstrap.sh` globs `shard_*.h5`. Relaunched (density
  `57176990`, depth `57176991`); watcher `bgjxm9aba` re-armed to fire qboot training on drain. (Lesson: any per-episode
  render job MUST shard + write incrementally.)
