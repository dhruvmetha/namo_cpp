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
- **2026-06-26 ~13:30 ET — ❌ STAGE 1 GATE FAILED (trained on ilab; GPU backlog forced the move). reactive@2 (n=1018,
  region):** qboot_density **30.3**, qboot_depth **34.1**, vs **NoHz-v3 40.7** (Hz-v3 45.6). **The bootstrap LOSES by
  6–10pp** — worse than my pre-registered "matches-not-beats" (that prediction = WRONG, it loses not matches). best-first@2
  partial: depth ~38.7, density ~30. **depth > density HELD** (34.1 > 30.3; my Stage-3 call). Ckpts (ilab-trained,
  best-val): density `qboot_density_s1/.../v5x21lsi/epoch012-val0.7152`, depth `.../xdbdc8vv/epoch014-val0.7192`.
  **⚠ CONFOUND (my error — violated one-change-at-a-time):** the qboot mix differs from NoHz-v3 in THREE ways, not one:
  (1) setup labels flat-0.9→γ·V_GT [intended], (2) **dropped `aug`**, (3) **finish data `exit_finish_v4` vs NoHz-v3's
  `exit_finish_valid`**. So "bootstrap regressed 10pp" = "this 3-change mix regressed", NOT cleanly "the bootstrap idea
  failed." Hint it IS partly the target: density (tiny ~0.1 targets) is the worst arm → grounded γ·V_GT may under-rank
  setups. **NEXT = CLEAN one-change re-run:** NoHz-v3's EXACT mix (`m2b + h2 + aug + exit_finish_valid`), change ONLY the
  h2 H=2 setup-cell labels (0.9 → γ·V_GT, matched via `frac_first_push`); train NoHz recipe; gate vs 40.7. Relabel script
  = new (relabel h2 H5 in place, NOT the separate boot_setup H5). Runs on ilab. [Ranking loss STILL parked — one change.]
- **2026-06-26 ~14:00 ET — gate table completed + made HONEST (user: "not really apples to apples, no?").** Full table
  (n=1018 pure-2, region; best-first COMPLETE + dedup-verified, lines==unique==1018, no dupes): reactive@2 / best-first@2 /
  solve@900 / med-sims — qboot_density **30.3 / 28.8 / 93.1 / 9**, qboot_depth **34.1 / 31.8 / 93.5 / 7**,
  NoHz-v3 **40.7 / 37.8 / 95.9 / 4**, Hz-v3 45.6 / 36.1 / 97.7 / 4. (CORRECTION: my first best-first read 31.2/35.6 was on
  PARTIAL still-running shards; complete = 28.8/31.8 — lower. reactive was complete throughout. NOT a dedup bug.) **solve@900 ties
  ~95% for ALL (the 2-push set is exhausted by budget-900 search → NOT a differentiator; my earlier "~58" was a stale
  guess — deleted).** **NoHz-v3 3-seed baseline is TIGHT: reactive [40.7,40.3,41.1]=40.7±0.4, best-first [38.0,37.3,38.0]
  =37.8±0.3 — seed noise on THIS metric is ±0.4, NOT the ±3-4pp I'd assumed; 1-seed-vs-1-seed is fair here.**
  **NOT-apples-to-apples = TWO things only: (1) the 3-change CONFOUND (dropped aug + finish v4≠valid) — since depth's setup
  labels ≈ status-quo 0.9, the entire depth−NoHz −6.6pp is the aug/finish swap, NOT the bootstrap; (2) machine (ilab vs
  Amarel, unmeasured).** **The ONE clean cell = density vs depth −3.8pp (same run/machine/seed, only V(s1) summary) → the
  density target genuinely hurts.** Clean re-run's **depth arm = control that MEASURES the machine gap** (should hit
  40.7±machine; if 38, subtract 2.7). Verdict to stand on: density<depth (real); everything-vs-NoHz waits for the re-run.

- **2026-06-29 ~04:00-04:30 ET — WALL-CLOCK TIMING INVESTIGATION → the render is the deploy bottleneck (overturns "minimize
  sims").** Smoke on one node (warm, perf_counter): **sim (env.step) ~160ms · NN forward (score_ctx) ~36ms · reachability
  ~7ms · BUT render_ctx (the model-input crop) ~2000ms.** So the per-state cost is **render-bound**, wall-clock ≈
  (#states-scored × ~2s), and "minimize sims = minimize time" is FALSE on CPU deploy. (I first mislabeled the 2s as "NN
  forward" — wrong; verified it's the RENDER, not the net.) **Render map (Explore agent):** `live_scorer.render_ctx` →
  `NAMODataVisualizer.generate_all_masks_highres` (sage `visualizer.py` L1068, 1024² canvas) → `WavefrontSnapshotExporter`
  (namo_cpp `python/namo/visualization/wavefront_snapshot.py`). The 2s = **pure-Python 8-conn region BFS in `_compute_regions`**
  (~0.5-1.5s) + discarded wide/global crops + 63MB zero-alloc/call + YAML×2 + XML parse/call. Channels: `static`(walls)=fully
  static, `movable`/`robot_region`/`goal_region`=static-within-state, `target_object`=dynamic. **FIX (output-preserving only —
  must NOT change the pixels the model trained on): replace the Python BFS with `cv2.connectedComponents` (the big ~10× win,
  but TRAINING-CRITICAL → must be bit-identical) + skip discarded crops + cache static reads.** Built: **`test_render_equiv.py`**
  (bit-compare GATE: capture original crops → assert `np.array_equal` after any change; ref captured = 29 crops),
  **`time_benchmark.py`** + `time_benchmark.slurm` (warm, interleaved Hz/NoHz/random, same node, component timing).
  **[AUTONOMOUS DECISION 2026-06-29]:** did NOT rewrite the BFS unattended (feeds all training data; bit-perfect rewrite on a
  29-sample gate while USER asleep = too risky). Fix is prepped+documented for USER to execute+review awake. Plan file:
  `~/.claude/plans/memoized-skipping-lampson.md`. Excluded `local_output_size=64` (changes pixels → needs retrain).
- **2026-06-29 — SEEDED stratified tables (3 seeds each, mean±spread; pure-2 n=1018, 1-push n=1323).**
  **1-push 1-deep best-first solve@{1,2,5,20,900}:** Hz `84.3±0.5 / 88.4±0.6 / 93.5±0.2 / 98.2 / 99.6` · NoHz `82.4 / 86.1 /
  91.1±1.0 / 97.1 / 99.7` · random `38.0 / 52.7±1.9 / 70.1 / 88.8 / 99.7`. → Hz>NoHz at low budget (outside bars), converges
  by @900 (model-independent ceiling, verified: @900 differs by 1 flaky episode). **2-push reactive@2** (easy/med/hard): Hz
  `62.4±0.4 / 47.3±4.1 / 25.1±4.0` ≈ NoHz `62.9±2.0 / 46.9±2.5 / 25.8±2.3` — **the single-seed "Hz wins reactive" was SEED
  NOISE; they TIE.** **2-push best-first@2:** Hz `43.3±5.6 / 36.0±4.0 / 20.3±4.6` < NoHz `55.6±2.8 / 42.9±2.3 / 24.4±2.0`
  (**NoHz>Hz real, outside bars easy/med**). random ≈1-10. Hard reactive@2 ~25% = the frontier. Uniform feasibility (earlier):
  raw unguided search SOLVES pure-2 cold (median ~43 sims) → warm-start NOT mandatory; the model buys ~10× sim-efficiency,
  not solvability. **Eval gains:** `eval_reactive_argmax.py` got `--h` (query budget) + `--leaf-out` (per-episode jsonl).

- **🤖 AUTONOMOUS RUN STATE [2026-06-29 ~05:00 ET, USER asleep ~3h — RESUME HERE if compacted].** Task: speed up the
  render (the deploy bottleneck) WITHOUT changing the model input (bit-compare gated), then the wall-clock timing stat.
  **Render finding (measured):** render ~2019ms dominates (sim ~160ms, NN forward ~36ms). The 2s is NOT the region BFS
  (only ~370ms) — cProfile: it's `circle_fully_within_region` (visualizer.py:1502) allocating a full **1024² array
  ×~3000/render** (robot/goal SAMPLING). **The model uses only 5 channels `static,movable,target_object,robot_region,
  goal_sample_region`** (live_scorer.py:47) — NOT robot/goal/goal_samples — so the sampling + discarded wide-crop +
  globals are ALL wasted for `render_ctx`.
  **DONE+COMMITTED (both BIT-IDENTICAL, gate `test_render_equiv.py` 29/29 diff=0):** (1) BFS→`scipy.ndimage.label`
  (namo_cpp `a2a826b`, br `feat/horizon-q-redesign`); (2) `circle_fully_within_region` windowed to circle bbox (sage
  `11dc6ac`, br **`feat/render-speedup`**). **Render 2019→322ms (6.3×).**
  **IN PROGRESS — `fast_scorer` flag** (skip wasted sampling+wide+global for render_ctx; opt-in → training byte-identical):
  signature DONE; PENDING = (a) 3 sampling guards in extract_local_crop (visualizer.py ~1590/1603/1624: prefix
  `(not fast_scorer) and ` to each `if`); (b) wide-crop+rewind guard ~1669 (`if not fast_scorer:`); (c) global-masks loop
  guard ~1494; (d) render_ctx live_scorer.py:141 add `fast_scorer=True` → then GATE
  (`python scripts/sandbox/test_render_equiv.py --mode compare --n 30`, MUST be bit-identical) → re-measure → commit →
  re-run fast timing. Expected ~150ms. If gate FAILS → revert fast_scorer edits (322ms already committed+safe).
  **TIMING:** before job 57505248 DONE (`/scratch/dm1487/eval/timebench/current_render.jsonl`): model t_wall ~3.6-4.0s
  render-bound vs random ~0.3s = 12× slower. Fast(322ms) job 57508274 RUNNING (watcher `b6r56u72i`, `.../fast_render.jsonl`)
  = the "after". `time_benchmark.py`=warm interleaved Hz/NoHz/random reactive@2 same node. Gate ref:
  `/scratch/dm1487/eval/render_equiv/ref_crops.npz`. Slack→DM `U07N1DR8S94` (3 sent), hourly heartbeat set. NO pushes.
- **✅ RENDER FIX COMPLETE [2026-06-29 ~05:30 ET]: `fast_scorer` LANDED + BIT-IDENTICAL (gate 29/29 diff=0).** render
  **2019→101ms = 19.9× total**, all 3 changes provably output-preserving (BFS, circle-window, fast_scorer). **101ms <
  sim ~160ms → scoring is no longer the deploy bottleneck.** Committed: sage `c0a00f7` (`feat/render-speedup`), namo_cpp
  `5e0c2ae`+`a2a826b` (`feat/horizon-q-redesign`). FINAL timing (101ms) job 57509898 RUNNING (watcher `bkk4xcnvx`,
  `/scratch/dm1487/eval/timebench/final_render.jsonl`). REMAINING: aggregate before(57505248)/after(57509898) timing →
  Slack the table → optionally the same `fast_scorer`-style speedup belongs in the data-collection render too (same
  visualizer; training uses the default-False path so it's untouched, but collection could opt in). NO pushes (user's call).
- **✅ TIMING STAT DONE [2026-06-29 ~05:20 ET] — the original goal.** Wall-clock reactive@2, same node, warm-only, 100/tier.
  Median `t_wall`/episode (sec) before(2019ms)→322→101: **Hz** 3.65/3.99/3.64 → 1.05/1.13/1.00 → **0.80/0.86/0.81**;
  **NoHz** 3.72/3.83/3.55 → 1.04/1.06/0.94 → **0.79/0.77/0.70**; **random** ~0.3 throughout (no scoring). At 101ms: 2 model
  scorings=`t_score`~0.20s, 2 sims=`t_sim`~0.6s → **scoring is now ~⅓ of sim; the SIM is the new bottleneck, not render.**
  **Hz/random wall ratio: before 11-15× → final 2.0-2.6×** while model solves 10-30× more (Hz hard 30% vs random 4%, easy
  61% vs 8%). Files: `/scratch/dm1487/eval/timebench/{current,fast,final}_render.jsonl`. Implication: the model's value
  (fewer sims via ranking) now converts DIRECTLY to wall-time — esp. for best-first. **AUTONOMOUS RUN CORE COMPLETE**
  (render 20× bit-identical + committed; timing before/after; durable here). Open follow-ups for USER: (1) push branches?
  (2) opt data-collection render into fast_scorer? (3) best-first wall-time curve now feasible (~0.3s/sim).
