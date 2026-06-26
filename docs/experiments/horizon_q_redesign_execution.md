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
**Status:** ⏳ IN PROGRESS (building the harness).

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
