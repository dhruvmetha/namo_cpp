# Horizon-Q — LevinTS Search-Ordering (frozen model) — Design Spec

> **Date:** 2026-06-29. **Branch:** `feat/horizon-q-redesign`. Companion to the redesign
> journals ([execution](horizon_q_redesign_execution.md), [search-first](horizon_q_search_redesign_journal.md),
> [registry](horizon_q_model_registry.md)). This is the FIRST step of adopting LevinTS/PHS\* as the
> search+learning framework — the **search half only, frozen model**.

## 1. Decision context

- **Horizon DROPPED — flag-off, reversible** [USER, 2026-06-29]. Run single-Q everywhere
  (`budget_cond=False` / `+data.budget_h=false`). **No model/data surgery** — the `budget_h`/H
  embedding stays dormant in `scorer_data.py` and `classifier_module.py`. "For a while, just
  no-horizon stuff." The qboot run is already this.
- **Adopt LevinTS/PHS\*** (policy-guided tree search with an expansion guarantee = a bound on
  node-expansions = a bound on **sims**). Code refs: `levilelis/h-levin`; LevinTS (Orseau et al.
  NeurIPS'18), PHS\* (Orseau & Lelis AAAI'21).
- **No-exhaustive-data honored.** LevinTS trains π and h from the search's OWN found solutions, never
  from the global `(setup×finish)` pairmap. This de-oracles the plan — see §6. (The current qboot's
  `γ·V_GT` target reads the pairmap = eval-luxury; LevinTS's h replaces it. Out of scope here, flagged
  for the learning step.)
- **Staging** [USER, one-change-at-a-time]: land the **search half first, frozen model**. Learning
  half (Levin loss, sims cost-to-go `h`, PHS\*) deferred to a later spec.

## 2. Goal of THIS step

Add a LevinTS priority (`d/π`) as a new ordering in the existing best-first search and **A/B it vs the
current `combine=q` ordering on avg-sims-to-solve**. Zero training.

## 3. Why this is "just best-first"

LevinTS = best-first on the cost `d(n)/π(n)` — same frontier/heap as `eval_bestfirst.py`. Differences
vs `combine=q`:

1. cost is a **PATH product** — carry cumulative `log π` on the node (not a per-node scalar `q`);
2. expand **lowest `d/π` first** (min-heap) vs highest `q`;
3. **`dive_bonus` disappears** — depth is baked into `d/π`.

Deeper reason it is *better-behaved* best-first: `d/π` is **monotone non-decreasing along a path**
(`d` grows, `π` shrinks), so best-first on it is principled (like A\* with a consistent heuristic).
`combine=q` is a local score, non-monotone down a path — which is exactly why it needed the
`dive_bonus` crutch. A swaps a hack for a guarantee, on the same heap.

## 4. Approach — A (minimal in-place)

- **File:** `scripts/sandbox/eval_bestfirst.py`.
- Add a `combine="levin"` branch to `priority()` (currently `:51-54`).
- Node carries **cumulative `log π`** = Σ `log softmax(score/τ)` along the chosen path; π is taken
  over the **full reachable pool** at the node (`candidates()`, `:39-48`).
- `d = ndone + 1`.
- **Levin cost** = `d / exp(cum_logπ)` = `d · exp(-cum_logπ)`. Push `+cost` into a **min-heap**; make
  the heap sign **mode-aware** (current code negates for a max-heap — gate on `combine`).
- `combine=q` path **untouched** — it is the baseline arm.
- New flag: `--tau` (default `1.0`).

## 5. Design calls

1. **LevinTS only** (`d/π`) this step. PHS\* (`(g+h)/π`) **deferred** — it needs `h` in sims-units, and
   the frozen qboot value is γ-discounted `[0,1]`, not sims.
2. **π = softmax(score/τ)** over the full reachable pool; **τ swept** (start `1.0`). `combine=q` stays
   the baseline — pure A/B.
3. **Log per run: solution path, per-step π, sims** — even frozen. Cost ~0 here; hands the future
   Levin-loss step its training data for free. (Where: extend `eval_bestfirst`'s result dump.)

## 6. Gate (pre-registered)

- **Metric:** avg-sims-to-solve (primary), solve-rate (secondary).
- **Arms:** `combine=levin` (τ sweep) vs `combine=q` (baseline). **Frozen NoHz qboot model** —
  ckpt path from [the registry](horizon_q_model_registry.md), do NOT glob.
- **Set:** n=1018, region criterion, object-constrained (same as the redesign gate).
- **Pre-registered prediction [CLAUDE]:** `levin ≈ combine=q` on avg-sims at H=2 — the within-node
  order is identical; only the **cross-depth dive** differs, and `dive_bonus` already approximates it.
  **ACCEPT-as-keeper iff** `levin ≥ combine=q` on avg-sims AND it removes the `dive_bonus`
  hyperparameter (one fewer knob, principled). The strategic value is **scaffolding for the learning
  half + depth≥3**, not an H=2 win. Don't rationalize a loss into a win.

## 7. Deferred (explicitly OUT of scope here)

- **Levin loss** — train π to MINIMIZE sims (replace/augment the InfoNCE/HL-Gauss classification). Next
  spec; uses the §5.3 logged solution paths.
- **Cost-to-go `h` in sims** — retarget qboot off **found solutions** (remaining-sims observed), NOT
  `γ·V_GT`. Next spec; also de-oracles the current qboot.
- **PHS\*** (`(g+h)/π`) — rides in with `h`.
- **Multi-push depth ≥ 3** — the real LevinTS payoff. Don't hardcode `depth==2` in the new ordering,
  but don't build depth≥3 finish-labeling here.

## 8. Touch-points (reference)

- `scripts/sandbox/eval_bestfirst.py` — `priority()` `:51-54`, `solve_scene` `:57-83`, `candidates()`
  `:39-48`, heap push / `dive_bonus` `~:82`.
- Baseline arm: `scripts/sandbox/eval_reactive_argmax.py`.
- Scorer: `scripts/sandbox/live_scorer.py` (`rank_first_pushes_h2` / `score_state`).
