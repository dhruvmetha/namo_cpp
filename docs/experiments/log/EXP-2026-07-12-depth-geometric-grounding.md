---
type: experiment
status: idea
created: 2026-07-12
metric: 2push-hard solve@sim-budget + avg-sims-to-solve (primary); H=2 chain-edge joint@1 + depth_top1_hist shift (offline secondary); 1-push hard@1 guardrail (±2pp)
thread: scorer-search
tags: [experiment, ranker, depth, geometric-grounding, idea, scorer]
---
# Give the DEPTH axis a geometric coordinate — ground each (edge,depth) in its known Δpose, don't leave it a bare output channel

> Analysis + one proposed A/B. NO training launched. Every number below is read-only from the CAR scorer H5s (`v4_hq_m1_scorer` H=1, `v4_hq_h2_scorer` H=2) and the registered M2b eval JSONs.

## The kink (what we asked)
We predict 5 numbers per contact as unstructured output channels — we never tell the model that the 5 depths are (a) ORDINAL or (b) geometrically meaningful (each is a known Δpose on the object). Should we encode this, and will it help reasoning on the depth axis (our flagged weak spot)?

## Current depth handling (verified in code)
- `EdgeCrossAttn` (`sage_learning/src/model/dit/edge_crossattn.py`): one edge vector per contact (B,60,D). A SINGLE shared head `Linear(D,D)→GELU→Linear(D, 5·51)` reads ALL 5 depths × 51 HL-Gauss bins from that one vector, reshaped to (B,60,5,51). Depth is PURELY an output-channel index.
- No ordinal/positional/geometric encoding of depth ANYWHERE — not on input, not on output. The only thing distinguishing d0 from d4 is the learned head weight rows. The 5 depths never "talk" beyond sharing the one edge vector + the head's first `Linear(D,D)`.
- HL-Gauss gives ordinal structure over the VALUE bins (Gaussian-smoothed 51-bin target), but NOT over depth.
- A depth-smoothing lever EXISTS but is OFF for current models: `classifier_module._build_soft_target` spreads label credit to adjacent depths via `soft_depth_sigma`, but the `hl_gauss` branch RETURNS before it is called (`_compute_masked_loss`), so no registered HL-Gauss model ever used depth smoothing.
- Precedent for per-query injection: `budget_embed` (an `nn.Embedding` added to every edge token) — the horizon input. Options (b)/(c) below are the per-DEPTH version of that pattern.

## Label structure over depth (verified, read-only)
**1-push (`v4_hq_m1_scorer` H=1; 40k rows, 416k positive edge-cells). f_grid is BINARY {0,1}.**
- Success-over-depth is **99.6% a single contiguous band**; **80.1% monotone non-decreasing** (0..01..1). Mean band length **3.29 / 5** (wide).
- Marginal success rate per depth d0..d4: **0.254 / 0.399 / 0.486 / 0.543 / 0.563** — monotone INCREASING (deeper opens more).
- `r_mask` is essentially contact-level: per-edge reachable-depth count is 0 (74%) or 5 (25%); only ~1% partial, of which 89% are shallow prefixes d0..dk (deep pushes blocked). So realistic≈oracle (matches the M2b JSON where they are equal).

**2-push (`v4_hq_h2_scorer`, H==2 rows; 40k rows, 86k positive edge-cells). f_grid ∈ {0, 0.9, 1.0} (γ=0.9).**
- **The depth profiles are OPPOSITE by success mode:**
  - DIRECT open (f=1) marginal d0..d4: **0.027 / 0.058 / 0.088 / 0.116 / 0.138** (INCREASING — deep opens directly).
  - CHAIN enable (f=0.9, the F₁′ setup pushes) marginal d0..d4: **0.106 / 0.069 / 0.047 / 0.029 / 0.019** (DECREASING — SHALLOW sets up chains).
- Pure-chain-enabler edges (best value 0.9, no direct opener at any depth): **37,745** vs 48,580 direct edges — chain-only edges are abundant.
- Their enabling band is **98.9% contiguous but mean length 1.38**, and **73% are a SINGLE depth** (band-length histogram 27703/6873/2182/695/292). Representative enabling depth concentrates shallow (d0 34%, d1 27%).

**Takeaway:** depth IS strongly ordinal (99% contiguous both horizons) — an ordinal prior is data-consistent. BUT the WINNING depth flips by mode: deep for direct-open, shallow for chain-enable; and chain depth is a NARROW, single-depth target, not the wide "iterate cheaply" band of 1-push.

## The model's actual depth behavior (M2b s3 eval JSON — current best 1-push model)
- On 1-push, depth is NOT the bottleneck: hard `failure_decomp@1` = success 29.6 / right_edge_wrong_depth **6.9** / wrong_edge **63.5** → **90% of misses are wrong-EDGE, only ~10% wrong-depth**. `depth_acc_given_right_edge` = 81.2% hard / 94.3% med (but "right" only means hitting the WIDE band).
- Its top-1 depth pick is **76.7% d4 (deepest)** on hard, 63% on med — a strong DEEP prior, learned from the abundant 1-push labels where deep wins (marginal 0.56 at d4). That prior is exactly WRONG for chain-enabling setups (which want shallow).
- Horizon-probe (`EXP-2026-07-09-horizon-role-probe`): told H=2, the Hz model demotes the deep direct-opener for a shallow setup (a "route knob"); its H=2 setup-ranking lifts best-first by up to **+13.5pp on 2push-hard** at budget ≥3. And at hmax=2 **even random reaches ~98% solve** → the whole 2-push game is **sims-to-solve (efficiency), not capability/ceiling**.

## Options (ranked, honest pros/cons)
1. **(c) GEOMETRIC GROUNDING of depth [RECOMMENDED FIRST].** Feed each (edge,depth) its known nominal Δpose (displacement magnitude + push-axis direction, Fourier-encoded) into a per-depth-conditioned head, so each depth is read from a geometry-conditioned representation instead of all-5-from-one-vector. PRO: injects real information the model currently lacks in usable form; directly attacks the measured opposite-profile miscalibration (small vs large displacement become geometrically separable); the depth-axis analog of `contact_px` (we already feed the action's "where"; this adds "how far"); cheap; subsumes option (d). CON: a small head restructure + a retrain; must feed NOMINAL (pre-sim) displacement only (see tension).
2. **(b) PER-DEPTH QUERY / depth embedding.** Expand the edge vector into 5 depth-conditioned tokens (learned `Embedding(5,D)`, budget_embed-style) or full 300 edge×depth tokens that attend to the scene. PRO: highest ceiling — lets each depth gather DIFFERENT scene evidence (shallow token → "is a nudge enough to set up," deep token → "does a big push clear it"), the direct cure for opposite profiles; the Hz result (horizon embedding switches depth profiles → helps search) is evidence this works. CON: learned-index-only variant is near-null (≈ current head capacity); full 300-token variant is 5× attention cost + higher variance. The GEOMETRIC version of (b) IS option (c) — so (c) is (b) done with real signal at low cost.
3. **(d) DEPTH AS CONTINUOUS FOURIER SCALAR vs categorical index.** PRO: smoothness/ordinality for free. CON: meaningless on its own (depth isn't even an INPUT today, only an output channel) — only matters once depth is a query/feature, i.e. it is a sub-choice inside (b)/(c), not a standalone lever.
4. **(a) ORDINAL / MONOTONE / UNIMODAL head or loss.** Monotone: REJECT — the global profile is non-monotone (opposite by mode). Unimodal / `soft_depth_sigma` depth smoothing: data-consistent (99% contiguous) and cheap (the code path exists, just disabled for hl_gauss), BUT the model's errors are wrong-EDGE not scattered-depth, and chain bands are single-depth so smoothing risks SMEARING the sharp signal; and a unimodal constraint teaches band SHAPE, not the peak LOCATION (which is the actual problem). Low EV; keep only as a secondary micro-ablation.

## Design-tension call (does grounding bake in a forward model?)
No. A forward model predicts the RESULTING state (new masks / new reachability / new object pose). Option (c) feeds only the primitive's OWN nominal displacement — a fixed pre-sim action parameter, exactly like `contact_px`. Bright line: feeding ACTION PARAMETERS (where + how far) = grounding ✓; feeding the SIMULATED post-pose = forward-model / verifier-leak ✗. Because `dynamic_direction=false` and depth = push-step count (`skill.max_push_steps=5`), the nominal Δ is deterministic and monotone along the fixed push axis; the exact per-(edge,depth) Δpose is stored in the motion-primitive DB (`data/motion_primitives_1x_car_*.dat`). The perfect verifier still computes true post-states during search. So (c) sharpens RANKING without turning the ranker into a forward model — as long as we feed the nominal displacement, never the simulated one.

## Will it help reasoning? (honest verdict)
- **1-push: NO measurable help expected.** Depth is not the 1-push bottleneck (wrong-edge is 90% of misses; the planner iterates the wide 3.3-depth band cheaply). Predict ±2pp.
- **2-push: this is where any payoff lives, and it is an EFFICIENCY payoff, not a ceiling payoff.** The model carries a deep prior that fights the shallow, single-depth chain target. Grounding the depth axis in geometry should let it rank the correct shallow enabling depth higher → best-first finds the chain in FEWER sims. Payoff shows in avg-sims-to-solve / solve@low-budget on 2push-hard, and in the offline chain-edge depth histogram shifting shallow — NOT in final solve rate (already ~ceiling at hmax=2).
- **Risk it's data-limited not bias-limited:** chain labels are only ~10% of reachable cells at H=2 and require post-state reasoning; if the limit is data, geometry helps only modestly. That is exactly what the A/B measures.

## Hypothesis (pre-registered)
Grounding each (edge,depth) in its known Δpose sharpens the first-push DEPTH ranking on chains (shifts it from the learned deep prior toward the shallow enabling depth), lowering sims-to-solve on 2push-hard, while leaving 1-push unchanged (depth isn't the 1-push bottleneck).

## The ONE A/B to run first
Add a `depth_geom` feature: per (edge,depth), the nominal displacement `(Δ‖ magnitude, cosθ, sinθ)` in the crop frame (from edge normal + depth via the primitive DB), Fourier-encoded, injected so the head reads each depth from `edge_vec ⊕ geomEmbed(edge,depth)` (shared head over 51 bins). Everything else identical to the NoHz/M2b recipe (same data, seeds, LR, HL-Gauss). Smoke 1 seed, then 3.
- **Primary metric:** best-first `eval_bestfirst` on `pure2push_HARD.json` — **avg-sims-to-solve** and **solve@{5,10,30}** (the horizon-probe axis).
- **Secondary (offline, cheap, no search):** on H=2 rows — **joint (edge,depth) hit@1 on pure-chain-enabler edges** and the **depth_top1_hist at chain edges**.
- **Guardrail:** 1-push `hard@1` (resolve_robust) — must stay within ±2pp.
- **Prediction:** avg-sims-to-solve on 2push-hard **↓ ≥15%**; solve@{5,10} **↑ ≥+3pp**; chain-edge top-1 depth histogram shifts from d4-dominant toward d0–d1; offline chain joint@1 **↑ ≥+5pp**; 1-push hard@1 within ±2pp.
- **Kill criterion (reject):** 1-push hard@1 drops **>2pp**, OR 2push-hard avg-sims fails to improve **≥10%**.

## Files / evidence
- Model / head: `sage_learning/src/model/dit/edge_crossattn.py` (head lines 125–126, 180–183; budget_embed precedent 103–104, 173–174).
- Loss (soft_depth_sigma disabled for hl_gauss): `sage_learning/src/model/classifier_module.py` (`_compute_masked_loss` 300–309, `_build_soft_target` 200–268).
- Data / semantics: `sage_learning/src/data/scorer_data.py`; H5s `/scratch/dm1487/h5/v4_hq_m1_scorer/data.h5`, `/scratch/dm1487/h5/v4_hq_h2_scorer/data.h5`.
- Geometry: `scripts/pipeline/add_contact_px.py` `contact_px()`; config `config/namo_config_complete_skill15_car_1x_d5.yaml` (`dynamic_direction=false`, `max_push_steps=5`).
- Model behavior: `/scratch/dm1487/eval/m2b_verdict/m2b_v4hq_s3__epoch013-val_loss0.6780.json`; `EXP-2026-07-09-horizon-role-probe.md`.
