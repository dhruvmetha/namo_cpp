# Scorer → HACMan-faithful: experiment journal

**Owner:** autonomous overnight session (started 2026-06-06, user asleep ~8h).
**Objective (do not drift):** a 1-push *push-success scorer* — given a scene, score every (edge, depth)
primitive by P(opens a path to the goal) — that (a) beats the diffusion baseline and the honest floor,
and (b) is built so the SAME net becomes the value function for 2-push search later. We are NOT
collecting exhaustive 2-push labels; multi-push is search over this scorer.

**Method — every decision is a hypothesis with a test.** Format below. A change is KEPT only if its
test accepts; otherwise reverted/iterated. Bars: held-out **by room**, binned by **true difficulty**,
honest **without-replacement** floor, success@k + the failure decomposition (wrong-edge% / wrong-depth%
/ rank-of-first-valid). Primary metric to move: **hard @1** and **hard wrong-edge%**.

---

## Baseline (E0) — global-readout DiT classifier  [DONE, established]
- Arch: DiT over 5×64×64 crop → CLS token → MLP → 60×5. 6.7M params. Supervised masked BCE+Dice.
- Result (hard / med / easy, success@1): **14.3 / 46.4 / 84.2**;  hard @20 = 82.6 (vs floor 47).
- **Diagnosis (the bottleneck):** top-1 errors are **96.6% wrong-EDGE**, only 3.4% wrong-depth; first
  valid push sits at **median rank 7/75**; depth collapsed to **always-d4 (98%)**. Score separation
  weak (+0.12, positive in only 63% of scenes). Viz: a smooth spatial *gradient* across edges.
- **Conclusion:** the failure is **edge-selection precision**, and its cause is the **global readout**
  (one scene vector forced to emit 60 scores → can only express "which side," not "which exact edge").
  Depth is NOT the bottleneck (it solved it via a max-push prior). → fix edge precision first.

---

## Hypotheses & experiment plan

**H1 (E2 — per-edge tokens + cross-attention):** giving each edge its OWN token (positional id of its
contact location + a feature gathered from the scene) that cross-attends to the scene will let each edge
reason independently → **wrong-edge% drops, hard @1 rises** above 14.3. *This is HACMan's per-point
critic, on our 60-edge "point cloud."*  TEST: train, eval; accept if hard @1 ↑ and wrong-edge% ↓ vs E0.

**H2 (E3 — + zoomed object crop):** if E2 helps but per-edge scores are still fuzzy (resolution-bound),
adding a sharp zoomed object crop (object-bbox sub-crop, resized) as the source of each edge's *local*
feature will further sharpen edge precision. TEST: vs E2. (Only run if E2's residual looks resolution-limited.)

**H3 (E4 — more data, 10× pool):** le10 is ~10% of the 211k-scene pool; the weak score-separation /
"lost" ~17% may be data-limited. Regen masks for more scenes → hard @1 ↑. TEST: vs best-so-far. (Expensive;
gated on E2/E3 showing the arch is right but data-hungry.)

**H4 (E5 — continuous-duration actor, the HACMan "how"):** depth collapsed to always-d4, which
structurally misses the ~36% of valids needing a short push (the d0/d1 cases). A per-edge continuous
duration (actor maximizing the critic) should recover those. TEST: wrong-depth% / those specific cases.
*Lower priority — depth is only 3.4% of errors; do after edge precision is solved.*

**Critical guardrails:** (1) always compare on the SAME held-out test episodes as E0/diffusion;
(2) re-check the honest (without-replacement) floor each time; (3) watch for new degenerate shortcuts;
(4) leak-check any new data (0 train/test room overlap); (5) if a hypothesis is rejected, say so plainly
and record why — negative results are results.

---

## Results log  (filled in as experiments complete)

### E2 — per-edge cross-attention critic   [TRAINING, job 55581167, started 2026-06-06 ~01:57]
- **Hypothesis (H1):** the global CLS readout causes the 96.6% wrong-edge failure; per-edge tokens that
  reason independently will drop wrong-edge% and raise hard @1 above 14.3.
- **Design (`src/model/dit/edge_crossattn.py`, 4.3M params):** DiT patch-embed (4×4) → 16×16 scene
  tokens → 4 self-attn blocks (scene context). 60 **edge tokens** = `grid_sample` of the scene feature
  map at each edge's contact pixel (local feature) **+** an MLP positional id of its (x,y) contact
  coord. 4 **cross-blocks**: edge tokens cross-attend to scene + self-attend among edges (point-
  transformer). Shared per-edge MLP head → 60×5. = HACMan's per-point critic on our 60-edge cloud.
- **Data:** same `v3_scorer_1push` H5 + new `contact_px` (60,2) added via `add_contact_px.py` (pose
  math, in-bounds 100%). Same room-grouped split, same masked BCE+Dice loss. *Only the readout changed*
  vs E0 — so a win is attributable to per-edge reasoning, not data/loss.
- **Test:** eval on the SAME test episodes → compare hard @1 + wrong-edge% + grid viz vs E0
  (14.3 @1, 96.6% wrong-edge). ACCEPT if @1 ↑ and wrong-edge ↓.
- **RESULT — H1 ACCEPTED (strongly).** Mid-training read (epoch 41, val_loss 0.60 vs E0's 0.86 plateau;
  still improving, so this is a *lower bound*):
  - success@1: hard **14.3 → 26.6**, med 46.4 → 71.1, easy 84.2 → 96.1.  (vs diffusion 5.9/28.9/64.6.)
  - first-valid **median rank 7 → 3**; rank≤3 29% → 51%; score separation **+0.12 → +0.38** (3× sharper).
  - wrong-edge% 96.6 → **90.2** (still the dominant residual error, but the model is far better *at* it).
  - depth still ~always-d4 (85%); wrong-depth rose 3.4 → 9.8% (relatively larger now that edges are better).
  - "lost core": positive-separation scenes still **64%** (unchanged from E0) → ~36% of hard scenes have
    NO signal regardless of arch → smells **data-limited**, not arch-limited.
- **FINAL (converged epoch045, val_loss 0.594):** hard **24.0** / med 70.7 / easy 94.9 @1; hard @20 89.8.
  NB: the best-val_loss ckpt's hard @1 (24.0) is slightly *below* the epoch-41 read (26.6) — val_loss is
  all-difficulty, hard@1 is a noisy n=413 slice. So E2 ≈ **24–26 @1 on hard** (~1.7× E0, ~4× diffusion,
  ~9× floor). Takeaway: **E2 has ~maxed out on le10 data** (converged, hard@1 flat mid→final).
- **Decision:** keep E2 as the backbone. The flat mid→final on hard@1 says the *architecture* converged
  on this data → test the two orthogonal levers: **E2b (more capacity, same data)** and **E4 (same arch,
  3.6× data)**. Both LAUNCHED. E3 (zoom) held unless these stall on the edge-precision residual.

### E4 — more data (the data lever for the lost core)   [DATA-GEN RUNNING, 2026-06-06 ~02:40]
- **Hypothesis (H3):** E2 left the ~36% no-signal "lost core" of hard scenes unchanged → it's
  data-limited, not arch-limited. ~3.6× more (hard-inclusive) data should raise positive-separation% and
  hard @1/coverage. *Uses the E2 architecture — isolates the DATA variable.*
- **Data:** 80k random pkls from the full 211k solvable TRAIN pool (le10 was 22k). mask-gen array
  `55584295` (27 shards, reusing the tested `run_batch_collection_smoke`, extra= empty = same config as
  le10) + per-episode labels `55584296`. Then build H5 (--tight-only) → join (build_scorer_dataset) →
  add_contact_px → train E2-arch. ~75-100k episodes expected (~3.5× le10). Leak-check vs test before training.
- **Test:** train E2-arch on E4 data, eval vs E2-on-le10. ACCEPT if positive-separation% ↑ and hard @1 ↑.
- **⚠️ CRITICAL REALIZATION (logged before training, 2026-06-06 ~04:40):** E4's difficulty mix is
  **4% hard / 21% med / 76% easy** (the pool's natural distribution), vs le10's hard-enriched 34/46/19.
  Because **le10 = all pkls that have a ≤10% episode, le10 already contains EVERY hard training scene** —
  random sampling can only add easy/med, NOT hard. So E4 has *fewer* hard episodes than le10 (~4k vs ~9.5k).
  → E4 actually tests "more *total/diverse* data (with the easy-skew)", NOT "more hard data". And it implies
  the lost core **cannot** be fixed by more data — *we already have all the hard data* → the lost core is
  most likely **feature-limited** (the 5 masks don't determine which push opens the path; needs richer
  features like the wavefront distance field, which this H5 lacks). Still running E4 to see if diversity
  helps or the skew hurts — but the strong prior is now: data is NOT the lost-core lever; **features are.**
- E4 data: 98,387 episodes, 79,626 rooms, contact_px ✓, gt_in_valid 100%, 0 test leakage. job 55588491.
- result: PENDING (training).

### E2b — scale the winner (capacity lever)   [DONE — H REJECTED]
- **Hypothesis:** E2 (4.3M) capacity/training-limited; a bigger edge model (dim 256, depth 6/6, 2× params)
  improves further.
- **RESULT — REJECTED.** E2b val_loss converged *worse* (0.636 vs E2's 0.594), and on the metric it is
  **no better: hard @1 24.2 vs 24.0** (flat), med 64.6 < 70.7, easy 93.4 < 94.9. The bigger model does not
  help — likely under-trained at the lower lr, but regardless, **E2 (4.3M) is already the right size.**
- **Conclusion:** the limit is NOT capacity. With the lost core also unchanged E0→E2 (65% positive-sep),
  the evidence converges on **data** as the lever → E4 is the decisive test. (Negative result = result.)

---

## Reading list (for the user on waking)
Grounding for the per-edge / point-critic architecture we built:
- **HACMan** (Zhou et al., CoRL 2023) — the per-point Q-map + actor-critic we replicate.
  https://arxiv.org/abs/2305.03942 · code: github.com/HACMan-2023/HACMan (cloned at /scratch/dm1487/refs/HACMan).
  Read §"per-point feature extractor" + the PointNet++ U-Net (SA encoder + FP decoder w/ skips) — that's
  exactly what our edge-token cross-attention does, on a 60-edge cloud.
- **HACMan++** (Jiang et al., RSS 2024) — spatially-grounded *parameterized* primitives (what/where/how),
  the chaining we'll use for 2-push. https://arxiv.org/abs/2407.08585 · code: github.com/JiangBowen0008/HACManPP.
  Note it uses a **point_transformer** over the points — validates our transformer-over-edges choice.
- **Point Transformer** (Zhao et al., ICCV 2021) — local self-attention → per-point features fusing
  local+global. https://arxiv.org/abs/2012.09164 . This is the backbone idea of our edge self-attention.

**My recommendation for what to read first:** the HACMan per-point feature-extractor + PointNet++ section
— it makes the "each edge gets its own local+global summary" idea concrete and shows why the global
readout (our E0) was the bottleneck.

## Key open question for the lost core (see E4 realization)
We have ALL the hard training data already (le10 = every ≤10% scene). So the ~35% "lost core" of hard
scenes is most likely **feature-limited**, not data-limited. The decisive next measurement is the
**geometric+wavefront oracle** (render each push's footprint → recompute reachability → does the goal
open?). If the oracle solves the lost-core scenes, the info is in the geometry → add wavefront/distance-
field features (this H5 lacks them — only a binary robot-region blob). If the oracle can't, those scenes
need multi-push. This is the highest-value next experiment after the current batch.
