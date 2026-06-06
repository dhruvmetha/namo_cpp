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
- result: PENDING (training).

---

## Reading list (for the user on waking)
- (curated below as I go — papers grounding the per-point critic / query-token design)
