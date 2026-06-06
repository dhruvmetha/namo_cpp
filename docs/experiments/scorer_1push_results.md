# 1-push SCORER (HACMan-critic / F-classifier) — v1 results

**What it is.** A discriminative per-push success scorer: `f(scene masks) → P(opens path)` for every
(edge, depth) primitive — the supervised version of HACMan's per-location Q-map (confirmed vs the real
HACMan/HACMan++ source). Revived from the existing sage_learning F-classifier; only changes were
car-d5 (60×5) + per-episode labels + room-grouped split (see `multi_episode_rooms.md`).

**Architecture.** DiT-classifier backbone (6.7M params), 5 scene masks (64×64) → 60×5 logits. Trained
supervised: BCE-on-all + Dice-on-reachable, **reachability-masked** (loss + eval only on reachable
primitives → reachability-safe by construction, structurally fixing what the diffusion model ignored).

**Data.** `v3_scorer_1push` — 27,931 episodes joined from the diffusion H5's correct car masks + the
per-episode `f_grid` (60×5, 1=valid/0=tried-fail/NaN=unreachable). Gates: gt_in_valid 100%, bad_match 0,
edge-align 0 errors. Room-grouped split (train 25,137 / val 2,794, 0 scenes straddle). Train = `v3_phase1`
pool, test = `test_feb_phase1` pool — **0 train/test room overlap** (verified).

**Training.** `base_lr=3e-4`, AdamW, 1 GPU, early-stopped at epoch 47 (best epoch 22, val_loss 0.864),
32 min. Plateaued early → recipe likely has headroom.

## Eval — exactly how (1-push)
Held out by room; binned by true difficulty (ratio = |valid|/|reachable|). For each test episode,
sigmoid the 60×5 logits, **rank the reachable candidates, top-k → hit if any is valid → success@k**.
Deterministic, exact (no sampling). Floor = reachability-aware random `1-(1-ratio)^k`. Same test
episodes as the diffusion baseline. (Oracle == realistic here: every reachable edge was tried at all 5
depths, so no depth-blocked "wasted" cells.)

## Results (success@k %, n = 413/491/752 hard/med/easy)
| | scorer @1 | scorer @20 | diffusion @1 | diffusion @20 | floor @1 | floor @20 |
|---|---|---|---|---|---|---|
| **hard** | **14.3** | **82.6** | 5.9 | 55.2 | 2.7 | 41.3 |
| **med** | **46.4** | 98.0 | 28.9 | 94.5 | 16.8 | 92.7 |
| **easy** | **84.2** | 100 | 64.6 | 99.9 | 65.4 | 100 |

- **Cleanest comparison = @1 (single best pick).** Hard: scorer 14.3 vs diffusion 5.9 vs floor 2.7 —
  **2.4× the diffusion, 5.3× the floor.** Wins @1 on every bin, decisively.
- **Coverage (@20) — judge against the *fair* floor.** Metric-calibration check: a random-score ranker
  reproduces the floor at @1 exactly (hard 2.9 vs analytic 2.7 ✓) but sits at **48.0** at @20, because
  ranking draws *distinct* candidates (without replacement) while the analytic floor `1-(1-sr)^k` assumes
  *with* replacement. So the honest @20 floor for the scorer is **48.0**, not 41.3. **Lift over the fair
  floor:** scorer **82.6 vs 48.0 = +34.6**, vs diffusion **55.2 vs 41.3 = +13.9** (diffusion samples, so
  its fair floor is the with-replacement 41.3). The scorer adds ~2.5× more coverage over its honest
  baseline. @1 remains the cleanest single number (both single-pick, no mechanism difference).

## Verdict
The discriminative scorer **beats both the floor and the generative diffusion baseline** on a clean
held-out set — the hypothesis holds. Reachability-safe by construction, deterministic, exact eval.

## Next levers (not yet done)
1. **10× data** — le10 is ~10% of the 211k-scene pool; generate masks for more scenes.
2. **Spatially-grounded head** — gather per-edge features at contact pixels (HACMan's per-point variant;
   the global-pool readout here is HACMan's weaker variant).
3. **Recipe** — plateaued at epoch 22; lr/capacity/longer-schedule has headroom.
4. **2-push = search over this scorer** (no new labels): roll push-1 forward, recompute reachability,
   score push-2 with the same net; validate against the existing 2-push winning chains.

_Ckpt: `…/scorer/scorer_1push_v1/namo-classifier/pwbgz10f/checkpoints/epoch022-val_loss0.8638.ckpt`.
Eval json: `/scratch/dm1487/eval_grounding/scorer_mid.json`. Scripts: `build_scorer_dataset.py`,
`scorer_data.py`, `train_scorer.slurm`, `eval_scorer.py`._
