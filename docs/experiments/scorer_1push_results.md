# 1-push SCORER (HACMan-critic / F-classifier) — results

> **HEADLINE (2026-06-06, after the overnight HACMan-faithful build).** The per-edge cross-attention
> scorer **E2** is the winner: **hard 24.0 / med 70.7 / easy 94.9 @1** (hard @20 89.8) — **~1.7× the
> DiT baseline (E0, below), ~4× the diffusion, ~9× the random floor, and ~4× a calibrated geometric
> oracle.** A full lever sweep (capacity E2b, data E4, FOV oracle; resolution E3 pending) shows
> **hard@1 is a genuine ~24 plateau** — not fixable by capacity/data/FOV. Recommended next step:
> **2-push search using E2 as the value function** (the objective), not more 1-push tuning.
> Full reasoning + the falsified-FOV correction: **`scorer_hacman_journal.md`**.
>
> | model | hard@1 | med@1 | easy@1 | hard@20 | what it tests | verdict |
> |---|---|---|---|---|---|---|
> | E0 DiT (global readout) | 14.3 | 46.4 | 84.2 | 82.6 | baseline | — |
> | **E2 per-edge cross-attn** | **24.0** | **70.7** | **94.9** | **89.8** | per-point critic (HACMan) | **WIN, kept** |
> | E2b 2× capacity | 24.2 | 64.6 | 93.4 | — | capacity lever | flat → reject |
> | E4 3.6× data | 24.0 | 80.4† | 99.3† | 88.1 | data lever | hard flat → data not the lever |
> | E3-fine patch=2 | 25.2 | 69.7 | 94.5 | 88.6 | resolution lever | flat → resolution not the lever |
> | geometric oracle (hard) | ~6%‡ | | | ~40%‡ | rigid-geom baseline | model beats it ~4× |
>
> † E4's med/easy rose only because it added med/easy data; hard was already saturated (le10 = every
> ≤10% scene). ‡ oracle precision / est. @20-as-ranker — it's a weak baseline, NOT a ceiling.

---

## v1 baseline (E0 — global-readout DiT)

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

## Levers — status after the overnight sweep
1. ✅ **Spatially-grounded head (E2)** — per-edge tokens cross-attending the scene at contact pixels
   (HACMan's per-point critic). **+10 pts hard@1 (14.3→24.0). The big win.**
2. ✅ **Capacity (E2b)** — 2× params: flat (24.2). Not the lever.
3. ✅ **Data (E4)** — 3.6× scenes: hard flat (24.0); med/easy up only from added med/easy data. Not the
   hard lever (le10 already had all ≤10% scenes).
4. ✅ **Resolution (E3-fine, patch=2)** — finer 32×32 per-edge gather: flat (hard 25.2 ≈ 24, within
   noise). Not the lever.
5. ✅ **FOV (geometric oracle, tight vs wide)** — *rejected*: widening to 1.2 m removes goal-clipping but
   doesn't move hard recall/precision; and the model already beats the rigid oracle ~4× → oracle is a
   weak baseline, not a ceiling. Real but secondary: ~27% of hard goals are clipped at 0.5 m.
6. → **NEXT: 2-push = search over this scorer** (no new labels): roll push-1 forward, recompute
   reachability, score push-2 with the SAME net; validate against existing 2-push winning chains.
   This is where hard scenes (few valid single pushes) actually get solved — hard@20 89.8% means the
   right first push is almost always inside the top-20 a search would expand.
7. (optional 1-push gains, low priority given plateau) **dual-crop** (tight edges + wide context;
   pipeline built & gate-verified) addresses the ~27% goal-clipping; **continuous-duration depth actor
   (H4)** for the always-d4 depth collapse.

_Ckpt: `…/scorer/scorer_1push_v1/namo-classifier/pwbgz10f/checkpoints/epoch022-val_loss0.8638.ckpt`.
Eval json: `/scratch/dm1487/eval_grounding/scorer_mid.json`. Scripts: `build_scorer_dataset.py`,
`scorer_data.py`, `train_scorer.slurm`, `eval_scorer.py`._
