---
status: snapshot
tags: [experiment]
updated: 2026-06-06
---

# Informative 1-push — 500-epoch + annealing training study

Follow-up to the informative-vs-baseline comparison. Trained the informative (≤10% solve-rate,
27,931 NPZs) goal-diffusion model to 500 epochs, two configs, and evaluated the grounding
breakdown every 100 epochs on the held-out test divisions (render-match decode; GT round-trip
validates it 95/91/78%).

## Setup (both runs: 2× GPU DDP, eff batch 256, no-SNR, bf16, warmup 1000)
- **Run A** `informative_le10_500ep_2gpu` — lr 2e-4, decay_steps 300k (LR barely anneals over run)
- **Run B** `informative_le10_500ep_2gpu_anneal` — lr 2e-4, decay_steps 53.5k (LR anneals to end_lr by ep500)

⚠️ Divergence cause was **base_lr=8e-4** (8× too hot), NOT min-SNR — see [[project_minsnr_divergence]].
Both runs here use lr 2e-4 and are stable (val → ~0.002).

## Result — hard success@20 (best-of-20 feasibility, render-match)

| epoch | Run A (no-anneal) | Run B (anneal) |
|---|---|---|
| 100 | 29.8 | 33.7 |
| 200 | 35.0 | 32.4 |
| 300 | 33.5 | 31.1 |
| 400 | 35.7 | **39.3** |
| 500 | 37.8 | 39.3 |

- **Best: Run B 39.3% @ep400** (hard); Run A 37.8% @ep500. (med ~74%, easy ~88% both.)
- **Annealing helps modestly and late** — A≈B through ep300, B pulls +1.5–3.6 pts at ep400–500
  (LR has annealed by then). Reaches ceiling faster. The annealing concern was valid but small.
- **val_loss plateau ≠ capability plateau**: val flattened ~0.002 while feasibility kept climbing
  to ~ep200–400 — exactly why we eval on the feasibility metric, not val_loss.
- **Mask cleanliness improves with training**: as-deployed `undecodable` (>2-blob speckle) fell
  ~87%→~21% from ep100→200, so the brittle rectangle decode unblocks on its own.

## The ceiling is model/data-limited (~38–39% hard best-of-20)

Failure decomposition at the best checkpoint (both runs nearly identical):
- **wrong-edge (reachable): ~54%** ← dominant — picks a reachable but non-solving edge
- **not-reachable (ungrounded): ~36%** ← predicts edges the robot can't reach
- right-edge-wrong-depth: ~6% ← depth is NOT the bottleneck
- success (per-sample): ~3.5%

**To push past ~39% hard, the levers are reachability conditioning + better edge selection
(architecture/data), not more epochs, not annealing, not depth precision.**

## Best checkpoint for downstream use
`v3_1push_le10/informative_le10_500ep_2gpu_anneal/2026-06-05/03-59-01/checkpoints/periodic-epoch399.ckpt`
(Run B ep400, 39.3% hard best-of-20).
