# Feasibility eval — corrected (per-episode, true-difficulty bins)

> **Superseded the old numbers below the line.** The original tables compared early checkpoints
> (informative_le10 **ep66** vs baseline_random **ep19**) under per-*room* matching — the multi-episode
> contamination (the `GT∈valid ~75%` you see in the archived section is its fingerprint). These
> **corrected** tables use per-*episode* matching + re-binning by true solve_rate (see
> [multi_episode_rooms.md](../pipeline/multi_episode_rooms.md)) on the models we actually carry forward.

**Method:** reachable-masked render-match, K=20 samples / 50 denoise steps. Each sample matched to its
own episode by `object_center` (gt_in_valid = 1.0, bad_match = 0); binned by that episode's solve_rate.
Clean test set: **413 hard / 491 med / 752 easy**. Numbers are **% success** (s@1 / s@5 / s@10 / s@20).
Models: **informative ep500** (`…/informative_le10_500ep_2gpu/…/periodic-epoch499.ckpt`), **annealing
ep400**, **general** (`…/v3_safe_fp32_…/epoch085-val_loss0.0016.ckpt`). Floor = reachability-aware random
(analytic 1−(1−sr)^k).

## hard  (n=413, sr_mean=2.75%)
| sampler | s@1 | s@5 | s@10 | s@20 | E[tries] (cov) |
|---|---|---|---|---|---|
| random floor | 2.7 | 12.9 | 23.9 | 41.3 | 45.9 (100%) |
| **informative ep500** | **5.9** | 22.2 | 33.7 | **55.2** | 12.8 (55%) |
| annealing ep400 | 5.4 | 20.8 | 31.9 | 54.7 | 13.7 (55%) |
| general | 3.1 | 12.4 | 20.0 | 39.7 | 16.3 (40%) |

## med  (n=491, sr_mean=16.8%)
| sampler | s@1 | s@5 | s@10 | s@20 | E[tries] (cov) |
|---|---|---|---|---|---|
| random floor | 16.8 | 57.2 | 78.6 | 92.7 | 7.5 (100%) |
| **informative ep500** | **28.9** | 68.6 | 81.9 | 94.5 | 5.6 (95%) |
| annealing ep400 | 29.5 | 69.7 | 83.1 | 95.5 | 5.5 (95%) |
| general | 19.9 | 57.2 | 73.5 | 91.6 | 7.5 (92%) |

## easy  (n=752, sr_mean=65.4%)
| sampler | s@1 | s@5 | s@10 | s@20 | E[tries] (cov) |
|---|---|---|---|---|---|
| random floor | 65.4 | 97.5 | 99.8 | 100.0 | 1.7 (100%) |
| informative ep500 | 64.6 | 95.1 | 98.4 | 99.9 | 2.0 (100%) |
| annealing ep400 | 62.3 | 95.0 | 98.5 | 99.9 | 2.0 (100%) |
| general | 63.4 | 95.2 | 98.7 | 99.7 | 1.9 (100%) |

**Takeaways:** specialist beats the (reachability-aware) floor on hard — 5.9 vs 2.7 @1 (~2.2×),
55.2 vs 41.3 @20. **General sits at the floor on hard** (3.1 / 39.7) → the *informative training* is what
teaches the hard-scene skill. Annealing ≈ no-anneal. EASY is a tie (random-reachable already wins).
**Coverage is the ceiling** — only ~55% of hard scenes crack within 20 samples.

Source JSONs: `/scratch/dm1487/eval_grounding/{informative_ep500,annealing_ep400,older_safe_fp32}_rebin.json`.

---

## ARCHIVED — early ep66/ep19 run, per-ROOM matching (CONTAMINATED, do not cite)
K=20 samples, 20 denoise steps, snap gate 0.2 m / 0.2 rad.
informative ckpt: `…/informative_le10/2026-06-04/23-30-23/checkpoints/epoch066-val_loss0.0037.ckpt`
baseline ckpt:    `…/baseline_random/2026-06-04/23-34-49/checkpoints/epoch019-val_loss0.0038.ckpt`
The `GT∈valid ~75%` per division is the contamination fingerprint; hard s@1 0.063 was inflated.

| division | sampler | s@1 | s@5 | s@10 | s@20 |
|---|---|---|---|---|---|
| hard (n=540, GT∈valid 76%) | floor / baseline / informative | 0.027 / 0.010 / 0.063 | 0.127 / 0.043 / 0.224 | 0.236 / 0.072 / 0.335 | 0.408 / 0.109 / 0.446 |
| med (n=547, GT∈valid 75%) | floor / baseline / informative | 0.172 / 0.020 / 0.099 | 0.581 / 0.087 / 0.334 | 0.794 / 0.146 / 0.470 | 0.932 / 0.216 / 0.584 |
| easy (n=591, GT∈valid 74%) | floor / baseline / informative | 0.655 / 0.034 / 0.115 | 0.973 / 0.144 / 0.393 | 0.997 / 0.236 / 0.563 | 1.000 / 0.341 / 0.711 |
