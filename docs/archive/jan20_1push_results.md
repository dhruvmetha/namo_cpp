# 1-Push Evaluation Results

Generated from evaluation config.

Difficulty thresholds (data-driven percentiles): Easy ≤ 4 pushes, Medium ≤ 14 pushes, Hard > 14 pushes

## Dataset Overview

Total env+region pairs evaluated: **1534**

| Category | Count | Percentage |
|----------|-------|------------|
| Easy | 529 | 34.5% |
| Medium | 508 | 33.1% |
| Hard | 497 | 32.4% |

## Overall Success Rates

| Model | Successes | Total | Success Rate |
|-------|-----------|-------|--------------|
| No Heuristic | 1699 | 1700 | **99.9%** |
| Diffusion Hybrid Voting5 | 1700 | 1700 | **100.0%** |

## Success Rates by Category

| Model | Easy | Medium | Hard |
|-------|--------|--------|--------|
| No Heuristic | 100.0% (529/529) | 99.9% (673/674) | 100.0% (497/497) |
| Diffusion Hybrid Voting5 | 100.0% (529/529) | 100.0% (674/674) | 100.0% (497/497) |

## Pushes to Success (Successful Runs Only)

Format: median [IQR]

| Model | Easy | Medium | Hard |
|-------|--------|--------|--------|
| No Heuristic | 1 [1, 8] | 4 [1, 14] | 28 [16, 45] |
| Diffusion Hybrid Voting5 | 1 [1, 1] | 1 [1, 5] | 8 [2, 19] |

## Time to Success in seconds (Successful Runs Only)

Format: median [IQR]

| Model | Easy | Medium | Hard |
|-------|--------|--------|--------|
| No Heuristic | 0.1 [0.1, 0.5] | 0.3 [0.1, 0.9] | 2.1 [1.0, 4.0] |
| Diffusion Hybrid Voting5 | 0.2 [0.2, 0.2] | 0.2 [0.2, 0.5] | 0.8 [0.3, 2.1] |

## Interaction Statistics (Successful Runs Only)

*Note: Statistics computed over successful runs only. Models with lower success rates may show different interaction patterns due to selection bias (failing on harder instances).*

### Wall Collision Rate

Percentage of successful runs that had collisions with walls.

| Model | Easy | Medium | Hard |
|-------|--------|--------|--------|
| No Heuristic | 23.3% (123/529) | 35.7% (240/673) | 45.5% (226/497) |
| Diffusion Hybrid Voting5 | 23.4% (124/529) | 35.9% (242/674) | 45.1% (224/497) |

### Movable Object Collision Rate

Percentage of successful runs that collided with other movable objects.

| Model | Easy | Medium | Hard |
|-------|--------|--------|--------|
| No Heuristic | 14.6% (77/529) | 20.8% (140/673) | 31.6% (157/497) |
| Diffusion Hybrid Voting5 | 15.9% (84/529) | 24.3% (164/674) | 33.4% (166/497) |

### Mean Movable Collisions

Average number of unique movable objects collided with per successful run.

| Model | Easy | Medium | Hard |
|-------|--------|--------|--------|
| No Heuristic | 0.15 | 0.22 | 0.34 |
| Diffusion Hybrid Voting5 | 0.16 | 0.26 | 0.36 |

## Success Rate by Collision Type

Collision type determined by oracle (search) solution.

| Model | No Collision | Wall Only | Movable Only | Both |
|-------|-------------|-------------|-------------|-------------|
| No Heuristic | 100.0% (734/734) | 100.0% (561/561) | 100.0% (361/361) | 97.7% (43/44) |
| Diffusion Hybrid Voting5 | 100.0% (734/734) | 100.0% (561/561) | 100.0% (361/361) | 100.0% (44/44) |

### Efficiency by Collision Type (Solved Cases Only)

*Note: Efficiency numbers are computed over solved cases only; models with lower success rates may appear more efficient due to selection bias (e.g., only succeeding on easier instances).*

| Model | Collision Type | N | Median Checks | Median Time (s) |
|-------|----------------|---|---------------|-----------------|
| No Heuristic | No Collision | 734 | 1 | 0.1 |
| No Heuristic | Wall Only | 561 | 13 | 0.8 |
| No Heuristic | Movable Only | 361 | 12 | 0.9 |
| No Heuristic | Both | 43 | 24 | 1.7 |
| Diffusion Hybrid Voting5 | No Collision | 734 | 1 | 0.2 |
| Diffusion Hybrid Voting5 | Wall Only | 561 | 2 | 0.3 |
| Diffusion Hybrid Voting5 | Movable Only | 361 | 2 | 0.3 |
| Diffusion Hybrid Voting5 | Both | 44 | 11 | 1.1 |

## Difficulty Stratification (by Oracle Push Counts)

*Problems split into thirds by oracle push counts: Easy (fewest 33%), Medium (middle 33%), Hard (most 33%).*

Oracle push ranges: **Easy**: 1–65 pushes, **Medium**: 1–157 pushes, **Hard**: 1–212 pushes

### Success Rate by Difficulty

| Model | Easy | Medium | Hard |
|-------|------------|------------|------------|
| No Heuristic | 100.0% (529/529) | 99.9% (673/674) | 100.0% (497/497) |
| Diffusion Hybrid Voting5 | 100.0% (529/529) | 100.0% (674/674) | 100.0% (497/497) |

### Efficiency by Difficulty (Solved Cases Only)

*Note: Efficiency computed over solved cases only; selection bias may apply.*

| Model | Difficulty | N | Median Checks | Median Time (s) |
|-------|------------|---|---------------|-----------------|
| No Heuristic | Easy | 529 | 1 | 0.1 |
| No Heuristic | Medium | 673 | 4 | 0.3 |
| No Heuristic | Hard | 497 | 28 | 2.1 |
| Diffusion Hybrid Voting5 | Easy | 529 | 1 | 0.2 |
| Diffusion Hybrid Voting5 | Medium | 674 | 1 | 0.2 |
| Diffusion Hybrid Voting5 | Hard | 497 | 8 | 0.8 |

## Hybrid Decomposition (Learned vs Fallback)

*Phase tracking: solved_in_phase='ML-only' → LEARNED, 'primitives' → FALLBACK*

| Model | Total | Learned | Fallback | Failed | Success Rate |
|-------|-------|---------|----------|--------|--------------|
| Diffusion Hybrid Voting5 | 1700 | 1617 (95.1%) | 83 (4.9%) | 0 | 100.0% |

### Hybrid Decomposition by Difficulty

| Model | Difficulty | N | Learned | Fallback | Failed |
|-------|------------|---|---------|----------|--------|
| Diffusion Hybrid Voting5 | Easy | 529 | 99.8% (528) | 0.2% (1) | 0.0% (0) |
| Diffusion Hybrid Voting5 | Medium | 674 | 95.4% (643) | 4.6% (31) | 0.0% (0) |
| Diffusion Hybrid Voting5 | Hard | 497 | 89.7% (446) | 10.3% (51) | 0.0% (0) |

## Reachable Attachment @ K (RA@K)

*Fraction of top-K ML-ranked primitives with reachable push attachments.*

| Model | RA@10 | RA@50 | RA@100 | RA@All | Random |
|-------|--------|--------|--------|--------|--------|
| Diffusion Hybrid Voting5 | 86.2% | 72.5% | 71.4% | 71.4% | 71.4% |

## Success @ Budget

*Success rate at fixed verification budget (constant-compute comparison).*

| Model | @5 | @10 | @20 |
|-------|--------|--------|--------|
| No Heuristic | 46.1% | 56.4% | 75.7% |
| Diffusion Hybrid Voting5 | 71.4% | 81.9% | 89.8% |

## Success @ Time

*Success rate at fixed time budget (constant-time comparison).*

| Model | @1s | @3s | @6s |
|-------|---------|---------|---------|
| No Heuristic | 66.1% | 87.7% | 94.4% |
| Diffusion Hybrid Voting5 | 80.6% | 92.4% | 96.4% |

## Detailed Per-Model Statistics

### No Heuristic

| Category | Success Rate | Med Pushes | Mean Pushes | Med Time | Mean Time | Wall Col | Mov Col |
|----------|--------------|------------|-------------|----------|-----------|----------|---------|
| Easy | 100.0% (529/529) | 1.0 | 4.7 | 85 | 333 | 23% | 15% |
| Medium | 99.9% (673/674) | 4.0 | 9.9 | 283 | 931 | 36% | 21% |
| Hard | 100.0% (497/497) | 28.0 | 32.2 | 2056 | 3143 | 45% | 32% |

**Failure Reasons:**

| Reason | Count |
|--------|-------|
| success | 2026 |
| all_pushes_failed | 157 |

### Diffusion Hybrid Voting5

| Category | Success Rate | Med Pushes | Mean Pushes | Med Time | Mean Time | Wall Col | Mov Col |
|----------|--------------|------------|-------------|----------|-----------|----------|---------|
| Easy | 100.0% (529/529) | 1.0 | 2.0 | 173 | 260 | 23% | 16% |
| Medium | 100.0% (674/674) | 1.0 | 6.1 | 187 | 795 | 36% | 24% |
| Hard | 100.0% (497/497) | 8.0 | 16.4 | 764 | 2021 | 45% | 33% |

**Failure Reasons:**

| Reason | Count |
|--------|-------|
| success | 1912 |
| all_pushes_failed | 112 |
