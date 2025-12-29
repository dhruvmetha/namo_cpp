# 1-Push Evaluation Results

Generated from evaluation config.

## Dataset Overview

Total env+region pairs evaluated: **1015**

| Category | Count | Percentage |
|----------|-------|------------|
| Easy | 86 | 8.5% |
| Medium | 309 | 30.4% |
| Hard | 620 | 61.1% |

## Overall Success Rates

| Model | Successes | Total | Success Rate |
|-------|-----------|-------|--------------|
| No Heuristic | 1009 | 1015 | **99.4%** |
| Geometric Heuristic | 1012 | 1015 | **99.7%** |
| Diffusion 5 Steps | 936 | 1015 | **92.2%** |
| Diffusion 2 Steps | 792 | 1015 | **78.0%** |

## Success Rates by Category

| Model | Easy | Medium | Hard |
|-------|--------|--------|--------|
| No Heuristic | 100.0% (86/86) | 99.7% (308/309) | 99.2% (615/620) |
| Geometric Heuristic | 100.0% (86/86) | 99.7% (308/309) | 99.7% (618/620) |
| Diffusion 5 Steps | 100.0% (86/86) | 98.7% (305/309) | 87.9% (545/620) |
| Diffusion 2 Steps | 91.9% (79/86) | 95.1% (294/309) | 67.6% (419/620) |

## Pushes to Success (Successful Runs Only)

### Median Pushes

| Model | Easy | Medium | Hard |
|-------|--------|--------|--------|
| No Heuristic | 1.0 | 1.0 | 16.0 |
| Geometric Heuristic | 1.0 | 1.0 | 14.0 |
| Diffusion 5 Steps | 1.0 | 1.0 | 4.0 |
| Diffusion 2 Steps | 1.0 | 1.0 | 2.0 |

### Mean Pushes

| Model | Easy | Medium | Hard |
|-------|--------|--------|--------|
| No Heuristic | 1.5 | 4.7 | 24.8 |
| Geometric Heuristic | 1.1 | 3.4 | 23.2 |
| Diffusion 5 Steps | 1.1 | 1.7 | 5.9 |
| Diffusion 2 Steps | 1.0 | 1.5 | 2.9 |

## Time to Success in ms (Successful Runs Only)

### Median Time

| Model | Easy | Medium | Hard |
|-------|--------|--------|--------|
| No Heuristic | 75 | 103 | 1015 |
| Geometric Heuristic | 142 | 198 | 892 |
| Diffusion 5 Steps | 398 | 417 | 632 |
| Diffusion 2 Steps | 379 | 399 | 501 |

### Mean Time

| Model | Easy | Medium | Hard |
|-------|--------|--------|--------|
| No Heuristic | 112 | 328 | 1969 |
| Geometric Heuristic | 148 | 336 | 1864 |
| Diffusion 5 Steps | 469 | 475 | 785 |
| Diffusion 2 Steps | 449 | 446 | 554 |

## Interaction Statistics (Successful Runs Only)

These metrics show collision rates among successful runs.

### Wall Collision Rate

Percentage of successful runs that had collisions with walls.

| Model | Easy | Medium | Hard |
|-------|--------|--------|--------|
| No Heuristic | 8.1% (7/86) | 30.5% (94/308) | 45.0% (277/615) |
| Geometric Heuristic | 10.5% (9/86) | 29.9% (92/308) | 45.3% (280/618) |
| Diffusion 5 Steps | 0.0% (0/86) | 0.0% (0/305) | 0.0% (0/545) |
| Diffusion 2 Steps | 0.0% (0/79) | 0.0% (0/294) | 0.0% (0/419) |

### Movable Object Collision Rate

Percentage of successful runs that collided with other movable objects.

| Model | Easy | Medium | Hard |
|-------|--------|--------|--------|
| No Heuristic | 15.1% (13/86) | 13.0% (40/308) | 22.8% (140/615) |
| Geometric Heuristic | 15.1% (13/86) | 13.3% (41/308) | 23.3% (144/618) |
| Diffusion 5 Steps | 0.0% (0/86) | 0.0% (0/305) | 0.0% (0/545) |
| Diffusion 2 Steps | 0.0% (0/79) | 0.0% (0/294) | 0.0% (0/419) |

### Mean Movable Collisions

Average number of unique movable objects collided with per successful run.

| Model | Easy | Medium | Hard |
|-------|--------|--------|--------|
| No Heuristic | 0.15 | 0.14 | 0.24 |
| Geometric Heuristic | 0.15 | 0.14 | 0.25 |
| Diffusion 5 Steps | 0.00 | 0.00 | 0.00 |
| Diffusion 2 Steps | 0.00 | 0.00 | 0.00 |

## Detailed Per-Model Statistics

### No Heuristic

| Category | Success Rate | Med Pushes | Mean Pushes | Med Time | Mean Time | Wall Col | Mov Col |
|----------|--------------|------------|-------------|----------|-----------|----------|---------|
| Easy | 100.0% (86/86) | 1.0 | 1.5 | 75 | 112 | 8% | 15% |
| Medium | 99.7% (308/309) | 1.0 | 4.7 | 103 | 328 | 31% | 13% |
| Hard | 99.2% (615/620) | 16.0 | 24.8 | 1015 | 1969 | 45% | 23% |

**Failure Reasons:**

| Reason | Count |
|--------|-------|
| unknown | 2863 |

### Geometric Heuristic

| Category | Success Rate | Med Pushes | Mean Pushes | Med Time | Mean Time | Wall Col | Mov Col |
|----------|--------------|------------|-------------|----------|-----------|----------|---------|
| Easy | 100.0% (86/86) | 1.0 | 1.1 | 142 | 148 | 10% | 15% |
| Medium | 99.7% (308/309) | 1.0 | 3.4 | 198 | 336 | 30% | 13% |
| Hard | 99.7% (618/620) | 14.0 | 23.2 | 892 | 1864 | 45% | 23% |

**Failure Reasons:**

| Reason | Count |
|--------|-------|
| unknown | 2864 |

### Diffusion 5 Steps

| Category | Success Rate | Med Pushes | Mean Pushes | Med Time | Mean Time | Wall Col | Mov Col |
|----------|--------------|------------|-------------|----------|-----------|----------|---------|
| Easy | 100.0% (86/86) | 1.0 | 1.1 | 398 | 469 | 0% | 0% |
| Medium | 98.7% (305/309) | 1.0 | 1.7 | 417 | 475 | 0% | 0% |
| Hard | 87.9% (545/620) | 4.0 | 5.9 | 632 | 785 | 0% | 0% |

**Failure Reasons:**

| Reason | Count |
|--------|-------|
| success | 2394 |
| all_pushes_failed | 432 |
| already_accessible | 86 |
| no_reachable_objects | 26 |
| no_valid_goals | 9 |

### Diffusion 2 Steps

| Category | Success Rate | Med Pushes | Mean Pushes | Med Time | Mean Time | Wall Col | Mov Col |
|----------|--------------|------------|-------------|----------|-----------|----------|---------|
| Easy | 91.9% (79/86) | 1.0 | 1.0 | 379 | 449 | 0% | 0% |
| Medium | 95.1% (294/309) | 1.0 | 1.5 | 399 | 446 | 0% | 0% |
| Hard | 67.6% (419/620) | 2.0 | 2.9 | 501 | 554 | 0% | 0% |

**Failure Reasons:**

| Reason | Count |
|--------|-------|
| success | 795 |
| all_pushes_failed | 269 |
| no_valid_goals | 39 |
| already_accessible | 27 |
| no_reachable_objects | 8 |
| ml_goals_not_aligned | 2 |
