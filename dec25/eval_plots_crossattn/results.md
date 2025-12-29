# 1-Push Evaluation Results

Generated from evaluation config.

## Dataset Overview

Total env+region pairs evaluated: **399**

| Category | Count | Percentage |
|----------|-------|------------|
| Easy | 34 | 8.5% |
| Medium | 132 | 33.1% |
| Hard | 233 | 58.4% |

## Overall Success Rates

| Model | Successes | Total | Success Rate |
|-------|-----------|-------|--------------|
| No Heuristic | 398 | 399 | **99.7%** |
| Diffusion 5 Steps | 371 | 399 | **93.0%** |
| Diffusion 5 Steps Faster BFS | 365 | 399 | **91.5%** |
| Diffusion 5 Steps Faster BFS Hybrid | 394 | 399 | **98.7%** |

## Success Rates by Category

| Model | Easy | Medium | Hard |
|-------|--------|--------|--------|
| No Heuristic | 100.0% (34/34) | 99.2% (131/132) | 100.0% (233/233) |
| Diffusion 5 Steps | 100.0% (34/34) | 98.5% (130/132) | 88.8% (207/233) |
| Diffusion 5 Steps Faster BFS | 100.0% (34/34) | 100.0% (132/132) | 85.4% (199/233) |
| Diffusion 5 Steps Faster BFS Hybrid | 100.0% (34/34) | 99.2% (131/132) | 98.3% (229/233) |

## Pushes to Success (Successful Runs Only)

### Median Pushes

| Model | Easy | Medium | Hard |
|-------|--------|--------|--------|
| No Heuristic | 1.0 | 1.0 | 16.0 |
| Diffusion 5 Steps | 1.0 | 1.0 | 4.0 |
| Diffusion 5 Steps Faster BFS | 1.0 | 1.0 | 4.0 |
| Diffusion 5 Steps Faster BFS Hybrid | 1.0 | 1.0 | 14.0 |

### Mean Pushes

| Model | Easy | Medium | Hard |
|-------|--------|--------|--------|
| No Heuristic | 2.0 | 3.9 | 24.1 |
| Diffusion 5 Steps | 1.2 | 1.6 | 5.9 |
| Diffusion 5 Steps Faster BFS | 1.2 | 1.7 | 5.7 |
| Diffusion 5 Steps Faster BFS Hybrid | 1.0 | 1.9 | 20.6 |

## Time to Success in ms (Successful Runs Only)

### Median Time

| Model | Easy | Medium | Hard |
|-------|--------|--------|--------|
| No Heuristic | 74 | 94 | 1048 |
| Diffusion 5 Steps | 412 | 400 | 656 |
| Diffusion 5 Steps Faster BFS | 161 | 164 | 370 |
| Diffusion 5 Steps Faster BFS Hybrid | 155 | 165 | 980 |

### Mean Time

| Model | Easy | Medium | Hard |
|-------|--------|--------|--------|
| No Heuristic | 136 | 260 | 1908 |
| Diffusion 5 Steps | 485 | 467 | 828 |
| Diffusion 5 Steps Faster BFS | 172 | 213 | 548 |
| Diffusion 5 Steps Faster BFS Hybrid | 161 | 267 | 1930 |

## Interaction Statistics (Successful Runs Only)

These metrics show collision rates among successful runs.

### Wall Collision Rate

Percentage of successful runs that had collisions with walls.

| Model | Easy | Medium | Hard |
|-------|--------|--------|--------|
| No Heuristic | 2.9% (1/34) | 29.8% (39/131) | 45.5% (106/233) |
| Diffusion 5 Steps | 0.0% (0/34) | 0.0% (0/130) | 0.0% (0/207) |
| Diffusion 5 Steps Faster BFS | 0.0% (0/34) | 0.0% (0/132) | 0.0% (0/199) |
| Diffusion 5 Steps Faster BFS Hybrid | 0.0% (0/34) | 0.0% (0/131) | 0.0% (0/229) |

### Movable Object Collision Rate

Percentage of successful runs that collided with other movable objects.

| Model | Easy | Medium | Hard |
|-------|--------|--------|--------|
| No Heuristic | 8.8% (3/34) | 11.5% (15/131) | 22.7% (53/233) |
| Diffusion 5 Steps | 0.0% (0/34) | 0.0% (0/130) | 0.0% (0/207) |
| Diffusion 5 Steps Faster BFS | 0.0% (0/34) | 0.0% (0/132) | 0.0% (0/199) |
| Diffusion 5 Steps Faster BFS Hybrid | 0.0% (0/34) | 0.0% (0/131) | 0.0% (0/229) |

### Mean Movable Collisions

Average number of unique movable objects collided with per successful run.

| Model | Easy | Medium | Hard |
|-------|--------|--------|--------|
| No Heuristic | 0.09 | 0.13 | 0.24 |
| Diffusion 5 Steps | 0.00 | 0.00 | 0.00 |
| Diffusion 5 Steps Faster BFS | 0.00 | 0.00 | 0.00 |
| Diffusion 5 Steps Faster BFS Hybrid | 0.00 | 0.00 | 0.00 |

## Detailed Per-Model Statistics

### No Heuristic

| Category | Success Rate | Med Pushes | Mean Pushes | Med Time | Mean Time | Wall Col | Mov Col |
|----------|--------------|------------|-------------|----------|-----------|----------|---------|
| Easy | 100.0% (34/34) | 1.0 | 2.0 | 74 | 136 | 3% | 9% |
| Medium | 99.2% (131/132) | 1.0 | 3.9 | 94 | 260 | 30% | 11% |
| Hard | 100.0% (233/233) | 16.0 | 24.1 | 1048 | 1908 | 45% | 23% |

**Failure Reasons:**

| Reason | Count |
|--------|-------|
| unknown | 2863 |

### Diffusion 5 Steps

| Category | Success Rate | Med Pushes | Mean Pushes | Med Time | Mean Time | Wall Col | Mov Col |
|----------|--------------|------------|-------------|----------|-----------|----------|---------|
| Easy | 100.0% (34/34) | 1.0 | 1.2 | 412 | 485 | 0% | 0% |
| Medium | 98.5% (130/132) | 1.0 | 1.6 | 400 | 467 | 0% | 0% |
| Hard | 88.8% (207/233) | 4.0 | 5.9 | 656 | 828 | 0% | 0% |

**Failure Reasons:**

| Reason | Count |
|--------|-------|
| success | 2394 |
| all_pushes_failed | 432 |
| already_accessible | 86 |
| no_reachable_objects | 26 |
| no_valid_goals | 9 |

### Diffusion 5 Steps Faster BFS

| Category | Success Rate | Med Pushes | Mean Pushes | Med Time | Mean Time | Wall Col | Mov Col |
|----------|--------------|------------|-------------|----------|-----------|----------|---------|
| Easy | 100.0% (34/34) | 1.0 | 1.2 | 161 | 172 | 0% | 0% |
| Medium | 100.0% (132/132) | 1.0 | 1.7 | 164 | 213 | 0% | 0% |
| Hard | 85.4% (199/233) | 4.0 | 5.7 | 370 | 548 | 0% | 0% |

**Failure Reasons:**

| Reason | Count |
|--------|-------|
| success | 2383 |
| all_pushes_failed | 445 |
| already_accessible | 83 |
| no_reachable_objects | 26 |
| no_valid_goals | 10 |

### Diffusion 5 Steps Faster BFS Hybrid

| Category | Success Rate | Med Pushes | Mean Pushes | Med Time | Mean Time | Wall Col | Mov Col |
|----------|--------------|------------|-------------|----------|-----------|----------|---------|
| Easy | 100.0% (34/34) | 1.0 | 1.0 | 155 | 161 | 0% | 0% |
| Medium | 99.2% (131/132) | 1.0 | 1.9 | 165 | 267 | 0% | 0% |
| Hard | 98.3% (229/233) | 14.0 | 20.6 | 980 | 1930 | 0% | 0% |

**Failure Reasons:**

| Reason | Count |
|--------|-------|
| success | 398 |
| all_pushes_failed | 32 |
| already_accessible | 15 |
| no_reachable_objects | 2 |
