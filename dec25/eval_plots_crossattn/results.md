# 1-Push Evaluation Results

Generated from evaluation config.

## Dataset Overview

Total env+region pairs evaluated: **3494**

| Category | Count | Percentage |
|----------|-------|------------|
| Easy | 247 | 7.1% |
| Medium | 1123 | 32.1% |
| Hard | 2124 | 60.8% |

## Overall Success Rates

| Model | Successes | Total | Success Rate |
|-------|-----------|-------|--------------|
| No Heuristic | 3492 | 3494 | **99.9%** |
| Diffusion | 3121 | 3494 | **89.3%** |
| Diffusion Hybrid | 3491 | 3494 | **99.9%** |

## Success Rates by Category

| Model | Easy | Medium | Hard |
|-------|--------|--------|--------|
| No Heuristic | 100.0% (247/247) | 100.0% (1123/1123) | 99.9% (2122/2124) |
| Diffusion | 99.2% (245/247) | 98.0% (1100/1123) | 83.6% (1776/2124) |
| Diffusion Hybrid | 100.0% (247/247) | 100.0% (1123/1123) | 99.9% (2121/2124) |

## Pushes to Success (Successful Runs Only)

### Median Pushes

| Model | Easy | Medium | Hard |
|-------|--------|--------|--------|
| No Heuristic | 1.0 | 1.0 | 15.0 |
| Diffusion | 1.0 | 1.0 | 2.0 |
| Diffusion Hybrid | 1.0 | 1.0 | 2.0 |

### Mean Pushes

| Model | Easy | Medium | Hard |
|-------|--------|--------|--------|
| No Heuristic | 1.2 | 4.7 | 20.5 |
| Diffusion | 1.0 | 1.6 | 3.5 |
| Diffusion Hybrid | 1.0 | 1.9 | 10.7 |

## Time to Success in ms (Successful Runs Only)

### Median Time

| Model | Easy | Medium | Hard |
|-------|--------|--------|--------|
| No Heuristic | 73 | 94 | 971 |
| Diffusion | 155 | 165 | 258 |
| Diffusion Hybrid | 155 | 165 | 318 |

### Mean Time

| Model | Easy | Medium | Hard |
|-------|--------|--------|--------|
| No Heuristic | 88 | 349 | 2069 |
| Diffusion | 164 | 219 | 457 |
| Diffusion Hybrid | 165 | 254 | 1424 |

## Interaction Statistics (Successful Runs Only)

These metrics show collision rates among successful runs.

### Wall Collision Rate

Percentage of successful runs that had collisions with walls.

| Model | Easy | Medium | Hard |
|-------|--------|--------|--------|
| No Heuristic | 8.9% (22/247) | 28.0% (314/1123) | 39.8% (845/2122) |
| Diffusion | 9.8% (24/245) | 28.8% (317/1100) | 39.6% (704/1776) |
| Diffusion Hybrid | 9.7% (24/247) | 28.9% (325/1123) | 41.0% (869/2121) |

### Movable Object Collision Rate

Percentage of successful runs that collided with other movable objects.

| Model | Easy | Medium | Hard |
|-------|--------|--------|--------|
| No Heuristic | 10.9% (27/247) | 14.2% (160/1123) | 27.2% (578/2122) |
| Diffusion | 11.0% (27/245) | 17.2% (189/1100) | 27.1% (482/1776) |
| Diffusion Hybrid | 10.9% (27/247) | 17.3% (194/1123) | 28.6% (607/2121) |

### Mean Movable Collisions

Average number of unique movable objects collided with per successful run.

| Model | Easy | Medium | Hard |
|-------|--------|--------|--------|
| No Heuristic | 0.11 | 0.15 | 0.29 |
| Diffusion | 0.11 | 0.18 | 0.29 |
| Diffusion Hybrid | 0.11 | 0.18 | 0.31 |

## Detailed Per-Model Statistics

### No Heuristic

| Category | Success Rate | Med Pushes | Mean Pushes | Med Time | Mean Time | Wall Col | Mov Col |
|----------|--------------|------------|-------------|----------|-----------|----------|---------|
| Easy | 100.0% (247/247) | 1.0 | 1.2 | 73 | 88 | 9% | 11% |
| Medium | 100.0% (1123/1123) | 1.0 | 4.7 | 94 | 349 | 28% | 14% |
| Hard | 99.9% (2122/2124) | 15.0 | 20.5 | 971 | 2069 | 40% | 27% |

**Failure Reasons:**

| Reason | Count |
|--------|-------|
| success | 3494 |
| all_pushes_failed | 450 |

### Diffusion

| Category | Success Rate | Med Pushes | Mean Pushes | Med Time | Mean Time | Wall Col | Mov Col |
|----------|--------------|------------|-------------|----------|-----------|----------|---------|
| Easy | 99.2% (245/247) | 1.0 | 1.0 | 155 | 164 | 10% | 11% |
| Medium | 98.0% (1100/1123) | 1.0 | 1.6 | 165 | 219 | 29% | 17% |
| Hard | 83.6% (1776/2124) | 2.0 | 3.5 | 258 | 457 | 40% | 27% |

**Failure Reasons:**

| Reason | Count |
|--------|-------|
| success | 3125 |
| all_pushes_failed | 798 |
| no_valid_goals | 21 |

### Diffusion Hybrid

| Category | Success Rate | Med Pushes | Mean Pushes | Med Time | Mean Time | Wall Col | Mov Col |
|----------|--------------|------------|-------------|----------|-----------|----------|---------|
| Easy | 100.0% (247/247) | 1.0 | 1.0 | 155 | 165 | 10% | 11% |
| Medium | 100.0% (1123/1123) | 1.0 | 1.9 | 165 | 254 | 29% | 17% |
| Hard | 99.9% (2121/2124) | 2.0 | 10.7 | 318 | 1424 | 41% | 29% |

**Failure Reasons:**

| Reason | Count |
|--------|-------|
| success | 3499 |
| all_pushes_failed | 445 |
