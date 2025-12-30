# 1-Push Evaluation Results

Generated from evaluation config.

## Dataset Overview

Total env+region pairs evaluated: **1274**

| Category | Count | Percentage |
|----------|-------|------------|
| Easy | 82 | 6.4% |
| Medium | 423 | 33.2% |
| Hard | 769 | 60.4% |

## Overall Success Rates

| Model | Successes | Total | Success Rate |
|-------|-----------|-------|--------------|
| Diffusion 5 Steps | 1146 | 1274 | **90.0%** |

## Success Rates by Category

| Model | Easy | Medium | Hard |
|-------|--------|--------|--------|
| Diffusion 5 Steps | 100.0% (82/82) | 98.1% (415/423) | 84.4% (649/769) |

## Pushes to Success (Successful Runs Only)

### Median Pushes

| Model | Easy | Medium | Hard |
|-------|--------|--------|--------|
| Diffusion 5 Steps | 1.0 | 1.0 | 2.0 |

### Mean Pushes

| Model | Easy | Medium | Hard |
|-------|--------|--------|--------|
| Diffusion 5 Steps | 1.0 | 1.5 | 3.7 |

## Time to Success in ms (Successful Runs Only)

### Median Time

| Model | Easy | Medium | Hard |
|-------|--------|--------|--------|
| Diffusion 5 Steps | 155 | 162 | 266 |

### Mean Time

| Model | Easy | Medium | Hard |
|-------|--------|--------|--------|
| Diffusion 5 Steps | 160 | 217 | 473 |

## Interaction Statistics (Successful Runs Only)

These metrics show collision rates among successful runs.

### Wall Collision Rate

Percentage of successful runs that had collisions with walls.

| Model | Easy | Medium | Hard |
|-------|--------|--------|--------|
| Diffusion 5 Steps | 7.3% (6/82) | 29.2% (121/415) | 40.1% (260/649) |

### Movable Object Collision Rate

Percentage of successful runs that collided with other movable objects.

| Model | Easy | Medium | Hard |
|-------|--------|--------|--------|
| Diffusion 5 Steps | 15.9% (13/82) | 17.1% (71/415) | 26.2% (170/649) |

### Mean Movable Collisions

Average number of unique movable objects collided with per successful run.

| Model | Easy | Medium | Hard |
|-------|--------|--------|--------|
| Diffusion 5 Steps | 0.16 | 0.18 | 0.28 |

## Detailed Per-Model Statistics

### Diffusion 5 Steps

| Category | Success Rate | Med Pushes | Mean Pushes | Med Time | Mean Time | Wall Col | Mov Col |
|----------|--------------|------------|-------------|----------|-----------|----------|---------|
| Easy | 100.0% (82/82) | 1.0 | 1.0 | 155 | 160 | 7% | 16% |
| Medium | 98.1% (415/423) | 1.0 | 1.5 | 162 | 217 | 29% | 17% |
| Hard | 84.4% (649/769) | 2.0 | 3.7 | 266 | 473 | 40% | 26% |

**Failure Reasons:**

| Reason | Count |
|--------|-------|
| success | 1148 |
| all_pushes_failed | 279 |
| no_valid_goals | 2 |
