# 2-Push Evaluation Results

Generated from evaluation config.

## Dataset Overview

Problems requiring exactly 2 sequential push actions to clear a path to the goal region, 
as determined by an exhaustive oracle search. Evaluates model ability to solve multi-step 
manipulation problems where object interactions must be planned in sequence.

Total 2-push problems evaluated: **243**

## Summary Table

**Definitions:**
- **Success**: First valid 2-push plan found satisfying clearance + executability
- **Checks**: # simulation-verified candidate push primitive evaluations until first solution
- **Time**: End-to-end wall-clock until first solution (includes inference+decode+scoring+verification)

| Model | Success % | Checks (median [IQR]) | Time (s) (median [IQR]) |
|-------|-----------|----------------------|-------------------------|
| Exhaustive Primitive Search | **100.0%** (243/243) | 219 [85, 1100] | 28.8 [10.1, 136.9] |
| 2 Push Learned Hybrid Voting 5 | **100.0%** (243/243) | 68 [30, 434] | 7.6 [3.2, 48.6] |

## Success@Budget (Constant-Compute Comparison)

**Definition:**
- **Success@B**: Success rate when limited to B simulation-verified push evaluations
- Enables fair comparison by fixing verification compute budget

| Model |@50 | @100 | @200 |
|-------|--------|--------|--------|
| Exhaustive Primitive Search | **12.8%** | **26.7%** | **46.5%** |
| 2 Push Learned Hybrid Voting 5 | **40.7%** | **58.8%** | **66.3%** |

## Success@Time (Constant-Time Comparison)

**Definition:**
- **Success@T**: Success rate when limited to T seconds of search time
- Enables fair comparison by fixing wall-clock time budget

| Model |@5s | @10s | @30s |
|-------|--------|--------|--------|
| Exhaustive Primitive Search | **9.9%** | **25.1%** | **51.0%** |
| 2 Push Learned Hybrid Voting 5 | **41.2%** | **56.0%** | **70.4%** |

## ML Prediction Grounding

**Definition:**
- **Grounding %**: Fraction of ML-aligned primitive slots where the robot can actually reach the push edge
- Measures how well ML predictions are grounded in physical reachability

| Model | Grounding % | Reachable / Aligned | N samples |
|-------|-------------|---------------------|-----------|
| Exhaustive Primitive Search | - | - | 0 |
| 2 Push Learned Hybrid Voting 5 | **60.4%** | 7819/13282 | 243 |

## ReachableAttachment@K

**Definition:**
- **RA@K**: Among top-K ML-ranked primitives (by vote count), fraction with reachable push attachments
- Measures how well ML ranking prioritizes physically feasible pushes

| Model |@10 | @50 | @100 | @All |
|-------|--------|--------|--------|--------|
| 2 Push Learned Hybrid Voting 5 | **73.3%** | **61.7%** | **60.4%** | **60.4%** |

### Detailed RA@K Statistics

| Model | K | Macro | Micro | Reachable/Total | N |
|-------|---|-------|-------|-----------------|---|
| 2 Push Learned Hybrid Voting 5 | 10 | 73.3% | 73.3% | 1780/2430 | 243 |
| 2 Push Learned Hybrid Voting 5 | 50 | 61.7% | 61.3% | 6978/11391 | 243 |
| 2 Push Learned Hybrid Voting 5 | 100 | 60.4% | 58.9% | 7819/13282 | 243 |
| 2 Push Learned Hybrid Voting 5 | All | 60.4% | 58.9% | 7819/13282 | 243 |

## Hybrid Decomposition

**Definitions:**
- **LEARNED**: Solved during ML-only phase (ML-scored primitives)
- **FALLBACK**: ML phase exhausted, solved during primitives phase
- **FAILED**: Neither phase found a solution

### Outcome Breakdown

| Model | N | Learned | Fallback | Failed |
|-------|---|---------|----------|--------|
| 2 Push Learned Hybrid Voting 5 | 243 | 79.0% (192) | 21.0% (51) | 0.0% (0) |

### Learned Cases: Efficiency

*Problems solved by ML-only phase.*

| Model | N | Checks (median [IQR]) | Time (s) (median [IQR]) |
|-------|---|----------------------|-------------------------|
| 2 Push Learned Hybrid Voting 5 | 192 | 46 [28, 105] | 4.9 [2.9, 12.4] |

### Fallback Cases: Efficiency

*Problems where ML phase exhausted, solved by primitives phase. Totals include both phases.*

| Model | N | Checks (median [IQR]) | Time (s) (median [IQR]) |
|-------|---|----------------------|-------------------------|
| 2 Push Learned Hybrid Voting 5 | 51 | 1097 [700, 2086] | 151.3 [55.1, 240.7] |

### Outcome by Difficulty

*Learned vs Fallback breakdown per difficulty bucket (based on oracle pushes).*

| Model | Difficulty | N | Learned | Fallback | Failed |
|-------|------------|---|---------|----------|--------|
| 2 Push Learned Hybrid Voting 5 | Easy | 69 | 91.3% (63) | 8.7% (6) | 0.0% (0) |
| 2 Push Learned Hybrid Voting 5 | Medium | 65 | 84.6% (55) | 15.4% (10) | 0.0% (0) |
| 2 Push Learned Hybrid Voting 5 | Hard | 81 | 75.3% (61) | 24.7% (20) | 0.0% (0) |

## Success Rate by Collision Type

Stratifies problems by the type of collisions encountered in the oracle solution. 
**No Collision**: Direct push to goal. **Wall Only**: Push causes wall contact. 
**Movable Only**: Push causes contact with other movable objects. 
**Both**: Push causes both wall and movable object contact.

| Model | No Collision | Wall Only | Movable Only | Both |
|-------|-------------|-------------|-------------|-------------|
| Exhaustive Primitive Search | 100.0% (4/4) | 100.0% (124/124) | 100.0% (47/47) | 100.0% (81/81) |
| 2 Push Learned Hybrid Voting 5 | 100.0% (4/4) | 100.0% (124/124) | 100.0% (47/47) | 100.0% (81/81) |

### Efficiency by Collision Type (Solved Cases Only)

*Note: Efficiency numbers are computed over solved cases only; models with lower success rates may appear more efficient due to selection bias (e.g., only succeeding on easier instances).*

| Model | Collision Type | N | Median Checks | Median Time (s) |
|-------|----------------|---|---------------|-----------------|
| Exhaustive Primitive Search | No Collision | 4 | 134 | 12.0 |
| Exhaustive Primitive Search | Wall Only | 124 | 166 | 17.3 |
| Exhaustive Primitive Search | Movable Only | 47 | 278 | 50.4 |
| Exhaustive Primitive Search | Both | 81 | 332 | 45.6 |
| 2 Push Learned Hybrid Voting 5 | No Collision | 4 | 56 | 6.3 |
| 2 Push Learned Hybrid Voting 5 | Wall Only | 124 | 51 | 4.7 |
| 2 Push Learned Hybrid Voting 5 | Movable Only | 47 | 64 | 8.1 |
| 2 Push Learned Hybrid Voting 5 | Both | 81 | 119 | 12.5 |

## Difficulty Stratification (by Oracle Pushes)

Categorizes problems by computational difficulty using the oracle's push count as a proxy. 
Problems requiring fewer simulation-verified checks are considered easier. 
Enables analysis of model performance across different difficulty levels.

*Problems split into thirds by oracle pushes: Easy (fewest 33%), Medium (middle 33%), Hard (most 33%).*

Oracle push ranges: **Easy**: 11–421 pushes, **Medium**: 45–2371 pushes, **Hard**: 95–14496 pushes

### Success Rate by Difficulty

| Model | Easy | Medium | Hard |
|-------|------------|------------|------------|
| Exhaustive Primitive Search | 100.0% (69/69) | 100.0% (65/65) | 100.0% (81/81) |
| 2 Push Learned Hybrid Voting 5 | 100.0% (69/69) | 100.0% (65/65) | 100.0% (81/81) |

### Efficiency by Difficulty (Solved Cases Only)

*Note: Efficiency computed over solved cases only; selection bias may apply.*

| Model | Difficulty | N | Median Checks | Median Time (s) |
|-------|------------|---|---------------|-----------------|
| Exhaustive Primitive Search | Easy | 69 | 74 | 8.0 |
| Exhaustive Primitive Search | Medium | 65 | 168 | 19.7 |
| Exhaustive Primitive Search | Hard | 81 | 1124 | 151.2 |
| 2 Push Learned Hybrid Voting 5 | Easy | 69 | 30 | 3.0 |
| 2 Push Learned Hybrid Voting 5 | Medium | 65 | 71 | 8.1 |
| 2 Push Learned Hybrid Voting 5 | Hard | 81 | 214 | 22.7 |

## Detailed Statistics

Granular performance metrics including success counts, efficiency distributions, 
and statistical summaries (median, IQR, mean) for checks and time.

### Success Rates

| Model | Successes | Total | Success Rate |
|-------|-----------|-------|--------------|
| Exhaustive Primitive Search | 243 | 243 | **100.0%** |
| 2 Push Learned Hybrid Voting 5 | 243 | 243 | **100.0%** |

### Checks to Success (Successful Runs Only)

| Model | Median | IQR [25%, 75%] | Mean |
|-------|--------|----------------|------|
| Exhaustive Primitive Search | 219 | [85, 1100] | 1132 |
| 2 Push Learned Hybrid Voting 5 | 68 | [30, 434] | 597 |

### Time to Success (s) (Successful Runs Only)

| Model | Median | IQR [25%, 75%] | Mean |
|-------|--------|----------------|------|
| Exhaustive Primitive Search | 28.8 | [10.1, 136.9] | 143.2 |
| 2 Push Learned Hybrid Voting 5 | 7.6 | [3.2, 48.6] | 70.2 |

## Interaction Statistics (Successful Runs Only)

Measures how often solutions involve object collisions during push execution. 
**Wall Collision Rate**: Fraction of solutions where pushed object contacts walls. 
**Movable Collision Rate**: Fraction where pushed object contacts other movable objects.

*Note: Statistics computed over successful runs only. Models with lower success rates may show different interaction patterns due to selection bias (failing on harder instances).*

| Model | Wall Collision Rate | Movable Collision Rate |
|-------|---------------------|------------------------|
| Exhaustive Primitive Search | 81.5% (198/243) | 48.6% (118/243) |
| 2 Push Learned Hybrid Voting 5 | 83.5% (203/243) | 47.3% (115/243) |

## Model Result Breakdown

How each model solved 2-push problems:

- **Model 1-Push**: Solved with 1 push (unexpected for true 2-push)
- **Model 2-Push**: Solved with exactly 2 pushes (optimal)
- **Model 2+ Push**: Solved with more than 2 pushes
- **Failed**: Did not find a solution

| Model | 1-Push | 2-Push | 2+ Push | Failed | Total |
|-------|--------|--------|---------|--------|-------|
| Exhaustive Primitive Search | 1 (0%) | 242 (100%) | 0 (0%) | 0 (0%) | 243 |
| 2 Push Learned Hybrid Voting 5 | 3 (1%) | 240 (99%) | 0 (0%) | 0 (0%) | 243 |

## Failure Reasons

Breakdown of why models failed to find solutions. Common reasons include: 
timeout (search exceeded time limit), no valid primitives (no feasible push actions found), 
and search exhausted (all candidates evaluated without finding a solution).

### Exhaustive Primitive Search

| Reason | Count |
|--------|-------|
| success | 266 |

### 2 Push Learned Hybrid Voting 5

| Reason | Count |
|--------|-------|
| success | 426 |
| all_pushes_failed | 3 |
