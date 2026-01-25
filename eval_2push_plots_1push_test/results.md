# 2-Push Evaluation Results

Generated from evaluation config.

## Dataset Overview

Problems requiring exactly 2 sequential push actions to clear a path to the goal region, 
as determined by an exhaustive oracle search. Evaluates model ability to solve multi-step 
manipulation problems where object interactions must be planned in sequence.

Total 2-push problems evaluated: **1125**

## Summary Table

**Definitions:**
- **Success**: First valid 2-push plan found satisfying clearance + executability
- **Checks**: # simulation-verified candidate push primitive evaluations until first solution
- **Time**: End-to-end wall-clock until first solution (includes inference+decode+scoring+verification)

| Model | Success % | Checks (median [IQR]) | Time (s) (median [IQR]) |
|-------|-----------|----------------------|-------------------------|
| Exhaustive Primitive Search | **99.0%** (1114/1125) | 140 [66, 626] | 19.5 [7.6, 71.8] |
| 2 Push Learned Hybrid Voting 5 | **99.6%** (1121/1125) | 71 [30, 391] | 7.9 [3.1, 38.6] |

## Success@Budget (Constant-Compute Comparison)

**Definition:**
- **Success@B**: Success rate when limited to B simulation-verified push evaluations
- Enables fair comparison by fixing verification compute budget

| Model |@50 | @100 | @200 |
|-------|--------|--------|--------|
| Exhaustive Primitive Search | **19.4%** | **39.7%** | **58.0%** |
| 2 Push Learned Hybrid Voting 5 | **41.1%** | **56.4%** | **66.8%** |

## Success@Time (Constant-Time Comparison)

**Definition:**
- **Success@T**: Success rate when limited to T seconds of search time
- Enables fair comparison by fixing wall-clock time budget

| Model |@5s | @10s | @30s |
|-------|--------|--------|--------|
| Exhaustive Primitive Search | **17.3%** | **32.8%** | **58.4%** |
| 2 Push Learned Hybrid Voting 5 | **37.8%** | **54.2%** | **71.1%** |

## ML Prediction Grounding

**Definition:**
- **Grounding %**: Fraction of ML-aligned primitive slots where the robot can actually reach the push edge
- Measures how well ML predictions are grounded in physical reachability

| Model | Grounding % | Reachable / Aligned | N samples |
|-------|-------------|---------------------|-----------|
| Exhaustive Primitive Search | - | - | 0 |
| 2 Push Learned Hybrid Voting 5 | **59.9%** | 35741/60403 | 1121 |

## ReachableAttachment@K

**Definition:**
- **RA@K**: Among top-K ML-ranked primitives (by vote count), fraction with reachable push attachments
- Measures how well ML ranking prioritizes physically feasible pushes

| Model |@10 | @50 | @100 | @All |
|-------|--------|--------|--------|--------|
| 2 Push Learned Hybrid Voting 5 | **72.4%** | **60.9%** | **59.9%** | **59.9%** |

### Detailed RA@K Statistics

| Model | K | Macro | Micro | Reachable/Total | N |
|-------|---|-------|-------|-----------------|---|
| 2 Push Learned Hybrid Voting 5 | 10 | 72.4% | 72.4% | 8117/11209 | 1121 |
| 2 Push Learned Hybrid Voting 5 | 50 | 60.9% | 60.7% | 31505/51869 | 1121 |
| 2 Push Learned Hybrid Voting 5 | 100 | 59.9% | 59.2% | 35741/60403 | 1121 |
| 2 Push Learned Hybrid Voting 5 | All | 59.9% | 59.2% | 35741/60403 | 1121 |

## Hybrid Decomposition

**Definitions:**
- **LEARNED**: Solved during ML-only phase (ML-scored primitives)
- **FALLBACK**: ML phase exhausted, solved during primitives phase
- **FAILED**: Neither phase found a solution

### Outcome Breakdown

| Model | N | Learned | Fallback | Failed |
|-------|---|---------|----------|--------|
| 2 Push Learned Hybrid Voting 5 | 1125 | 78.3% (881) | 21.3% (240) | 0.4% (4) |

### Learned Cases: Efficiency

*Problems solved by ML-only phase.*

| Model | N | Checks (median [IQR]) | Time (s) (median [IQR]) |
|-------|---|----------------------|-------------------------|
| 2 Push Learned Hybrid Voting 5 | 881 | 47 [25, 120] | 5.4 [2.6, 13.5] |

### Fallback Cases: Efficiency

*Problems where ML phase exhausted, solved by primitives phase. Totals include both phases.*

| Model | N | Checks (median [IQR]) | Time (s) (median [IQR]) |
|-------|---|----------------------|-------------------------|
| 2 Push Learned Hybrid Voting 5 | 240 | 1120 [588, 2294] | 118.5 [56.1, 244.4] |

### Outcome by Difficulty

*Learned vs Fallback breakdown per difficulty bucket (based on oracle pushes).*

| Model | Difficulty | N | Learned | Fallback | Failed |
|-------|------------|---|---------|----------|--------|
| 2 Push Learned Hybrid Voting 5 | Easy | 332 | 91.6% (304) | 8.4% (28) | 0.0% (0) |
| 2 Push Learned Hybrid Voting 5 | Medium | 332 | 81.9% (272) | 18.1% (60) | 0.0% (0) |
| 2 Push Learned Hybrid Voting 5 | Hard | 342 | 69.9% (239) | 29.8% (102) | 0.3% (1) |

## Success Rate by Collision Type

Stratifies problems by the type of collisions encountered in the oracle solution. 
**No Collision**: Direct push to goal. **Wall Only**: Push causes wall contact. 
**Movable Only**: Push causes contact with other movable objects. 
**Both**: Push causes both wall and movable object contact.

| Model | No Collision | Wall Only | Movable Only | Both |
|-------|-------------|-------------|-------------|-------------|
| Exhaustive Primitive Search | 100.0% (54/54) | 98.9% (533/539) | 99.6% (231/232) | 98.9% (363/367) |
| 2 Push Learned Hybrid Voting 5 | 100.0% (54/54) | 99.6% (537/539) | 100.0% (232/232) | 99.5% (365/367) |

### Efficiency by Collision Type (Solved Cases Only)

*Note: Efficiency numbers are computed over solved cases only; models with lower success rates may appear more efficient due to selection bias (e.g., only succeeding on easier instances).*

| Model | Collision Type | N | Median Checks | Median Time (s) |
|-------|----------------|---|---------------|-----------------|
| Exhaustive Primitive Search | No Collision | 54 | 60 | 7.8 |
| Exhaustive Primitive Search | Wall Only | 533 | 114 | 13.5 |
| Exhaustive Primitive Search | Movable Only | 231 | 141 | 24.4 |
| Exhaustive Primitive Search | Both | 363 | 192 | 30.6 |
| 2 Push Learned Hybrid Voting 5 | No Collision | 54 | 22 | 2.4 |
| 2 Push Learned Hybrid Voting 5 | Wall Only | 537 | 56 | 5.7 |
| 2 Push Learned Hybrid Voting 5 | Movable Only | 232 | 62 | 7.8 |
| 2 Push Learned Hybrid Voting 5 | Both | 365 | 124 | 14.4 |

## Difficulty Stratification (by Oracle Pushes)

Categorizes problems by computational difficulty using the oracle's push count as a proxy. 
Problems requiring fewer simulation-verified checks are considered easier. 
Enables analysis of model performance across different difficulty levels.

*Problems split into thirds by oracle pushes: Easy (fewest 33%), Medium (middle 33%), Hard (most 33%).*

Oracle push ranges: **Easy**: 5–597 pushes, **Medium**: 45–3234 pushes, **Hard**: 68–30179 pushes

### Success Rate by Difficulty

| Model | Easy | Medium | Hard |
|-------|------------|------------|------------|
| Exhaustive Primitive Search | 100.0% (332/332) | 99.7% (331/332) | 98.5% (337/342) |
| 2 Push Learned Hybrid Voting 5 | 100.0% (332/332) | 100.0% (332/332) | 99.7% (341/342) |

### Efficiency by Difficulty (Solved Cases Only)

*Note: Efficiency computed over solved cases only; selection bias may apply.*

| Model | Difficulty | N | Median Checks | Median Time (s) |
|-------|------------|---|---------------|-----------------|
| Exhaustive Primitive Search | Easy | 332 | 62 | 7.1 |
| Exhaustive Primitive Search | Medium | 331 | 137 | 19.7 |
| Exhaustive Primitive Search | Hard | 337 | 569 | 66.8 |
| 2 Push Learned Hybrid Voting 5 | Easy | 332 | 27 | 2.7 |
| 2 Push Learned Hybrid Voting 5 | Medium | 332 | 82 | 9.6 |
| 2 Push Learned Hybrid Voting 5 | Hard | 341 | 260 | 27.2 |

## Detailed Statistics

Granular performance metrics including success counts, efficiency distributions, 
and statistical summaries (median, IQR, mean) for checks and time.

### Success Rates

| Model | Successes | Total | Success Rate |
|-------|-----------|-------|--------------|
| Exhaustive Primitive Search | 1114 | 1125 | **99.0%** |
| 2 Push Learned Hybrid Voting 5 | 1121 | 1125 | **99.6%** |

### Checks to Success (Successful Runs Only)

| Model | Median | IQR [25%, 75%] | Mean |
|-------|--------|----------------|------|
| Exhaustive Primitive Search | 140 | [66, 626] | 586 |
| 2 Push Learned Hybrid Voting 5 | 71 | [30, 391] | 537 |

### Time to Success (s) (Successful Runs Only)

| Model | Median | IQR [25%, 75%] | Mean |
|-------|--------|----------------|------|
| Exhaustive Primitive Search | 19.5 | [7.6, 71.8] | 85.0 |
| 2 Push Learned Hybrid Voting 5 | 7.9 | [3.1, 38.6] | 64.6 |

## Interaction Statistics (Successful Runs Only)

Measures how often solutions involve object collisions during push execution. 
**Wall Collision Rate**: Fraction of solutions where pushed object contacts walls. 
**Movable Collision Rate**: Fraction where pushed object contacts other movable objects.

*Note: Statistics computed over successful runs only. Models with lower success rates may show different interaction patterns due to selection bias (failing on harder instances).*

| Model | Wall Collision Rate | Movable Collision Rate |
|-------|---------------------|------------------------|
| Exhaustive Primitive Search | 76.1% (848/1114) | 51.0% (568/1114) |
| 2 Push Learned Hybrid Voting 5 | 78.3% (878/1121) | 51.1% (573/1121) |

## Model Result Breakdown

How each model solved 2-push problems:

- **Model 1-Push**: Solved with 1 push (unexpected for true 2-push)
- **Model 2-Push**: Solved with exactly 2 pushes (optimal)
- **Model 2+ Push**: Solved with more than 2 pushes
- **Failed**: Did not find a solution

| Model | 1-Push | 2-Push | 2+ Push | Failed | Total |
|-------|--------|--------|---------|--------|-------|
| Exhaustive Primitive Search | 197 (18%) | 917 (82%) | 0 (0%) | 11 (1%) | 1125 |
| 2 Push Learned Hybrid Voting 5 | 19 (2%) | 1102 (98%) | 0 (0%) | 4 (0%) | 1125 |

## Failure Reasons

Breakdown of why models failed to find solutions. Common reasons include: 
timeout (search exceeded time limit), no valid primitives (no feasible push actions found), 
and search exhausted (all candidates evaluated without finding a solution).

### Exhaustive Primitive Search

| Reason | Count |
|--------|-------|
| success | 1296 |
| all_pushes_failed | 21 |

### 2 Push Learned Hybrid Voting 5

| Reason | Count |
|--------|-------|
| success | 1245 |
| all_pushes_failed | 13 |
