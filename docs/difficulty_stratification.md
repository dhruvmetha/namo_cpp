# Difficulty Stratification in 2-Push Evaluation

This document explains how problems are stratified by difficulty in `eval_2push.py`.

## Overview

Difficulty stratification allows analyzing model performance across problems of varying complexity. The key insight is that **oracle (search) performance** serves as a proxy for problem difficulty - problems that require more simulation-verified pushes from the oracle are inherently harder.

Since search is stochastic (primitives are explored in shuffled order), the same problem might take different numbers of pushes on different runs. To get a stable difficulty estimate, we run multiple oracle evaluations with different random seeds and use the **median pushes** across runs.

## Difficulty Categories

Problems are divided into three terciles based on median oracle push counts:

| Category | Percentile Range | Description |
|----------|------------------|-------------|
| **Easy** | Bottom 33% (≤ p33) | Problems solved quickly by oracle |
| **Medium** | Middle 33% (p33 - p66) | Moderately difficult problems |
| **Hard** | Top 33% (> p66) | Problems requiring many oracle pushes |

## Multi-Reference Stratification

When multiple oracle runs with different random seeds are available, **median pushes** across runs determines difficulty. This is robust to the stochastic nature of search.

**Implementation** (`categorize_by_consistency`, lines 1399-1454):

```python
# For each problem, compute median pushes across all oracle runs
median_pushes_list = [tc.median_pushes for tc in consistency_data.values()]

# Compute percentiles from median values
p33 = np.percentile(median_pushes_list, 33.33)
p66 = np.percentile(median_pushes_list, 66.67)

# Assign difficulty based on median
if tc.median_pushes <= p33:
    difficulty = 'easy'
elif tc.median_pushes <= p66:
    difficulty = 'medium'
else:
    difficulty = 'hard'
```

## Key Data Structures

### TripletConsistency (lines 1269-1335)

Stores consistency metrics for a single problem across multiple oracle runs:

```python
@dataclass
class TripletConsistency:
    env: str                    # Environment identifier
    key: str                    # region_label::object_id
    pushes: List[int]           # Pushes from each oracle
    times: List[float]          # Times from each oracle
    chain_depths: List[int]     # Chain depth from each oracle
    all_successful: bool        # True if all oracles solved it

    @property
    def median_pushes(self) -> float:
        return float(np.median(self.pushes))

    @property
    def cv_pushes(self) -> float:
        """Coefficient of variation (std/mean) - measures variability."""
        return self.std_pushes / self.mean_pushes
```

### Difficulty Mapping

A pre-computed dictionary mapping `(env, key) -> difficulty`:

```python
difficulty_mapping: Dict[Tuple[str, str], str] = {
    ('env1.xml', 'region_A::obj_1'): 'easy',
    ('env1.xml', 'region_B::obj_2'): 'hard',
    ...
}
```

This mapping is passed to analysis functions to ensure consistent difficulty assignment across different metrics.

## Depth Filtering

Difficulty can be computed separately for 1-push and 2-push problems:

- `depth_filter=1`: Only 1-push problems (oracle solved with chain_depth=1)
- `depth_filter=2`: Only 2-push problems (oracle solved with chain_depth=2)
- `depth_filter=None`: All problems combined

## Example Output

For a typical 2-push evaluation with ~500 problems and 3 oracle seeds:

```
[2-Push] 487 triplets, 3 refs, CV=0.24
Thresholds: Easy ≤5, Medium 5-12, Hard >12 pushes
Distribution: Easy 162 (33%), Medium 163 (33%), Hard 162 (33%)
```

## Metrics Stratified by Difficulty

The following metrics are computed per difficulty bucket:

1. **Success Rate**: Fraction of problems solved
2. **Pushes Distribution**: Box plots of simulation-verified push counts
3. **Time Distribution**: Box plots of search time
4. **Success@B**: Success rate at fixed push budget (e.g., B=50, 100, 200)
5. **Success@T**: Success rate at fixed time budget (e.g., T=5s, 10s, 30s)
6. **Hybrid Decomposition**: Learned vs fallback success rates

## Visualization

The `plot_consistency_scatter` function (lines 1646-1683) visualizes the relationship between difficulty (median pushes) and oracle variance (coefficient of variation):

```
      ^
  CV  |    * * *     (variable problems)
      |   *   *
      |  *  *  *
      | ***** **     (consistent problems)
      +-------------->
         Median Pushes (difficulty)
```

## Related Functions

| Function | Description |
|----------|-------------|
| `compute_percentile_thresholds` (107-119) | Helper to compute p33/p66 thresholds |
| `assign_difficulty` (122-128) | Assign category from value and thresholds |
| `build_difficulty_mapping` (1457-1473) | Convert categories dict to mapping dict |
| `compute_multi_reference_consistency` (1337-1396) | Aggregate consistency across oracle runs |
| `categorize_by_consistency` (1399-1454) | Assign difficulty categories |
| `compute_consistency_stats` (1509-1563) | Aggregate statistics for reporting |
