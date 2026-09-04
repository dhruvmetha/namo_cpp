# Combined Region-Opening Cost Ridgeline

## Goal

Extend the existing real-data ridgeline renderer with one overall population that pools the exact common valid one-push and two-push test episodes while retaining the existing horizon-separated figure.

## Population and aggregation

The combined population contains the 1,310 one-push and 973 two-push episodes already admitted by the seven-arm common-set checks, for 2,283 horizon-tagged observations. HY5U and Uniform Random are reduced to one observation per episode with the existing censoring-aware across-seed median before pooling; Geometric remains deterministic. Pooling is by episode rather than by horizon, so every test episode has equal weight and key reuse across horizons cannot merge observations.

## Rendering

The plotter will continue writing the existing two-panel simulator-push and wall-time figures. It will additionally write one single-panel combined simulator-push figure and one single-panel combined wall-time figure, using the existing method order, colors, log axes, KDE bandwidths, and conditional-on-success density semantics. Each combined unsolved annotation uses the full 2,283-episode denominator for that method.

## Interface

The existing command-line arguments and defaults remain unchanged. The output stem controls the existing simulator-push figure; the new files append `_combined` and `_wall_time_combined` to that stem.

## Validation

Focused tests will assert that pooling preserves every horizon-tagged observation, reports the weighted combined unsolved fraction, and renders both combined outputs. Existing separate-panel behavior must remain unchanged, and the production loader must still enforce the exact 1,310/973 common populations before pooling.
