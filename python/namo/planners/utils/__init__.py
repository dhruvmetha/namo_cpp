"""Shared utilities for planning and data collection.

Modules:
    solution_smoother: Finds minimal subsequences of solution actions that still solve the task.
    failure_codes: Standardized failure classification for data collection episodes.
"""

from .solution_smoother import SolutionSmoother
from .failure_codes import (
    FailureCode,
    FailureClassifier,
    create_failure_info,
    get_failure_statistics,
)

__all__ = [
    "SolutionSmoother",
    "FailureCode",
    "FailureClassifier",
    "create_failure_info",
    "get_failure_statistics",
]
