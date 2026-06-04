"""Shared push-attempt budget for full-environment solves."""

from __future__ import annotations

from dataclasses import dataclass


class PushBudgetExceeded(RuntimeError):
    """Raised when the shared per-environment push budget is exhausted."""

    def __init__(self, limit: int, used: int):
        super().__init__(f"Simulation budget exhausted after {used}/{limit} env.step calls")
        self.limit = int(limit)
        self.used = int(used)


@dataclass
class PushAttemptBudget:
    """Mutable per-environment budget counted in env.step calls."""

    limit: int
    used: int = 0

    def __post_init__(self) -> None:
        self.limit = int(self.limit)
        self.used = int(self.used)
        if self.limit < 0:
            raise ValueError("PushAttemptBudget.limit must be non-negative")

    @property
    def remaining(self) -> int:
        return max(0, self.limit - self.used)

    @property
    def exhausted(self) -> bool:
        return self.used >= self.limit

    def consume_or_raise(self) -> None:
        if self.exhausted:
            raise PushBudgetExceeded(self.limit, self.used)
        self.used += 1
