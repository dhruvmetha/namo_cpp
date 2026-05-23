"""Goal selection strategies for IDFS planners.

This module provides different strategies for generating goals for selected objects
during IDFS search, allowing for different approaches while keeping the core
search algorithm unchanged.
"""

import math
import random
from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any, Tuple
from dataclasses import dataclass
import namo_rl


@dataclass
class Goal:
    """Goal representation for IDFS actions."""
    x: float
    y: float
    theta: float
    score: float = 0.0
    edge_idx: int = -1   # -1 means "let C++ search", >=0 means specific edge
    depth: int = -1      # -1 means "let C++ search", 0-indexed (depth=0 means push_steps=1)
    # Optional ML-debug metadata. These are ignored by planners that don't use them.
    sample_index: int = -1
    ml_call_id: int = -1
    mask_path: Optional[str] = None


class GoalSelectionStrategy(ABC):
    """Abstract base class for goal selection strategies."""

    @abstractmethod
    def generate_goals(self,
                      object_id: str,
                      state: namo_rl.RLState,
                      env: namo_rl.RLEnvironment,
                      max_goals: int,
                      region_goals_sampled: Optional[List[Tuple[float, float, float]]] = None) -> List[Goal]:
        """Generate goals for the given object in the given state.

        Args:
            object_id: ID of object to generate goals for
            state: Current environment state
            env: Environment instance for querying object positions
            max_goals: Maximum number of goals to generate
            region_goals_sampled: Optional list of (x, y, theta) tuples representing
                                  goal samples for the target neighbor region.
                                  Used for computing goal_sample_region mask in ML inference.

        Returns:
            List of goals to try for this object (can be fewer than max_goals)
        """
        pass
    
    @property
    @abstractmethod
    def strategy_name(self) -> str:
        """Return human-readable name of this strategy."""
        pass


