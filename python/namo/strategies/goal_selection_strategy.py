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


class RandomGoalStrategy(GoalSelectionStrategy):
    """Default random goal generation strategy.
    
    This preserves the original IDFS behavior where goals are sampled randomly
    using polar coordinates around the object position.
    """
    
    def __init__(self, min_distance: float = 0.2, max_distance: float = 0.8, 
                 theta_min: float = 0.0, theta_max: float = 2 * math.pi):
        """Initialize with action constraints.
        
        Args:
            min_distance: Minimum push distance
            max_distance: Maximum push distance  
            theta_min: Minimum push angle (radians)
            theta_max: Maximum push angle (radians)
        """
        self.min_distance = min_distance
        self.max_distance = max_distance
        self.theta_min = theta_min
        self.theta_max = theta_max
    
    def generate_goals(self,
                      object_id: str,
                      state: namo_rl.RLState,
                      env: namo_rl.RLEnvironment,
                      max_goals: int,
                      region_goals_sampled: Optional[List[Tuple[float, float, float]]] = None) -> List[Goal]:
        """Generate random goals around object using polar sampling."""
        # Save current environment state to restore later
        original_state = env.get_full_state()
        
        try:
            # Set state to get object position
            env.set_full_state(state)
            obs = env.get_observation()
            
            # Get object position
            pose_key = f"{object_id}_pose"
            if pose_key not in obs:
                return []  # Object not found
            
            obj_x, obj_y = obs[pose_key][0], obs[pose_key][1]
            
            goals = []
            # random.seed(0)
            for _ in range(max_goals):
                # Sample from continuous action space using polar coordinates
                distance = random.uniform(self.min_distance, self.max_distance)
                theta = random.uniform(self.theta_min, self.theta_max)
                
                target_x = obj_x + distance * math.cos(theta)
                target_y = obj_y + distance * math.sin(theta)
                
                goals.append(Goal(x=target_x, y=target_y, theta=theta))
            
            random.shuffle(goals)
            return goals
            
        finally:
            # Always restore original state to avoid corrupting search
            env.set_full_state(original_state)
    
    @property
    def strategy_name(self) -> str:
        return "Random Goal Generation"



