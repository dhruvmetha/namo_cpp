"""Goal selection strategies for IDFS planners.

This module provides different strategies for generating goals for selected objects
during IDFS search, allowing for different approaches while keeping the core
search algorithm unchanged.
"""

import math
import random
from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
from dataclasses import dataclass
import namo_rl


@dataclass
class Goal:
    """Goal representation for IDFS actions."""
    x: float
    y: float 
    theta: float


class GoalSelectionStrategy(ABC):
    """Abstract base class for goal selection strategies."""
    
    @abstractmethod
    def generate_goals(self, 
                      object_id: str,
                      state: namo_rl.RLState,
                      env: namo_rl.RLEnvironment,
                      max_goals: int) -> List[Goal]:
        """Generate goals for the given object in the given state.
        
        Args:
            object_id: ID of object to generate goals for
            state: Current environment state
            env: Environment instance for querying object positions
            max_goals: Maximum number of goals to generate
            
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
                      max_goals: int) -> List[Goal]:
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
            for _ in range(max_goals):
                # Sample from continuous action space using polar coordinates
                distance = random.uniform(self.min_distance, self.max_distance)
                theta = random.uniform(self.theta_min, self.theta_max)
                
                target_x = obj_x + distance * math.cos(theta)
                target_y = obj_y + distance * math.sin(theta)
                
                goals.append(Goal(x=target_x, y=target_y, theta=theta))
            
            return goals
            
        finally:
            # Always restore original state to avoid corrupting search
            env.set_full_state(original_state)
    
    @property
    def strategy_name(self) -> str:
        return "Random Goal Generation"


class GridGoalStrategy(GoalSelectionStrategy):
    """Grid-based goal generation strategy.
    
    Generates goals in a systematic grid pattern around the object,
    which can provide more thorough exploration compared to random sampling.
    """
    
    def __init__(self, min_distance: float = 0.2, max_distance: float = 0.8,
                 num_angles: int = 8, num_distances: int = 2):
        """Initialize with grid parameters.
        
        Args:
            min_distance: Minimum push distance
            max_distance: Maximum push distance
            num_angles: Number of angles to sample uniformly
            num_distances: Number of distances to sample uniformly  
        """
        self.min_distance = min_distance
        self.max_distance = max_distance
        self.num_angles = num_angles
        self.num_distances = num_distances
    
    def generate_goals(self, 
                      object_id: str,
                      state: namo_rl.RLState,
                      env: namo_rl.RLEnvironment,
                      max_goals: int) -> List[Goal]:
        """Generate goals in grid pattern around object."""
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
            
            # Generate grid of angles and distances
            angles = [2 * math.pi * i / self.num_angles for i in range(self.num_angles)]
            if self.num_distances == 1:
                distances = [(self.min_distance + self.max_distance) / 2]
            else:
                distances = [self.min_distance + (self.max_distance - self.min_distance) * i / (self.num_distances - 1) 
                           for i in range(self.num_distances)]
            
            # Generate all combinations
            for distance in distances:
                for theta in angles:
                    if len(goals) >= max_goals:
                        break
                    
                    target_x = obj_x + distance * math.cos(theta)
                    target_y = obj_y + distance * math.sin(theta)
                    
                    goals.append(Goal(x=target_x, y=target_y, theta=theta))
                
                if len(goals) >= max_goals:
                    break
            
            return goals
            
        finally:
            # Always restore original state to avoid corrupting search
            env.set_full_state(original_state)
    
    @property
    def strategy_name(self) -> str:
        return f"Grid Goal Generation ({self.num_angles}x{self.num_distances})"


class AdaptiveGoalStrategy(GoalSelectionStrategy):
    """Adaptive goal generation that adjusts based on environment constraints.
    
    This strategy attempts to be smarter about goal placement by considering
    environment boundaries and avoiding obviously invalid goal positions.
    """
    
    def __init__(self, min_distance: float = 0.2, max_distance: float = 0.8,
                 boundary_margin: float = 0.1):
        """Initialize with adaptive parameters.
        
        Args:
            min_distance: Minimum push distance
            max_distance: Maximum push distance
            boundary_margin: Margin to keep away from boundaries
        """
        self.min_distance = min_distance
        self.max_distance = max_distance
        self.boundary_margin = boundary_margin
    
    def generate_goals(self, 
                      object_id: str,
                      state: namo_rl.RLState,
                      env: namo_rl.RLEnvironment,
                      max_goals: int) -> List[Goal]:
        """Generate adaptive goals based on environment constraints."""
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
            
            # Try to get environment bounds (this is environment-specific)
            bounds = self._estimate_environment_bounds(obs)
            
            goals = []
            attempts = 0
            max_attempts = max_goals * 5  # Allow some failed attempts
            
            while len(goals) < max_goals and attempts < max_attempts:
                attempts += 1
                
                # Sample candidate goal
                distance = random.uniform(self.min_distance, self.max_distance)
                theta = random.uniform(0, 2 * math.pi)
                
                target_x = obj_x + distance * math.cos(theta)
                target_y = obj_y + distance * math.sin(theta)
                
                # Check if goal is within bounds (if we could determine them)
                if bounds and not self._is_within_bounds(target_x, target_y, bounds):
                    continue
                
                goals.append(Goal(x=target_x, y=target_y, theta=theta))
            
            return goals
            
        finally:
            # Always restore original state to avoid corrupting search
            env.set_full_state(original_state)
    
    def _estimate_environment_bounds(self, obs: Dict[str, Any]) -> Optional[Dict[str, float]]:
        """Attempt to estimate environment boundaries from observations.
        
        This is a best-effort approach that may not work for all environments.
        """
        # Look for common boundary indicators in observations
        # This is heuristic and may need adjustment based on environment
        
        # Try to find all object positions to estimate workspace
        all_positions = []
        
        for key, value in obs.items():
            if key.endswith('_pose') and isinstance(value, (list, tuple)) and len(value) >= 2:
                all_positions.append((value[0], value[1]))
        
        if len(all_positions) < 2:
            return None  # Can't estimate bounds
        
        xs, ys = zip(*all_positions)
        
        # Add margins to estimated bounds  
        margin = self.boundary_margin + self.max_distance
        return {
            'x_min': min(xs) - margin,
            'x_max': max(xs) + margin,
            'y_min': min(ys) - margin,
            'y_max': max(ys) + margin
        }
    
    def _is_within_bounds(self, x: float, y: float, bounds: Dict[str, float]) -> bool:
        """Check if position is within estimated bounds."""
        return (bounds['x_min'] <= x <= bounds['x_max'] and
                bounds['y_min'] <= y <= bounds['y_max'])
    
    @property
    def strategy_name(self) -> str:
        return "Adaptive Goal Generation"