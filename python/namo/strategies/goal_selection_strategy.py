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
    score: float = 0.0


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


class DiscretizedGridGoalStrategy(GoalSelectionStrategy):
    """Discretized grid goal generation strategy with per-cell sampling.

    Generates goals in a grid around the object's current position (including center),
    then samples multiple orientations per selected grid cell. This provides both
    spatial diversity (different cells) and orientational diversity (multiple samples per cell).
    """

    def __init__(self, cell_size: float = 0.3, grid_size: int = 5,
                 samples_per_cell: int = 1, use_nominal_orientation: bool = False):
        """Initialize with grid parameters.

        Args:
            cell_size: Size of each grid cell in meters
            grid_size: Size of the grid (grid_size x grid_size)
            samples_per_cell: Number of orientation samples per grid cell
            use_nominal_orientation: If True, use zero orientation (θ = 0);
                                   If False, use random orientations
        """
        self.cell_size = cell_size
        self.grid_size = grid_size
        self.samples_per_cell = samples_per_cell
        self.use_nominal_orientation = use_nominal_orientation

        # Calculate grid offset from center
        self.grid_offset = (grid_size - 1) // 2

    def generate_goals(self,
                      object_id: str,
                      state: namo_rl.RLState,
                      env: namo_rl.RLEnvironment,
                      max_goals: int) -> List[Goal]:
        """Generate goals by sampling from discretized grid around object."""
        # Get current object positions from environment
        obs = env.get_observation()

        # Get object position
        pose_key = f"{object_id}_pose"
        if pose_key not in obs:
            return []  # Object not found

        obj_x, obj_y = obs[pose_key][0], obs[pose_key][1]

        # Generate all grid cell positions (including center)
        grid_cells = []
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                # Calculate grid cell center relative to object
                grid_x = obj_x + (i - self.grid_offset) * self.cell_size
                grid_y = obj_y + (j - self.grid_offset) * self.cell_size
                grid_cells.append((grid_x, grid_y))

        # Generate multiple orientation samples for each grid cell
        goals = []
        for grid_x, grid_y in grid_cells:
            for _ in range(self.samples_per_cell):
                if self.use_nominal_orientation:
                    # Use nominal orientation (zero rotation)
                    theta = 0.0
                else:
                    # Use random orientation
                    theta = random.uniform(0, 2 * math.pi)
                goals.append(Goal(x=grid_x, y=grid_y, theta=theta))

        # Respect max_goals limit
        if len(goals) > max_goals:
            goals = random.sample(goals, max_goals)

        return goals

    @property
    def strategy_name(self) -> str:
        return f"Discretized Grid ({self.grid_size}x{self.grid_size}, {self.samples_per_cell} samples/cell)"