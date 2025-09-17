"""MCTS configuration for NAMO planning."""

import math
from dataclasses import dataclass


@dataclass
class MCTSConfig:
    """Configuration for MCTS with progressive widening."""
    
    # Progressive widening parameters (correct formula: k * N^alpha)
    k: float = 10.0                 # Progressive widening constant
    alpha: float = 0.5              # Widening exponent
    
    # UCB1 exploration
    c_exploration: float = 1.414    # UCB1 exploration constant
    
    # Search budget
    simulation_budget: int = 100    # Simulations per decision
    max_rollout_steps: int = 5      # Random rollout depth
    
    # Termination
    max_tree_depth: int = 50        # Maximum search depth
    
    # Object selection strategy
    object_selection_strategy: str = "no_heuristic"  # Strategy for ordering objects
    
    # Reward shaping
    reachability_reward: float = 0.5  # Bonus reward for improving object reachability
    
    # Debug
    verbose: bool = False           # Print search progress


@dataclass 
class ActionConstraints:
    """Action space constraints from environment."""
    min_distance: float
    max_distance: float  
    theta_min: float
    theta_max: float