"""Abstract base classes for NAMO planning algorithms.

This module provides the interface for pluggable planning algorithms
that can be used with the parallel data collection system.
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Tuple, Dict, Any
from dataclasses import dataclass
import namo_rl


@dataclass
class PlannerResult:
    """Result from a planning algorithm execution."""
    
    # Core results
    success: bool
    solution_found: bool
    action_sequence: Optional[List[namo_rl.Action]] = None
    solution_depth: Optional[int] = None
    
    # State information - SE(2) poses before each action
    state_observations: Optional[List[Dict[str, List[float]]]] = None
    
    # State information - SE(2) poses after each action is executed
    post_action_state_observations: Optional[List[Dict[str, List[float]]]] = None
    
    # Performance metrics
    search_time_ms: Optional[float] = None
    nodes_expanded: Optional[int] = None
    terminal_checks: Optional[int] = None
    max_depth_reached: Optional[int] = None
    
    # Algorithm-specific metrics
    algorithm_stats: Optional[Dict[str, Any]] = None
    
    # Error information
    error_message: str = ""


@dataclass
class PlannerConfig:
    """Base configuration for planning algorithms."""

    # Search limits
    max_depth: int = 5
    max_goals_per_object: int = 5
    max_terminal_checks: Optional[int] = 5000  # Cap on expensive terminal checks (default 5000)
    max_search_time_seconds: Optional[float] = 300.0  # 5 minute timeout per search (default)
    goals_per_region: int = 5  # Number of robot goal samples per region for validation
    
    # Randomization
    random_seed: Optional[int] = None
    
    # Debugging
    verbose: bool = False
    collect_stats: bool = True
    
    # Algorithm-specific parameters
    algorithm_params: Optional[Dict[str, Any]] = None


class BasePlanner(ABC):
    """Abstract base class for NAMO planning algorithms.
    
    This interface allows different planning algorithms (IDFS, Tree-IDFS, MCTS, etc.)
    to be used interchangeably in the parallel data collection system.
    """
    
    def __init__(self, env: namo_rl.RLEnvironment, config: PlannerConfig):
        """Initialize planner with environment and configuration.
        
        Args:
            env: NAMO RL environment
            config: Algorithm configuration
        """
        self.env = env
        self.config = config
        self._setup_constraints()
        self._initialize_algorithm()
    
    @abstractmethod
    def _setup_constraints(self):
        """Setup action constraints from environment."""
        pass
    
    @abstractmethod
    def _initialize_algorithm(self):
        """Initialize algorithm-specific components."""
        pass
    
    @abstractmethod
    def search(self, robot_goal: Tuple[float, float, float]) -> PlannerResult:
        """Execute planning algorithm to find action sequence.
        
        Args:
            robot_goal: Target robot position (x, y, theta)
            
        Returns:
            PlannerResult containing solution and performance metrics
        """
        pass
    
    @abstractmethod
    def reset(self):
        """Reset internal algorithm state for new planning episode."""
        pass
    
    @property
    @abstractmethod
    def algorithm_name(self) -> str:
        """Return human-readable algorithm name."""
        pass
    
    @property
    @abstractmethod
    def algorithm_version(self) -> str:
        """Return algorithm version/variant identifier."""
        pass


class PlannerFactory:
    """Factory for creating planner instances."""
    
    _planners = {}
    
    @classmethod
    def register_planner(cls, name: str, planner_class: type):
        """Register a planner algorithm.
        
        Args:
            name: Algorithm identifier (e.g., "idfs", "tree_idfs", "mcts")
            planner_class: Class implementing BasePlanner interface
        """
        cls._planners[name] = planner_class
    
    @classmethod
    def create_planner(cls, name: str, env: namo_rl.RLEnvironment, 
                      config: PlannerConfig) -> BasePlanner:
        """Create planner instance by name.
        
        Args:
            name: Algorithm identifier
            env: NAMO RL environment
            config: Algorithm configuration
            
        Returns:
            Planner instance
            
        Raises:
            KeyError: If algorithm name not registered
        """
        if name not in cls._planners:
            available = list(cls._planners.keys())
            raise KeyError(f"Unknown planner '{name}'. Available: {available}")
        
        planner_class = cls._planners[name]
        return planner_class(env, config)
    
    @classmethod
    def list_available_planners(cls) -> List[str]:
        """Return list of registered planner names."""
        return list(cls._planners.keys())


# Utility functions for algorithm comparison
def create_planner_from_legacy_config(algorithm: str, env: namo_rl.RLEnvironment,
                                     max_depth: int = 5, max_goals_per_object: int = 5,
                                     random_seed: Optional[int] = None,
                                     verbose: bool = False,
                                     collect_stats: bool = True) -> BasePlanner:
    """Create planner instance from legacy parameters.
    
    This function provides backward compatibility with existing code.
    """
    config = PlannerConfig(
        max_depth=max_depth,
        max_goals_per_object=max_goals_per_object,
        random_seed=random_seed,
        verbose=verbose,
        collect_stats=collect_stats
    )
    
    return PlannerFactory.create_planner(algorithm, env, config)


def compare_planners(env: namo_rl.RLEnvironment, 
                    robot_goal: Tuple[float, float, float],
                    planner_configs: Dict[str, PlannerConfig],
                    num_trials: int = 5) -> Dict[str, List[PlannerResult]]:
    """Compare multiple planning algorithms on the same problem.
    
    Args:
        env: NAMO RL environment
        robot_goal: Target robot position
        planner_configs: Dict mapping algorithm names to configurations
        num_trials: Number of trials per algorithm
        
    Returns:
        Dict mapping algorithm names to lists of results
    """
    results = {}
    
    for algo_name, config in planner_configs.items():
        planner = PlannerFactory.create_planner(algo_name, env, config)
        algo_results = []
        
        for trial in range(num_trials):
            env.reset()  # Reset environment state
            planner.reset()  # Reset planner state
            result = planner.search(robot_goal)
            algo_results.append(result)
        
        results[algo_name] = algo_results
    
    return results