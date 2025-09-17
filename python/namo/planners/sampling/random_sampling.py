"""Random Sampling Planner for NAMO planning with Strategy Support.

This implementation provides a single-stream random sampling approach with configurable strategies:
1. Select object using configurable object selection strategy
2. Generate goals using configurable goal selection strategy  
3. Execute action and repeat up to max-depth
4. Report episode results

Key characteristics:
- No iterative deepening (single stream to max-depth)
- Configurable object selection strategies (random, nearest, goal-proximity, farthest, ML)
- Configurable goal selection strategies (random, grid, adaptive, ML)
- Fresh sampling at each step
- Baseline for comparison with more sophisticated algorithms
"""

import math
import random
import time
from typing import List, Optional, Tuple, Dict, TYPE_CHECKING
from dataclasses import dataclass

if TYPE_CHECKING:
    from namo.strategies import ObjectSelectionStrategy
    from namo.strategies import GoalSelectionStrategy

import namo_rl
from namo.config import ActionConstraints
from namo.core import BasePlanner, PlannerConfig, PlannerResult
from namo.strategies import ObjectSelectionStrategy, NoHeuristicStrategy, NearestFirstStrategy, GoalProximityStrategy, FarthestFirstStrategy
from namo.strategies import GoalSelectionStrategy, RandomGoalStrategy, GridGoalStrategy, AdaptiveGoalStrategy


@dataclass
class Goal:
    """Push goal representation."""
    x: float
    y: float
    theta: float


@dataclass
class Action:
    """Random sampling action representation."""
    object_id: str
    goal: Goal
    
    def to_namo_action(self) -> namo_rl.Action:
        """Convert to NAMO RL action."""
        action = namo_rl.Action()
        action.object_id = self.object_id
        action.x = self.goal.x
        action.y = self.goal.y
        action.theta = self.goal.theta
        return action


class RandomSamplingPlanner(BasePlanner):
    """Random Sampling planner for NAMO planning with configurable strategies.
    
    This planner provides a baseline approach that uses configurable strategies for
    object selection and goal generation in a single stream until max-depth is reached
    or solution found. No iterative deepening, but supports various strategies.
    """
    
    def __init__(self, env: namo_rl.RLEnvironment, config: PlannerConfig,
                 object_selection_strategy: Optional['ObjectSelectionStrategy'] = None,
                 goal_selection_strategy: Optional['GoalSelectionStrategy'] = None):
        
        # Initialize strategy attributes first (before super().__init__ calls _setup_constraints)
        self.object_selection_strategy = None
        self.goal_selection_strategy = None
        
        super().__init__(env, config)
        
        # Object selection strategy - can come from config or parameter
        if object_selection_strategy is not None:
            # Explicit strategy parameter takes precedence
            self.object_selection_strategy = object_selection_strategy
        elif config.algorithm_params and 'object_selection_strategy' in config.algorithm_params:
            # Get strategy from config with proper error handling
            strategy_name = config.algorithm_params['object_selection_strategy']
            try:
                self.object_selection_strategy = self._create_object_strategy_from_name(strategy_name)
            except Exception as e:
                if self.config.verbose:
                    print(f"Warning: Failed to create object strategy '{strategy_name}': {e}")
                    print("Falling back to default no heuristic strategy")
                # Fall back to default strategy
                self.object_selection_strategy = NoHeuristicStrategy()
        else:
            # Default to no heuristic (pure random)
            self.object_selection_strategy = NoHeuristicStrategy()
        
        # Goal selection strategy - can come from config or parameter
        if goal_selection_strategy is not None:
            # Explicit strategy parameter takes precedence
            self.goal_selection_strategy = goal_selection_strategy
        elif config.algorithm_params and 'goal_selection_strategy' in config.algorithm_params:
            # Get strategy from config with proper error handling
            strategy_name = config.algorithm_params['goal_selection_strategy']
            try:
                self.goal_selection_strategy = self._create_goal_strategy_from_name(strategy_name)
            except Exception as e:
                if self.config.verbose:
                    print(f"Warning: Failed to create goal strategy '{strategy_name}': {e}")
                    print("Falling back to default random strategy")
                # Fall back to default strategy
                self.goal_selection_strategy = RandomGoalStrategy(
                    min_distance=self.constraints.min_distance,
                    max_distance=self.constraints.max_distance,
                    theta_min=self.constraints.theta_min,
                    theta_max=self.constraints.theta_max
                )
        else:
            # Default to random strategy
            self.goal_selection_strategy = RandomGoalStrategy(
                min_distance=self.constraints.min_distance,
                max_distance=self.constraints.max_distance,
                theta_min=self.constraints.theta_min,
                theta_max=self.constraints.theta_max
            )
        
        # Statistics tracking
        self.stats = {}
    
    def _create_object_strategy_from_name(self, strategy_name: str) -> 'ObjectSelectionStrategy':
        """Create object selection strategy from string name."""
        if strategy_name == "no_heuristic":
            return NoHeuristicStrategy()
        elif strategy_name == "nearest_first":
            return NearestFirstStrategy()
        elif strategy_name == "goal_proximity":
            return GoalProximityStrategy()
        elif strategy_name == "farthest_first":
            return FarthestFirstStrategy()
        elif strategy_name == "ml":
            # Import ML strategy
            from namo.strategies.ml_strategies import MLObjectSelectionStrategy
            
            # Get ML parameters from config
            ml_object_model_path = self.config.algorithm_params.get('ml_object_model_path')
            ml_samples = self.config.algorithm_params.get('ml_samples', 32)
            ml_device = self.config.algorithm_params.get('ml_device', 'cuda')
            xml_file = self.config.algorithm_params.get('xml_file', '')
            preloaded_model = self.config.algorithm_params.get('preloaded_object_model')
            
            if not ml_object_model_path and not preloaded_model:
                raise ValueError("ML object strategy requires ml_object_model_path or preloaded_object_model")
            
            return MLObjectSelectionStrategy(
                object_model_path=ml_object_model_path or "",
                samples=ml_samples,
                device=ml_device,
                xml_path_relative=xml_file,
                verbose=self.config.verbose,
                preloaded_model=preloaded_model
            )
        else:
            raise ValueError(f"Unknown object selection strategy: {strategy_name}")
    
    def _create_goal_strategy_from_name(self, strategy_name: str) -> 'GoalSelectionStrategy':
        """Create goal selection strategy from string name."""
        if strategy_name == "random":
            return RandomGoalStrategy(
                min_distance=self.constraints.min_distance,
                max_distance=self.constraints.max_distance,
                theta_min=self.constraints.theta_min,
                theta_max=self.constraints.theta_max
            )
        elif strategy_name == "grid":
            return GridGoalStrategy(
                min_distance=self.constraints.min_distance,
                max_distance=self.constraints.max_distance
            )
        elif strategy_name == "adaptive":
            return AdaptiveGoalStrategy(
                min_distance=self.constraints.min_distance,
                max_distance=self.constraints.max_distance
            )
        elif strategy_name == "ml":
            # Import ML strategy
            from namo.strategies.ml_strategies import MLGoalSelectionStrategy
            
            # Get ML parameters from config
            ml_goal_model_path = self.config.algorithm_params.get('ml_goal_model_path')
            ml_samples = self.config.algorithm_params.get('ml_samples', 32)
            ml_device = self.config.algorithm_params.get('ml_device', 'cuda')
            xml_file = self.config.algorithm_params.get('xml_file', '')
            epsilon = self.config.algorithm_params.get('epsilon')
            preloaded_model = self.config.algorithm_params.get('preloaded_goal_model')
            
            if not ml_goal_model_path and not preloaded_model:
                raise ValueError("ML goal strategy requires ml_goal_model_path or preloaded_goal_model")
            
            return MLGoalSelectionStrategy(
                goal_model_path=ml_goal_model_path or "",
                samples=ml_samples,
                device=ml_device,
                xml_path_relative=xml_file,
                epsilon=epsilon,
                fallback_strategy=RandomGoalStrategy(
                    min_distance=self.constraints.min_distance,
                    max_distance=self.constraints.max_distance,
                    theta_min=self.constraints.theta_min,
                    theta_max=self.constraints.theta_max
                ),
                verbose=self.config.verbose,
                preloaded_model=preloaded_model
            )
        else:
            raise ValueError(f"Unknown goal selection strategy: {strategy_name}")
    
    def _setup_constraints(self):
        """Setup action constraints from environment."""
        env_constraints = self.env.get_action_constraints()
        self.constraints = ActionConstraints(
            min_distance=env_constraints.min_distance,
            max_distance=env_constraints.max_distance,
            theta_min=env_constraints.theta_min,
            theta_max=env_constraints.theta_max
        )
    
    def _initialize_algorithm(self):
        """Initialize algorithm-specific components."""
        if self.config.random_seed is not None:
            random.seed(self.config.random_seed)
    
    def reset(self):
        """Reset internal algorithm state for new planning episode."""
        self.stats = {
            'nodes_expanded': 0,
            'terminal_checks': 0,
            'max_depth_reached': 0,
        } if self.config.collect_stats else {}
        
        # Reset random seed for consistency
        if self.config.random_seed is not None:
            random.seed(self.config.random_seed)
    
    @property
    def algorithm_name(self) -> str:
        obj_strategy = self.object_selection_strategy.strategy_name if self.object_selection_strategy else "Unknown"
        goal_strategy = self.goal_selection_strategy.strategy_name if self.goal_selection_strategy else "Unknown"
        return f"Random Sampling ({obj_strategy} + {goal_strategy})"
    
    @property
    def algorithm_version(self) -> str:
        return "random_sampling_v2.0_with_strategies"
    
    def search(self, robot_goal: Tuple[float, float, float]) -> PlannerResult:
        """Run single-stream random sampling to find action sequence."""
        start_time = time.time()
        timeout_seconds = self.config.max_search_time_seconds
        
        # Reset planner state for new search
        self.reset()
        
        # Set robot goal
        self.env.set_robot_goal(*robot_goal)
        
        if self.config.verbose:
            print(f"Starting random sampling (max depth: {self.config.max_depth})")
        
        # Check if root is already terminal
        root_state = self.env.get_full_state()
        if self._is_terminal_state(root_state):
            if self.config.verbose:
                print("Goal already reachable at root!")
            
            search_time_ms = (time.time() - start_time) * 1000
            return PlannerResult(
                success=True,
                solution_found=True,
                action_sequence=[],
                solution_depth=0,
                state_observations=[],  # Empty state sequence for already-terminal state
                post_action_state_observations=[],  # Empty post-action state sequence for already-terminal state
                search_time_ms=search_time_ms,
                nodes_expanded=self.stats.get('nodes_expanded', 0),
                terminal_checks=self.stats.get('terminal_checks', 0),
                max_depth_reached=self.stats.get('max_depth_reached', 0),
                algorithm_stats={'single_stream': True, 'random_sampling': True}
            )
        
        # Single-stream random sampling
        solution_result = self._random_sampling_search(root_state, start_time, timeout_seconds)
        if solution_result is not None:
            solution_actions, solution_states, post_action_states = solution_result
            if self.config.verbose:
                actions_str = [f'{a.object_id}->({a.goal.x:.2f},{a.goal.y:.2f})' for a in solution_actions]
                print(f"Solution found at depth {len(solution_actions)}: {actions_str}")
            
            # Convert to namo_rl.Action list
            namo_actions = [action.to_namo_action() for action in solution_actions]
            
            search_time_ms = (time.time() - start_time) * 1000
            return PlannerResult(
                success=True,
                solution_found=True,
                action_sequence=namo_actions,
                solution_depth=len(solution_actions),
                state_observations=solution_states,
                post_action_state_observations=post_action_states,
                search_time_ms=search_time_ms,
                nodes_expanded=self.stats.get('nodes_expanded', 0),
                terminal_checks=self.stats.get('terminal_checks', 0),
                max_depth_reached=self.stats.get('max_depth_reached', 0),
                algorithm_stats={'single_stream': True, 'random_sampling': True}
            )
        
        if self.config.verbose:
            print("No solution found within depth/time limits")
        
        search_time_ms = (time.time() - start_time) * 1000
        return PlannerResult(
            success=True,
            solution_found=False,
            state_observations=None,  # No state observations when no solution found
            post_action_state_observations=None,  # No post-action state observations when no solution found
            search_time_ms=search_time_ms,
            nodes_expanded=self.stats.get('nodes_expanded', 0),
            terminal_checks=self.stats.get('terminal_checks', 0),
            max_depth_reached=self.stats.get('max_depth_reached', 0),
            algorithm_stats={'single_stream': True, 'random_sampling': True}
        )
    
    def _is_terminal_state(self, state: namo_rl.RLState) -> bool:
        """Check if robot goal is reachable from given state."""
        # Check terminal check limit before expensive operation
        if (self.config.max_terminal_checks is not None and 
            self.config.collect_stats and 
            self.stats.get('terminal_checks', 0) >= self.config.max_terminal_checks):
            return False  # Treat as non-terminal to stop search
        
        self.env.set_full_state(state)
        is_terminal = self.env.is_robot_goal_reachable()
        
        if self.config.collect_stats:
            self.stats['terminal_checks'] += 1
        
        return is_terminal
    
    def _get_reachable_objects(self, state: namo_rl.RLState) -> List[str]:
        """Get list of reachable objects from given state."""
        self.env.set_full_state(state)
        return self.env.get_reachable_objects()
    
    def _get_state_observation(self, state: namo_rl.RLState) -> Dict[str, List[float]]:
        """Get SE(2) pose observation from given state."""
        self.env.set_full_state(state)
        return self.env.get_observation()
    
    
    def _execute_action(self, state: namo_rl.RLState, action: Action) -> namo_rl.RLState:
        """Execute action and return resulting state."""
        self.env.set_full_state(state)
        self.env.step(action.to_namo_action())
        new_state = self.env.get_full_state()
        
        if self.config.collect_stats:
            self.stats['nodes_expanded'] += 1
        
        return new_state
    
    def _random_sampling_search(self, start_state: namo_rl.RLState, 
                               start_time: float, 
                               timeout_seconds: Optional[float]) -> Optional[Tuple[List[Action], List[Dict[str, List[float]]], List[Dict[str, List[float]]]]]:
        """Perform single-stream random sampling search."""
        
        current_state = start_state
        action_sequence = []
        state_sequence = []
        post_action_state_sequence = []
        current_depth = 0
        
        while current_depth < self.config.max_depth:
            # Check timeout
            if timeout_seconds and (time.time() - start_time) > timeout_seconds:
                if self.config.verbose:
                    print(f"Search timed out after {timeout_seconds}s")
                break
            
            # Update max depth reached
            if self.config.collect_stats:
                self.stats['max_depth_reached'] = max(
                    self.stats['max_depth_reached'], current_depth
                )
            
            # Check if current state is terminal
            if self._is_terminal_state(current_state):
                if self.config.verbose:
                    print(f"Solution found at depth {current_depth}")
                return (action_sequence, state_sequence, post_action_state_sequence)
            
            # Get reachable objects from current state
            reachable_objects = self._get_reachable_objects(current_state)
            if not reachable_objects:
                if self.config.verbose:
                    print(f"No reachable objects at depth {current_depth}")
                break
            
            # Select object using configured strategy
            selected_objects = self.object_selection_strategy.select_objects(
                reachable_objects, current_state, self.env
            )
            if not selected_objects:
                if self.config.verbose:
                    print(f"Object selection strategy returned no objects at depth {current_depth}")
                break
            
            # For random sampling, just take the first object from strategy's ordered list
            object_id = selected_objects[0]
            
            # Generate goals for this object using configured strategy
            goals = self.goal_selection_strategy.generate_goals(
                object_id, current_state, self.env, max_goals=1
            )
            if not goals:
                if self.config.verbose:
                    print(f"Goal selection strategy generated no goals for {object_id} at depth {current_depth}")
                break
            
            # Convert from goal_selection_strategy.Goal to our Goal class
            strategy_goal = goals[0]
            goal = Goal(x=strategy_goal.x, y=strategy_goal.y, theta=strategy_goal.theta)
            
            action = Action(object_id=object_id, goal=goal)
            
            # Get state observation BEFORE executing the action
            state_obs = self._get_state_observation(current_state)
            
            # Execute action to get new state
            try:
                current_state = self._execute_action(current_state, action)
                # Get state observation AFTER executing the action
                post_action_state_obs = self._get_state_observation(current_state)
                
                action_sequence.append(action)
                state_sequence.append(state_obs)
                post_action_state_sequence.append(post_action_state_obs)
                current_depth += 1
                
                if self.config.verbose:
                    print(f"Depth {current_depth}: moved {object_id} to ({goal.x:.2f}, {goal.y:.2f})")
                
            except Exception as e:
                if self.config.verbose:
                    print(f"Action execution failed at depth {current_depth}: {e}")
                break
        
        # Update final max depth reached
        if self.config.collect_stats:
            self.stats['max_depth_reached'] = max(
                self.stats['max_depth_reached'], current_depth
            )
        
        return None  # No solution found


# Register the planner with the factory
from namo.core import PlannerFactory
PlannerFactory.register_planner("random_sampling", RandomSamplingPlanner)


# Convenience function
def plan_with_random_sampling(env: namo_rl.RLEnvironment,
                             robot_goal: Tuple[float, float, float],
                             max_depth: int = 5,
                             object_strategy: str = "no_heuristic",
                             goal_strategy: str = "random",
                             random_seed: Optional[int] = None,
                             verbose: bool = False,
                             collect_stats: bool = True) -> Optional[List[namo_rl.Action]]:
    """Plan action sequence using random sampling with configurable strategies.
    
    Args:
        env: NAMO environment
        robot_goal: Target robot position (x, y, theta)
        max_depth: Maximum search depth
        object_strategy: Object selection strategy ("no_heuristic", "nearest_first", "goal_proximity", "farthest_first", "ml")
        goal_strategy: Goal selection strategy ("random", "grid", "adaptive", "ml")
        random_seed: Random seed for reproducibility
        verbose: Enable verbose output
        collect_stats: Collect algorithm statistics
        
    Returns:
        List of actions if solution found, None otherwise
    """
    
    config = PlannerConfig(
        max_depth=max_depth,
        max_goals_per_object=1,  # Always 1 for random sampling
        random_seed=random_seed,
        verbose=verbose,
        collect_stats=collect_stats,
        algorithm_params={
            'object_selection_strategy': object_strategy,
            'goal_selection_strategy': goal_strategy
        }
    )
    
    planner = RandomSamplingPlanner(env, config)
    result = planner.search(robot_goal)
    
    return result.action_sequence if result.solution_found else None