"""Optimal Iterative Deepening DFS that finds all minimum-depth solutions.

This implementation extends the standard IDFS to find ALL solutions at the
minimum depth, with intelligent pruning to avoid exploring branches that
exceed the best solution depth found so far.
"""

import math
import random
import time
from typing import List, Optional, Tuple, Dict
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from namo.strategies.object_selection_strategy import ObjectSelectionStrategy

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import namo_rl
from namo.config.mcts_config import ActionConstraints
from namo.core import BasePlanner, PlannerConfig, PlannerResult
from namo.strategies.object_selection_strategy import ObjectSelectionStrategy, NoHeuristicStrategy, NearestFirstStrategy, GoalProximityStrategy, FarthestFirstStrategy
from namo.strategies.goal_selection_strategy import GoalSelectionStrategy, RandomGoalStrategy

from dataclasses import dataclass

@dataclass
class Goal:
    """Push goal representation."""
    x: float
    y: float
    theta: float


@dataclass
class Action:
    """IDFS Action representation."""
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


class OptimalIterativeDeepeningDFS(BasePlanner):
    """Optimal Iterative Deepening DFS planner that finds all minimum-depth solutions.
    
    This planner extends the standard IDFS by:
    1. Continuing search after finding the first solution
    2. Pruning branches that exceed the best solution depth found so far
    3. Returning all solutions at the minimum depth
    4. Providing access to all minimum solutions via get_all_minimum_solutions()
    """
    
    def __init__(self, env: namo_rl.RLEnvironment, config: PlannerConfig,
                 object_selection_strategy: Optional['ObjectSelectionStrategy'] = None,
                 goal_selection_strategy: Optional['GoalSelectionStrategy'] = None,
                 visualize_search: bool = False,
                 search_delay: float = 0.5,
                 step_mode: bool = False):
        
        # Initialize strategy attributes first (before super().__init__ calls _setup_constraints)
        self.object_selection_strategy = None
        self.goal_selection_strategy = None
        
        # Visualization parameters
        self.visualize_search = visualize_search
        self.search_delay = search_delay
        self.step_mode = step_mode
        
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
            # Default to no heuristic
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
                    print("Falling back to default random goal generation")
                # Set to None so _setup_constraints will create default strategy
                self.goal_selection_strategy = None
        else:
            # Default to random goal generation with environment constraints
            self.goal_selection_strategy = None  # Will create after constraints are set
        
        # Statistics tracking
        self.stats = {
            'nodes_expanded': 0,
            'terminal_checks': 0,
            'max_depth_reached': 0,
        } if self.config.collect_stats else {}
        
        # Store all minimum solutions from last search
        self.last_all_minimum_solutions = []
    
    def _create_object_strategy_from_name(self, strategy_name: str) -> 'ObjectSelectionStrategy':
        """Create object strategy instance from string name."""
        if strategy_name == 'no_heuristic':
            return NoHeuristicStrategy()
        elif strategy_name == 'nearest_first':
            return NearestFirstStrategy()
        elif strategy_name == 'goal_proximity':
            return GoalProximityStrategy()
        elif strategy_name == 'farthest_first':
            return FarthestFirstStrategy()
        elif strategy_name == 'ml':
            return self._create_ml_object_strategy()
        else:
            available = ['no_heuristic', 'nearest_first', 'goal_proximity', 'farthest_first', 'ml']
            raise ValueError(f"Unknown object strategy '{strategy_name}'. Available: {available}")
    
    def _create_goal_strategy_from_name(self, strategy_name: str) -> 'GoalSelectionStrategy':
        """Create goal strategy instance from string name."""
        if strategy_name == 'random':
            return RandomGoalStrategy(
                min_distance=self.constraints.min_distance,
                max_distance=self.constraints.max_distance,
                theta_min=self.constraints.theta_min,
                theta_max=self.constraints.theta_max
            )
        elif strategy_name == 'grid':
            GridGoalStrategy = self._import_and_create('idfs.goal_selection_strategy', 'GridGoalStrategy')
            return GridGoalStrategy()
        elif strategy_name == 'adaptive':
            AdaptiveGoalStrategy = self._import_and_create('idfs.goal_selection_strategy', 'AdaptiveGoalStrategy')
            return AdaptiveGoalStrategy()
        elif strategy_name == 'ml':
            return self._create_ml_goal_strategy()
        else:
            available = ['random', 'grid', 'adaptive', 'ml']
            raise ValueError(f"Unknown goal strategy '{strategy_name}'. Available: {available}")
    
    def _import_and_create(self, module_name: str, class_name: str):
        """Helper to dynamically import and return strategy class."""
        import importlib
        module = importlib.import_module(module_name)
        return getattr(module, class_name)
    
    def _create_ml_object_strategy(self) -> 'ObjectSelectionStrategy':
        """Create ML object selection strategy from config parameters."""
        params = self.config.algorithm_params or {}
        
        object_model_path = params.get('ml_object_model_path')
        if not object_model_path:
            raise ValueError("ML object strategy requires 'ml_object_model_path' in algorithm_params")
        
        # Import ML strategies
        MLObjectSelectionStrategy = self._import_and_create('idfs.ml_strategies', 'MLObjectSelectionStrategy')
        
        # Use preloaded model if available
        preloaded_model = params.get('preloaded_object_model')
        return MLObjectSelectionStrategy(
            object_model_path=object_model_path,
            samples=params.get('ml_samples', 32),
            device= params.get('ml_device', 'cuda'),
            xml_path_relative=params.get('xml_file'),
            verbose=self.config.verbose,
            preloaded_model=preloaded_model
        )
    
    def _create_ml_goal_strategy(self) -> 'GoalSelectionStrategy':
        """Create ML goal selection strategy from config parameters."""
        params = self.config.algorithm_params or {}
        
        goal_model_path = params.get('ml_goal_model_path')
        
        if not goal_model_path:
            raise ValueError("ML goal strategy requires 'ml_goal_model_path' in algorithm_params")
        
        # Check if epsilon is specified for epsilon-greedy strategy
        epsilon = params.get('epsilon')
        
        if epsilon is not None:
            # Create epsilon-greedy strategy
            from namo.strategies.ml_strategies import EpsilonGreedyGoalStrategy, MLGoalSelectionStrategy
            from namo.strategies.goal_selection_strategy import RandomGoalStrategy
            
            # Create ML strategy
            preloaded_model = params.get('preloaded_goal_model')
            ml_strategy = MLGoalSelectionStrategy(
                goal_model_path=goal_model_path,
                samples=params.get('ml_samples', 32),
                device=params.get('ml_device', 'cuda'),
                xml_path_relative=params.get('xml_file'),
                verbose=self.config.verbose,
                preloaded_model=preloaded_model
            )
            
            # Create random strategy with same constraints
            random_strategy = RandomGoalStrategy(
                min_distance=self.constraints.min_distance,
                max_distance=self.constraints.max_distance,
                theta_min=self.constraints.theta_min,
                theta_max=self.constraints.theta_max
            )
            
            if self.config.verbose:
                print(f"Using epsilon-greedy goal strategy with epsilon={epsilon}")
            
            return EpsilonGreedyGoalStrategy(
                ml_strategy=ml_strategy,
                random_strategy=random_strategy,
                epsilon=epsilon,
                verbose=self.config.verbose
            )
        
        else:
            # Create pure ML strategy (original behavior)
            MLGoalSelectionStrategy = self._import_and_create('idfs.ml_strategies', 'MLGoalSelectionStrategy')
            
            if self.config.verbose:
                print("ml samples", params.get('ml_samples', 32))
            # Use preloaded model if available
            preloaded_model = params.get('preloaded_goal_model')
            
            return MLGoalSelectionStrategy(
                goal_model_path=goal_model_path,
                samples=params.get('ml_samples', 32),
                device=params.get('ml_device', 'cuda'),
                xml_path_relative=params.get('xml_file'),
                verbose=self.config.verbose,
                preloaded_model=preloaded_model
            )
    
    def _setup_constraints(self):
        """Setup action constraints from environment."""
        env_constraints = self.env.get_action_constraints()
        self.constraints = ActionConstraints(
            min_distance=env_constraints.min_distance,
            max_distance=env_constraints.max_distance,
            theta_min=env_constraints.theta_min,
            theta_max=env_constraints.theta_max
        )
        
        # Setup default goal selection strategy if not provided
        if self.goal_selection_strategy is None:
            self.goal_selection_strategy = RandomGoalStrategy(
                min_distance=self.constraints.min_distance,
                max_distance=self.constraints.max_distance,
                theta_min=self.constraints.theta_min,
                theta_max=self.constraints.theta_max
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
        
        # Reset solution tracking
        self.last_all_minimum_solutions = []
    
    @property
    def algorithm_name(self) -> str:
        return f"Optimal Iterative Deepening DFS ({self.object_selection_strategy.strategy_name}, {self.goal_selection_strategy.strategy_name})"
    
    @property
    def algorithm_version(self) -> str:
        return "optimal_idfs_v1.0"
    
    def _visualize_search_state(self, state: namo_rl.RLState, depth: int, message: str):
        """Visualize current state during search process."""
        if not self.visualize_search:
            return
        
        # Set environment to the state we want to visualize
        self.env.set_full_state(state)
        
        # Print debug message with indentation for depth
        indent = "  " * depth
        print(f"{indent}🔍 Depth {depth}: {message}")
        
        if self.step_mode:
            # Manual step-through mode
            input(f"{indent}Press Enter to continue...")
        else:
            # Automatic mode with delay
            self.env.render()
            if self.search_delay > 0:
                time.sleep(self.search_delay)
    
    def search(self, robot_goal: Tuple[float, float, float]) -> PlannerResult:
        """Run optimal iterative deepening DFS to find all minimum-depth solutions."""
        start_time = time.time()
        timeout_seconds = self.config.max_search_time_seconds
        
        # Set robot goal
        self.env.set_robot_goal(*robot_goal)
        
        if self.config.verbose:
            print(f"Starting optimal IDFS (depths 1-{self.config.max_depth}) - finding all minimum solutions")
        
        if self.visualize_search and self.step_mode:
            print("👆 STEP MODE: Press Enter to advance through search steps")
        
        all_minimum_solutions = []  # Store all solutions at minimum depth
        best_depth = None  # Track the best (minimum) solution depth found
        
        # Iterative deepening: try each depth limit
        for depth_limit in range(1, self.config.max_depth + 1):
            # Check timeout before starting new depth
            if timeout_seconds and (time.time() - start_time) > timeout_seconds:
                if self.config.verbose:
                    print(f"Search timed out after {timeout_seconds}s")
                break
            
            # If we already found solutions and this depth exceeds the best depth, stop
            if best_depth is not None and depth_limit > best_depth:
                if self.config.verbose:
                    print(f"Stopping search: depth {depth_limit} > best depth {best_depth}")
                break
                
            if self.config.verbose:
                print(f"Trying depth limit: {depth_limit}")
            
            # Check if root is already terminal
            root_state = self.env.get_full_state()
            
            # Visualize initial search state
            self._visualize_search_state(root_state, 0, f"Starting search from root (depth limit {depth_limit})")
            
            if self._is_terminal_state(root_state):
                if self.config.verbose:
                    print("Goal already reachable at root!")
                if self.config.collect_stats:
                    self.stats['terminal_checks'] += 1
                
                # Root is terminal - this is the best possible solution (depth 0)
                search_time_ms = (time.time() - start_time) * 1000
                self.last_all_minimum_solutions = [([], [], [])]  # Store for access
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
                    algorithm_stats={'optimal_search': True, 'num_minimum_solutions': 1}
                )
            
            # Depth-limited DFS (restarting from root each time)
            solutions_at_depth = self._depth_limited_dfs(root_state, depth_limit, 0, start_time, timeout_seconds, best_depth)
            
            # Process solutions found at this depth
            for solution_actions, solution_states, post_action_states in solutions_at_depth:
                solution_depth = len(solution_actions)
                
                # Update best depth if this is the first solution or better than previous
                if best_depth is None or solution_depth < best_depth:
                    # Found solutions at a better (shallower) depth
                    best_depth = solution_depth
                    all_minimum_solutions = [(solution_actions, solution_states, post_action_states)]
                    if self.config.verbose:
                        print(f"New minimum depth found: {best_depth}")
                elif solution_depth == best_depth:
                    # Found another solution at the same minimum depth
                    all_minimum_solutions.append((solution_actions, solution_states, post_action_states))
                    if self.config.verbose:
                        print(f"Additional solution found at minimum depth {best_depth}")
                
                if self.config.verbose:
                    actions_str = [f'{a.object_id}->({a.goal.x:.2f},{a.goal.y:.2f})' for a in solution_actions]
                    print(f"Solution {len(all_minimum_solutions)} found at depth {solution_depth}: {actions_str}")
        
        # Store all minimum solutions for potential access later
        self.last_all_minimum_solutions = all_minimum_solutions
        
        # Return results
        if all_minimum_solutions:
            if self.config.verbose:
                print(f"Search completed: found {len(all_minimum_solutions)} solutions at minimum depth {best_depth}")
            
            # Return the first minimum solution (could be extended to return all)
            solution_actions, solution_states, post_action_states = all_minimum_solutions[0]
            
            # Convert to namo_rl.Action list
            namo_actions = [action.to_namo_action() for action in solution_actions]
            
            search_time_ms = (time.time() - start_time) * 1000
            return PlannerResult(
                success=True,
                solution_found=True,
                action_sequence=namo_actions,
                solution_depth=best_depth,
                state_observations=solution_states,
                post_action_state_observations=post_action_states,
                search_time_ms=search_time_ms,
                nodes_expanded=self.stats.get('nodes_expanded', 0),
                terminal_checks=self.stats.get('terminal_checks', 0),
                max_depth_reached=self.stats.get('max_depth_reached', 0),
                algorithm_stats={'optimal_search': True, 'num_minimum_solutions': len(all_minimum_solutions)}
            )
        else:
            if self.config.verbose:
                print("No solution found within depth limits")
            
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
                algorithm_stats={'optimal_search': True, 'num_minimum_solutions': 0}
            )
    
    def get_all_minimum_solutions(self) -> List[Tuple[List[namo_rl.Action], List[Dict[str, List[float]]], List[Dict[str, List[float]]]]]:
        """Return all solutions found at the minimum depth from the last search.
        
        Returns:
            List of tuples, each containing:
            - action_sequence: List of namo_rl.Action objects
            - state_observations: List of state observations 
            - post_action_state_observations: List of post-action state observations
        """
        result = []
        for solution_actions, solution_states, post_action_states in self.last_all_minimum_solutions:
            namo_actions = [action.to_namo_action() for action in solution_actions]
            result.append((namo_actions, solution_states, post_action_states))
        return result
    
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
    
    def _sample_goal_for_object(self, state: namo_rl.RLState, object_id: str) -> Optional[Goal]:
        """Sample a random goal for the given object."""
        self.env.set_full_state(state)
        obs = self.env.get_observation()
        
        # Get object position
        pose_key = f"{object_id}_pose"
        if pose_key not in obs:
            return None
        
        obj_x, obj_y = obs[pose_key][0], obs[pose_key][1]
        
        # Sample from continuous action space using polar coordinates
        distance = random.uniform(self.constraints.min_distance, self.constraints.max_distance)
        theta = random.uniform(self.constraints.theta_min, self.constraints.theta_max)
        
        target_x = obj_x + distance * math.cos(theta)
        target_y = obj_y + distance * math.sin(theta)
        
        return Goal(x=target_x, y=target_y, theta=theta)
    
    def _execute_action(self, state: namo_rl.RLState, action: Action) -> namo_rl.RLState:
        """Execute action and return resulting state."""
        self.env.set_full_state(state)
        self.env.step(action.to_namo_action())
        new_state = self.env.get_full_state()
        
        if self.config.collect_stats:
            self.stats['nodes_expanded'] += 1
        
        return new_state
    
    def _depth_limited_dfs(self, state: namo_rl.RLState, depth_limit: int, 
                          current_depth: int, start_time: float, 
                          timeout_seconds: Optional[float], best_depth: Optional[int] = None) -> List[Tuple[List[Action], List[Dict[str, List[float]]], List[Dict[str, List[float]]]]]:
        """Perform depth-limited DFS from given state, returning all solutions at minimum depth."""
        
        # Check timeout
        if timeout_seconds and (time.time() - start_time) > timeout_seconds:
            return []  # Timeout reached
        
        # Prune if current depth already exceeds best solution depth found so far
        if best_depth is not None and current_depth >= best_depth:
            self._visualize_search_state(state, current_depth, f"✂️ Pruning: depth {current_depth} >= best depth {best_depth}")
            return []  # Prune this branch
        
        # Update max depth reached
        if self.config.collect_stats:
            self.stats['max_depth_reached'] = max(
                self.stats['max_depth_reached'], current_depth
            )
        
        # Visualize current state exploration
        self._visualize_search_state(state, current_depth, "Exploring state (checking if terminal)")
        
        # Check if current state is terminal
        if self._is_terminal_state(state):
            self._visualize_search_state(state, current_depth, "🎉 GOAL REACHABLE! Terminal state found")
            return [([], [], [])]  # Return list with one empty solution
        
        # Check depth limit
        if current_depth >= depth_limit:
            self._visualize_search_state(state, current_depth, f"❌ Depth limit {depth_limit} reached")
            return []  # Depth limit reached
        
        # Get reachable objects from current state
        reachable_objects = self._get_reachable_objects(state)
        self._visualize_search_state(state, current_depth, f"Found {len(reachable_objects)} reachable objects: {reachable_objects}")
        
        # Apply object selection strategy to order objects
        ordered_objects = self.object_selection_strategy.select_objects(
            reachable_objects, state, self.env
        )
        
        all_solutions = []  # Collect all solutions found at this level
        
        # Try each reachable object in strategy-determined order
        for obj_idx, object_id in enumerate(ordered_objects):
            # Check timeout before trying each object
            if timeout_seconds and (time.time() - start_time) > timeout_seconds:
                return all_solutions  # Return solutions found so far
            
            self._visualize_search_state(state, current_depth, f"Trying object {obj_idx+1}/{len(ordered_objects)}: {object_id}")
            
            # Generate goals for this object using goal selection strategy
            goals = self.goal_selection_strategy.generate_goals(
                object_id, state, self.env, self.config.max_goals_per_object
            )
            
            # Try each generated goal
            for goal_idx, goal in enumerate(goals):
                action = Action(object_id=object_id, goal=goal)
                
                self._visualize_search_state(state, current_depth, 
                    f"Trying goal {goal_idx+1}/{len(goals)} for {object_id}: ({goal.x:.2f},{goal.y:.2f})")
                
                # Get state observation BEFORE executing the action
                state_obs = self._get_state_observation(state)
                
                # Execute action to get new state
                try:
                    new_state = self._execute_action(state, action)
                    # Get state observation AFTER executing the action
                    post_action_state_obs = self._get_state_observation(new_state)
                    
                    self._visualize_search_state(new_state, current_depth, 
                        f"✅ Action executed: {object_id} -> ({goal.x:.2f},{goal.y:.2f})")
                    
                except Exception as e:
                    self._visualize_search_state(state, current_depth, f"❌ Action failed: {e}")
                    continue  # Skip failed actions
                
                # Check timeout before expensive recursive call
                if timeout_seconds and (time.time() - start_time) > timeout_seconds:
                    return all_solutions  # Return solutions found so far
                
                # Recursively search from new state
                sub_solutions = self._depth_limited_dfs(new_state, depth_limit, current_depth + 1, start_time, timeout_seconds, best_depth)
                
                # Check if we got timeout vs no solutions found
                if timeout_seconds and (time.time() - start_time) > timeout_seconds:
                    return all_solutions  # Return solutions found so far
                
                # Process all solutions found in recursive call
                for sub_actions, sub_states, sub_post_action_states in sub_solutions:
                    self._visualize_search_state(state, current_depth, f"🎉 Solution found via {object_id}!")
                    # Prepend current action, state observation, and post-action state observation
                    complete_solution = ([action] + sub_actions, [state_obs] + sub_states, [post_action_state_obs] + sub_post_action_states)
                    all_solutions.append(complete_solution)
                
                if not sub_solutions:
                    self._visualize_search_state(state, current_depth, f"⏪ Backtracking from {object_id} goal {goal_idx+1}")
        
        if not all_solutions:
            self._visualize_search_state(state, current_depth, f"❌ No solution found at depth {current_depth}")
        else:
            self._visualize_search_state(state, current_depth, f"✅ Found {len(all_solutions)} solutions at depth {current_depth}")
        
        return all_solutions


# Import dataclass after other imports to avoid circular dependencies
from dataclasses import dataclass

# Register the planner with the factory
from namo.core import PlannerFactory
PlannerFactory.register_planner("optimal_idfs", OptimalIterativeDeepeningDFS)


# Convenience function for optimal IDFS
def plan_with_optimal_idfs(env: namo_rl.RLEnvironment,
                          robot_goal: Tuple[float, float, float],
                          max_depth: int = 5,
                          max_goals_per_object: int = 5,
                          random_seed: Optional[int] = None,
                          verbose: bool = False,
                          collect_stats: bool = True,
                          object_selection_strategy: Optional['ObjectSelectionStrategy'] = None,
                          goal_selection_strategy: Optional['GoalSelectionStrategy'] = None) -> Optional[List[namo_rl.Action]]:
    """Plan action sequence using optimal iterative deepening DFS (finds all minimum solutions)."""
    
    config = PlannerConfig(
        max_depth=max_depth,
        max_goals_per_object=max_goals_per_object,
        random_seed=random_seed,
        verbose=verbose,
        collect_stats=collect_stats
    )
    
    planner = OptimalIterativeDeepeningDFS(env, config, object_selection_strategy, goal_selection_strategy)
    result = planner.search(robot_goal)
    
    return result.action_sequence if result.solution_found else None