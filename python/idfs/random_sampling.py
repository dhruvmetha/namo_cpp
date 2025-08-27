"""Simple Random Sampling Planner for NAMO planning.

This implementation provides a baseline single-stream random sampling approach:
1. Sample random object from reachable objects
2. Sample 1 goal for that object  
3. Execute action and repeat up to max-depth
4. Report episode results

Key characteristics:
- No iterative deepening (single stream to max-depth)
- No object selection strategy (pure random sampling)  
- Fresh random sampling at each step
- Simple baseline for comparison with more sophisticated algorithms
"""

import math
import random
import time
from typing import List, Optional, Tuple, Dict
from dataclasses import dataclass

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import namo_rl
from mcts_config import ActionConstraints
from idfs.base_planner import BasePlanner, PlannerConfig, PlannerResult


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
    """Simple Random Sampling planner for NAMO planning.
    
    This planner provides a baseline approach that samples random objects
    and goals in a single stream until max-depth is reached or solution found.
    No iterative deepening or sophisticated object selection strategies.
    """
    
    def __init__(self, env: namo_rl.RLEnvironment, config: PlannerConfig):
        super().__init__(env, config)
        
        # Statistics tracking
        self.stats = {}
    
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
        return "Simple Random Sampling"
    
    @property
    def algorithm_version(self) -> str:
        return "random_sampling_v1.0"
    
    def search(self, robot_goal: Tuple[float, float, float]) -> PlannerResult:
        """Run single-stream random sampling to find action sequence."""
        start_time = time.time()
        timeout_seconds = self.config.max_search_time_seconds
        
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
            
            # Sample random object (key difference from IDFS - pure random)
            object_id = random.choice(reachable_objects)
            
            # Sample goal for this object
            goal = self._sample_goal_for_object(current_state, object_id)
            if not goal:
                if self.config.verbose:
                    print(f"Failed to sample goal for {object_id} at depth {current_depth}")
                break
            
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
from idfs.base_planner import PlannerFactory
PlannerFactory.register_planner("random_sampling", RandomSamplingPlanner)


# Convenience function
def plan_with_random_sampling(env: namo_rl.RLEnvironment,
                             robot_goal: Tuple[float, float, float],
                             max_depth: int = 5,
                             random_seed: Optional[int] = None,
                             verbose: bool = False,
                             collect_stats: bool = True) -> Optional[List[namo_rl.Action]]:
    """Plan action sequence using simple random sampling."""
    
    config = PlannerConfig(
        max_depth=max_depth,
        max_goals_per_object=1,  # Always 1 for random sampling
        random_seed=random_seed,
        verbose=verbose,
        collect_stats=collect_stats
    )
    
    planner = RandomSamplingPlanner(env, config)
    result = planner.search(robot_goal)
    
    return result.action_sequence if result.solution_found else None