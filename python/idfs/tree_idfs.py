"""Tree-Maintained Iterative Deepening DFS for NAMO planning.

This implementation maintains a persistent tree structure across depth iterations,
avoiding re-sampling of goals and re-execution of actions for previously explored nodes.

Key differences from standard IDFS:
1. Persistent tree structure with cached children
2. Each ObjectNode samples goals exactly once
3. No re-execution of env.step() for cached states
4. Progressive deepening of the SAME tree
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
    """Tree-IDFS Action representation."""
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


class TreeStateNode:
    """Persistent state node that caches its object children."""
    
    def __init__(self, state: namo_rl.RLState, action_taken: Optional[Action] = None, 
                 parent: Optional['TreeObjectNode'] = None, depth: int = 0):
        self.state = state
        self.action_taken = action_taken
        self.parent = parent
        self.depth = depth
        
        # Persistent caching
        self.object_children: Dict[str, 'TreeObjectNode'] = {}
        self.reachable_objects_cache: Optional[List[str]] = None
        self.is_terminal_cache: Optional[bool] = None
        
        # Expansion tracking
        self.is_expanded = False
    
    def get_reachable_objects(self, env: namo_rl.RLEnvironment) -> List[str]:
        """Get cached list of reachable objects from this state."""
        if self.reachable_objects_cache is None:
            env.set_full_state(self.state)
            self.reachable_objects_cache = env.get_reachable_objects()
        return self.reachable_objects_cache
    
    def is_terminal(self, env: namo_rl.RLEnvironment) -> bool:
        """Check if robot goal is reachable (cached)."""
        
        if self.is_terminal_cache is None:
            env.set_full_state(self.state)
            self.is_terminal_cache = env.is_robot_goal_reachable()
        return self.is_terminal_cache
    
    def get_object_children(self, env: namo_rl.RLEnvironment) -> List['TreeObjectNode']:
        """Get cached object children for all reachable objects."""
        if not self.is_expanded:
            reachable_objects = self.get_reachable_objects(env)
            
            for obj_id in reachable_objects:
                if obj_id not in self.object_children:
                    self.object_children[obj_id] = TreeObjectNode(
                        self.state, obj_id, parent=self, depth=self.depth
                    )
            
            self.is_expanded = True
        
        return list(self.object_children.values())
    
    def reconstruct_path(self) -> List[Action]:
        """Reconstruct action sequence from root to this node."""
        path = []
        current = self
        
        while current.parent is not None:
            if current.action_taken is not None:
                path.insert(0, current.action_taken)
            current = current.parent.parent  # ObjectNode -> StateNode
        
        return path


class TreeObjectNode:
    """Persistent object node that caches its goal-based state children."""
    
    def __init__(self, parent_state: namo_rl.RLState, object_id: str, 
                 parent: TreeStateNode, depth: int = 0):
        self.parent_state = parent_state
        self.object_id = object_id
        self.parent = parent
        self.depth = depth
        
        # Persistent caching
        self.state_children: List[TreeStateNode] = []
        self.sampled_goals: List[Goal] = []
        
        # Expansion tracking
        self.is_expanded = False
    
    def is_terminal(self, env: namo_rl.RLEnvironment) -> bool:
        """Terminal if parent state is terminal."""
        return self.parent.is_terminal(env)
    
    def sample_goal(self, env: namo_rl.RLEnvironment, 
                   constraints: ActionConstraints) -> Optional[Goal]:
        """Sample a random goal for this object using same method as IDFS."""
        env.set_full_state(self.parent_state)
        obs = env.get_observation()
        
        # Get object position
        pose_key = f"{self.object_id}_pose"
        if pose_key not in obs:
            return None
        
        obj_x, obj_y = obs[pose_key][0], obs[pose_key][1]
        
        # Sample from continuous action space using polar coordinates
        distance = random.uniform(constraints.min_distance, constraints.max_distance)
        theta = random.uniform(constraints.theta_min, constraints.theta_max)
        
        target_x = obj_x + distance * math.cos(theta)
        target_y = obj_y + distance * math.sin(theta)
        
        return Goal(x=target_x, y=target_y, theta=theta)
    
    def get_state_children(self, env: namo_rl.RLEnvironment, 
                          constraints: ActionConstraints, 
                          num_goals: int,
                          stats: Optional[Dict] = None) -> List[TreeStateNode]:
        """Get cached state children by sampling and executing goals ONCE."""
        if not self.is_expanded:
            # Sample goals once and cache results
            for _ in range(num_goals):
                goal = self.sample_goal(env, constraints)
                if not goal:
                    continue
                
                # Store the goal for reproducibility
                self.sampled_goals.append(goal)
                
                action = Action(object_id=self.object_id, goal=goal)
                
                # Execute action to get new state (EXPENSIVE - done only once)
                env.set_full_state(self.parent_state)
                env.step(action.to_namo_action())
                new_state = env.get_full_state()
                
                # Create persistent child StateNode
                child_state_node = TreeStateNode(
                    new_state, action_taken=action, 
                    parent=self, depth=self.depth + 1
                )
                self.state_children.append(child_state_node)
                
                # Track statistics
                if stats is not None:
                    stats['nodes_expanded'] = stats.get('nodes_expanded', 0) + 1
                    stats['max_depth_reached'] = max(
                        stats.get('max_depth_reached', 0), self.depth + 1
                    )
            
            self.is_expanded = True
        
        return self.state_children


class TreeIterativeDeepeningDFS(BasePlanner):
    """Tree-Maintained Iterative Deepening DFS planner.
    
    Maintains persistent tree structure across depth iterations to avoid
    re-sampling goals and re-executing actions for previously explored nodes.
    """
    
    def __init__(self, env: namo_rl.RLEnvironment, config: PlannerConfig):
        super().__init__(env, config)
        
        # Tree state
        self.root: Optional[TreeStateNode] = None
        self.stats = {
            'nodes_expanded': 0,
            'terminal_checks': 0,
            'max_depth_reached': 0,
        } if self.config.collect_stats else {}
    
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
        self.root = None
        self.stats = {
            'nodes_expanded': 0,
            'terminal_checks': 0,
            'max_depth_reached': 0,
        } if self.config.collect_stats else {}
    
    @property
    def algorithm_name(self) -> str:
        return "Tree-Maintained Iterative Deepening DFS"
    
    @property
    def algorithm_version(self) -> str:
        return "tree_idfs_v1.0"
    
    def search(self, robot_goal: Tuple[float, float, float]) -> PlannerResult:
        """Run tree-maintained iterative deepening DFS to find action sequence."""
        start_time = time.time()
        
        # Set robot goal
        self.env.set_robot_goal(*robot_goal)
        
        # Initialize root if not exists
        if self.root is None:
            root_state = self.env.get_full_state()
            self.root = TreeStateNode(state=root_state, depth=0)
        
        if self.config.verbose:
            print(f"Starting tree-maintained IDFS (depths 1-{self.config.max_depth})")
        
        # Iterative deepening: try each depth limit
        for depth_limit in range(1, self.config.max_depth + 1):
            if self.config.verbose:
                print(f"Trying depth limit: {depth_limit}")
            
            # Check if root is already terminal
            is_terminal = self.root.is_terminal(self.env)
            if is_terminal:
                if self.config.verbose:
                    print("Goal already reachable at root!")
                if self.config.collect_stats:
                    self.stats['terminal_checks'] += 1
                
                search_time_ms = (time.time() - start_time) * 1000
                return PlannerResult(
                    success=True,
                    solution_found=True,
                    action_sequence=[],
                    solution_depth=0,
                    search_time_ms=search_time_ms,
                    nodes_expanded=self.stats.get('nodes_expanded', 0),
                    terminal_checks=self.stats.get('terminal_checks', 0),
                    max_depth_reached=self.stats.get('max_depth_reached', 0),
                    algorithm_stats={'tree_maintained': True}
                )
            
            # Depth-limited DFS on persistent tree
            solution_path = self._depth_limited_dfs(self.root, depth_limit)
            if solution_path is not None:
                if self.config.verbose:
                    actions_str = [f'{a.object_id}->({a.goal.x:.2f},{a.goal.y:.2f})' for a in solution_path]
                    print(f"Solution found at depth {len(solution_path)}: {actions_str}")
                
                # Convert to namo_rl.Action list
                namo_actions = [action.to_namo_action() for action in solution_path]
                
                search_time_ms = (time.time() - start_time) * 1000
                return PlannerResult(
                    success=True,
                    solution_found=True,
                    action_sequence=namo_actions,
                    solution_depth=len(solution_path),
                    search_time_ms=search_time_ms,
                    nodes_expanded=self.stats.get('nodes_expanded', 0),
                    terminal_checks=self.stats.get('terminal_checks', 0),
                    max_depth_reached=self.stats.get('max_depth_reached', 0),
                    algorithm_stats={'tree_maintained': True}
                )
        
        if self.config.verbose:
            print("No solution found within depth limits")
        
        search_time_ms = (time.time() - start_time) * 1000
        return PlannerResult(
            success=True,
            solution_found=False,
            search_time_ms=search_time_ms,
            nodes_expanded=self.stats.get('nodes_expanded', 0),
            terminal_checks=self.stats.get('terminal_checks', 0),
            max_depth_reached=self.stats.get('max_depth_reached', 0),
            algorithm_stats={'tree_maintained': True}
        )
    
    def _depth_limited_dfs(self, node: TreeStateNode, depth_limit: int) -> Optional[List[Action]]:
        """Perform depth-limited DFS on persistent tree structure."""
        # Check if current state is terminal
        if node.is_terminal_cache is None:
            if self.config.collect_stats:
                self.stats['terminal_checks'] += 1
        is_terminal = node.is_terminal(self.env)
        
        if is_terminal:
            return []  # Empty action sequence - goal already reachable
        
        # Check depth limit
        if node.depth >= depth_limit:
            return None  # Depth limit reached
        
        # Expand StateNode -> ObjectNodes (cached)
        object_children = node.get_object_children(self.env)
        
        for obj_node in object_children:
            # Expand ObjectNode -> StateNodes (cached after first expansion)
            state_children = obj_node.get_state_children(
                self.env, self.constraints, self.config.max_goals_per_object, 
                self.stats if self.config.collect_stats else None
            )
            
            for state_child in state_children:
                # Recursively search from this state
                sub_solution = self._depth_limited_dfs(state_child, depth_limit)
                
                if sub_solution is not None:
                    # Found solution! Reconstruct full path
                    return state_child.reconstruct_path()
        
        return None  # No solution found at this branch


# Register the planner with the factory
from idfs.base_planner import PlannerFactory
PlannerFactory.register_planner("tree_idfs", TreeIterativeDeepeningDFS)


# Convenience function for backward compatibility
def plan_with_tree_idfs(env: namo_rl.RLEnvironment,
                       robot_goal: Tuple[float, float, float],
                       max_depth: int = 5,
                       max_goals_per_object: int = 5,
                       random_seed: Optional[int] = None,
                       verbose: bool = False,
                       collect_stats: bool = True) -> Optional[List[namo_rl.Action]]:
    """Plan action sequence using tree-maintained iterative deepening DFS."""
    
    config = PlannerConfig(
        max_depth=max_depth,
        max_goals_per_object=max_goals_per_object,
        random_seed=random_seed,
        verbose=verbose,
        collect_stats=collect_stats
    )
    
    planner = TreeIterativeDeepeningDFS(env, config)
    result = planner.search(robot_goal)
    
    return result.action_sequence if result.solution_found else None