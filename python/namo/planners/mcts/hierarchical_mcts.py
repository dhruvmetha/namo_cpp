"""Clean 2-Level Hierarchical MCTS for NAMO planning.

This implementation uses a clean 2-level hierarchy:
- StateNode: Environment states (simulation targets)
- ObjectNode: Decision nodes for object selection (NEVER simulated from)

Key principles:
1. ObjectNodes are pure aggregation/selection nodes
2. Only StateNodes (post-action) are simulation targets  
3. Progressive Widening only at ObjectNode level (goal selection)
4. Clean MCTS semantics: simulate only from executed action results

References:
- Coulom, R. (2006). Efficient Selectivity and Backup Operators in Monte-Carlo Tree Search
- Browne et al. (2012). A Survey of Monte Carlo Tree Search Methods
"""

import math
import random
from typing import List, Optional, Tuple, Dict
from dataclasses import dataclass
from abc import ABC, abstractmethod

import namo_rl
from namo.config.mcts_config import MCTSConfig, ActionConstraints

# Import object selection strategies from IDFS
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from namo.strategies.object_selection_strategy import (
    ObjectSelectionStrategy, NoHeuristicStrategy, NearestFirstStrategy, 
    GoalProximityStrategy, FarthestFirstStrategy
)

# Rich library for tree visualization
try:
    from rich.tree import Tree
    from rich.console import Console
    from rich.live import Live
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False


@dataclass
class Goal:
    """Push goal representation."""
    x: float
    y: float
    theta: float


@dataclass
class Action:
    """MCTS Action representation."""
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


class MCTSNode(ABC):
    """Abstract base class for all MCTS nodes."""
    
    def __init__(self, parent: Optional['MCTSNode'] = None):
        self.parent = parent
        self.visit_count = 0
        self.reward_sum = 0.0
    
    @property
    def q_value(self) -> float:
        """Average reward (Q-value) for this node."""
        return self.reward_sum / self.visit_count if self.visit_count > 0 else 0.0
    
    def ucb1_value(self, config: MCTSConfig) -> float:
        """Calculate UCB1 value for node selection."""
        if self.visit_count == 0:
            return float('inf')
        
        if self.parent is None or self.parent.visit_count == 0:
            return float('inf')
        
        exploitation = self.q_value
        exploration = config.c_exploration * math.sqrt(
            math.log(self.parent.visit_count) / self.visit_count
        )
        return exploitation + exploration
    
    def update_statistics(self, reward: float):
        """Update visit count and reward sum."""
        self.visit_count += 1
        self.reward_sum += reward
    
    @abstractmethod
    def is_terminal(self, env: 'namo_rl.RLEnvironment') -> bool:
        """Check if this node represents a terminal state."""
        pass
    
    @abstractmethod
    def can_expand(self, config: MCTSConfig, env: 'namo_rl.RLEnvironment') -> bool:
        """Check if this node can be expanded."""
        pass


class StateNode(MCTSNode):
    """MCTS node representing a full environment state.
    
    These are the ONLY nodes that can be simulation targets.
    Children are ObjectNodes (decision nodes for object selection).
    """
    
    def __init__(self, state: namo_rl.RLState, action_taken: Optional[Action] = None, parent: Optional['ObjectNode'] = None):
        super().__init__(parent)
        self.state = state
        self.action_taken = action_taken  # Action that led to this state (None for root)
        self.object_children: Dict[str, 'ObjectNode'] = {}
        self._reachable_objects_cache: Optional[List[str]] = None
    
    def get_reachable_objects(self, env: 'namo_rl.RLEnvironment') -> List[str]:
        """Get list of reachable objects from this state."""
        if self._reachable_objects_cache is None:
            env.set_full_state(self.state)
            self._reachable_objects_cache = env.get_reachable_objects()
        return self._reachable_objects_cache
    
    def is_terminal(self, env: 'namo_rl.RLEnvironment') -> bool:
        """Check if robot goal is reachable (terminal condition)."""
        env.set_full_state(self.state)
        return env.is_robot_goal_reachable()
    
    def can_expand(self, config: MCTSConfig, env: 'namo_rl.RLEnvironment') -> bool:
        """Check if we can create more ObjectNode children."""
        if self.is_terminal(env):
            return False
        
        reachable_objects = self.get_reachable_objects(env)
        if not reachable_objects:
            return False
        
        # Deterministic expansion: one ObjectNode per reachable object
        return len(self.object_children) < len(reachable_objects)
    
    def expand(self, env: 'namo_rl.RLEnvironment', 
               object_selection_strategy: Optional[ObjectSelectionStrategy] = None) -> Optional['ObjectNode']:
        """Create a new ObjectNode child for an untried reachable object."""
        reachable_objects = self.get_reachable_objects(env)
        
        if object_selection_strategy is not None:
            # Use strategy to order objects
            ordered_objects = object_selection_strategy.select_objects(
                reachable_objects, self.state, env
            )
        else:
            # Default order (original behavior)
            ordered_objects = reachable_objects
        
        # Find first ordered object without an ObjectNode
        for obj_id in ordered_objects:
            if obj_id not in self.object_children:
                obj_node = ObjectNode(self.state, obj_id, parent=self)
                self.object_children[obj_id] = obj_node
                return obj_node
        
        return None
    
    def select_best_child(self, config: MCTSConfig) -> Optional['ObjectNode']:
        """Select best ObjectNode child using UCB1."""
        if not self.object_children:
            return None
        
        max_value = max(node.ucb1_value(config) for node in self.object_children.values())
        best_nodes = [node for node in self.object_children.values() if node.ucb1_value(config) == max_value]
        return random.choice(best_nodes)
    
        # return max(self.object_children.values(), key=lambda node: node.ucb1_value(config))
    
    def is_fully_expanded(self, env: 'namo_rl.RLEnvironment') -> bool:
        """Check if all reachable objects have ObjectNode children."""
        reachable_objects = self.get_reachable_objects(env)
        return len(self.object_children) >= len(reachable_objects)


class ObjectNode(MCTSNode):
    """MCTS node representing object selection within a state.
    
    CRITICAL: These nodes are NEVER simulation targets!
    They are pure decision/aggregation nodes that only get Q-values 
    through backpropagation from their children (post-action StateNodes).
    
    Children are StateNodes (resulting from executed push actions).
    Progressive Widening is applied here for goal selection.
    """
    
    def __init__(self, parent_state: namo_rl.RLState, object_id: str, parent: StateNode):
        super().__init__(parent)
        self.parent_state = parent_state
        self.object_id = object_id
        self.goal_children: List[StateNode] = []  # Post-action StateNodes
    
    def is_terminal(self, env: 'namo_rl.RLEnvironment') -> bool:
        """Terminal if parent state is terminal."""
        return self.parent.is_terminal(env)
    
    def can_expand(self, config: MCTSConfig, env: 'namo_rl.RLEnvironment') -> bool:
        """Check if we can create more goal children using Progressive Widening."""
        if self.is_terminal(env):
            return False
        
        # Progressive Widening formula: k * N^α
        max_children = max(1, int(config.k * (self.visit_count ** config.alpha)))
        return len(self.goal_children) < max_children
    
    def sample_goal(self, env: 'namo_rl.RLEnvironment', 
                   constraints: ActionConstraints) -> Optional[Goal]:
        """Sample a random goal for this object from continuous action space."""
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
    
    def expand(self, env: 'namo_rl.RLEnvironment', 
              constraints: ActionConstraints) -> Optional[StateNode]:
        """Create a new post-action StateNode child by executing a sampled goal."""
        goal = self.sample_goal(env, constraints)
        if not goal:
            return None
        
        action = Action(object_id=self.object_id, goal=goal)
        
        # Execute action to get new state
        env.set_full_state(self.parent_state)
        env.step(action.to_namo_action())
        new_state = env.get_full_state()
        
        # Create post-action StateNode
        child_state_node = StateNode(new_state, action_taken=action, parent=self)
        self.goal_children.append(child_state_node)
        return child_state_node
    
    def select_best_child(self, config: MCTSConfig) -> Optional[StateNode]:
        """Select best goal child (StateNode) using UCB1."""
        if not self.goal_children:
            return None
        return max(self.goal_children, 
                  key=lambda node: node.ucb1_value(config))
    
    def is_fully_expanded(self, config: MCTSConfig, env: 'namo_rl.RLEnvironment') -> bool:
        """Check if we've reached the Progressive Widening limit."""
        if self.is_terminal(env):
            return True
        
        max_children = max(1, int(config.k * (self.visit_count ** config.alpha)))
        return len(self.goal_children) >= max_children


class CleanHierarchicalMCTS:
    """Clean 2-level Hierarchical MCTS with proper simulation semantics.
    
    Key design principles:
    1. StateNode → ObjectNode → StateNode (2-level hierarchy)
    2. Only StateNodes are simulation targets (post-action states)
    3. ObjectNodes are pure aggregation/decision nodes
    4. Progressive Widening only for goal selection (continuous space)
    """
    
    def __init__(self, env: namo_rl.RLEnvironment, config: MCTSConfig, verbose: bool = False,
                 object_selection_strategy: Optional[ObjectSelectionStrategy] = None):
        self.env = env
        self.config = config
        self.verbose = verbose
        self.constraints = self._get_action_constraints()
        
        # Set up object selection strategy
        if object_selection_strategy is not None:
            self.object_selection_strategy = object_selection_strategy
        else:
            self.object_selection_strategy = self._create_object_strategy_from_name(
                config.object_selection_strategy
            )
        
        # Store robot goal for reward calculations
        self.robot_goal: Optional[Tuple[float, float, float]] = None
    
    def _get_action_constraints(self) -> ActionConstraints:
        """Get action constraints from environment."""
        env_constraints = self.env.get_action_constraints()
        return ActionConstraints(
            min_distance=env_constraints.min_distance,
            max_distance=env_constraints.max_distance,
            theta_min=env_constraints.theta_min,
            theta_max=env_constraints.theta_max
        )
    
    def _create_object_strategy_from_name(self, strategy_name: str) -> ObjectSelectionStrategy:
        """Create object selection strategy from string name."""
        if strategy_name == 'no_heuristic':
            return NoHeuristicStrategy()
        elif strategy_name == 'nearest_first':
            return NearestFirstStrategy()
        elif strategy_name == 'goal_proximity':
            return GoalProximityStrategy()
        elif strategy_name == 'farthest_first':
            return FarthestFirstStrategy()
        else:
            available = ['no_heuristic', 'nearest_first', 'goal_proximity', 'farthest_first']
            raise ValueError(f"Unknown object strategy '{strategy_name}'. Available: {available}")
    
    def _get_min_distance_to_goal(self, reachable_objects: List[str]) -> float:
        """Calculate minimum distance from any reachable object to robot goal.
        
        Args:
            reachable_objects: List of object IDs that are reachable
            
        Returns:
            Minimum distance from any reachable object to goal, or infinity if no objects
        """
        if not reachable_objects or self.robot_goal is None:
            return float('inf')
        
        obs = self.env.get_observation()
        goal_x, goal_y = self.robot_goal[0], self.robot_goal[1]
        
        min_distance = float('inf')
        for obj_id in reachable_objects:
            pose_key = f"{obj_id}_pose"
            if pose_key in obs:
                obj_x, obj_y = obs[pose_key][0], obs[pose_key][1]
                distance = math.sqrt((obj_x - goal_x)**2 + (obj_y - goal_y)**2)
                min_distance = min(min_distance, distance)
        
        return min_distance
    
    def search(self, robot_goal: Tuple[float, float, float], 
              visualize_tree: bool = False) -> Optional[Action]:
        """Run MCTS search to find best action."""
        if visualize_tree:
            return self._search_with_visualization(robot_goal)
        else:
            return self._search_without_visualization(robot_goal)
    
    def search_with_root_access(self, robot_goal: Tuple[float, float, float], 
                               visualize_tree: bool = False) -> Tuple[Optional[Action], StateNode]:
        """Run MCTS search and return both best action and root node for data extraction."""
        # Set robot goal
        self.robot_goal = robot_goal
        self.env.set_robot_goal(*robot_goal)
        
        # Initialize root state node
        root_state = self.env.get_full_state()
        root = StateNode(state=root_state)
        
        if visualize_tree and RICH_AVAILABLE:
            # Run with visualization
            from rich.console import Console
            from rich.live import Live
            console = Console()
            
            with Live(console=console, refresh_per_second=2) as live:
                for iteration in range(self.config.simulation_budget):
                    self._mcts_iteration(root)
                    
                    if iteration % 10 == 0:
                        tree_display = self._create_tree_display(root, iteration)
                        live.update(tree_display)
            
            # Show final tree
            final_tree = self._create_tree_display(root, self.config.simulation_budget, final=True)
            console.print(final_tree)
        else:
            # Run without visualization
            for iteration in range(self.config.simulation_budget):
                self._mcts_iteration(root)
        
        # Select best action and return both action and root
        best_action = self._select_best_action(root)
        return best_action, root
    
    def _search_without_visualization(self, robot_goal: Tuple[float, float, float]) -> Optional[Action]:
        """Run MCTS search without visualization."""
        # Set robot goal
        self.robot_goal = robot_goal
        self.env.set_robot_goal(*robot_goal)
        
        # Initialize root state node
        root_state = self.env.get_full_state()
        root = StateNode(state=root_state)
        
        # Run MCTS iterations
        for iteration in range(self.config.simulation_budget):
            self._mcts_iteration(root)
        
        # Select best action
        return self._select_best_action(root)
    
    def _search_with_visualization(self, robot_goal: Tuple[float, float, float]) -> Optional[Action]:
        """Run MCTS search with live tree visualization."""
        if not RICH_AVAILABLE:
            return self._search_without_visualization(robot_goal)
        
        # Set robot goal
        self.robot_goal = robot_goal
        self.env.set_robot_goal(*robot_goal)
        
        # Initialize root state node
        root_state = self.env.get_full_state()
        root = StateNode(state=root_state)
        
        console = Console()
        
        with Live(console=console, refresh_per_second=2) as live:
            # Run MCTS iterations with live updates
            for iteration in range(self.config.simulation_budget):
                self._mcts_iteration(root)
                
                # Update visualization every 10 iterations
                if iteration % 10 == 0:
                    tree_display = self._create_tree_display(root, iteration)
                    live.update(tree_display)
        
        # Show final tree
        final_tree = self._create_tree_display(root, self.config.simulation_budget, final=True)
        console.print(final_tree)
        
        # Select best action
        return self._select_best_action(root)
    
    def _mcts_iteration(self, root: StateNode):
        """Single MCTS iteration with clean 2-level semantics:
        Selection → Expansion → Simulation → Backpropagation
        
        CRITICAL: Only simulate from StateNodes (post-action states)
        """
        # Phase 1: Selection - traverse tree to find expandable node
        leaf_node, path = self._select(root)
        
        # Phase 2: Expansion - ensure we end with a post-action StateNode
        while leaf_node.can_expand(self.config, self.env):
            expanded_node = self._expand(leaf_node)
            if expanded_node:
                path.append(expanded_node)
                leaf_node = expanded_node
                
                # If we created a StateNode (post-action), stop expanding
                if isinstance(leaf_node, StateNode) and leaf_node.action_taken:
                    break
            else:
                break
        
        # Phase 3: Simulation - ONLY from StateNodes (post-action states)
        if isinstance(leaf_node, StateNode):
            reward = self._simulate(leaf_node)
        else:
            # Should never happen with proper expansion logic
            reward = 0.0
        
        # Phase 4: Backpropagation - update statistics along path
        self._backpropagate(path, reward)
    
    def _select(self, root: StateNode) -> Tuple[MCTSNode, List[MCTSNode]]:
        """Selection phase: traverse tree using UCB1 until reaching expandable node."""
        path = [root]
        node = root
        
        while True:
            # If node can expand, stop here for expansion
            if node.can_expand(self.config, self.env):
                break
            
            # If terminal, stop here
            if node.is_terminal(self.env):
                break
            
            # Navigate through hierarchy based on node type
            next_node = None
            
            if isinstance(node, StateNode):
                if node.object_children:
                    next_node = node.select_best_child(self.config)
                
            elif isinstance(node, ObjectNode):
                if node.goal_children:
                    next_node = node.select_best_child(self.config)
            
            if next_node is None:
                break
            
            path.append(next_node)
            node = next_node
        
        return node, path
    
    def _expand(self, node: MCTSNode) -> Optional[MCTSNode]:
        """Expansion phase: add new child based on node type."""
        if isinstance(node, StateNode):
            return node.expand(self.env, self.object_selection_strategy)
        elif isinstance(node, ObjectNode):
            return node.expand(self.env, self.constraints)
        return None
    
    def _simulate(self, node: StateNode) -> float:
        """Simulation phase: random rollout from StateNode (post-action state).
        
        CRITICAL: Only called with StateNodes - never ObjectNodes!
        """
        # Set environment to node's state
        self.env.set_full_state(node.state)
        
        # Check if already at goal
        if self.env.is_robot_goal_reachable():
            return 1.0
        
        reward = [-0.5]
        
        # Perform random rollout
        for ct in range(self.config.max_rollout_steps):
            # if self.env.is_robot_goal_reachable():
            #     return 1.0
            
            # Sample random action
            reachable_objects_before = self.env.get_reachable_objects()
            if not reachable_objects_before:
                reward.append(-10.0)
                break
            
            # Calculate minimum distance to goal from current reachable set
            min_dist_before = self._get_min_distance_to_goal(reachable_objects_before)
            
            # Use object selection strategy instead of random selection
            ordered_objects = self.object_selection_strategy.select_objects(
                reachable_objects_before, self.env.get_full_state(), self.env
            )
            # Take first object from strategy ordering (could add randomness here if desired)
            obj_id = ordered_objects[0] if ordered_objects else random.choice(reachable_objects_before)
            
            # Random goal sampling
            obs = self.env.get_observation()
            pose_key = f"{obj_id}_pose"
            if pose_key not in obs:
                reward.append(-10.0)
                break
            
            obj_x, obj_y = obs[pose_key][0], obs[pose_key][1]
            distance = random.uniform(self.constraints.min_distance, self.constraints.max_distance)
            theta = random.uniform(self.constraints.theta_min, self.constraints.theta_max)
            
            target_x = obj_x + distance * math.cos(theta)
            target_y = obj_y + distance * math.sin(theta)
            
            # Execute action
            action = namo_rl.Action()
            action.object_id = obj_id
            action.x = target_x
            action.y = target_y
            action.theta = theta
            
            result = self.env.step(action)
            
            if result.reward > 0:  # Goal reached
                reward.append(1.0)
                break
            else:
                # Check if we opened access to new objects closer to goal
                reachable_objects_after = self.env.get_reachable_objects()
                
                # Find newly reachable objects (not in the previous set)
                newly_reachable = [obj for obj in reachable_objects_after 
                                 if obj not in reachable_objects_before]
                
                step_reward = -0.5  # Default negative reward for push
                
                if newly_reachable:
                    # Calculate minimum distance from newly reachable objects to goal
                    min_dist_new_objects = self._get_min_distance_to_goal(newly_reachable)
                    
                    # Check if any newly reachable object is closer than all previously reachable ones
                    if min_dist_new_objects < min_dist_before:
                        # Found a newly reachable object closer to goal!
                        step_reward = -0.1 + self.config.reachability_reward  # Net positive reward
                        if self.config.verbose:
                            print(f"  🎯 New reachable object closer to goal! "
                                  f"Previous best: {min_dist_before:.3f}, "
                                  f"New best: {min_dist_new_objects:.3f} (+{self.config.reachability_reward})")
                            print(f"    Newly reachable: {newly_reachable}")
                
                if ct == self.config.max_rollout_steps - 1:
                    step_reward = min(step_reward, -1.0)  # Final step penalty overrides if worse
                    
                reward.append(step_reward)
        
        ret = 0.0
        for r in reward[::-1]:
            ret = ret * 0.99 + r
        return ret
    
    def _backpropagate(self, path: List[MCTSNode], reward: float):
        """Backpropagation phase: update statistics for all nodes in path.
        
        ObjectNodes get Q-values purely through aggregation from children.
        """
        for node in path:
            node.update_statistics(reward)
    
    def _select_best_action(self, root: StateNode) -> Optional[Action]:
        """Select best action from root using robust child selection."""
        if not root.object_children:
            return None
        
        # Find best object (highest visit count)
        best_object_node = max(root.object_children.values(), 
                              key=lambda node: node.visit_count)
        
        if not best_object_node.goal_children:
            return None
        
        # Find best goal for that object (highest visit count)
        best_goal_state = max(best_object_node.goal_children, 
                             key=lambda node: node.visit_count)
        
        return best_goal_state.action_taken
    
    def _create_tree_display(self, root: StateNode, iteration: int, final: bool = False) -> Tree:
        """Create rich tree display for clean hierarchical MCTS tree."""
        title = f"🌳 Clean 2-Level MCTS {'(Final)' if final else f'- Iteration {iteration}'}"
        tree = Tree(title)
        
        # Add root state info
        self._add_state_node_to_tree(tree, root, "🏠 ROOT", max_depth=3)
        
        return tree
    
    def _add_state_node_to_tree(self, parent_branch, state_node: StateNode, label_prefix: str, max_depth: int = 2):
        """Recursively add StateNode and its children to tree display."""
        if max_depth <= 0:
            return
        
        # Create StateNode label
        if state_node.action_taken:
            action_info = f" [Action: {state_node.action_taken.object_id} → ({state_node.action_taken.goal.x:.2f}, {state_node.action_taken.goal.y:.2f})]"
        else:
            action_info = ""
        
        state_info = f"{label_prefix} STATE (V:{state_node.visit_count}, Q:{state_node.q_value:.3f}, Objects:{len(state_node.object_children)}){action_info}"
        state_branch = parent_branch.add(state_info)
        
        # Add ObjectNode children
        sorted_objects = sorted(state_node.object_children.items(), 
                               key=lambda x: x[1].visit_count, reverse=True)
        
        for i, (obj_id, obj_node) in enumerate(sorted_objects[:4]):  # Show top 4 objects
            obj_emoji = "🥇" if i == 0 else "🥈" if i == 1 else "🥉" if i == 2 else "📦"
            obj_info = f"{obj_emoji} OBJ {obj_id} (V:{obj_node.visit_count}, Q:{obj_node.q_value:.3f}, Goals:{len(obj_node.goal_children)})"
            obj_branch = state_branch.add(obj_info)
            
            # Add goal StateNode children (post-action states)
            sorted_goals = sorted(obj_node.goal_children, 
                                key=lambda x: x.visit_count, reverse=True)
            
            for j, goal_state in enumerate(sorted_goals[:3]):  # Show top 3 goals per object
                goal_emoji = "🎯" if j == 0 else "📍"
                
                # Check if this goal state has further expansions
                has_children = len(goal_state.object_children) > 0
                child_indicator = f" (+{len(goal_state.object_children)} objs)" if has_children else ""
                
                if goal_state.action_taken:
                    goal_info = f"{goal_emoji} GOAL ({goal_state.action_taken.goal.x:.2f}, {goal_state.action_taken.goal.y:.2f}) (V:{goal_state.visit_count}, Q:{goal_state.q_value:.3f}){child_indicator}"
                else:
                    goal_info = f"{goal_emoji} GOAL (unknown) (V:{goal_state.visit_count}, Q:{goal_state.q_value:.3f}){child_indicator}"
                
                goal_branch = obj_branch.add(goal_info)
                
                # Recursively add deeper levels if they exist
                if has_children and max_depth > 1:
                    self._add_state_node_to_tree(goal_branch, goal_state, "🔄 NEXT", max_depth - 1)


def plan_with_clean_hierarchical_mcts(env: namo_rl.RLEnvironment,
                                     robot_goal: Tuple[float, float, float],
                                     config: MCTSConfig = None,
                                     visualize_tree: bool = False,
                                     verbose: bool = False,
                                     object_selection_strategy: Optional[ObjectSelectionStrategy] = None) -> Optional[namo_rl.Action]:
    """Plan single action using clean 2-level hierarchical MCTS.
    
    Args:
        env: NAMO RL environment
        robot_goal: Target robot position (x, y, theta)
        config: MCTS configuration (uses defaults if None)
        visualize_tree: Whether to show live tree visualization
        verbose: Whether to print debug information
        
    Returns:
        Best NAMO action or None if no solution found
    """
    if config is None:
        config = MCTSConfig()
    
    mcts = CleanHierarchicalMCTS(env, config, verbose=verbose, 
                                object_selection_strategy=object_selection_strategy)
    action = mcts.search(robot_goal, visualize_tree=visualize_tree)
    
    # Convert to namo_rl.Action if found
    if action:
        return action.to_namo_action()
    return None