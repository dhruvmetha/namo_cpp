"""MCTS with Progressive Widening for NAMO planning."""

import math
import random
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

import namo_rl
from mcts_config import MCTSConfig, ActionConstraints

# Rich library for tree visualization
try:
    from rich.tree import Tree
    from rich.console import Console
    from rich.live import Live
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False
    print("Warning: rich library not available. Tree visualization disabled.")


@dataclass
class MCTSNode:
    """MCTS tree node."""
    state: namo_rl.RLState
    action: Optional[namo_rl.Action] = None
    parent: Optional['MCTSNode'] = None
    children: List['MCTSNode'] = None
    visit_count: int = 0
    q_value: float = 0.0
    available_actions: List[namo_rl.Action] = None
    
    def __post_init__(self):
        if self.children is None:
            self.children = []
        if self.available_actions is None:
            self.available_actions = []
    
    def is_fully_expanded(self, config: MCTSConfig) -> bool:
        """Check if node has reached progressive widening limit."""
        # Ensure at least 1 child is allowed for unvisited nodes
        max_children = max(1, int(config.c_pw * (self.visit_count ** config.alpha)))
        return len(self.children) >= max_children
    
    def ucb1_value(self, config: MCTSConfig) -> float:
        """Calculate UCB1 value for action selection."""
        if self.visit_count == 0:
            return float('inf')
        
        # Avoid log(0) by ensuring parent has at least 1 visit
        if self.parent is None or self.parent.visit_count == 0:
            return float('inf')
        
        exploitation = self.q_value / self.visit_count
        exploration = config.c_exploration * math.sqrt(
            math.log(self.parent.visit_count) / self.visit_count
        )
        return exploitation + exploration


class MCTS:
    """MCTS with Progressive Widening for NAMO."""
    
    def __init__(self, env: namo_rl.RLEnvironment, config: MCTSConfig):
        self.env = env
        self.config = config
        self.constraints = self._get_action_constraints()
    
    def _get_action_constraints(self) -> ActionConstraints:
        """Get action constraints from environment."""
        env_constraints = self.env.get_action_constraints()
        return ActionConstraints(
            min_distance=env_constraints.min_distance,
            max_distance=env_constraints.max_distance,
            theta_min=env_constraints.theta_min,
            theta_max=env_constraints.theta_max
        )
    
    def search(self, robot_goal: Tuple[float, float, float], visualize_tree: bool = False) -> Optional[namo_rl.Action]:
        """Run MCTS search to find best action.
        
        Args:
            robot_goal: Target robot position (x, y, theta)
            visualize_tree: Whether to show live tree visualization
            
        Returns:
            Best action or None if no solution found
        """
        if visualize_tree:
            return self._search_with_visualization(robot_goal)
        else:
            return self._search_without_visualization(robot_goal)
    
    def _search_without_visualization(self, robot_goal: Tuple[float, float, float]) -> Optional[namo_rl.Action]:
        """Run MCTS search without visualization."""
        # Set robot goal in environment
        self.env.set_robot_goal(*robot_goal)
        
        # Initialize root node
        root_state = self.env.get_full_state()
        root = MCTSNode(state=root_state)
        
        # Run simulations
        for _ in range(self.config.simulation_budget):
            self._simulate(root)
        
        # Return best action
        if not root.children:
            return None
        
        best_child = max(root.children, key=lambda c: c.visit_count)
        return best_child.action
    
    def _search_with_visualization(self, robot_goal: Tuple[float, float, float]) -> Optional[namo_rl.Action]:
        """Run MCTS search with live tree visualization."""
        if not RICH_AVAILABLE:
            print("Rich library not available. Running without visualization.")
            return self._search_without_visualization(robot_goal)
        
        # Set robot goal in environment
        self.env.set_robot_goal(*robot_goal)
        
        # Initialize root node
        root_state = self.env.get_full_state()
        root = MCTSNode(state=root_state)
        
        console = Console()
        
        with Live(console=console, refresh_per_second=4) as live:
            # Run simulations with live updates
            for i in range(self.config.simulation_budget):
                self._simulate(root)
                
                # Update visualization every 10 iterations
                if i % 10 == 0:
                    tree_display = self._create_tree_display(root, i)
                    live.update(tree_display)
        
        # Show final tree
        final_tree = self._create_tree_display(root, self.config.simulation_budget, final=True)
        console.print(final_tree)
        
        # Return best action
        if not root.children:
            return None
        
        best_child = max(root.children, key=lambda c: c.visit_count)
        return best_child.action
    
    def _create_tree_display(self, root: MCTSNode, iteration: int, final: bool = False) -> Tree:
        """Create rich tree display for MCTS search tree."""
        title = f"🌳 MCTS Search Tree {'(Final)' if final else f'- Iteration {iteration}'}"
        tree = Tree(title)
        
        # Add root node info
        root_info = f"🏠 ROOT (Visits: {root.visit_count}, Q: {root.q_value:.2f})"
        root_branch = tree.add(root_info)
        
        # Add children recursively
        self._add_children_to_tree(root_branch, root, max_depth=3)
        
        return tree
    
    def _add_children_to_tree(self, rich_node, mcts_node: MCTSNode, max_depth: int = 3):
        """Recursively add MCTS children to rich tree display."""
        if max_depth <= 0 or not mcts_node.children:
            return
        
        # Sort children by visit count (most visited first)
        sorted_children = sorted(mcts_node.children, key=lambda c: c.visit_count, reverse=True)
        
        for i, child in enumerate(sorted_children[:5]):  # Show top 5 children
            # Create child label
            action_str = child.action.object_id if child.action else "UNKNOWN"
            ucb_value = child.ucb1_value(self.config) if child.parent else 0.0
            
            # Add emoji based on visit count rank
            emoji = "🥇" if i == 0 else "🥈" if i == 1 else "🥉" if i == 2 else "📍"
            
            label = (f"{emoji} {action_str} "
                    f"(V:{child.visit_count}, Q:{child.q_value:.2f}, UCB:{ucb_value:.2f})")
            
            child_rich = rich_node.add(label)
            self._add_children_to_tree(child_rich, child, max_depth - 1)
    
    def _simulate(self, node: MCTSNode):
        """Single MCTS simulation: select -> expand -> rollout -> backpropagate."""
        path = []
        current = node
        
        # Selection: traverse tree using UCB1
        while current.children and current.is_fully_expanded(self.config):
            current = max(current.children, key=lambda c: c.ucb1_value(self.config))
            path.append(current)
            
        # Expansion: add new child if under progressive widening limit
        if not current.is_fully_expanded(self.config):
            self.env.set_full_state(current.state)
            action = self._generate_action()
            if action:
                child_state = self._execute_action(action)
                child = MCTSNode(
                    state=child_state,
                    action=action,
                    parent=current
                )
                current.children.append(child)
                path.append(child)
                current = child
        
        # Rollout: random simulation from current node
        self.env.set_full_state(current.state)
        reward = self._rollout()
        
        # Backpropagation: update Q-values along path
        for node_in_path in reversed(path):
            node_in_path.visit_count += 1
            node_in_path.q_value += reward
    
    def _generate_action(self) -> Optional[namo_rl.Action]:
        """Generate random valid action."""
        reachable_objects = self.env.get_reachable_objects()
        if not reachable_objects:
            return None
        
        # Pick random object
        object_name = random.choice(reachable_objects)
        
        # Sample target position
        distance = random.uniform(self.constraints.min_distance, self.constraints.max_distance)
        theta = random.uniform(self.constraints.theta_min, self.constraints.theta_max)
        
        # Get object position and compute target
        obs = self.env.get_observation()
        pose_key = f"{object_name}_pose"
        if pose_key not in obs:
            return None
        
        obj_x, obj_y = obs[pose_key][0], obs[pose_key][1]
        target_x = obj_x + distance * math.cos(theta)
        target_y = obj_y + distance * math.sin(theta)
        
        action = namo_rl.Action()
        action.object_id = object_name
        action.x = target_x
        action.y = target_y
        action.theta = theta
        return action
    
    def _execute_action(self, action: namo_rl.Action) -> namo_rl.RLState:
        """Execute action and return resulting state."""
        self.env.step(action)
        return self.env.get_full_state()
    
    def _rollout(self) -> float:
        """Random rollout from current state."""
        for _ in range(self.config.max_rollout_steps):
            if self.env.is_robot_goal_reachable():
                # print("Goal reached")
                return 1.0
            
            action = self._generate_action()
            if not action:
                break
            
            result = self.env.step(action)
            # print(result.reward, result.done, result.info)
            if result.reward > 0:  # Goal reached
                return 1.0
        
        return -1.0  # Goal not reached


def plan_with_mcts(env: namo_rl.RLEnvironment, 
                   robot_goal: Tuple[float, float, float],
                   config: MCTSConfig = None,
                   visualize_tree: bool = False) -> Optional[namo_rl.Action]:
    """Plan single action using MCTS.
    
    Args:
        env: NAMO RL environment
        robot_goal: Target robot position (x, y, theta)
        config: MCTS configuration (uses defaults if None)
        visualize_tree: Whether to show live tree visualization
        
    Returns:
        Best action or None if no solution found
    """
    if config is None:
        config = MCTSConfig()
    
    mcts = MCTS(env, config)
    return mcts.search(robot_goal, visualize_tree=visualize_tree)