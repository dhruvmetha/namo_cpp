#!/usr/bin/env python3
"""
AlphaZero-style data collection from hierarchical MCTS for NAMO planning.

This module collects training data from MCTS tree search to train:
1. Object proposal policy: P(obj|root_state)  
2. Goal proposal policy: P(goal|obj, root_state)
3. Object Q-values: Q(root_state, obj)
4. Goal Q-values: Q(root_state, obj, goal) 
5. State value function: V(root_state)
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass, field
import json
import pickle
from pathlib import Path

import namo_rl
from namo.planners.mcts.hierarchical_mcts import CleanHierarchicalMCTS, StateNode, ObjectNode, Action

@dataclass
class ObjectProposal:
    """Object selection proposal with probability."""
    object_id: str
    probability: float
    visit_count: int
    q_value: float


@dataclass 
class GoalProposal:
    """Goal proposal for a specific object."""
    goal_position: Tuple[float, float, float]  # x, y, theta
    probability: float
    visit_count: int
    q_value: float


@dataclass
class MCTSTrainingData:
    """Complete training data extracted from one MCTS tree at a root state."""
    
    # State information
    root_state: namo_rl.RLState
    scene_observation: Dict[str, Any]  # Observable state (object poses, etc.)
    robot_goal: Tuple[float, float, float]
    
    # Ground truth labels from MCTS tree
    object_proposals: List[ObjectProposal]  # P(obj|state) supervision
    goal_proposals: Dict[str, List[GoalProposal]]  # P(goal|obj,state) per object
    
    # Q-value targets
    state_value: float  # V(state) from root node
    object_q_values: Dict[str, float]  # Q(state, obj) for each object
    goal_q_values: Dict[str, Dict[Tuple[float, float, float], float]]  # Q(state, obj, goal)
    
    # Metadata
    mcts_iterations: int
    step_in_episode: int
    reachable_objects: List[str]
    total_objects: int
    
    # Static environment information for mask generation (same format as IDFS)
    static_object_info: Dict[str, Dict[str, Any]] = field(default_factory=dict)


@dataclass
class EpisodeData:
    """Complete episode data from multi-step MCTS planning."""
    episode_id: str
    xml_file: str
    config_file: str
    robot_goal: Tuple[float, float, float]
    
    # Training data for each step/root state
    step_data: List[MCTSTrainingData] = field(default_factory=list)
    
    # Episode outcome
    success: bool = False
    total_steps: int = 0
    final_action_sequence: List[Dict[str, Any]] = field(default_factory=list)


class MCTSDataExtractor:
    """Extracts training data from hierarchical MCTS trees."""
    
    def __init__(self, top_k_goals: int = 3):
        """
        Args:
            top_k_goals: Number of top goals per object to collect Q-values for
        """
        self.top_k_goals = top_k_goals
    
    def extract_training_data(self, 
                            root_node: StateNode,
                            env: namo_rl.RLEnvironment,
                            robot_goal: Tuple[float, float, float],
                            mcts_iterations: int,
                            step_in_episode: int) -> MCTSTrainingData:
        """Extract all training data from MCTS root node after search completion."""
        
        # Get state information
        env.set_full_state(root_node.state)
        scene_obs = env.get_observation()
        # Note: robot_goal should be passed in since env may not have get_robot_goal method
        reachable_objects = env.get_reachable_objects()
        
        # Count total objects in scene
        total_objects = len([k for k in scene_obs.keys() if k.endswith('_pose')])
        
        # Extract object proposals (visit counts → probabilities)
        object_proposals = self._extract_object_proposals(root_node)
        
        # Extract goal proposals per object
        goal_proposals = self._extract_goal_proposals(root_node)
        
        # Extract Q-values
        state_value = root_node.q_value
        object_q_values = self._extract_object_q_values(root_node)
        goal_q_values = self._extract_goal_q_values(root_node)
        
        # Extract static environment information for mask generation (same as IDFS)
        try:
            static_object_info = env.get_object_info()
        except:
            static_object_info = {}

        return MCTSTrainingData(
            root_state=root_node.state,
            scene_observation=scene_obs,
            robot_goal=robot_goal,
            object_proposals=object_proposals,
            goal_proposals=goal_proposals,
            state_value=state_value,
            object_q_values=object_q_values,
            goal_q_values=goal_q_values,
            mcts_iterations=mcts_iterations,
            step_in_episode=step_in_episode,
            reachable_objects=reachable_objects,
            total_objects=total_objects,
            static_object_info=static_object_info
        )
    
    def _extract_object_proposals(self, root_node: StateNode) -> List[ObjectProposal]:
        """Extract object selection probabilities from ObjectNode visit counts."""
        if not root_node.object_children:
            return []
        
        # Get visit counts for all objects
        object_visits = []
        total_visits = sum(node.visit_count for node in root_node.object_children.values())
        
        if total_visits == 0:
            return []
        
        for obj_id, obj_node in root_node.object_children.items():
            probability = obj_node.visit_count / total_visits
            proposal = ObjectProposal(
                object_id=obj_id,
                probability=probability,
                visit_count=obj_node.visit_count,
                q_value=obj_node.q_value
            )
            object_visits.append(proposal)
        
        # Sort by visit count (descending)
        object_visits.sort(key=lambda x: x.visit_count, reverse=True)
        return object_visits
    
    def _extract_goal_proposals(self, root_node: StateNode) -> Dict[str, List[GoalProposal]]:
        """Extract TOP-K goal selection probabilities per object, renormalized to sum to 1."""
        goal_proposals = {}
        
        for obj_id, obj_node in root_node.object_children.items():
            if not obj_node.goal_children:
                goal_proposals[obj_id] = []
                continue
            
            # Collect all goal proposals with visit counts
            goal_visits = []
            for state_node in obj_node.goal_children:
                if state_node.action_taken:
                    goal_pos = (
                        state_node.action_taken.goal.x,
                        state_node.action_taken.goal.y, 
                        state_node.action_taken.goal.theta
                    )
                    goal_visits.append({
                        'position': goal_pos,
                        'visit_count': state_node.visit_count,
                        'q_value': state_node.q_value
                    })
            
            if not goal_visits:
                goal_proposals[obj_id] = []
                continue
            
            # Sort by visit count (descending) and take top-K
            goal_visits.sort(key=lambda x: x['visit_count'], reverse=True)
            top_k_goals = goal_visits[:self.top_k_goals]
            
            # Renormalize probabilities for top-K to sum to 1
            top_k_visits = sum(goal['visit_count'] for goal in top_k_goals)
            
            if top_k_visits == 0:
                goal_proposals[obj_id] = []
                continue
            
            top_k_proposals = []
            for goal in top_k_goals:
                probability = goal['visit_count'] / top_k_visits  # Renormalized probability
                proposal = GoalProposal(
                    goal_position=goal['position'],
                    probability=probability,
                    visit_count=goal['visit_count'],
                    q_value=goal['q_value']
                )
                top_k_proposals.append(proposal)
            
            goal_proposals[obj_id] = top_k_proposals
        
        return goal_proposals
    
    def _extract_object_q_values(self, root_node: StateNode) -> Dict[str, float]:
        """Extract Q(state, obj) for each object."""
        object_q_values = {}
        
        for obj_id, obj_node in root_node.object_children.items():
            object_q_values[obj_id] = obj_node.q_value
        
        return object_q_values
    
    def _extract_goal_q_values(self, root_node: StateNode) -> Dict[str, Dict[Tuple[float, float, float], float]]:
        """Extract Q(state, obj, goal) for ALL goals per object to learn good/bad differentiation."""
        goal_q_values = {}
        
        for obj_id, obj_node in root_node.object_children.items():
            if not obj_node.goal_children:
                goal_q_values[obj_id] = {}
                continue
            
            # Collect Q-values for ALL goals (not just top-K)
            obj_goal_q_values = {}
            for state_node in obj_node.goal_children:
                if state_node.action_taken:
                    goal_pos = (
                        state_node.action_taken.goal.x,
                        state_node.action_taken.goal.y,
                        state_node.action_taken.goal.theta
                    )
                    obj_goal_q_values[goal_pos] = state_node.q_value
            
            goal_q_values[obj_id] = obj_goal_q_values
        
        return goal_q_values


class SingleEnvironmentDataCollector:
    """Collects training data from single environment multi-step MCTS episodes."""
    
    def __init__(self, 
                 xml_file: str,
                 config_file: str,
                 data_extractor: MCTSDataExtractor,
                 output_dir: str = "training_data"):
        """
        Args:
            xml_file: MuJoCo XML environment file
            config_file: YAML configuration file
            data_extractor: MCTS tree data extraction utility
            output_dir: Directory to save collected data
        """
        self.xml_file = xml_file
        self.config_file = config_file
        self.data_extractor = data_extractor
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize environments (same pattern as test_multi_step_mcts.py)
        self.planning_env = None
        self.execution_env = None
        self._initialize_environments()
    
    def _initialize_environments(self):
        """Initialize planning and execution environments."""
        try:
            self.planning_env = namo_rl.RLEnvironment(self.xml_file, self.config_file, False)
            self.planning_env.reset()
            
            self.execution_env = namo_rl.RLEnvironment(self.xml_file, self.config_file, False)
            self.execution_env.reset()
            
        except Exception as e:
            raise RuntimeError(f"Failed to initialize environments: {e}")
    
    def collect_episode_data(self,
                           robot_goal: Tuple[float, float, float],
                           mcts_config,
                           max_steps: int = 10,
                           episode_id: Optional[str] = None,
                           save_json_summary: bool = True) -> EpisodeData:
        """Collect training data from a complete multi-step MCTS episode."""
        
        if episode_id is None:
            episode_id = f"episode_{len(list(self.output_dir.glob('episode_*.pkl')))}"
        
        # Initialize episode data
        episode_data = EpisodeData(
            episode_id=episode_id,
            xml_file=self.xml_file.split("models/")[1],
            config_file=self.config_file,
            robot_goal=robot_goal
        )
        
        # Reset environments and set goals
        self.planning_env.reset()
        self.execution_env.reset()
        self.planning_env.set_robot_goal(*robot_goal)
        self.execution_env.set_robot_goal(*robot_goal)
        
        # Check if already solved
        if self.execution_env.is_robot_goal_reachable():
            episode_data.success = True
            episode_data.total_steps = 0
            return episode_data
        
        
        # Multi-step planning loop with data collection
        step = 0
        action_sequence = []
        
        while step < max_steps:
            step += 1
            
            # Check termination
            if self.execution_env.is_robot_goal_reachable():
                episode_data.success = True
                break
            
            # Sync environments
            execution_state = self.execution_env.get_full_state()
            self.planning_env.set_full_state(execution_state)
            self.planning_env.set_robot_goal(*robot_goal)
            
            # Create MCTS instance and run search with root access
            mcts = CleanHierarchicalMCTS(self.planning_env, mcts_config, verbose=False)
            best_action, root_node = mcts.search_with_root_access(robot_goal, visualize_tree=False)
            
            # Extract training data from MCTS tree
            training_data = self.data_extractor.extract_training_data(
                root_node, self.planning_env, robot_goal, mcts_config.simulation_budget, step
            )
            episode_data.step_data.append(training_data)
            if best_action:
                # Convert to namo_rl.Action and execute
                namo_action = best_action.to_namo_action()
                result = self.execution_env.step(namo_action)
                
                # Capture post-action SE(2) poses (only what we need for training)
                post_action_observation = self.execution_env.get_observation()
                
                # Extract only SE(2) poses for movable objects and robot
                post_action_poses = {}
                for key, value in post_action_observation.items():
                    if key == 'robot_pose' or key.endswith('_movable_pose'):
                        # Keep SE(2) pose: [x, y, theta]
                        post_action_poses[key] = list(value)
                
                # Store action info with post-action poses
                action_info = {
                    'step': step,
                    'object_id': best_action.object_id,
                    'target': (best_action.goal.x, best_action.goal.y, best_action.goal.theta),
                    'reward': result.reward,
                    'post_action_poses': post_action_poses
                }
                action_sequence.append(action_info)
            else:
                break
        
        # Finalize episode data
        episode_data.total_steps = len(action_sequence)
        episode_data.final_action_sequence = action_sequence
        episode_data.success = self.execution_env.is_robot_goal_reachable()
        
        # Only save episode data if we have training samples
        if len(episode_data.step_data) > 0:
            self.save_episode_data(episode_data, save_json_summary)
        else:
            # Log skipped episode but don't save
            pass
        
        
        return episode_data
    
    def save_episode_data(self, episode_data: EpisodeData, save_json_summary: bool = True):
        """Save episode data to disk (only if it contains training samples)."""
        # Skip saving episodes with no training samples
        if len(episode_data.step_data) == 0:
            return
            
        # Create a pickleable version by excluding RLState objects
        pickleable_data = self._make_episode_data_pickleable(episode_data)
        
        filename = f"{episode_data.episode_id}.pkl"
        filepath = self.output_dir / filename
        
        with open(filepath, 'wb') as f:
            pickle.dump(pickleable_data, f)
        
        
        # Optionally save a human-readable summary
        if save_json_summary:
            summary_path = self.output_dir / f"{episode_data.episode_id}_summary.json"
            summary = self._create_episode_summary(episode_data)
            
            with open(summary_path, 'w') as f:
                json.dump(summary, f, indent=2)
    
    def _make_episode_data_pickleable(self, episode_data: EpisodeData) -> Dict:
        """Convert episode data to pickleable format by excluding RLState objects."""
        pickleable_step_data = []
        
        for step_data in episode_data.step_data:
            pickleable_step = {
                'scene_observation': step_data.scene_observation,
                'robot_goal': step_data.robot_goal,
                'object_proposals': [
                    {
                        'object_id': prop.object_id,
                        'probability': prop.probability,
                        'visit_count': prop.visit_count,
                        'q_value': prop.q_value
                    } for prop in step_data.object_proposals
                ],
                'goal_proposals': {
                    obj_id: [
                        {
                            'goal_position': goal.goal_position,
                            'probability': goal.probability,
                            'visit_count': goal.visit_count,
                            'q_value': goal.q_value
                        } for goal in goals
                    ] for obj_id, goals in step_data.goal_proposals.items()
                },
                'state_value': step_data.state_value,
                'object_q_values': step_data.object_q_values,
                'goal_q_values': step_data.goal_q_values,
                'mcts_iterations': step_data.mcts_iterations,
                'step_in_episode': step_data.step_in_episode,
                'reachable_objects': step_data.reachable_objects,
                'total_objects': step_data.total_objects,
                'static_object_info': step_data.static_object_info
            }
            pickleable_step_data.append(pickleable_step)
        
        return {
            'episode_id': episode_data.episode_id,
            'xml_file': episode_data.xml_file,
            'config_file': episode_data.config_file,
            'robot_goal': episode_data.robot_goal,
            'step_data': pickleable_step_data,
            'success': episode_data.success,
            'total_steps': episode_data.total_steps,
            'final_action_sequence': episode_data.final_action_sequence
        }
    
    def _create_episode_summary(self, episode_data: EpisodeData) -> Dict:
        """Create human-readable summary of episode data."""
        summary = {
            'episode_id': episode_data.episode_id,
            'xml_file': episode_data.xml_file,
            'robot_goal': episode_data.robot_goal,
            'success': episode_data.success,
            'total_steps': episode_data.total_steps,
            'training_samples': len(episode_data.step_data),
            'action_sequence': episode_data.final_action_sequence
        }
        
        # Add detailed training data summary
        if episode_data.step_data:
            summary['training_data_summary'] = []
            for i, step_data in enumerate(episode_data.step_data, 1):
                step_summary = {
                    'step': i,
                    'state_value': step_data.state_value,
                    'num_objects_proposed': len(step_data.object_proposals),
                    'num_reachable_objects': len(step_data.reachable_objects),
                    'total_goals_proposed': sum(len(goals) for goals in step_data.goal_proposals.values()),
                    'top_objects': [
                        {
                            'object_id': prop.object_id,
                            'probability': prop.probability,
                            'q_value': prop.q_value
                        } for prop in step_data.object_proposals[:3]  # Top 3
                    ]
                }
                summary['training_data_summary'].append(step_summary)
        
        return summary


if __name__ == "__main__":
    # Example usage
    from namo.config.mcts_config import MCTSConfig
    
    # Configuration
    xml_file = "../ml4kp_ktamp/resources/models/custom_walled_envs/aug9/easy/set2/benchmark_3/env_config_1375a.xml"
    config_file = "config/namo_config_complete.yaml"
    robot_goal = (-0.4993882613295453, 1.3595015590581654, 0.0)
    
    # MCTS configuration
    mcts_config = MCTSConfig(
        simulation_budget=30,
        max_rollout_steps=5,
        k=2.0,
        alpha=0.5,
        c_exploration=1.414
    )
    
    # Create data collection pipeline
    data_extractor = MCTSDataExtractor(top_k_goals=3)
    collector = SingleEnvironmentDataCollector(
        xml_file, config_file, data_extractor, output_dir="alphazero_data"
    )
    
    # Collect data from one episode
    episode_data = collector.collect_episode_data(robot_goal, mcts_config, max_steps=10)
    
    # Final summary
    status = "SUCCESS" if episode_data.success else "FAILED"
    samples = len(episode_data.step_data)
    steps = episode_data.total_steps
    
    if samples > 0:
        print(f"Episode {episode_data.episode_id}: {status} ({steps} steps, {samples} samples)")
    else:
        print(f"Episode {episode_data.episode_id}: SKIPPED (0 samples collected)")