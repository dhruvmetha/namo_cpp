#!/usr/bin/env python3
"""
Test script for the AlphaZero-style data collection pipeline.

This script runs a single episode data collection to verify that all components work correctly.
"""

import sys
import argparse
from pathlib import Path

# Add current directory to path for imports
sys.path.append(str(Path(__file__).parent))

from alphazero_data_collection import MCTSDataExtractor, SingleEnvironmentDataCollector
from mcts_config import MCTSConfig


def main():
    """Test data collection pipeline with a single episode."""
    parser = argparse.ArgumentParser(description="Test AlphaZero Data Collection Pipeline")
    parser.add_argument("--budget", type=int, default=20,
                       help="MCTS simulation budget per step (default: 20)")
    parser.add_argument("--max-steps", type=int, default=5,
                       help="Maximum planning steps (default: 5)")
    parser.add_argument("--xml", type=str, 
                       default="../ml4kp_ktamp/resources/models/custom_walled_envs/aug9/easy/set2/benchmark_3/env_config_1375a.xml",
                       help="XML environment file")
    parser.add_argument("--config", type=str, default="config/namo_config_complete.yaml",
                       help="YAML config file")
    parser.add_argument("--output-dir", type=str, default="test_alphazero_data",
                       help="Output directory for collected data")
    
    args = parser.parse_args()
    
    print("🧪 Testing AlphaZero-style Data Collection Pipeline")
    print("="*60)
    print(f"Configuration:")
    print(f"  XML file: {args.xml}")
    print(f"  Config file: {args.config}")
    print(f"  MCTS budget: {args.budget}")
    print(f"  Max steps: {args.max_steps}")
    print(f"  Output dir: {args.output_dir}")
    
    # Robot goal (same as in test_multi_step_mcts.py)
    robot_goal = (-0.4993882613295453, 1.3595015590581654, 0.0)
    print(f"  Robot goal: {robot_goal}")
    
    # MCTS configuration
    mcts_config = MCTSConfig(
        simulation_budget=args.budget,
        max_rollout_steps=5,
        k=2.0,
        alpha=0.5,
        c_exploration=1.414,
        verbose=False
    )
    
    try:
        # Create data collection pipeline
        print(f"\n🔧 Initializing data collection pipeline...")
        data_extractor = MCTSDataExtractor(top_k_goals=3)
        collector = SingleEnvironmentDataCollector(
            args.xml, args.config, data_extractor, output_dir=args.output_dir
        )
        
        print(f"✓ Data collection pipeline initialized")
        
        # Collect data from one episode
        print(f"\n🚀 Starting data collection...")
        episode_data = collector.collect_episode_data(
            robot_goal, mcts_config, max_steps=args.max_steps, episode_id="test_episode_001"
        )
        
        # Print detailed results
        print(f"\n📊 Data Collection Results:")
        print(f"="*60)
        print(f"Episode ID: {episode_data.episode_id}")
        print(f"Success: {'✅ YES' if episode_data.success else '❌ NO'}")
        print(f"Total Steps: {episode_data.total_steps}")
        print(f"Training Samples: {len(episode_data.step_data)}")
        
        if episode_data.final_action_sequence:
            print(f"\n📋 Action Sequence:")
            for i, action in enumerate(episode_data.final_action_sequence, 1):
                print(f"  {i}. Push {action['object_id']} to {action['target']} (reward: {action['reward']:.3f})")
        
        # Analyze collected training data
        print(f"\n🔍 Training Data Analysis:")
        print(f"="*60)
        
        for i, sample in enumerate(episode_data.step_data, 1):
            print(f"\nStep {i} Training Data:")
            print(f"  State value: {sample.state_value:.3f}")
            print(f"  MCTS iterations: {sample.mcts_iterations}")
            print(f"  Reachable objects: {len(sample.reachable_objects)} - {sample.reachable_objects}")
            print(f"  Total objects in scene: {sample.total_objects}")
            
            # Object proposals
            print(f"  Object proposals: {len(sample.object_proposals)}")
            for j, prop in enumerate(sample.object_proposals[:3], 1):  # Show top 3
                print(f"    {j}. {prop.object_id}: prob={prop.probability:.3f}, Q={prop.q_value:.3f}, visits={prop.visit_count}")
            
            # Goal proposals (show for top object)
            if sample.object_proposals:
                top_obj = sample.object_proposals[0].object_id
                if top_obj in sample.goal_proposals:
                    goals = sample.goal_proposals[top_obj]
                    print(f"  Goal proposals for '{top_obj}': {len(goals)}")
                    for j, goal in enumerate(goals[:2], 1):  # Show top 2 goals
                        pos = goal.goal_position
                        print(f"    {j}. ({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}): prob={goal.probability:.3f}, Q={goal.q_value:.3f}")
            
            # Q-values summary
            obj_q_count = len(sample.object_q_values)
            goal_q_count = sum(len(goals) for goals in sample.goal_q_values.values())
            print(f"  Q-values collected: {obj_q_count} objects, {goal_q_count} goals")
        
        print(f"\n✅ Test completed successfully!")
        print(f"📁 Data saved to: {args.output_dir}")
        
        return 0
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())