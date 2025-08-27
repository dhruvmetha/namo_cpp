#!/usr/bin/env python3
"""
Example script showing how to use the NAMO Data Visualizer programmatically.
"""

import sys
import os
import pickle
import numpy as np

from ..visualizer import NAMODataVisualizer, NAMOXMLParser


def visualize_single_episode_example():
    """Example of visualizing a single episode."""
    print("=== Single Episode Visualization Example ===")
    
    # Load data file
    data_file = "/common/users/dm1487/namo_data/idfs_train_set/easy/modular_data_ilab1/ilab1_env_000000_results.pkl"
    
    if not os.path.exists(data_file):
        print(f"Data file not found: {data_file}")
        return
    
    with open(data_file, 'rb') as f:
        data = pickle.load(f)
    
    # Get first episode
    episodes = data.get('episode_results', [])
    if not episodes:
        print("No episodes found")
        return
    
    episode = episodes[0]
    
    # Create visualizer
    visualizer = NAMODataVisualizer(figsize=(10, 8))
    
    # Visualize episode with different options
    print("Creating visualization with trajectory and actions...")
    visualizer.visualize_episode(
        episode, 
        save_path="/tmp/example_full_viz.png",
        show_trajectory=True,
        show_actions=True
    )
    
    print("Creating visualization without trajectory...")
    visualizer.visualize_episode(
        episode, 
        save_path="/tmp/example_no_trajectory.png",
        show_trajectory=False,
        show_actions=True
    )
    
    # Generate 224x224 masks
    print("Generating 224x224 masks...")
    masks = visualizer.generate_episode_masks(episode)
    episode_id = episode.get('episode_id', 'example_episode')
    visualizer.save_masks(masks, "/tmp/example_masks", episode_id)
    
    # Print mask information
    print(f"\nGenerated {len(masks)} masks:")
    for mask_name, mask in masks.items():
        nonzero_pixels = np.count_nonzero(mask)
        print(f"  {mask_name}: shape={mask.shape}, dtype={mask.dtype}, "
              f"nonzero_pixels={nonzero_pixels}, max_value={mask.max():.3f}")
    
    print("Visualizations saved to /tmp/example_*.png")
    print("Masks saved to /tmp/example_masks/")


def analyze_episode_data_example():
    """Example of analyzing episode data structure."""
    print("\n=== Episode Data Analysis Example ===")
    
    # Load data file
    data_file = "/common/users/dm1487/namo_data/idfs_train_set/easy/modular_data_ilab1/ilab1_env_000000_results.pkl"
    
    with open(data_file, 'rb') as f:
        data = pickle.load(f)
    
    print(f"Task data keys: {list(data.keys())}")
    print(f"Number of episodes: {len(data.get('episode_results', []))}")
    
    episodes = data.get('episode_results', [])
    if episodes:
        episode = episodes[0]
        print(f"\nFirst episode keys: {list(episode.keys())}")
        print(f"Episode ID: {episode.get('episode_id')}")
        print(f"Success: {episode.get('solution_found')}")
        print(f"Search time: {episode.get('search_time_ms')} ms")
        print(f"Solution depth: {episode.get('solution_depth')}")
        print(f"XML file: {episode.get('xml_file')}")
        
        # Analyze state observations
        state_obs = episode.get('state_observations', [])
        print(f"\nNumber of state observations: {len(state_obs)}")
        if state_obs:
            print(f"State observation keys: {list(state_obs[0].keys())}")
            
            # Show robot trajectory
            robot_poses = [state.get('robot_pose', [0, 0, 0]) for state in state_obs]
            print(f"Robot trajectory (first 3 poses): {robot_poses[:3]}")
        
        # Analyze action sequence
        actions = episode.get('action_sequence', [])
        print(f"\nNumber of actions: {len(actions)}")
        if actions:
            for i, action in enumerate(actions):
                print(f"  Action {i+1}: object={action.get('object_id')}, target={action.get('target')}")


def xml_parsing_example():
    """Example of parsing XML environment files."""
    print("\n=== XML Parsing Example ===")
    
    # Get XML file path from episode data
    data_file = "/common/users/dm1487/namo_data/idfs_train_set/easy/modular_data_ilab1/ilab1_env_000000_results.pkl"
    
    with open(data_file, 'rb') as f:
        data = pickle.load(f)
    
    episodes = data.get('episode_results', [])
    if episodes:
        xml_file = episodes[0].get('xml_file')
        print(f"XML file: {xml_file}")
        
        if xml_file and os.path.exists(xml_file):
            # Parse XML file
            parser = NAMOXMLParser(xml_file)
            env_info = parser.parse_environment()
            
            print(f"\nEnvironment info:")
            print(f"Static objects: {len(env_info.static_objects)}")
            for obj in env_info.static_objects:
                print(f"  {obj.name}: pos=({obj.x:.2f}, {obj.y:.2f}), size=({obj.size_x:.2f}, {obj.size_y:.2f})")
            
            print(f"\nMovable objects: {len(env_info.movable_objects)}")
            for obj in env_info.movable_objects:
                print(f"  {obj.name}: size=({obj.size_x:.2f}, {obj.size_y:.2f})")
            
            print(f"\nRobot start: {env_info.robot_start}")
            print(f"Robot goal: {env_info.robot_goal}")
            print(f"World bounds: {env_info.world_bounds}")
        else:
            print(f"XML file not found or accessible: {xml_file}")


def batch_analysis_example():
    """Example of batch processing multiple data files."""
    print("\n=== Batch Analysis Example ===")
    
    data_dir = "/common/users/dm1487/namo_data/idfs_train_set/easy/modular_data_ilab1"
    
    import glob
    pickle_files = glob.glob(os.path.join(data_dir, "*_results.pkl"))[:5]  # Limit to 5 files
    
    total_episodes = 0
    successful_episodes = 0
    total_search_time = 0
    
    for pickle_file in pickle_files:
        try:
            with open(pickle_file, 'rb') as f:
                data = pickle.load(f)
            
            episodes = data.get('episode_results', [])
            total_episodes += len(episodes)
            
            for episode in episodes:
                if episode.get('solution_found'):
                    successful_episodes += 1
                
                search_time = episode.get('search_time_ms', 0)
                if search_time:
                    total_search_time += search_time
        
        except Exception as e:
            print(f"Warning: Failed to process {pickle_file}: {e}")
    
    print(f"Processed {len(pickle_files)} files")
    print(f"Total episodes: {total_episodes}")
    print(f"Successful episodes: {successful_episodes}")
    print(f"Success rate: {successful_episodes/total_episodes*100:.1f}%")
    print(f"Average search time: {total_search_time/total_episodes:.1f} ms")


if __name__ == "__main__":
    # Run all examples
    visualize_single_episode_example()
    analyze_episode_data_example()
    xml_parsing_example()
    batch_analysis_example()
    
    print("\n=== Example Complete ===")
    print("Check /tmp/ directory for generated visualizations:")
    print("  - example_full_viz.png: Full visualization with trajectory and actions")
    print("  - example_no_trajectory.png: Visualization without robot trajectory")
    print("  - example_masks/: Directory containing 224x224 masks:")
    print("    * robot_mask.png: Robot position (circle)")
    print("    * goal_mask.png: Goal position (circle)")
    print("    * movable_mask.png: All movable objects")
    print("    * static_mask.png: Static walls and obstacles")
    print("    * reachable_mask.png: Reachable objects")
    print("    * target_object_mask.png: Target object being manipulated")
    print("    * target_goal_mask.png: Target object at goal position")
    print("    * robot_distance_mask.png: Distance field from robot (wavefront)")
    print("    * goal_distance_mask.png: Distance field from goal (wavefront)")
    print("    * combined_distance_mask.png: Sum of robot + goal distance fields")
    print("    * masks_composite.png: All masks visualized together")