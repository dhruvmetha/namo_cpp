#!/usr/bin/env python3
"""Generate masks from fresh MCTS data."""

import pickle
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.insert(0, 'python')
import namo_rl
from namo_image_converter import NAMOImageConverter

def create_visualizations(masks, output_dir, prefix=""):
    """Save individual mask visualizations."""
    os.makedirs(output_dir, exist_ok=True)
    
    channel_names = [
        'robot', 'goal', 'movable', 'static', 'reachable',
        'target_object', 'target_goal', 'robot_distance', 'goal_distance', 'combined_distance'
    ]
    
    # Save individual masks
    for i, (mask, name) in enumerate(zip(masks, channel_names)):
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        im = ax.imshow(mask, cmap='viridis', origin='upper')
        ax.set_title(f"{prefix}{name}\\nShape: {mask.shape}, Sum: {mask.sum():.1f}, Non-zero: {np.count_nonzero(mask)}")
        ax.axis('off')
        plt.colorbar(im, ax=ax, shrink=0.8)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/{prefix}{name}.png", dpi=150, bbox_inches='tight')
        plt.close()
        print(f"💾 Saved: {prefix}{name}.png")

def generate_masks_from_fresh_data():
    """Generate masks from the fresh MCTS episode data."""
    print("🎨 Generating masks from fresh MCTS data...")
    
    # Load fresh episode data
    with open('alphazero_data/episode_0.pkl', 'rb') as f:
        episode_data = pickle.load(f)
    
    print(f"Episode: {episode_data['episode_id']}")
    print(f"XML file: {episode_data['xml_file']}")
    print(f"Robot goal: {episode_data['robot_goal']}")
    print(f"Total steps: {episode_data['total_steps']}")
    
    # Create output directory
    output_dir = "fresh_mcts_masks"
    os.makedirs(output_dir, exist_ok=True)
    
    # Process each step
    for step_idx, step_data in enumerate(episode_data['step_data']):
        print(f"\n📊 Processing Step {step_idx}:")
        
        # Get pre-action data
        scene_obs = step_data['scene_observation']
        robot_goal = step_data['robot_goal']
        static_object_info = step_data['static_object_info']
        reachable_objects = step_data.get('reachable_objects', [])
        
        print(f"  Pre-action objects: {list(scene_obs.keys())}")
        print(f"  Static object info: {list(static_object_info.keys())}")
        print(f"  Reachable objects: {reachable_objects}")
        
        # Create environment for image conversion
        xml_file = f"/common/home/dm1487/robotics_research/ktamp/ml4kp_ktamp/resources/models/{episode_data['xml_file']}"
        config_file = episode_data['config_file']
        env = namo_rl.RLEnvironment(xml_file, config_file, visualize=False)
        
        # Create image converter  
        converter = NAMOImageConverter(env)
        
        # Generate pre-action masks (10 channels)
        pre_action_masks = converter.convert_state_to_image_with_goal_info(
            scene_obs, robot_goal, reachable_objects, 
            target_object="obstacle_2_movable",  # From the action sequence
            goal_position=[0.0, 0.0, 0.0]  # Placeholder
        )
        
        print(f"  Pre-action masks shape: {pre_action_masks.shape}")
        
        # Save pre-action masks
        create_visualizations(pre_action_masks, output_dir, f"step_{step_idx:02d}_pre_")
        
        # Check if we have post-action data from the final action sequence
        if step_idx < len(episode_data['final_action_sequence']):
            action = episode_data['final_action_sequence'][step_idx]
            
            if 'post_action_poses' in action:
                print(f"  ✅ Post-action poses found!")
                post_action_obs = action['post_action_poses']
                
                # Show rotation changes
                print(f"  🔄 ROTATION CHANGES:")
                for obj_name in ['obstacle_1_movable_pose', 'obstacle_2_movable_pose', 'obstacle_3_movable_pose', 'obstacle_4_movable_pose']:
                    if obj_name in scene_obs and obj_name in post_action_obs:
                        pre_theta = scene_obs[obj_name][2]
                        post_theta = post_action_obs[obj_name][2]
                        delta_theta = post_theta - pre_theta
                        print(f"    {obj_name}: {pre_theta:.2f} → {post_theta:.2f} rad (Δ={delta_theta:.2f})")
                
                # Generate post-action masks (3 channels: robot, movable, robot_distance)
                post_action_masks = converter.convert_state_to_image(post_action_obs, robot_goal)
                
                # Extract relevant channels for post-action
                post_robot_mask = post_action_masks[0]  # Robot position
                post_movable_mask = post_action_masks[2]  # Movable objects  
                post_robot_distance = converter._compute_distance_field(post_action_obs['robot_pose'][0], post_action_obs['robot_pose'][1])
                
                post_masks = [post_robot_mask, post_movable_mask, post_robot_distance]
                post_names = ['robot', 'movable', 'robot_distance']
                
                # Save post-action masks
                for mask, name in zip(post_masks, post_names):
                    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
                    im = ax.imshow(mask, cmap='viridis', origin='upper')
                    ax.set_title(f"step_{step_idx:02d}_post_{name}\\nShape: {mask.shape}, Sum: {mask.sum():.1f}, Non-zero: {np.count_nonzero(mask)}")
                    ax.axis('off')
                    plt.colorbar(im, ax=ax, shrink=0.8)
                    plt.tight_layout()
                    plt.savefig(f"{output_dir}/step_{step_idx:02d}_post_{name}.png", dpi=150, bbox_inches='tight')
                    plt.close()
                    print(f"💾 Saved: step_{step_idx:02d}_post_{name}.png")
                
                # Create comparison image
                fig, axes = plt.subplots(2, 3, figsize=(15, 10))
                
                # Pre-action
                axes[0, 0].imshow(pre_action_masks[0], cmap='viridis', origin='upper')
                axes[0, 0].set_title(f"Pre Robot\\nSum: {pre_action_masks[0].sum():.1f}")
                axes[0, 0].axis('off')
                
                axes[0, 1].imshow(pre_action_masks[2], cmap='viridis', origin='upper')  
                axes[0, 1].set_title(f"Pre Movable\\nSum: {pre_action_masks[2].sum():.1f}")
                axes[0, 1].axis('off')
                
                axes[0, 2].imshow(pre_action_masks[3], cmap='viridis', origin='upper')
                axes[0, 2].set_title(f"Pre Static\\nSum: {pre_action_masks[3].sum():.1f}")
                axes[0, 2].axis('off')
                
                # Post-action
                axes[1, 0].imshow(post_robot_mask, cmap='viridis', origin='upper')
                axes[1, 0].set_title(f"Post Robot\\nSum: {post_robot_mask.sum():.1f}")
                axes[1, 0].axis('off')
                
                axes[1, 1].imshow(post_movable_mask, cmap='viridis', origin='upper')
                axes[1, 1].set_title(f"Post Movable\\nSum: {post_movable_mask.sum():.1f}")
                axes[1, 1].axis('off')
                
                axes[1, 2].imshow(pre_action_masks[3], cmap='viridis', origin='upper')  # Static doesn't change
                axes[1, 2].set_title(f"Static (unchanged)\\nSum: {pre_action_masks[3].sum():.1f}")
                axes[1, 2].axis('off')
                
                plt.tight_layout()
                plt.savefig(f"{output_dir}/step_{step_idx:02d}_comparison.png", dpi=150, bbox_inches='tight')
                plt.close()
                print(f"💾 Saved: step_{step_idx:02d}_comparison.png")
    
    print(f"\n🎉 Fresh MCTS masks generated successfully!")
    print(f"📁 Check './{output_dir}/' for all mask visualizations")

if __name__ == "__main__":
    generate_masks_from_fresh_data()