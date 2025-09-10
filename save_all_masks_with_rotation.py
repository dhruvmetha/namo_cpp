#!/usr/bin/env python3
"""
Generate and save all individual masks, including object rotations.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add python directory to path  
sys.path.insert(0, 'python')

def save_all_masks_with_rotation():
    """Generate and save all individual masks, showing object rotations."""
    
    print("🎨 Generating All Masks with Object Rotations...")
    
    output_dir = Path("./all_masks_with_rotation")
    output_dir.mkdir(exist_ok=True)
    
    try:
        # Create test data with significant rotations
        scene_observation = {
            'robot_pose': [0.0, 0.0, 0.0],  # Robot starts at origin, no rotation
            'obstacle_1_movable_pose': [1.0, 1.0, 0.0],  # No rotation initially
            'obstacle_2_movable_pose': [2.0, 2.0, 1.57],  # 90 degrees rotated
            'obstacle_3_movable_pose': [-1.0, -1.0, -0.78]  # -45 degrees
        }

        # Post-action: objects moved AND rotated significantly
        post_action_poses = {
            'robot_pose': [0.8, 0.8, 0.0],  # Robot moved, no rotation
            'obstacle_1_movable_pose': [1.5, 1.8, 2.36],  # Moved AND rotated 135°
            'obstacle_2_movable_pose': [2.2, 1.8, 0.78],   # Moved AND rotated to 45°
            'obstacle_3_movable_pose': [-0.8, -1.2, 0.52]  # Moved AND rotated to 30°
        }

        static_object_info = {
            # MOVABLE OBJECTS (only need size info)
            'obstacle_1_movable': {'size_x': 0.6, 'size_y': 0.3, 'size_z': 0.3},  # Rectangle - rotation visible
            'obstacle_2_movable': {'size_x': 0.4, 'size_y': 0.2, 'size_z': 0.3},  # Rectangle - rotation visible  
            'obstacle_3_movable': {'size_x': 0.5, 'size_y': 0.25, 'size_z': 0.3}, # Rectangle - rotation visible
            'robot': {'size_x': 0.15, 'size_y': 0.15, 'size_z': 0.15},
            
            # STATIC OBJECTS (need position, size, and quaternion)
            'wall_1': {
                'pos_x': -2.0, 'pos_y': 0.0, 'pos_z': 0.3,
                'size_x': 0.05, 'size_y': 3.0, 'size_z': 0.3,
                'quat_w': 1.0, 'quat_x': 0.0, 'quat_y': 0.0, 'quat_z': 0.0
            },
            'wall_2': {
                'pos_x': 2.0, 'pos_y': 0.0, 'pos_z': 0.3,
                'size_x': 0.05, 'size_y': 3.0, 'size_z': 0.3,
                'quat_w': 1.0, 'quat_x': 0.0, 'quat_y': 0.0, 'quat_z': 0.0
            },
            'wall_3': {
                'pos_x': 0.0, 'pos_y': -3.0, 'pos_z': 0.3,
                'size_x': 2.0, 'size_y': 0.05, 'size_z': 0.3,
                'quat_w': 1.0, 'quat_x': 0.0, 'quat_y': 0.0, 'quat_z': 0.0
            },
            'wall_4': {
                'pos_x': 0.0, 'pos_y': 3.0, 'pos_z': 0.3,
                'size_x': 2.0, 'size_y': 0.05, 'size_z': 0.3,
                'quat_w': 1.0, 'quat_x': 0.0, 'quat_y': 0.0, 'quat_z': 0.0
            },
            # Static obstacle with rotation (45 degrees)
            'obstacle_static_1': {
                'pos_x': 0.8, 'pos_y': -1.2, 'pos_z': 0.2,
                'size_x': 0.4, 'size_y': 0.3, 'size_z': 0.2,
                'quat_w': 0.924, 'quat_x': 0.0, 'quat_y': 0.0, 'quat_z': 0.383  # 45° rotation
            }
        }

        robot_goal = [3.0, 3.0, 0.0]
        goal_proposals = [{'goal_position': [1.8, 1.8, 2.36], 'probability': 1.0, 'visit_count': 10, 'q_value': 0.8}]

        # Generate masks
        from mcts_mask_generation.mcts_visualizer import MCTSMaskGenerator

        xml_file = '../ml4kp_ktamp/resources/models/custom_walled_envs/aug9/easy/set1/benchmark_1/env_config_7c65f.xml'
        config_file = 'config/namo_config_complete.yaml'

        generator = MCTSMaskGenerator(xml_file, config_file)
        
        training_samples = generator.generate_goal_proposal_data(
            scene_observation=scene_observation,
            robot_goal=robot_goal,
            object_id='obstacle_1_movable',
            goal_proposals=goal_proposals,
            post_action_poses=post_action_poses,
            static_object_info=static_object_info
        )
        
        if not training_samples:
            print("❌ No training samples generated")
            return False
        
        sample = training_samples[0]
        
        # Print rotation analysis
        print_rotation_analysis(scene_observation, post_action_poses)
        
        # Save all individual masks
        mask_keys = [k for k, v in sample.items() if isinstance(v, np.ndarray) and v.ndim == 2]
        print(f"📊 Saving {len(mask_keys)} individual masks...")
        
        for mask_key in mask_keys:
            save_individual_mask(sample[mask_key], mask_key, output_dir)
        
        # Create rotation-focused comparison
        create_rotation_comparison(sample, output_dir, scene_observation, post_action_poses)
        
        # Create comprehensive grid of all masks
        create_all_masks_grid(sample, output_dir)
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def print_rotation_analysis(scene_observation, post_action_poses):
    """Print analysis of object rotations."""
    
    print(f"\\n🔄 ROTATION ANALYSIS:")
    
    for obj_key in scene_observation:
        if obj_key.endswith('_pose') and obj_key in post_action_poses:
            pre_pose = scene_observation[obj_key]
            post_pose = post_action_poses[obj_key]
            
            pre_theta = pre_pose[2]
            post_theta = post_pose[2]
            delta_theta = post_theta - pre_theta
            
            # Normalize angle difference to [-π, π]
            while delta_theta > np.pi:
                delta_theta -= 2 * np.pi
            while delta_theta < -np.pi:
                delta_theta += 2 * np.pi
            
            print(f"  {obj_key.replace('_pose', '')}:")
            print(f"    Pre:  θ={pre_theta:+5.2f} rad ({np.degrees(pre_theta):+6.1f}°)")
            print(f"    Post: θ={post_theta:+5.2f} rad ({np.degrees(post_theta):+6.1f}°)")
            print(f"    Δθ:   {delta_theta:+5.2f} rad ({np.degrees(delta_theta):+6.1f}°)")
            
            if abs(delta_theta) > 0.1:
                print(f"    ✅ Significant rotation detected")
            else:
                print(f"    ⚠️  Little/no rotation")

def save_individual_mask(mask, mask_name, output_dir):
    """Save individual mask as PNG file."""
    
    plt.figure(figsize=(8, 8))
    
    # Choose appropriate colormap
    if 'distance' in mask_name.lower():
        cmap = 'plasma'
        vmin, vmax = None, None
    else:
        cmap = 'viridis'
        vmin, vmax = 0, 1
    
    im = plt.imshow(mask, cmap=cmap, vmin=vmin, vmax=vmax)
    plt.title(f'{mask_name}\\nShape: {mask.shape}, Sum: {mask.sum():.1f}, Non-zero: {np.count_nonzero(mask)}', 
              fontsize=14, fontweight='bold')
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.axis('off')
    
    # Save with descriptive filename
    filename = output_dir / f'{mask_name}.png'
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"💾 Saved: {filename}")

def create_rotation_comparison(sample, output_dir, scene_observation, post_action_poses):
    """Create focused comparison highlighting rotations."""
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Object Rotation Comparison: Pre vs Post Action\\nRectangular Objects Show Rotation Clearly', 
                 fontsize=16, fontweight='bold')
    
    # Movable objects comparison
    if 'movable' in sample and 'post_action_movable' in sample:
        # Pre-action movable
        im1 = axes[0, 0].imshow(sample['movable'], cmap='viridis', vmin=0, vmax=1)
        axes[0, 0].set_title('PRE-ACTION: Movable Objects\\n(Original Rotations)', fontweight='bold')
        axes[0, 0].axis('off')
        plt.colorbar(im1, ax=axes[0, 0], fraction=0.046)
        
        # Post-action movable
        im2 = axes[0, 1].imshow(sample['post_action_movable'], cmap='viridis', vmin=0, vmax=1)
        axes[0, 1].set_title('POST-ACTION: Movable Objects\\n(After Rotation + Translation)', fontweight='bold')
        axes[0, 1].axis('off')
        plt.colorbar(im2, ax=axes[0, 1], fraction=0.046)
        
        # Difference
        diff = sample['post_action_movable'] - sample['movable']
        im3 = axes[0, 2].imshow(diff, cmap='RdBu_r', vmin=-1, vmax=1)
        axes[0, 2].set_title('DIFFERENCE: Object Movement\\n+ Rotation Changes', fontweight='bold')
        axes[0, 2].axis('off')
        plt.colorbar(im3, ax=axes[0, 2], fraction=0.046)
        
        # Add statistics
        diff_sum = np.sum(np.abs(diff))
        axes[0, 2].text(0.02, 0.98, f'Total Change: {diff_sum:.0f}\\n(Movement + Rotation)', 
                       transform=axes[0, 2].transAxes, 
                       bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8),
                       fontsize=10, fontweight='bold', va='top')
    
    # Robot comparison
    if 'robot' in sample and 'post_action_robot' in sample:
        # Pre-action robot
        im4 = axes[1, 0].imshow(sample['robot'], cmap='viridis', vmin=0, vmax=1)
        axes[1, 0].set_title('PRE-ACTION: Robot Position', fontweight='bold')
        axes[1, 0].axis('off')
        plt.colorbar(im4, ax=axes[1, 0], fraction=0.046)
        
        # Post-action robot
        im5 = axes[1, 1].imshow(sample['post_action_robot'], cmap='viridis', vmin=0, vmax=1)
        axes[1, 1].set_title('POST-ACTION: Robot Position', fontweight='bold')
        axes[1, 1].axis('off')
        plt.colorbar(im5, ax=axes[1, 1], fraction=0.046)
        
        # Difference
        robot_diff = sample['post_action_robot'] - sample['robot']
        im6 = axes[1, 2].imshow(robot_diff, cmap='RdBu_r', vmin=-1, vmax=1)
        axes[1, 2].set_title('DIFFERENCE: Robot Movement', fontweight='bold')
        axes[1, 2].axis('off')
        plt.colorbar(im6, ax=axes[1, 2], fraction=0.046)
        
        robot_diff_sum = np.sum(np.abs(robot_diff))
        axes[1, 2].text(0.02, 0.98, f'Robot Movement: {robot_diff_sum:.0f}', 
                       transform=axes[1, 2].transAxes, 
                       bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8),
                       fontsize=10, fontweight='bold', va='top')
    
    plt.tight_layout()
    
    filename = output_dir / 'rotation_focused_comparison.png'
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"💾 Saved: {filename}")

def create_all_masks_grid(sample, output_dir):
    """Create comprehensive grid showing all masks."""
    
    mask_arrays = {k: v for k, v in sample.items() if isinstance(v, np.ndarray) and v.ndim == 2}
    n_masks = len(mask_arrays)
    
    # Calculate grid size
    cols = 4
    rows = (n_masks + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(20, 5*rows))
    fig.suptitle(f'All {n_masks} Generated Masks (Pre-Action + Post-Action)', 
                 fontsize=16, fontweight='bold')
    
    # Flatten axes for easier indexing
    if rows == 1:
        axes = axes.reshape(1, -1)
    axes_flat = axes.flatten()
    
    for i, (mask_name, mask) in enumerate(mask_arrays.items()):
        if i < len(axes_flat):
            ax = axes_flat[i]
            
            # Choose colormap
            if 'distance' in mask_name.lower():
                cmap = 'plasma'
                vmin, vmax = None, None
            else:
                cmap = 'viridis'  
                vmin, vmax = 0, 1
            
            im = ax.imshow(mask, cmap=cmap, vmin=vmin, vmax=vmax)
            ax.set_title(f'{mask_name}\\nΣ={mask.sum():.0f}', fontsize=10, fontweight='bold')
            ax.axis('off')
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    
    # Hide unused subplots
    for i in range(n_masks, len(axes_flat)):
        axes_flat[i].set_visible(False)
    
    plt.tight_layout()
    
    filename = output_dir / 'all_masks_grid.png'
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"💾 Saved: {filename}")

if __name__ == "__main__":
    success = save_all_masks_with_rotation()
    if success:
        print(f"\\n🎉 All masks with rotation analysis generated successfully!")
        print(f"📁 Check './all_masks_with_rotation/' for all individual masks")
        print(f"🔄 Object rotations are now included in the masks")
    else:
        print(f"\\n❌ Mask generation failed")