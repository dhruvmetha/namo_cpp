#!/usr/bin/env python3
"""
ML Inference vs Visualizer Comparison Test

This script runs a single inference with:
- Object selection: no_heuristic
- Goal selection: ml with specified model
- XML: custom_walled_envs/aug9/easy/set1/benchmark_2/env_config_1275a.xml

It then compares the image masks produced by ML inference logic
with those produced by our unified visualizer interface.
"""

import sys
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path

# Import torch for CUDA check
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None

# Add necessary paths
sys.path.append('/common/home/dm1487/robotics_research/ktamp/namo/python')
sys.path.append('/common/home/dm1487/robotics_research/ktamp/learning')

# Import NAMO RL environment
import namo_rl

# Import our unified converter and ML strategies
from unified_image_converter import UnifiedImageConverter, ObjectInfo, create_converter_from_xml
from ml_image_converter_adapter import MLImageConverterAdapter
from mask_generation.visualizer import NAMODataVisualizer

# Import IDFS strategies
from idfs.ml_strategies import MLGoalSelectionStrategy
from idfs.object_selection_strategy import NoHeuristicStrategy


def setup_environment():
    """Set up the NAMO environment with specified XML file."""
    print("=== Setting up Environment ===")
    
    # XML and config paths
    xml_path = "/common/home/dm1487/robotics_research/ktamp/ml4kp_ktamp/resources/models/custom_walled_envs/aug9/easy/set1/benchmark_2/env_config_1275a.xml"
    config_path = "/common/home/dm1487/robotics_research/ktamp/namo/config/namo_config_complete.yaml"
    
    print(f"XML: {xml_path}")
    print(f"Config: {config_path}")
    
    # Check if files exist
    xml_full_path = xml_path
    if not os.path.exists(xml_full_path):
        print(f"ERROR: XML file not found at {xml_full_path}")
        return None, None, None
    
    if not os.path.exists(config_path):
        print(f"ERROR: Config file not found at {config_path}")
        return None, None, None
    
    try:
        # Initialize environment
        env = namo_rl.RLEnvironment(xml_full_path, config_path)
        print("✓ NAMO environment initialized")
        
        # Reset environment and get initial state
        env.reset()
        initial_state = env.get_full_state()
        obs = env.get_observation()
        
        print(f"✓ Environment reset complete")
        print(f"  Observation keys: {list(obs.keys())}")
        
        # Get robot goal from environment
        robot_goal = None
        if hasattr(env, '_robot_goal'):
            robot_goal = getattr(env, '_robot_goal')
        elif hasattr(env, 'get_robot_goal'):
            try:
                robot_goal = env.get_robot_goal()
            except:
                robot_goal = (2.0, 2.0, 0.0)  # Default fallback
        else:
            robot_goal = (2.0, 2.0, 0.0)  # Default fallback
            
        print(f"  Robot goal: {robot_goal}")
        
        return env, initial_state, robot_goal
        
    except Exception as e:
        print(f"ERROR: Failed to initialize environment: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None


def run_object_selection(env, state):
    """Run no_heuristic object selection."""
    print("\n=== Running Object Selection (no_heuristic) ===")
    
    try:
        # Get reachable objects from environment (without state argument)
        reachable_objects = env.get_reachable_objects()
        print(f"  Reachable objects: {reachable_objects}")
        
        # Use no_heuristic strategy (returns objects in original order)
        no_heuristic_strategy = NoHeuristicStrategy()
        selected_objects = no_heuristic_strategy.select_objects(reachable_objects, state, env)
        
        print(f"  Selected objects (no_heuristic): {selected_objects}")
        
        # Return first object for goal selection
        target_object = selected_objects[0] if selected_objects else None
        print(f"  Target object for goal selection: {target_object}")
        
        return target_object, reachable_objects
        
    except Exception as e:
        print(f"ERROR: Object selection failed: {e}")
        import traceback
        traceback.print_exc()
        return None, []


def run_goal_selection(env, state, target_object, robot_goal):
    """Run ML goal selection with specified model."""
    print("\n=== Running Goal Selection (ML) ===")
    
    goal_model_path = "/common/home/dm1487/robotics_research/ktamp/learning/outputs/rel_coords_goal/mse_dice/2025-08-25_01-19-40"
    
    print(f"  Goal model path: {goal_model_path}")
    print(f"  Target object: {target_object}")
    
    if not target_object:
        print("  No target object - skipping goal selection")
        return []
    
    try:
        # Check if model path exists
        if not os.path.exists(goal_model_path):
            print(f"ERROR: Model path not found: {goal_model_path}")
            return []
        
        # Initialize ML goal selection strategy
        ml_goal_strategy = MLGoalSelectionStrategy(
            goal_model_path=goal_model_path,
            samples=16,  # Use fewer samples for faster testing
            device="cuda" if TORCH_AVAILABLE and torch.cuda.is_available() else "cpu",
            xml_path_relative="custom_walled_envs/aug9/easy/set1/benchmark_2/env_config_1275a.xml",
            min_goals_threshold=1,
            verbose=True
        )
        
        # Generate goals
        goals = ml_goal_strategy.generate_goals(target_object, state, env, max_goals=5)
        
        print(f"  Generated {len(goals)} goals:")
        for i, goal in enumerate(goals):
            print(f"    Goal {i+1}: x={goal.x:.3f}, y={goal.y:.3f}, theta={goal.theta:.3f}")
        
        return goals
        
    except Exception as e:
        print(f"ERROR: Goal selection failed: {e}")
        import traceback
        traceback.print_exc()
        return []


def create_ml_inference_images(env, state, robot_goal, target_object, reachable_objects):
    """Create images using ML inference logic (via adapter)."""
    print("\n=== Creating ML Inference Images ===")
    
    try:
        # Create ML adapter with relative path (adapter will handle the full path resolution)
        xml_rel_path = "custom_walled_envs/aug9/easy/set1/benchmark_2/env_config_1275a.xml"
        ml_adapter = MLImageConverterAdapter(xml_rel_path)
        
        # Set environment state
        env.set_full_state(state)
        obs = env.get_observation()
        
        # Create JSON message format for ML inference
        objects_dict = {}
        for obj_name, pose in obs.items():
            if obj_name != 'robot_pose' and obj_name.endswith('_pose'):
                obj_base_name = obj_name.replace('_pose', '')
                if len(pose) >= 3:
                    objects_dict[obj_base_name] = {
                        "position": [float(pose[0]), float(pose[1]), float(pose[2])],
                        "quaternion": [1.0, 0.0, 0.0, 0.0]  # Default quaternion
                    }
        
        # Add robot info
        if 'robot_pose' in obs:
            robot_pos = obs['robot_pose']
        else:
            robot_pos = [0.0, 0.0, 0.0]
        
        json_message = {
            "xml_path": xml_rel_path,
            "robot_goal": [float(robot_goal[0]), float(robot_goal[1])],
            "reachable_objects": reachable_objects,
            "robot": {
                "position": [float(robot_pos[0]), float(robot_pos[1]), float(robot_pos[2])]
            },
            "objects": objects_dict
        }
        
        print(f"  Processing data with ML adapter...")
        print(f"  Robot position: {robot_pos}")
        print(f"  Robot goal: {robot_goal[:2]}")
        print(f"  Objects: {list(objects_dict.keys())}")
        print(f"  Reachable objects: {reachable_objects}")
        
        # Process with ML adapter
        ml_result = ml_adapter.process_datapoint(json_message, robot_goal[:2])
        
        print(f"  ML adapter result keys: {list(ml_result.keys())}")
        
        # Extract individual channels
        ml_channels = {
            'robot': ml_result['robot_image'][:, :, 0],
            'goal': ml_result['goal_image'][:, :, 0],
            'movable': ml_result['movable_objects_image'][:, :, 0],
            'static': ml_result['static_objects_image'][:, :, 0],
            'reachable': ml_result['reachable_objects_image'][:, :, 0]
        }
        
        # Print channel stats
        for name, channel in ml_channels.items():
            nonzero = np.count_nonzero(channel)
            print(f"    {name}: {nonzero} nonzero pixels")
        
        return ml_channels, ml_adapter.converter.world_bounds
        
    except Exception as e:
        print(f"ERROR: ML inference image creation failed: {e}")
        import traceback
        traceback.print_exc()
        return None, None


def create_visualizer_images(env, state, robot_goal, target_object, reachable_objects):
    """Create images using our unified visualizer."""
    print("\n=== Creating Visualizer Images ===")
    
    try:
        # Create a mock episode data structure for the visualizer
        env.set_full_state(state)
        obs = env.get_observation()
        
        # Create mock static object info since environment doesn't have get_object_info
        static_object_info = {}
        
        # Add default sizes for movable objects
        movable_objects = []
        for key in obs.keys():
            if key.endswith('_pose') and key != 'robot_pose':
                obj_name = key.replace('_pose', '')
                movable_objects.append(obj_name)
                # Add default size info for movable objects
                static_object_info[obj_name] = {
                    'size_x': 0.25,  # Default half-extent
                    'size_y': 0.25,  # Default half-extent
                    'size_z': 0.25   # Default half-extent
                }
        
        print(f"    Found movable objects: {movable_objects}")
        
        # Create episode data structure
        episode_data = {
            'episode_id': 'test_comparison',
            'solution_found': True,
            'robot_goal': robot_goal,
            'static_object_info': static_object_info,
            'state_observations': [obs],  # Single state observation
            'action_sequence': []  # Empty for this test
        }
        
        print(f"  Created episode data with:")
        print(f"    Robot goal: {robot_goal}")
        print(f"    Static object info keys: {list(static_object_info.keys())}")
        print(f"    Observation keys: {list(obs.keys())}")
        
        # Create visualizer and generate masks
        visualizer = NAMODataVisualizer()
        masks = visualizer.generate_episode_masks_batch(episode_data)  # Use batch version (excludes combined distance)
        
        print(f"  Visualizer mask keys: {list(masks.keys())}")
        
        # Extract relevant channels (convert to same format as ML)
        vis_channels = {}
        for mask_name, mask in masks.items():
            if mask_name in ['robot', 'goal', 'movable', 'static', 'reachable']:
                vis_channels[mask_name] = mask
                nonzero = np.count_nonzero(mask)
                print(f"    {mask_name}: {nonzero} nonzero pixels")
        
        # Get world bounds from visualizer
        env_info = visualizer._extract_env_info_from_episode(episode_data)
        world_bounds = env_info.world_bounds
        
        return vis_channels, world_bounds
        
    except Exception as e:
        print(f"ERROR: Visualizer image creation failed: {e}")
        import traceback
        traceback.print_exc()
        return None, None


def create_comparison_visualization(ml_channels, vis_channels, ml_bounds, vis_bounds, output_path):
    """Create a side-by-side comparison visualization."""
    print(f"\n=== Creating Comparison Visualization ===")
    
    try:
        # Channel names to compare
        channel_names = ['robot', 'goal', 'movable', 'static', 'reachable']
        
        # Create figure with 3 rows (ML, Visualizer, Difference) and 5 columns (channels)
        fig, axes = plt.subplots(3, 5, figsize=(20, 12))
        
        print(f"  ML bounds: {ml_bounds}")
        print(f"  Visualizer bounds: {vis_bounds}")
        
        for col, channel_name in enumerate(channel_names):
            if channel_name in ml_channels and channel_name in vis_channels:
                ml_img = ml_channels[channel_name]
                vis_img = vis_channels[channel_name]
                
                # ML image (top row)
                axes[0, col].imshow(ml_img, cmap='gray', vmin=0, vmax=1)
                axes[0, col].set_title(f'ML: {channel_name.title()}')
                axes[0, col].axis('off')
                
                # Visualizer image (middle row)
                axes[1, col].imshow(vis_img, cmap='gray', vmin=0, vmax=1)
                axes[1, col].set_title(f'Visualizer: {channel_name.title()}')
                axes[1, col].axis('off')
                
                # Difference image (bottom row)
                diff_img = np.abs(ml_img - vis_img)
                max_diff = diff_img.max()
                axes[2, col].imshow(diff_img, cmap='hot', vmin=0, vmax=max(max_diff, 0.1))
                axes[2, col].set_title(f'Diff: {channel_name.title()} (max: {max_diff:.3f})')
                axes[2, col].axis('off')
                
                # Print statistics
                ml_nonzero = np.count_nonzero(ml_img)
                vis_nonzero = np.count_nonzero(vis_img)
                print(f"  {channel_name}:")
                print(f"    ML nonzero: {ml_nonzero}, Visualizer nonzero: {vis_nonzero}")
                print(f"    Max difference: {max_diff:.6f}")
                print(f"    Mean difference: {diff_img.mean():.6f}")
            else:
                # Mark missing channels
                for row in range(3):
                    axes[row, col].text(0.5, 0.5, f'Missing\n{channel_name}', 
                                      ha='center', va='center', transform=axes[row, col].transAxes)
                    axes[row, col].set_title(f'{channel_name.title()}')
                    axes[row, col].axis('off')
        
        plt.suptitle('ML Inference vs Unified Visualizer Comparison', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"  Saved comparison to: {output_path}")
        
        return True
        
    except Exception as e:
        print(f"ERROR: Visualization creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main test function."""
    print("ML Inference vs Visualizer Comparison Test")
    print("=" * 60)
    
    # Import torch here to check CUDA availability
    try:
        import torch
        print(f"PyTorch available: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA device: {torch.cuda.get_device_name()}")
    except ImportError:
        print("PyTorch not available - ML inference may fail")
    
    # Step 1: Set up environment
    env, initial_state, robot_goal = setup_environment()
    if not env:
        return 1
    
    # Step 2: Run object selection
    target_object, reachable_objects = run_object_selection(env, initial_state)
    if not target_object:
        print("No target object selected - continuing with visualization comparison")
    
    # Step 3: Run goal selection (optional - just for demonstration)
    if target_object:
        goals = run_goal_selection(env, initial_state, target_object, robot_goal)
        print(f"Generated {len(goals)} goals for {target_object}")
    
    # Step 4: Create images with ML inference logic
    ml_channels, ml_bounds = create_ml_inference_images(env, initial_state, robot_goal, 
                                                       target_object, reachable_objects)
    if not ml_channels:
        print("Failed to create ML inference images")
        return 1
    
    # Step 5: Create images with unified visualizer
    vis_channels, vis_bounds = create_visualizer_images(env, initial_state, robot_goal, 
                                                       target_object, reachable_objects)
    if not vis_channels:
        print("Failed to create visualizer images")
        return 1
    
    # Step 6: Create comparison visualization
    output_path = "ml_vs_visualizer_comparison.png"
    success = create_comparison_visualization(ml_channels, vis_channels, 
                                            ml_bounds, vis_bounds, output_path)
    
    if success:
        print(f"\n✅ Comparison complete! Check {output_path}")
        return 0
    else:
        print(f"\n❌ Comparison failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())