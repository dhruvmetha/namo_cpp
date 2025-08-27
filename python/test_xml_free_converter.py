#!/usr/bin/env python3
"""
Test the XML-free NAMOImageConverter.

This script tests the updated NAMOImageConverter that uses only the environment interface
and doesn't require any XML parsing.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import namo_rl
from namo_image_converter import NAMOImageConverter

def main():
    # Use one of the available XML files
    config_file = "../config/namo_config_complete.yaml"
    
    # Try available XML files
    xml_candidates = [
        "/common/home/dm1487/robotics_research/ktamp/ml4kp_ktamp/resources/models/custom_walled_envs/set1/benchmark_1/env_config_1.xml",
        "/common/home/dm1487/robotics_research/ktamp/ml4kp_ktamp/resources/models/custom_walled_envs/set1/benchmark_1/env_config_2.xml",
        "../data/test_scene.xml"
    ]
    
    xml_file = None
    for candidate in xml_candidates:
        if os.path.exists(candidate):
            xml_file = candidate
            print(f"Using XML file: {xml_file}")
            break
    
    if xml_file is None:
        print("No suitable XML file found")
        return
    
    print("Creating NAMO RL Environment...")
    try:
        # Create environment with visualization disabled for testing
        env = namo_rl.RLEnvironment(xml_file, config_file, False)
        print("✓ Environment created successfully")
        
        # Create XML-free image converter
        print("\nCreating XML-free NAMOImageConverter...")
        converter = NAMOImageConverter(env)
        print("✓ Converter created successfully")
        
        # Test environment interface calls
        print("\nTesting environment interface calls...")
        
        # Test object info
        object_info = env.get_object_info()
        print(f"✓ Object info retrieved: {len(object_info)} objects")
        for obj_name, obj_data in object_info.items():
            if 'pos_x' in obj_data:  # Static object
                print(f"  Static: {obj_name} at ({obj_data['pos_x']:.2f}, {obj_data['pos_y']:.2f})")
            else:  # Movable object
                print(f"  Movable: {obj_name} size ({obj_data['size_x']:.2f}, {obj_data['size_y']:.2f})")
        
        # Test world bounds
        world_bounds = env.get_world_bounds()
        print(f"✓ World bounds: [{world_bounds[0]:.2f}, {world_bounds[1]:.2f}, {world_bounds[2]:.2f}, {world_bounds[3]:.2f}]")
        
        # Test observations
        obs = env.get_observation()
        print(f"✓ Observations retrieved: {len(obs)} poses")
        
        # Test reachable objects
        reachable = env.get_reachable_objects()
        print(f"✓ Reachable objects: {reachable}")
        
        # Test robot goal
        env.set_robot_goal(1.0, 1.0, 0.0)
        goal = env.get_robot_goal()
        print(f"✓ Robot goal set and retrieved: ({goal[0]:.2f}, {goal[1]:.2f}, {goal[2]:.2f})")
        
        # Test image conversion
        print("\nTesting image conversion...")
        image_channels = converter.convert_state_to_image(env, [goal[0], goal[1]], reachable)
        print(f"✓ Image conversion successful: shape {image_channels.shape}")
        
        # Save visualization
        output_path = "test_xml_free_image.png"
        converter.save_image_visualization(image_channels, output_path)
        print(f"✓ Visualization saved to: {output_path}")
        
        print("\n🎉 All tests passed! XML-free ImageConverter working correctly.")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()