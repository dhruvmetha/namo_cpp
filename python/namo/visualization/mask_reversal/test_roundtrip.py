#!/usr/bin/env python3
"""
Round-Trip Test: XML → NPZ → XML

This test validates the mask reversal system by:
1. Converting XML to NPZ masks (forward)
2. Converting NPZ back to XML (reverse)
3. Comparing the generated XML with the original
"""

import sys
import os
from pathlib import Path
import numpy as np
import argparse

import namo_rl
from namo.visualization.namo_image_converter import NAMOImageConverter
from namo.visualization.mask_reversal import (
    NPZLoader,
    MaskAnalyzer,
    SceneReconstructor,
    XMLGenerator,
    reverse_masks_to_xml
)


def step1_xml_to_npz(TEST_DIR, INPUT_XML, CONFIG_PATH):
    """Step 1: Convert XML to NPZ masks (forward process)."""
    print("="*70)
    print("STEP 1: XML → NPZ (Forward)")
    print("="*70)
    
    # Create output directory
    TEST_DIR.mkdir(parents=True, exist_ok=True)
    
    print(f"\nInput XML: {INPUT_XML}")
    print(f"Config: {CONFIG_PATH}")
    
    # Initialize environment
    print("\n[1.1] Initializing environment...")
    env = namo_rl.RLEnvironment(INPUT_XML, CONFIG_PATH)
    print("  ✓ Environment loaded")
    
    # Create converter
    print("\n[1.2] Creating image converter...")
    converter = NAMOImageConverter(env)
    print("  ✓ Converter created")
    
    # Get robot goal from environment or XML
    obs = env.get_observation()
    robot_goal = (0.0, 0.0, 0.0)
    if hasattr(env, 'get_robot_goal'):
        try:
            robot_goal = tuple(env.get_robot_goal())
        except:
            pass
    
    # Try to parse goal from XML
    import xml.etree.ElementTree as ET
    try:
        tree = ET.parse(INPUT_XML)
        root = tree.getroot()
        goal_site = root.find('.//site[@name="goal"]')
        if goal_site is not None:
            pos_str = goal_site.get('pos', '0 0 0')
            pos = [float(x) for x in pos_str.split()]
            robot_goal = (pos[0], pos[1], 0.0)
            print(f"  ✓ Goal from XML: ({robot_goal[0]:.3f}, {robot_goal[1]:.3f})")
    except Exception as e:
        print(f"  ⚠ Could not parse goal from XML: {e}")
    
    # Get reachable objects
    reachable_objects = env.get_reachable_objects()
    print(f"  ✓ Reachable objects: {len(reachable_objects)}")
    
    # Convert to image masks
    print("\n[1.3] Converting state to masks...")
    image_channels = converter.convert_state_to_image(env, robot_goal, reachable_objects)
    print(f"  ✓ Generated {image_channels.shape[0]} mask channels")
    print(f"  ✓ Mask shape: {image_channels.shape[1]}×{image_channels.shape[2]}")
    
    # Extract individual masks
    masks = {
        'robot': image_channels[0],
        'goal': image_channels[1],
        'movable': image_channels[2],
        'static': image_channels[3],
        'reachable': image_channels[4]
    }
    
    # Print mask statistics
    print("\n[1.4] Mask statistics:")
    for name, mask in masks.items():
        nonzero = np.count_nonzero(mask)
        print(f"  {name:12s}: {nonzero:5d} non-zero pixels (max={mask.max():.3f})")
    
    # Save to NPZ
    output_npz = TEST_DIR / "forward_masks.npz"
    print(f"\n[1.5] Saving to NPZ: {output_npz}")
    
    # Add metadata
    save_dict = {
        'robot': masks['robot'],
        'goal': masks['goal'],
        'movable': masks['movable'],
        'static': masks['static'],
        'reachable': masks['reachable'],
        'episode_id': np.array(['test_env_config_9982e']),
        'robot_goal': np.array(robot_goal, dtype=np.float32),
        'xml_file': np.array([INPUT_XML])
    }
    
    np.savez_compressed(output_npz, **save_dict)
    print(f"  ✓ Saved: {output_npz}")
    print(f"  ✓ File size: {output_npz.stat().st_size / 1024:.1f} KB")
    
    # Save visualization
    vis_path = TEST_DIR / "forward_masks_visualization.png"
    converter.save_image_visualization(image_channels, str(vis_path))
    
    print("\n" + "="*70)
    print("✅ STEP 1 COMPLETE: NPZ masks generated")
    print("="*70)
    
    return output_npz, robot_goal


def step2_npz_to_xml(TEST_DIR, INPUT_XML, CONFIG_PATH, npz_path, original_robot_goal):
    """Step 2: Convert NPZ back to XML (reverse process)."""
    print("\n" + "="*70)
    print("STEP 2: NPZ → XML (Reverse)")
    print("="*70)
    
    output_xml = TEST_DIR / "reconstructed.xml"
    
    # Get world bounds from original XML
    import xml.etree.ElementTree as ET
    try:
        tree = ET.parse(INPUT_XML)
        root = tree.getroot()
        
        # Collect all object positions to estimate bounds
        x_coords = []
        y_coords = []
        
        # Parse walls/geoms
        for geom in root.findall('.//geom[@pos]'):
            pos_str = geom.get('pos', '0 0 0')
            pos = [float(x) for x in pos_str.split()]
            if len(pos) >= 2:
                x_coords.append(pos[0])
                y_coords.append(pos[1])
        
        # Parse bodies with positions
        for body in root.findall('.//body[@pos]'):
            pos_str = body.get('pos', '0 0 0')
            pos = [float(x) for x in pos_str.split()]
            if len(pos) >= 2:
                x_coords.append(pos[0])
                y_coords.append(pos[1])
        
        if x_coords and y_coords:
            padding = 0.5
            world_bounds = (
                min(x_coords) - padding,
                max(x_coords) + padding,
                min(y_coords) - padding,
                max(y_coords) + padding
            )
            print(f"\nEstimated world bounds from XML: {world_bounds}")
        else:
            world_bounds = (-5.5, 5.5, -5.5, 5.5)
            print(f"\nUsing default world bounds: {world_bounds}")
    except Exception as e:
        world_bounds = (-5.5, 5.5, -5.5, 5.5)
        print(f"\nUsing default world bounds: {world_bounds}")
        print(f"  (Could not parse XML: {e})")
    
    print(f"\nInput NPZ: {npz_path}")
    print(f"Output XML: {output_xml}")
    
    # Run reversal
    success = reverse_masks_to_xml(
        str(npz_path),
        str(output_xml),
        world_bounds=world_bounds,
        visualize=True
    )
    
    if success:
        print("\n" + "="*70)
        print("✅ STEP 2 COMPLETE: XML reconstructed")
        print("="*70)
        return output_xml
    else:
        print("\n" + "="*70)
        print("❌ STEP 2 FAILED: XML reconstruction failed")
        print("="*70)
        return None


def step3_compare_xmls(TEST_DIR, INPUT_XML, CONFIG_PATH, reconstructed_xml):
    original_xml = INPUT_XML
    """Step 3: Compare original and reconstructed XML files."""
    print("\n" + "="*70)
    print("STEP 3: Validation (Compare XMLs)")
    print("="*70)
    
    print(f"\nOriginal XML: {original_xml}")
    print(f"Reconstructed XML: {reconstructed_xml}")
    
    import xml.etree.ElementTree as ET
    
    # Parse both XMLs
    try:
        original_tree = ET.parse(original_xml)
        original_root = original_tree.getroot()
        print("  ✓ Parsed original XML")
    except Exception as e:
        print(f"  ❌ Failed to parse original XML: {e}")
        return False
    
    try:
        reconstructed_tree = ET.parse(reconstructed_xml)
        reconstructed_root = reconstructed_tree.getroot()
        print("  ✓ Parsed reconstructed XML")
    except Exception as e:
        print(f"  ❌ Failed to parse reconstructed XML: {e}")
        return False
    
    # Compare structure
    print("\n[3.1] Comparing XML structure...")
    
    # Count objects
    def count_elements(root, tag):
        return len(root.findall(f'.//{tag}'))
    
    comparison = {
        'geoms': ('geom', 'Static objects + walls'),
        'bodies': ('body', 'Bodies (robot + movables)'),
        'freejoints': ('freejoint', 'Movable objects'),
        'sites': ('site', 'Goal markers'),
    }
    
    all_match = True
    for name, (tag, description) in comparison.items():
        orig_count = count_elements(original_root, tag)
        recon_count = count_elements(reconstructed_root, tag)
        match = "✓" if orig_count == recon_count else "✗"
        if orig_count != recon_count:
            all_match = False
        print(f"  {match} {description:30s}: {orig_count} → {recon_count}")
    
    # Compare robot position
    print("\n[3.2] Comparing robot position...")
    orig_robot = original_root.find('.//body[@name="robot"]')
    recon_robot = reconstructed_root.find('.//body[@name="robot"]')
    
    if orig_robot is not None and recon_robot is not None:
        # Get robot position - check body pos first, then geom pos if body pos is default
        def get_robot_world_pos(robot_body):
            body_pos = [float(x) for x in robot_body.get('pos', '0 0 0').split()]
            # If body is at origin, check if geom has position
            if body_pos[0] == 0.0 and body_pos[1] == 0.0:
                robot_geom = robot_body.find('.//geom[@name="robot"]')
                if robot_geom is not None:
                    geom_pos = [float(x) for x in robot_geom.get('pos', '0 0 0').split()]
                    # Geom pos is relative to body, so add them
                    return [body_pos[i] + geom_pos[i] for i in range(3)]
            return body_pos
        
        orig_pos = get_robot_world_pos(orig_robot)
        recon_pos = get_robot_world_pos(recon_robot)
        
        diff = np.linalg.norm(np.array(orig_pos[:2]) - np.array(recon_pos[:2]))
        print(f"  Original: ({orig_pos[0]:.3f}, {orig_pos[1]:.3f})")
        print(f"  Reconstructed: ({recon_pos[0]:.3f}, {recon_pos[1]:.3f})")
        print(f"  Difference: {diff:.4f} m")
        
        if diff < 0.1:  # 10cm tolerance
            print("  ✓ Robot position matches (within 10cm tolerance)")
        else:
            print("  ✗ Robot position differs significantly")
            all_match = False
    else:
        print("  ⚠ Could not find robot in one or both XMLs")
    
    # Compare goal position
    print("\n[3.3] Comparing goal position...")
    orig_goal = original_root.find('.//site[@name="goal"]')
    recon_goal = reconstructed_root.find('.//site[@name="goal"]')
    
    if orig_goal is not None and recon_goal is not None:
        orig_pos = [float(x) for x in orig_goal.get('pos', '0 0 0').split()]
        recon_pos = [float(x) for x in recon_goal.get('pos', '0 0 0').split()]
        
        diff = np.linalg.norm(np.array(orig_pos[:2]) - np.array(recon_pos[:2]))
        print(f"  Original: ({orig_pos[0]:.3f}, {orig_pos[1]:.3f})")
        print(f"  Reconstructed: ({recon_pos[0]:.3f}, {recon_pos[1]:.3f})")
        print(f"  Difference: {diff:.4f} m")
        
        if diff < 0.1:  # 10cm tolerance
            print("  ✓ Goal position matches (within 10cm tolerance)")
        else:
            print("  ✗ Goal position differs significantly")
            all_match = False
    else:
        print("  ⚠ Could not find goal in one or both XMLs")
    
    # Compare movable objects
    print("\n[3.4] Comparing movable objects...")
    orig_movables = [b for b in original_root.findall('.//body') 
                     if b.find('freejoint') is not None and b.get('name') != 'robot']
    recon_movables = [b for b in reconstructed_root.findall('.//body') 
                      if b.find('freejoint') is not None and b.get('name') != 'robot']
    
    print(f"  Original: {len(orig_movables)} movable objects")
    print(f"  Reconstructed: {len(recon_movables)} movable objects")
    
    if len(orig_movables) == len(recon_movables):
        print(f"  ✓ Movable object count matches")
    else:
        print(f"  ✗ Movable object count differs")
        all_match = False
    
    # Save comparison report
    report_path = TEST_DIR / "comparison_report.txt"
    with open(report_path, 'w') as f:
        f.write("="*70 + "\n")
        f.write("XML Comparison Report\n")
        f.write("="*70 + "\n\n")
        f.write(f"Original: {original_xml}\n")
        f.write(f"Reconstructed: {reconstructed_xml}\n\n")
        f.write(f"Structure match: {'YES' if all_match else 'NO'}\n")
        f.write(f"Object counts:\n")
        for name, (tag, description) in comparison.items():
            orig_count = count_elements(original_root, tag)
            recon_count = count_elements(reconstructed_root, tag)
            f.write(f"  {description}: {orig_count} → {recon_count}\n")
    
    print(f"\n  ✓ Comparison report saved: {report_path}")
    
    print("\n" + "="*70)
    if all_match:
        print("✅ STEP 3 COMPLETE: XMLs match!")
    else:
        print("⚠️  STEP 3 COMPLETE: XMLs differ (see details above)")
    print("="*70)
    
    return all_match


def main():
    _default_test_dir = Path(__file__).parent / "test_output"
    _default_input_xml = "/common/users/shared/robot_learning/dm1487/namo/mj_env_configs/aug9/easy/set1/benchmark_1/env_config_9982e.xml"
    _default_config = "/common/users/tdn39/Robotics/Mujoco/namo_cpp/config/namo_config_complete.yaml"
    parser = argparse.ArgumentParser(description="Round-trip mask reversal test")
    parser.add_argument("-i", "--input-xml", dest="input_xml", type=str,
                        default=os.environ.get("TEST_INPUT_XML", _default_input_xml),
                        help="Path to input XML to run the round-trip on")
    parser.add_argument("-c", "--config", dest="config_path", type=str,
                        default=os.environ.get("NAMO_CONFIG_PATH", _default_config),
                        help="Path to NAMO config YAML")
    parser.add_argument("-o", "--test-dir", dest="test_dir", type=str,
                        default=os.environ.get("TEST_OUTPUT_DIR", str(_default_test_dir)),
                        help="Output directory for round-trip artifacts")
    args = parser.parse_args()
    
    TEST_DIR = Path(args.test_dir)
    INPUT_XML = str(args.input_xml)
    CONFIG_PATH = str(args.config_path)

    """Run complete round-trip test."""
    print("\n" + "="*70)
    print("ROUND-TRIP TEST: XML → NPZ → XML")
    print("="*70)
    print(f"\nTest input: {INPUT_XML}")
    print(f"Test output directory: {TEST_DIR}")
    print("="*70)
    
    try:
        # Step 1: XML → NPZ
        npz_path, robot_goal = step1_xml_to_npz(TEST_DIR, INPUT_XML, CONFIG_PATH)
        
        # Step 2: NPZ → XML
        reconstructed_xml = step2_npz_to_xml(TEST_DIR, INPUT_XML, CONFIG_PATH, npz_path, robot_goal)
        
        if reconstructed_xml is None:
            print("\n❌ TEST FAILED: Could not reconstruct XML")
            return False
        
        # Step 3: Compare
        match = step3_compare_xmls(TEST_DIR, INPUT_XML, CONFIG_PATH, reconstructed_xml)
        
        # Final summary
        print("\n" + "="*70)
        print("ROUND-TRIP TEST SUMMARY")
        print("="*70)
        print(f"Input XML: {INPUT_XML}")
        print(f"Generated NPZ: {npz_path}")
        print(f"Reconstructed XML: {reconstructed_xml}")
        print(f"Comparison result: {'✅ MATCH' if match else '⚠️  DIFFER'}")
        print("\nGenerated files:")
        for file in sorted(TEST_DIR.glob("*")):
            print(f"  - {file.name}")
        print("="*70)
        
        return match
        
    except Exception as e:
        print(f"\n❌ TEST FAILED WITH EXCEPTION:")
        print(f"{type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
