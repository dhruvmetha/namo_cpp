"""
Mask Reversal Pipeline - Main entry point for reversing masks to XML.

This module provides the high-level API and CLI for converting NPZ mask files
back to MuJoCo XML environment files.
"""

import argparse
import sys
from pathlib import Path
from typing import List, Optional, Tuple

from .npz_loader import NPZLoader, MaskData
from .mask_analyzer import MaskAnalyzer
from .scene_reconstructor import SceneReconstructor
from .xml_generator import XMLGenerator


def reverse_masks_to_xml(npz_path: str, 
                         output_path: str,
                         world_bounds: Optional[Tuple[float, float, float, float]] = None,
                         visualize: bool = False) -> bool:
    """Reverse a single NPZ file to XML.
    
    Args:
        npz_path: Path to input NPZ file
        output_path: Path to save output XML file
        world_bounds: Optional world bounds (x_min, x_max, y_min, y_max)
        visualize: Whether to save visualization of detected objects
        
    Returns:
        True if successful, False otherwise
    """
    try:
        print(f"\n{'='*60}")
        print(f"Processing: {npz_path}")
        print(f"{'='*60}")
        
        # Step 1: Load NPZ file
        print("\n[1/5] Loading NPZ file...")
        mask_data = NPZLoader.load(npz_path)
        print(f"  ✓ Loaded masks: robot, goal, movable, static")
        if mask_data.episode_id:
            print(f"  Episode ID: {mask_data.episode_id}")
        
        # Step 2: Analyze masks to detect objects
        print("\n[2/5] Analyzing masks...")
        analyzer = MaskAnalyzer(min_area=10, max_area=20000)
        detections = analyzer.analyze_all_masks(
            mask_data.robot,
            mask_data.goal,
            mask_data.movable,
            mask_data.static
        )
        
        print(f"  ✓ Detected {len(detections['robot'])} robot(s)")
        print(f"  ✓ Detected {len(detections['goal'])} goal(s)")
        print(f"  ✓ Detected {len(detections['movable'])} movable object(s)")
        print(f"  ✓ Detected {len(detections['static'])} static object(s)")
        
        # Optional: visualize detections
        if visualize:
            vis_path = str(output_path).replace('.xml', '_detections.png')
            analyzer.visualize_detections(detections, vis_path)
        
        # Step 3: Reconstruct scene in world coordinates
        print("\n[3/5] Reconstructing world coordinates...")
        reconstructor = SceneReconstructor(world_bounds=world_bounds)
        scene = reconstructor.reconstruct_scene(detections, mask_data.robot_goal)
        
        print(f"  ✓ Robot at: ({scene.robot_position[0]:.3f}, {scene.robot_position[1]:.3f})")
        print(f"  ✓ Goal at: ({scene.goal_position[0]:.3f}, {scene.goal_position[1]:.3f})")
        print(f"  ✓ {len(scene.movable_objects)} movable objects")
        print(f"  ✓ {len(scene.static_objects)} static objects")
        print(f"  World bounds: {scene.world_bounds}")
        
        # Step 4: Generate XML
        print("\n[4/5] Generating MuJoCo XML...")
        generator = XMLGenerator()
        
        # Use episode_id as model name if available
        model_name = mask_data.episode_id if mask_data.episode_id else "reconstructed_environment"
        
        # Step 5: Save XML
        print("\n[5/5] Saving XML file...")
        generator.save_xml(scene, output_path, model_name)
        print(f"  ✓ XML saved to: {output_path}")
        
        print(f"\n{'='*60}")
        print(f"✅ SUCCESS: Converted {npz_path} -> {output_path}")
        print(f"{'='*60}\n")
        
        return True
        
    except Exception as e:
        print(f"\n❌ ERROR: Failed to process {npz_path}")
        print(f"  {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False


def batch_reverse_masks(input_dir: str,
                       output_dir: str,
                       pattern: str = "*.npz",
                       world_bounds: Optional[Tuple[float, float, float, float]] = None,
                       visualize: bool = False) -> Tuple[int, int]:
    """Reverse multiple NPZ files to XML.
    
    Args:
        input_dir: Directory containing NPZ files
        output_dir: Directory to save XML files
        pattern: File pattern to match (default "*.npz")
        world_bounds: Optional world bounds for all files
        visualize: Whether to save visualizations
        
    Returns:
        (successful_count, failed_count)
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    if not input_path.exists():
        print(f"Error: Input directory does not exist: {input_dir}")
        return (0, 0)
    
    # Find all NPZ files
    npz_files = list(input_path.rglob(pattern))
    
    if not npz_files:
        print(f"No files found matching pattern '{pattern}' in {input_dir}")
        return (0, 0)
    
    print(f"\n{'='*60}")
    print(f"Batch Mask Reversal")
    print(f"{'='*60}")
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Found {len(npz_files)} NPZ files")
    print(f"{'='*60}\n")
    
    # Create output directory
    output_path.mkdir(parents=True, exist_ok=True)
    
    successful = 0
    failed = 0
    
    for i, npz_file in enumerate(npz_files, 1):
        print(f"\n[{i}/{len(npz_files)}] Processing: {npz_file.name}")
        
        # Generate output path maintaining directory structure
        rel_path = npz_file.relative_to(input_path)
        xml_path = output_path / rel_path.with_suffix('.xml')
        
        # Create subdirectory if needed
        xml_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Process file
        if reverse_masks_to_xml(str(npz_file), str(xml_path), world_bounds, visualize):
            successful += 1
        else:
            failed += 1
    
    # Summary
    print(f"\n{'='*60}")
    print(f"Batch Processing Complete")
    print(f"{'='*60}")
    print(f"✅ Successful: {successful}/{len(npz_files)}")
    print(f"❌ Failed: {failed}/{len(npz_files)}")
    print(f"{'='*60}\n")
    
    return (successful, failed)


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Reverse NPZ mask files to MuJoCo XML environments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single file
  python -m namo.visualization.mask_reversal.reverse_masks \\
      --input episode.npz --output reconstructed.xml
  
  # Batch processing
  python -m namo.visualization.mask_reversal.reverse_masks \\
      --batch --input-dir ./masks --output-dir ./xml_files
  
  # With known world bounds
  python -m namo.visualization.mask_reversal.reverse_masks \\
      --input episode.npz --output reconstructed.xml \\
      --world-bounds -5.5 5.5 -5.5 5.5
        """
    )
    
    # Mode selection
    parser.add_argument('--batch', action='store_true',
                       help='Batch processing mode')
    
    # Single file mode
    parser.add_argument('--input', type=str,
                       help='Input NPZ file (single file mode)')
    parser.add_argument('--output', type=str,
                       help='Output XML file (single file mode)')
    
    # Batch mode
    parser.add_argument('--input-dir', type=str,
                       help='Input directory with NPZ files (batch mode)')
    parser.add_argument('--output-dir', type=str,
                       help='Output directory for XML files (batch mode)')
    parser.add_argument('--pattern', type=str, default='*.npz',
                       help='File pattern to match (batch mode, default: *.npz)')
    
    # Optional parameters
    parser.add_argument('--world-bounds', type=float, nargs=4,
                       metavar=('X_MIN', 'X_MAX', 'Y_MIN', 'Y_MAX'),
                       help='World bounds: x_min x_max y_min y_max')
    parser.add_argument('--visualize', action='store_true',
                       help='Save visualization of detected objects')
    
    args = parser.parse_args()
    
    # Parse world bounds
    world_bounds = None
    if args.world_bounds:
        world_bounds = tuple(args.world_bounds)
    
    # Validate arguments
    if args.batch:
        if not args.input_dir or not args.output_dir:
            parser.error("Batch mode requires --input-dir and --output-dir")
        
        successful, failed = batch_reverse_masks(
            args.input_dir,
            args.output_dir,
            args.pattern,
            world_bounds,
            args.visualize
        )
        
        sys.exit(0 if failed == 0 else 1)
    else:
        if not args.input or not args.output:
            parser.error("Single file mode requires --input and --output")
        
        success = reverse_masks_to_xml(
            args.input,
            args.output,
            world_bounds,
            args.visualize
        )
        
        sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
