#!/usr/bin/env python3
"""
Convenience runner script for NAMO mask generation tools.

This script provides easy access to mask generation functionality without
needing to manage imports or package paths.

Usage:
    # Run batch collection
    python run_mask_generation.py batch --input-dir /path/to/pkl/files --output-dir /path/to/output

    # Run example visualization  
    python run_mask_generation.py example

    # Show help
    python run_mask_generation.py --help
"""

import sys
import os
import argparse
from pathlib import Path

# Add python directory to path for imports
python_dir = Path(__file__).parent
sys.path.insert(0, str(python_dir))

def run_batch_collection():
    """Run the batch mask collection pipeline."""
    from mask_generation.batch_collection import main
    main()

def run_example():
    """Run the example visualization script."""
    try:
        from mask_generation.examples.example_visualization import (
            visualize_single_episode_example,
            analyze_episode_data_example,
            xml_parsing_example,
            batch_analysis_example
        )
        
        print("Running NAMO mask generation examples...")
        visualize_single_episode_example()
        analyze_episode_data_example()
        xml_parsing_example()
        batch_analysis_example()
        
    except ImportError as e:
        print(f"Error importing example functions: {e}")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(
        description='NAMO Mask Generation Tools',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s batch --input-dir /data/pkl --output-dir /output
  %(prog)s example
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Batch collection subcommand
    batch_parser = subparsers.add_parser('batch', help='Run batch mask collection')
    batch_parser.add_argument('--input-dir', required=True, help='Directory containing .pkl files')
    batch_parser.add_argument('--output-dir', required=True, help='Output directory for .npz files')
    batch_parser.add_argument('--pattern', default='**/*_results.pkl', help='File pattern to match')
    batch_parser.add_argument('--workers', type=int, default=None, 
                             help='Number of parallel workers (default: auto-detect)')
    batch_parser.add_argument('--serial', action='store_true', 
                             help='Use serial processing (for debugging)')
    batch_parser.add_argument('--filter-minimum-length', action='store_true',
                             help='Only process episodes with minimum action sequence length per environment')
    
    # Example subcommand
    example_parser = subparsers.add_parser('example', help='Run example visualization')
    
    args = parser.parse_args()
    
    if args.command == 'batch':
        # Set up arguments for batch collection
        batch_args = [
            'batch_collection.py',
            '--input-dir', args.input_dir,
            '--output-dir', args.output_dir,
            '--pattern', args.pattern
        ]
        
        # Add worker arguments if specified
        if args.workers is not None:
            batch_args.extend(['--workers', str(args.workers)])
        if args.serial:
            batch_args.append('--serial')
        if args.filter_minimum_length:
            batch_args.append('--filter-minimum-length')
        
        sys.argv = batch_args
        run_batch_collection()
        
    elif args.command == 'example':
        run_example()
        
    else:
        parser.print_help()

if __name__ == "__main__":
    main()