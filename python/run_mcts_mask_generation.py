#!/usr/bin/env python3
"""
MCTS Mask Generation Tools for Neural Network Training

This script generates mask-based datasets from MCTS/AlphaZero data collection
for training neural networks. Similar to run_mask_generation.py but adapted
for the MCTS data format.

Generates datasets for:
- Goal Proposal Networks: P(goal|object,state) as spatial heatmaps
- Value Networks: V(s) and Q(s,a) using masks as input features
- TODO: Object Selection P(o|s) - approach TBD for mask representation

Usage:
    # Run batch collection on MCTS data
    python run_mcts_mask_generation.py batch --input-dir /path/to/alphazero/data --output-dir /path/to/masks
    
    # Run example visualization
    python run_mcts_mask_generation.py example
"""

import sys
import os
import argparse
from pathlib import Path

# Add python directory to path for imports
python_dir = Path(__file__).parent
sys.path.insert(0, str(python_dir))

def run_mcts_batch_collection():
    """Run the MCTS mask collection pipeline."""
    from mcts_mask_generation.batch_collection import main
    main()

def run_mcts_example():
    """Run the MCTS mask generation example."""
    try:
        from mcts_mask_generation.examples.mcts_example_visualization import (
            visualize_mcts_single_episode_example,
            analyze_mcts_goal_proposals_example,
            mcts_value_network_data_example,
            mcts_batch_analysis_example
        )
        
        print("Running MCTS mask generation examples...")
        visualize_mcts_single_episode_example()
        analyze_mcts_goal_proposals_example()
        mcts_value_network_data_example()
        mcts_batch_analysis_example()
        
    except ImportError as e:
        print(f"Error importing MCTS example functions: {e}")
        print("Make sure to create the mcts_mask_generation module first")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(
        description='MCTS/AlphaZero Mask Generation Tools for Neural Network Training',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s batch --input-dir ./alphazero_data --output-dir ./mcts_masks
  %(prog)s example

Generated Datasets:
  - Goal Proposal: P(goal|object,state) as spatial heatmaps
  - Value Networks: V(s) and Q(s,a) with mask features
  - TODO: Object Selection P(o|s) - mask approach TBD
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Batch collection subcommand
    batch_parser = subparsers.add_parser('batch', help='Run MCTS mask collection')
    batch_parser.add_argument('--input-dir', required=True, 
                             help='Directory containing MCTS .pkl files from alphazero_data_collection')
    batch_parser.add_argument('--output-dir', required=True, 
                             help='Output directory for .npz mask files')
    batch_parser.add_argument('--pattern', default='*.pkl', 
                             help='File pattern to match (default: *.pkl)')
    batch_parser.add_argument('--workers', type=int, default=None,
                             help='Number of parallel workers (default: auto-detect)')
    batch_parser.add_argument('--serial', action='store_true',
                             help='Use serial processing (for debugging)')
    batch_parser.add_argument('--goal-proposal-only', action='store_true',
                             help='Only generate goal proposal heatmaps')
    batch_parser.add_argument('--value-network-only', action='store_true', 
                             help='Only generate value network training data')
    
    # Example subcommand
    example_parser = subparsers.add_parser('example', help='Run MCTS mask example')
    
    args = parser.parse_args()
    
    if args.command == 'batch':
        # Set up arguments for MCTS batch collection
        batch_args = [
            'mcts_batch_collection.py',
            '--input-dir', args.input_dir,
            '--output-dir', args.output_dir, 
            '--pattern', args.pattern
        ]
        
        # Add worker arguments if specified
        if args.workers is not None:
            batch_args.extend(['--workers', str(args.workers)])
        if args.serial:
            batch_args.append('--serial')
        if args.goal_proposal_only:
            batch_args.append('--goal-proposal-only')
        if args.value_network_only:
            batch_args.append('--value-network-only')
        
        sys.argv = batch_args
        run_mcts_batch_collection()
        
    elif args.command == 'example':
        run_mcts_example()
        
    else:
        parser.print_help()

if __name__ == "__main__":
    main()