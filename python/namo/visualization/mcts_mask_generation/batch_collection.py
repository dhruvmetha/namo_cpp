"""
Batch processing for MCTS mask generation.

This module handles parallel processing of MCTS episode files to generate
mask-based training datasets for neural networks.
"""

import os
import sys
import pickle
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import argparse

from .mcts_visualizer import MCTSMaskGenerator


def extract_mcts_training_samples(episode_file: Path) -> List[Dict[str, Any]]:
    """Extract training samples from MCTS episode data file."""
    try:
        with open(episode_file, 'rb') as f:
            episode_data = pickle.load(f)
        
        samples = []
        
        # MCTS data has step_data list with training data per planning step
        for step_idx, step_data in enumerate(episode_data.get('step_data', [])):
            sample = {
                'episode_id': episode_data.get('episode_id', 'unknown'),
                'step': step_idx,
                'xml_file': episode_data.get('xml_file', ''),
                'scene_observation': step_data['scene_observation'],
                'robot_goal': step_data['robot_goal'],
                'object_proposals': step_data['object_proposals'],
                'goal_proposals': step_data['goal_proposals'],
                'state_value': step_data['state_value'],
                'object_q_values': step_data['object_q_values'],
                'goal_q_values': step_data['goal_q_values'],
                'reachable_objects': step_data['reachable_objects'],
                'static_object_info': step_data.get('static_object_info', {}),
                'success': episode_data.get('success', False),
                'total_steps': episode_data.get('total_steps', 0)
            }
            
            # Add post-action poses if available (from final_action_sequence)
            post_action_poses = None
            if (step_idx < len(episode_data.get('final_action_sequence', [])) and 
                'post_action_poses' in episode_data['final_action_sequence'][step_idx]):
                post_action_poses = episode_data['final_action_sequence'][step_idx]['post_action_poses']
            
            sample['post_action_poses'] = post_action_poses
            samples.append(sample)
        
        return samples
        
    except Exception as e:
        print(f"Error extracting samples from {episode_file}: {e}")
        return []


def process_dynamics_transitions(sample, visualizer, xml_base_path: str):
    """Generate dynamics transition data from MCTS sample.
    
    NOTE: Currently returns empty list because transition data is not yet collected during MCTS.
    This is a placeholder for when we modify the MCTS collection to capture state transitions.
    """
    # TODO: Implement when MCTS collection includes transition data
    # For now, return empty list since transitions aren't captured yet
    return []
    
    # Future implementation when transitions are available:
    # transitions = []
    # if 'mcts_transitions' in sample:
    #     for transition in sample['mcts_transitions']:
    #         before_masks = visualizer.generate_state_masks(transition['before_state'])
    #         after_masks = visualizer.generate_state_masks(transition['after_state'])
    #         
    #         transition_data = {
    #             **{f'before_{k}': v for k, v in before_masks.items()},
    #             **{f'after_{k}': v for k, v in after_masks.items()},
    #             'action_object_id': transition['action']['object_id'],
    #             'action_target': np.array(transition['action']['target']),
    #             'reward': transition.get('reward', 0.0),
    #             'terminal': transition.get('terminal', False)
    #         }
    #         transitions.append(transition_data)
    # 
    # return transitions


def process_mcts_episode_file(episode_file: Path, output_dir: Path, 
                            xml_base_path: str = "../ml4kp_ktamp/resources/models/",
                            goal_proposal_only: bool = False,
                            value_network_only: bool = False,
                            dynamics_only: bool = False) -> bool:
    """Process single MCTS episode file and generate mask datasets."""
    try:
        # Extract training samples
        samples = extract_mcts_training_samples(episode_file)
        
        if not samples:
            print(f"No training samples in {episode_file}")
            return False
        
        processed_samples = 0
        
        for sample in samples:
            try:
                # Check if we have static_object_info (no XML needed)
                if 'static_object_info' in sample and sample['static_object_info']:
                    # Use NAMODataVisualizer directly with static_object_info
                    from mask_generation.visualizer import NAMODataVisualizer
                    visualizer = NAMODataVisualizer()
                    
                    # Create fake episode data for existing visualizer
                    fake_episode = {
                        'episode_id': sample['episode_id'],
                        'xml_file': sample['xml_file'],  # Keep for reference but not used
                        'robot_goal': sample['robot_goal'],
                        'state_observations': [sample['scene_observation']],
                        'static_object_info': sample['static_object_info'],
                        'action_sequence': []  # Empty for current state
                    }
                    
                    mask_gen = None  # We'll use visualizer directly
                    
                else:
                    # Fallback to XML-based approach
                    xml_file = sample['xml_file']
                    if not xml_file.startswith('/'):
                        full_xml_path = os.path.join(xml_base_path, xml_file)
                    else:
                        full_xml_path = xml_file
                    
                    if not os.path.exists(full_xml_path):
                        print(f"XML file not found: {full_xml_path}")
                        continue
                    
                    # Initialize mask generator (old way)
                    mask_gen = MCTSMaskGenerator(full_xml_path, "config/namo_config_complete.yaml")
                    visualizer = None
                
                episode_id = sample['episode_id']
                step = sample['step']
                
                # Generate different types of training data
                if not value_network_only:
                    # Generate goal proposal data for diffusion training (one file per proposal)
                    for object_id, goal_proposals in sample['goal_proposals'].items():
                        if goal_proposals:  # Only if object has goal proposals
                            if visualizer:
                                # Use visualizer directly with static_object_info
                                for i, goal_data in enumerate(goal_proposals):
                                    goal_pos = goal_data['goal_position']
                                    probability = goal_data['probability']
                                    visit_count = goal_data['visit_count']
                                    q_value = goal_data['q_value']
                                    
                                    # Create fake episode with target goal
                                    goal_episode = fake_episode.copy()
                                    goal_episode['action_sequence'] = [{'object_id': object_id, 'target': goal_pos}]
                                    
                                    # Generate masks using existing visualizer
                                    masks = visualizer.generate_episode_masks(goal_episode)
                                    
                                    # Add MCTS statistics (no post-action masks for clean dataset)
                                    goal_sample = {
                                        **masks,
                                        'goal_probability': np.float32(probability),
                                        'goal_q_value': np.float32(q_value),
                                        'goal_visit_count': np.int32(visit_count),
                                        'object_id': object_id,
                                        'proposal_index': i,
                                        'goal_coordinates': np.array(goal_pos, dtype=np.float32)
                                    }
                                    
                                    output_file = output_dir / f"{episode_id}_step_{step:02d}_goal_{object_id}_proposal_{i}.npz"
                                    np.savez_compressed(output_file, **goal_sample)
                            else:
                                # Use old MCTSMaskGenerator with post-action poses
                                training_samples = mask_gen.generate_goal_proposal_data(
                                    scene_observation=sample['scene_observation'],
                                    robot_goal=sample['robot_goal'],
                                    object_id=object_id,
                                    goal_proposals=goal_proposals,
                                    post_action_poses=sample.get('post_action_poses'),
                                    static_object_info=sample.get('static_object_info', {})
                                )
                                
                                # Save each goal proposal as separate file for diffusion training
                                for i, goal_sample in enumerate(training_samples):
                                    output_file = output_dir / f"{episode_id}_step_{step:02d}_goal_{object_id}_proposal_{i}.npz"
                                    np.savez_compressed(output_file, **goal_sample)
                
                if not goal_proposal_only:
                    # Generate value network training data
                    if visualizer:
                        # Use visualizer directly
                        value_masks = visualizer.generate_episode_masks(fake_episode)
                        value_data = {
                            **value_masks,
                            'state_value': np.float32(sample['state_value']),
                            'object_q_values': sample['object_q_values']
                        }
                        
                        # FIXED: Add post-action masks to value data if available
                        if sample.get('post_action_poses'):
                            # Use MCTSMaskGenerator which has the proper post-action logic
                            xml_file = sample['xml_file']
                            if not xml_file.startswith('/'):
                                full_xml_path = os.path.join(xml_base_path, xml_file)
                            else:
                                full_xml_path = xml_file
                                
                            # Clean value dataset - no post-action masks needed
                    else:
                        # Use old MCTSMaskGenerator
                        value_data = mask_gen.generate_value_network_data(
                            scene_observation=sample['scene_observation'],
                            robot_goal=sample['robot_goal'],
                            state_value=sample['state_value'],
                            object_q_values=sample['object_q_values']
                        )
                        
                        # Clean value dataset - no post-action masks
                    
                    # Save value network data
                    output_file = output_dir / f"{episode_id}_step_{step:02d}_value.npz"
                    np.savez_compressed(output_file, **value_data)
                
                # Dataset 3: Dynamics transitions (if available and requested)
                if not goal_proposal_only and not value_network_only:
                    dynamics_data = process_dynamics_transitions(sample, visualizer, xml_base_path)
                    if dynamics_data:
                        dynamics_count = len(dynamics_data)
                        print(f"  Generated {dynamics_count} dynamics transitions")
                        for i, transition_data in enumerate(dynamics_data):
                            output_file = output_dir / f"{episode_id}_step_{step:02d}_transition_{i:03d}.npz"
                            np.savez_compressed(output_file, **transition_data)
                
                # TODO: Object selection data (when mask approach is designed)
                # object_data = mask_gen.generate_object_selection_data(...)
                
                processed_samples += 1
                
            except Exception as e:
                print(f"Error processing sample {sample.get('episode_id', 'unknown')}_step_{sample.get('step', 0)}: {e}")
                continue
        
        if processed_samples > 0:
            print(f"✅ Processed {processed_samples} samples from {episode_file.name}")
            return True
        else:
            print(f"❌ No samples processed from {episode_file.name}")
            return False
            
    except Exception as e:
        print(f"❌ Error processing {episode_file}: {e}")
        return False


def process_worker(args):
    """Worker function for parallel processing."""
    episode_file, output_dir, xml_base_path, goal_proposal_only, value_network_only = args
    return process_mcts_episode_file(episode_file, output_dir, xml_base_path, 
                                   goal_proposal_only, value_network_only)


def process_mcts_episode_batch(input_dir: str, output_dir: str,
                             pattern: str = "*.pkl",
                             workers: Optional[int] = None,
                             serial: bool = False,
                             goal_proposal_only: bool = False,
                             value_network_only: bool = False,
                             xml_base_path: str = "../ml4kp_ktamp/resources/models/"):
    """Process batch of MCTS episode files to generate mask datasets."""
    
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    # Create output directory
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Find MCTS episode files
    episode_files = list(input_path.glob(pattern))
    
    if not episode_files:
        print(f"No files found matching pattern '{pattern}' in {input_dir}")
        return
    
    print(f"🔍 Found {len(episode_files)} MCTS episode files")
    print(f"📂 Output directory: {output_path}")
    if goal_proposal_only:
        print("🎯 Mode: Goal proposal heatmaps only")
    elif value_network_only:
        print("💰 Mode: Value network data only")
    else:
        print("🌟 Mode: All datasets (goal proposal + value networks)")
    
    # Determine number of workers
    if workers is None:
        workers = max(1, cpu_count() - 2)
    
    print(f"👥 Using {workers} workers")
    
    # Process files
    successful = 0
    failed = 0
    
    if serial or workers == 1:
        # Serial processing
        for episode_file in tqdm(episode_files, desc="Processing episodes"):
            if process_mcts_episode_file(episode_file, output_path, xml_base_path,
                                       goal_proposal_only, value_network_only):
                successful += 1
            else:
                failed += 1
    else:
        # Parallel processing
        args_list = [
            (episode_file, output_path, xml_base_path, goal_proposal_only, value_network_only)
            for episode_file in episode_files
        ]
        
        with Pool(processes=workers) as pool:
            results = list(tqdm(
                pool.imap(process_worker, args_list),
                total=len(args_list),
                desc="Processing episodes"
            ))
            
            successful = sum(results)
            failed = len(results) - successful
    
    print(f"\n🎉 MCTS mask generation complete!")
    print(f"✅ Successfully processed: {successful}")
    print(f"❌ Failed: {failed}")
    
    # Show output summary
    output_files = list(output_path.glob("*.npz"))
    goal_files = len([f for f in output_files if "_goal_" in f.name])
    value_files = len([f for f in output_files if "_value.npz" in f.name])
    
    print(f"\n📊 Generated datasets:")
    print(f"   Goal proposal files: {goal_files}")
    print(f"   Value network files: {value_files}")
    print(f"   Total .npz files: {len(output_files)}")


def main():
    """Main entry point for MCTS batch mask generation."""
    parser = argparse.ArgumentParser(description="MCTS Mask Generation Batch Processing")
    
    parser.add_argument('--input-dir', required=True,
                       help='Directory containing MCTS .pkl files')
    parser.add_argument('--output-dir', required=True,
                       help='Output directory for .npz mask files')
    parser.add_argument('--pattern', default='*.pkl',
                       help='File pattern to match (default: *.pkl)')
    parser.add_argument('--workers', type=int, default=None,
                       help='Number of parallel workers (default: auto-detect)')
    parser.add_argument('--serial', action='store_true',
                       help='Use serial processing (for debugging)')
    parser.add_argument('--goal-proposal-only', action='store_true',
                       help='Only generate goal proposal heatmaps')
    parser.add_argument('--value-network-only', action='store_true',
                       help='Only generate value network training data')
    parser.add_argument('--xml-base-path', default="../ml4kp_ktamp/resources/models/",
                       help='Base path for XML files')
    
    args = parser.parse_args()
    
    if args.goal_proposal_only and args.value_network_only:
        print("❌ Error: Cannot specify both --goal-proposal-only and --value-network-only")
        return 1
    
    try:
        process_mcts_episode_batch(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            pattern=args.pattern,
            workers=args.workers,
            serial=args.serial,
            goal_proposal_only=args.goal_proposal_only,
            value_network_only=args.value_network_only,
            xml_base_path=args.xml_base_path
        )
        return 0
        
    except KeyboardInterrupt:
        print("\n🛑 Processing interrupted by user")
        return 1
    except Exception as e:
        print(f"💥 Fatal error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())