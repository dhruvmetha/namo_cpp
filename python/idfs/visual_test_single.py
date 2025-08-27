#!/usr/bin/env python3
"""Visual Single-Run IDFS Tester

A simplified script for running single IDFS planning iterations with visualization.
Perfect for visual testing and debugging different algorithms and strategies.

Usage:
    python visual_test_single.py --xml-file ../data/test_scene.xml --algorithm idfs --object-strategy no_heuristic
"""

import os
import sys
import argparse
import time
import traceback
from typing import List, Dict, Tuple, Optional

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# NAMO imports
import namo_rl
from idfs.base_planner import BasePlanner, PlannerConfig, PlannerResult, PlannerFactory
from xml_goal_parser import extract_goal_with_fallback

# Import and register all available planners
from idfs.standard_idfs import StandardIterativeDeepeningDFS
from idfs.tree_idfs import TreeIterativeDeepeningDFS
from idfs.random_sampling import RandomSamplingPlanner


def get_available_algorithms() -> List[str]:
    """Get list of available planning algorithms."""
    return PlannerFactory.list_available_planners()


def get_available_object_strategies() -> List[str]:
    """Get list of available object selection strategies."""
    return ["no_heuristic", "nearest_first", "goal_proximity", "farthest_first", "ml"]


def get_available_goal_strategies() -> List[str]:
    """Get list of available goal selection strategies."""
    return ["random", "grid", "adaptive", "ml"]


def preload_ml_models(object_model_path: Optional[str], 
                     goal_model_path: Optional[str],
                     device: str = "cuda") -> Tuple[Optional[any], Optional[any]]:
    """Preload ML models if paths are provided."""
    object_model = None
    goal_model = None
    
    if object_model_path:
        try:
            # Add learning package to path
            learning_path = "/common/home/dm1487/robotics_research/ktamp/learning"
            if learning_path not in sys.path:
                sys.path.append(learning_path)
            
            from ktamp_learning.object_inference_model import ObjectInferenceModel
            print(f"🔮 Loading ObjectInferenceModel from {object_model_path}")
            object_model = ObjectInferenceModel(model_path=object_model_path, device=device)
            print(f"✅ Object model loaded successfully")
        except Exception as e:
            print(f"❌ Failed to load object model: {e}")
            return None, None
    
    if goal_model_path:
        try:
            # Add learning package to path
            learning_path = "/common/home/dm1487/robotics_research/ktamp/learning"
            if learning_path not in sys.path:
                sys.path.append(learning_path)
            
            from ktamp_learning.goal_inference_model import GoalInferenceModel
            print(f"🎯 Loading GoalInferenceModel from {goal_model_path}")
            goal_model = GoalInferenceModel(model_path=goal_model_path, device=device)
            print(f"✅ Goal model loaded successfully")
        except Exception as e:
            print(f"❌ Failed to load goal model: {e}")
            return object_model, None  # Return object model even if goal model fails
    
    return object_model, goal_model


def print_solution_summary(result: PlannerResult):
    """Print a formatted summary of the planning result."""
    print("\n" + "="*60)
    print("🎯 PLANNING RESULT SUMMARY")
    print("="*60)
    
    # Success status
    status_emoji = "✅" if result.success else "❌"
    print(f"{status_emoji} Success: {result.success}")
    print(f"🔍 Solution Found: {result.solution_found}")
    
    if result.error_message:
        print(f"💥 Error: {result.error_message}")
    
    # Solution details
    if result.solution_found:
        print(f"📏 Solution Depth: {result.solution_depth}")
        if result.action_sequence:
            print(f"🎬 Actions in Solution: {len(result.action_sequence)}")
            for i, action in enumerate(result.action_sequence):
                print(f"   {i+1}. Move object {action.object_id} to ({action.x:.2f}, {action.y:.2f}, {action.theta:.2f})")
    
    # Performance metrics
    print(f"⏱️  Search Time: {result.search_time_ms:.1f}ms" if result.search_time_ms else "⏱️  Search Time: N/A")
    print(f"🔢 Nodes Expanded: {result.nodes_expanded}" if result.nodes_expanded else "🔢 Nodes Expanded: N/A")
    print(f"🎯 Terminal Checks: {result.terminal_checks}" if result.terminal_checks else "🎯 Terminal Checks: N/A")
    print(f"🏔️  Max Depth Reached: {result.max_depth_reached}" if result.max_depth_reached else "🏔️  Max Depth Reached: N/A")
    
    # Algorithm-specific stats
    if result.algorithm_stats:
        print("📊 Algorithm Stats:")
        for key, value in result.algorithm_stats.items():
            print(f"   {key}: {value}")


def visualize_solution(env: namo_rl.RLEnvironment, result: PlannerResult, step_mode: bool = False):
    """Visualize the solution by executing actions in the environment."""
    if not result.solution_found or not result.action_sequence:
        print("❌ No solution to visualize")
        return
    
    print(f"\n🎬 Visualizing solution with {len(result.action_sequence)} actions...")
    
    if step_mode:
        print("👆 STEP MODE: Press Enter to advance to next action, 'q' to quit")
        input("Press Enter to start...")
    
    for i, action in enumerate(result.action_sequence):
        print(f"Step {i+1}/{len(result.action_sequence)}: Moving object {action.object_id} to ({action.x:.2f}, {action.y:.2f}, {action.theta:.2f})")
        
        # Execute the action using the proper step() method
        namo_action = namo_rl.Action()
        namo_action.object_id = action.object_id
        namo_action.x = action.x
        namo_action.y = action.y
        namo_action.theta = action.theta
        step_result = env.step(namo_action)
        if hasattr(step_result, 'info') and step_result.info:
            print(f"   Action result: {step_result.info}")
        else:
            print(f"   Action executed (done: {step_result.done if hasattr(step_result, 'done') else 'unknown'})")
        
        # Render the current state
        env.render()
        
        if step_mode:
            user_input = input("Press Enter for next step (or 'q' to quit): ").strip().lower()
            if user_input == 'q':
                break
        else:
            # Automatic mode - wait a bit between steps
            time.sleep(1.0)
    
    print("🎉 Solution visualization complete!")


def main():
    """Main entry point for visual single-run IDFS testing."""
    parser = argparse.ArgumentParser(description="Visual Single-Run IDFS Tester")
    
    # Required arguments
    parser.add_argument("--xml-file", type=str, required=True,
                        help="Path to XML environment file to test")
    
    # Algorithm selection
    available_algorithms = get_available_algorithms()
    parser.add_argument("--algorithm", type=str, default="idfs", choices=available_algorithms,
                        help=f"Planning algorithm to use. Options: {available_algorithms}")
    
    # Strategy selection
    available_obj_strategies = get_available_object_strategies()
    parser.add_argument("--object-strategy", type=str, default="no_heuristic", choices=available_obj_strategies,
                        help=f"Object selection strategy. Options: {available_obj_strategies}")
    
    available_goal_strategies = get_available_goal_strategies()
    parser.add_argument("--goal-strategy", type=str, default="random", choices=available_goal_strategies,
                        help=f"Goal selection strategy. Options: {available_goal_strategies}")
    
    # ML-specific arguments
    parser.add_argument("--ml-object-model", type=str,
                        help="Path to ML object inference model (required for ML object strategy)")
    parser.add_argument("--ml-goal-model", type=str,
                        help="Path to ML goal inference model (required for ML goal strategy)")
    parser.add_argument("--ml-samples", type=int, default=32,
                        help="Number of ML inference samples (default: 32)")
    parser.add_argument("--ml-device", type=str, default="cuda", choices=["cuda", "cpu"],
                        help="ML inference device (default: cuda)")
    
    # Planning parameters
    parser.add_argument("--max-depth", type=int, default=5,
                        help="Maximum search depth (default: 5)")
    parser.add_argument("--max-goals-per-object", type=int, default=5,
                        help="Maximum goals to sample per object (default: 5)")
    parser.add_argument("--max-terminal-checks", type=int, default=5000,
                        help="Maximum terminal checks before stopping search (default: 5000)")
    parser.add_argument("--search-timeout", type=float, default=60.0,
                        help="Search timeout in seconds (default: 60.0)")
    
    # Environment settings
    parser.add_argument("--config-file", type=str, 
                        default="config/namo_config_complete.yaml",
                        help="NAMO configuration file")
    parser.add_argument("--robot-goal", type=float, nargs=3, metavar=('X', 'Y', 'THETA'),
                        help="Custom robot goal (x, y, theta). If not provided, extracts from XML")
    
    # Visualization settings
    parser.add_argument("--step-mode", action="store_true",
                        help="Enable step-by-step visualization (press Enter to advance)")
    parser.add_argument("--no-visualization", action="store_true",
                        help="Disable solution visualization (search only)")
    parser.add_argument("--auto-visualize", action="store_true",
                        help="Automatically visualize solution without prompting")
    parser.add_argument("--verbose", action="store_true",
                        help="Enable verbose algorithm output")
    
    args = parser.parse_args()
    
    # Validate ML strategy requirements
    if args.object_strategy == "ml" and not args.ml_object_model:
        print("❌ Error: --ml-object-model is required when using ML object strategy")
        return 1
    
    if args.goal_strategy == "ml" and not args.ml_goal_model:
        print("❌ Error: --ml-goal-model is required when using ML goal strategy")
        return 1
    
    # Check if XML file exists
    if not os.path.exists(args.xml_file):
        print(f"❌ Error: XML file not found: {args.xml_file}")
        return 1
    
    try:
        # Print configuration summary
        print("🚀 Visual Single-Run IDFS Tester")
        print("="*50)
        print(f"📁 Environment: {args.xml_file}")
        print(f"🧠 Algorithm: {args.algorithm}")
        print(f"📦 Object Strategy: {args.object_strategy}")
        print(f"🎯 Goal Strategy: {args.goal_strategy}")
        print(f"🔍 Max Depth: {args.max_depth}")
        print(f"⏰ Timeout: {args.search_timeout}s")
        print("="*50)
        
        # Initialize environment with visualization
        print("🌍 Initializing environment...")
        env = namo_rl.RLEnvironment(args.xml_file, args.config_file, visualize=True)
        env.reset()
        
        # Extract or use custom robot goal
        if args.robot_goal:
            robot_goal = tuple(args.robot_goal)
            print(f"🎯 Using custom robot goal: ({robot_goal[0]:.2f}, {robot_goal[1]:.2f}, {robot_goal[2]:.2f})")
        else:
            robot_goal = extract_goal_with_fallback(args.xml_file, (-0.5, 1.3, 0.0))
            print(f"🎯 Extracted robot goal: ({robot_goal[0]:.2f}, {robot_goal[1]:.2f}, {robot_goal[2]:.2f})")
        
        # Set robot goal in environment
        env.set_robot_goal(*robot_goal)
        
        # Preload ML models if needed
        preloaded_object_model = None
        preloaded_goal_model = None
        if args.object_strategy == "ml" or args.goal_strategy == "ml":
            preloaded_object_model, preloaded_goal_model = preload_ml_models(
                args.ml_object_model if args.object_strategy == "ml" else None,
                args.ml_goal_model if args.goal_strategy == "ml" else None,
                args.ml_device
            )
        
        # Create planner configuration
        algorithm_params = {
            'object_selection_strategy': args.object_strategy,
            'goal_selection_strategy': args.goal_strategy,
            'ml_samples': args.ml_samples,
            'ml_device': args.ml_device
        }
        
        # Add ML model paths and preloaded models to parameters
        if args.object_strategy == "ml" and args.ml_object_model:
            algorithm_params['ml_object_model_path'] = args.ml_object_model
        if args.goal_strategy == "ml" and args.ml_goal_model:
            algorithm_params['ml_goal_model_path'] = args.ml_goal_model
            
        if preloaded_object_model is not None:
            algorithm_params['preloaded_object_model'] = preloaded_object_model
        if preloaded_goal_model is not None:
            algorithm_params['preloaded_goal_model'] = preloaded_goal_model
        
        # Add XML file path for ML strategies
        if args.object_strategy == "ml" or args.goal_strategy == "ml":
            xml_path = args.xml_file
            # Convert absolute path to relative path expected by ML models
            if '/ml4kp_ktamp/resources/models/' in xml_path:
                # Extract the relative path after 'resources/models/'
                xml_relative_path = xml_path.split('/ml4kp_ktamp/resources/models/')[1]
                algorithm_params['xml_file'] = xml_relative_path
            else:
                # For other paths, use the full path
                algorithm_params['xml_file'] = xml_path
        
        planner_config = PlannerConfig(
            max_depth=args.max_depth,
            max_goals_per_object=args.max_goals_per_object,
            max_terminal_checks=args.max_terminal_checks,
            max_search_time_seconds=args.search_timeout,
            verbose=args.verbose,
            collect_stats=True,
            algorithm_params=algorithm_params
        )
        
        # Create planner
        print(f"🧠 Creating {args.algorithm} planner...")
        planner = PlannerFactory.create_planner(args.algorithm, env, planner_config)
        
        # Initial render to show starting state
        print("📸 Initial state:")
        env.render()
        
        # Run planning
        print(f"\n🔍 Running {args.algorithm} search...")
        start_time = time.time()
        result = planner.search(robot_goal)
        search_duration = time.time() - start_time
        
        # Print results
        print_solution_summary(result)
        print(f"⏱️  Total Runtime: {search_duration:.2f}s")
        
        # Visualize solution if found and not disabled
        if not args.no_visualization and result.solution_found:
            if args.auto_visualize:
                visualize_requested = True
                print("\n🎬 Auto-visualizing solution...")
            else:
                try:
                    print(f"\n🎬 Would you like to visualize the solution? (y/N): ", end="")
                    user_input = input().strip().lower()
                    visualize_requested = user_input in ['y', 'yes']
                except (EOFError, KeyboardInterrupt):
                    print("N")  # Default to no visualization
                    visualize_requested = False
            
            if visualize_requested:
                # Reset environment to initial state
                env.reset()
                env.set_robot_goal(*robot_goal)
                visualize_solution(env, result, args.step_mode)
        
        return 0
        
    except KeyboardInterrupt:
        print("\n🛑 Interrupted by user")
        return 1
    except Exception as e:
        print(f"💥 Fatal error: {e}")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())