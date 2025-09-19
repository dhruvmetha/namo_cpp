#!/usr/bin/env python3
"""Visual Single-Run IDFS Tester

A script for running single IDFS planning iterations with flexible visualization controls.
Supports separate control over planning visualization and solution visualization.

Usage Examples:
    # Show only solution (no planning visualization)
    python visual_test_single.py --xml-file ../data/test_scene.xml --show-solution auto
    
    # Show only search tree visualization (no solution visualization)
    python visual_test_single.py --xml-file ../data/test_scene.xml --visualize-search --show-solution none
    
    # Show search tree + solution (no general planning visualization)
    python visual_test_single.py --xml-file ../data/test_scene.xml --visualize-search --show-solution auto
    
    # Show all visualizations with step-by-step controls
    python visual_test_single.py --xml-file ../data/test_scene.xml --show-planning --visualize-search --planning-step-mode --show-solution step
    
    # Completely silent run
    python visual_test_single.py --xml-file ../data/test_scene.xml --show-solution none
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
from namo.core import BasePlanner, PlannerConfig, PlannerResult, PlannerFactory
from namo.core.xml_goal_parser import extract_goal_with_fallback

# Import and register all available planners
from namo.planners.idfs.standard_idfs import StandardIterativeDeepeningDFS
from namo.planners.idfs.tree_idfs import TreeIterativeDeepeningDFS
from namo.planners.sampling.random_sampling import RandomSamplingPlanner


def get_available_algorithms() -> List[str]:
    """Get list of available planning algorithms."""
    return PlannerFactory.list_available_planners()


def get_available_object_strategies() -> List[str]:
    """Get list of available object selection strategies."""
    return ["no_heuristic", "nearest_first", "goal_proximity", "farthest_first", "ml"]


def get_available_goal_strategies() -> List[str]:
    """Get list of available goal selection strategies."""
    return ["random", "grid", "adaptive", "discretized", "ml"]


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


def reset_environment_for_visualization(env: namo_rl.RLEnvironment, robot_goal: Tuple[float, float, float]):
    """Reset environment to initial state for visualization."""
    env.reset()
    env.set_robot_goal(*robot_goal)


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


def visualize_solution(env: namo_rl.RLEnvironment, result: PlannerResult, step_mode: bool = False, delay: float = 1.0):
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
            # Automatic mode - wait specified delay between steps
            time.sleep(delay)
    
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
    
    # Planning visualization settings
    planning_group = parser.add_argument_group('Planning Visualization')
    planning_group.add_argument("--show-planning", action="store_true",
                        help="Show real-time search visualization during planning")
    planning_group.add_argument("--planning-delay", type=float, default=0.5,
                        help="Delay between planning visualization steps in seconds (default: 0.5)")
    planning_group.add_argument("--planning-step-mode", action="store_true",
                        help="Step-by-step planning visualization (press Enter to advance)")
    planning_group.add_argument("--visualize-search", action="store_true",
                        help="Enable search tree visualization (shows search state exploration)")
    
    # Solution visualization settings  
    solution_group = parser.add_argument_group('Solution Visualization')
    solution_group.add_argument("--show-solution", choices=["auto", "prompt", "step", "none"], default="prompt",
                        help="Solution visualization mode: auto (automatic), prompt (ask user), step (step-by-step), none (disable)")
    solution_group.add_argument("--solution-delay", type=float, default=1.0,
                        help="Delay between solution steps in auto mode (default: 1.0)")
    
    # General settings
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
        if args.show_planning:
            planning_mode = "step-through" if args.planning_step_mode else f"auto ({args.planning_delay}s delay)"
            print(f"🔍 Planning Visualization: {planning_mode}")
        if args.visualize_search:
            search_mode = "step-through" if args.planning_step_mode else f"auto ({args.planning_delay}s delay)"
            print(f"🌳 Search Tree Visualization: {search_mode}")
        print(f"🎬 Solution Visualization: {args.show_solution}")
        print("="*50)
        
        # Initialize environment for planning (with visualization if needed)
        print("🌍 Initializing planning environment...")
        needs_planning_viz = args.show_planning or args.visualize_search
        if needs_planning_viz:
            viz_reason = []
            if args.show_planning:
                viz_reason.append("planning")
            if args.visualize_search:
                viz_reason.append("search tree")
            print(f"   (With visualization for {' + '.join(viz_reason)})")
            planning_env = namo_rl.RLEnvironment(args.xml_file, args.config_file, visualize=True)
        else:
            print("   (Headless mode for planning)")
            planning_env = namo_rl.RLEnvironment(args.xml_file, args.config_file, visualize=False)
        
        planning_env.reset()
        
        # Extract or use custom robot goal
        if args.robot_goal:
            robot_goal = tuple(args.robot_goal)
            print(f"🎯 Using custom robot goal: ({robot_goal[0]:.2f}, {robot_goal[1]:.2f}, {robot_goal[2]:.2f})")
        else:
            robot_goal = extract_goal_with_fallback(args.xml_file, (-0.5, 1.3, 0.0))
            print(f"🎯 Extracted robot goal: ({robot_goal[0]:.2f}, {robot_goal[1]:.2f}, {robot_goal[2]:.2f})")
        
        # Set robot goal in planning environment
        planning_env.set_robot_goal(*robot_goal)
        
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
        
        # Create planner using planning environment
        print(f"🧠 Creating {args.algorithm} planner...")
        planner = PlannerFactory.create_planner(args.algorithm, planning_env, planner_config)
        
        # Configure planner visualization parameters
        if hasattr(planner, 'visualize_search'):
            # Enable search visualization if either show-planning or visualize-search is requested
            planner.visualize_search = args.show_planning or args.visualize_search
            planner.search_delay = args.planning_delay
            planner.step_mode = args.planning_step_mode
            
            if args.visualize_search:
                print("🌳 Search tree visualization enabled (shows search state exploration)")
            elif args.show_planning:
                print("🔍 Planning visualization enabled")
            else:
                print("🔍 Planning visualization disabled (search will run silently)")
        
        # Initial render to show starting state (only if planning visualization is enabled)
        if args.show_planning:
            print("📸 Initial state:")
            planning_env.render()
        
        # Run planning
        print(f"\n🔍 Running {args.algorithm} search...")
        start_time = time.time()
        result = planner.search(robot_goal)
        search_duration = time.time() - start_time
        
        # Print results
        print_solution_summary(result)
        print(f"⏱️  Total Runtime: {search_duration:.2f}s")
        
        # Visualize solution based on mode
        if result.solution_found and args.show_solution != "none":
            # Create separate visualization environment for solution
            print("🌍 Creating visualization environment for solution...")
            solution_env = namo_rl.RLEnvironment(args.xml_file, args.config_file, visualize=True)
            reset_environment_for_visualization(solution_env, robot_goal)
            
            if args.show_solution == "auto":
                print("\n🎬 Auto-visualizing solution...")
                visualize_solution(solution_env, result, step_mode=False, delay=args.solution_delay)
            elif args.show_solution == "step":
                print("\n🎬 Step-by-step solution visualization...")
                visualize_solution(solution_env, result, step_mode=True, delay=0)
            elif args.show_solution == "prompt":
                try:
                    print(f"\n🎬 Would you like to visualize the solution? (y/N): ", end="")
                    user_input = input().strip().lower()
                    if user_input in ['y', 'yes']:
                        visualize_solution(solution_env, result, step_mode=False, delay=1.0)
                except (EOFError, KeyboardInterrupt):
                    print("N")  # Default to no visualization
        
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