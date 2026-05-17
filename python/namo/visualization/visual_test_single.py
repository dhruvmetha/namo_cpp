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

# Add visualization directory to path for ML adapters (CRITICAL FIX)
# This directory contains ml_image_converter_adapter.py which ML models import as 'ml_image_converter_adapter'
namo_viz_path = os.path.dirname(os.path.abspath(__file__))
if namo_viz_path not in sys.path:
    sys.path.append(namo_viz_path)

# NAMO imports
import namo_rl
from namo.core import BasePlanner, PlannerConfig, PlannerResult, PlannerFactory
from namo.core.xml_goal_parser import extract_goal_with_fallback

# Import and register all available planners
from namo.planners.sampling.random_sampling import RandomSamplingPlanner
from namo.planners.opening.region_opening import RegionOpeningPlanner
from namo.planners.full_namo.full_namo_planner import FullNAMOPlanner

# Import solution smoothing system
from namo.planners.utils.solution_smoother import SolutionSmoother


def get_available_algorithms() -> List[str]:
    """Get list of available planning algorithms."""
    return PlannerFactory.list_available_planners()


def get_available_object_strategies() -> List[str]:
    """Get list of available object selection strategies."""
    return ["no_heuristic", "nearest_first", "goal_proximity", "farthest_first", "ml"]


def get_available_goal_strategies() -> List[str]:
    """Get list of available goal selection strategies."""
    return ["random", "grid", "adaptive", "discretized", "ml", "primitive"]


def create_goal_checker(robot_goal):
    """Create a goal checker function for the smoother."""
    def check_goal(env):
        # Use the environment's built-in reachability checking
        # which uses wavefront planning to determine if robot can reach goal
        return env.is_robot_goal_reachable()
    return check_goal


def preload_ml_models(object_model_path: Optional[str],
                     goal_model_path: Optional[str],
                     device: str = "cuda",
                     sampler_method: Optional[str] = None,
                     num_steps: Optional[int] = None) -> Tuple[Optional[any], Optional[any]]:
    """Preload ML models if paths are provided.

    Args:
        object_model_path: Path to object inference model
        goal_model_path: Path to goal inference model
        device: Device to load models on
        sampler_method: Override sampler method (euler, midpoint, rk4, dopri5 for flow matching)
        num_steps: Override number of sampling steps
    """
    object_model = None
    goal_model = None

    # Support local editable checkouts where `sage_learning` isn't installed globally.
    mujoco_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
    local_sage_root = os.path.join(mujoco_root, "sage_learning")
    if os.path.isdir(local_sage_root) and local_sage_root not in sys.path:
        sys.path.insert(0, local_sage_root)
    snapshot_sage_root = os.path.join(mujoco_root, "eval_pipeline_snapshot", "sage_learning")
    if os.path.isdir(snapshot_sage_root) and snapshot_sage_root not in sys.path:
        # Append so local code takes precedence; snapshot is only for legacy `ktamp_learning.*`.
        sys.path.append(snapshot_sage_root)

    if object_model_path:
        try:
            try:
                from sage_learning.object_inference_model import ObjectInferenceModel
            except Exception:
                from ktamp_learning.object_inference_model import ObjectInferenceModel
            print(f"🔮 Loading ObjectInferenceModel from {object_model_path}")
            object_model = ObjectInferenceModel(model_path=object_model_path, device=device)
            print(f"✅ Object model loaded successfully")
        except Exception as e:
            print(f"❌ Failed to load object model: {e}")
            return None, None

    if goal_model_path:
        try:
            try:
                from sage_learning.goal_inference_model import GoalInferenceModel
            except Exception:
                from ktamp_learning.goal_inference_model import GoalInferenceModel
            print(f"🎯 Loading GoalInferenceModel from {goal_model_path}")
            goal_model = GoalInferenceModel(
                model_path=goal_model_path,
                device=device,
                sampler_method=sampler_method,
                num_steps=num_steps
            )
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


def visualize_region_opening_sequence(env: namo_rl.RLEnvironment, result: PlannerResult, step_mode: bool = False, delay: float = 1.0):
    """Visualize Full NAMO solution using structured region opening sequence.

    This handles the hierarchical structure where each region opening may contain
    multiple actions (skill chain) that should be executed sequentially.
    """
    if not result.algorithm_stats or "region_opening_sequence" not in result.algorithm_stats:
        print("❌ No region_opening_sequence found, falling back to flat visualization")
        return visualize_solution(env, result, step_mode, delay)

    region_opening_sequence = result.algorithm_stats["region_opening_sequence"]
    if not region_opening_sequence:
        print("❌ Empty region_opening_sequence")
        return

    total_actions = sum(len(ro.actions) for ro in region_opening_sequence)
    print(f"\n🎬 Visualizing Full NAMO solution: {len(region_opening_sequence)} region openings, {total_actions} total actions...")

    if step_mode:
        print("👆 STEP MODE: Press Enter to advance to next action, 'q' to quit")
        input("Press Enter to start...")

    # Print the robot goal being used for this visualization
    robot_goal = env.get_robot_goal()
    print(f"🎯 Robot goal for visualization: ({robot_goal[0]:.2f}, {robot_goal[1]:.2f}, {robot_goal[2]:.2f})")

    action_counter = 0
    for ro_idx, region_opening in enumerate(region_opening_sequence):
        print(f"\n📍 Region Opening {ro_idx + 1}/{len(region_opening_sequence)}: Opening path to '{region_opening.target_region}' by pushing {region_opening.object_id}")

        for action_idx, action in enumerate(region_opening.actions):
            action_counter += 1
            print(f"   Step {action_counter}/{total_actions}: Push {action.object_id} to ({action.x:.2f}, {action.y:.2f}, {action.theta:.2f})")

            # Execute the action
            namo_action = namo_rl.Action()
            namo_action.object_id = action.object_id
            namo_action.x = action.x
            namo_action.y = action.y
            namo_action.theta = action.theta
            namo_action.edge_idx = getattr(action, 'edge_idx', -1)
            namo_action.depth = getattr(action, 'depth', -1)

            step_result = env.step(namo_action)
            if hasattr(step_result, 'info') and step_result.info:
                print(f"      Result: {step_result.info}")

            # Render the current state
            env.render()

            if step_mode:
                user_input = input("Press Enter for next step (or 'q' to quit): ").strip().lower()
                if user_input == 'q':
                    return
            else:
                time.sleep(delay)

        print(f"   ✓ Region opening complete")

    print("\n🎉 Full NAMO solution visualization complete!")


def visualize_solution(env: namo_rl.RLEnvironment, result: PlannerResult, step_mode: bool = False, delay: float = 1.0):
    """Visualize the solution by executing actions in the environment."""
    if not result.solution_found or not result.action_sequence:
        print("❌ No solution to visualize")
        return

    print(f"\n🎬 Visualizing solution with {len(result.action_sequence)} actions...")

    if step_mode:
        print("👆 STEP MODE: Press Enter to advance to next action, 'q' to quit")
        input("Press Enter to start...")

    # Print the robot goal being used for this visualization
    robot_goal = env.get_robot_goal()
    print(f"🎯 Robot goal for visualization: ({robot_goal[0]:.2f}, {robot_goal[1]:.2f}, {robot_goal[2]:.2f})")

    for i, action in enumerate(result.action_sequence):
        print(
            f"Step {i+1}/{len(result.action_sequence)}: Moving object {action.object_id} to "
            f"({action.x:.2f}, {action.y:.2f}, {action.theta:.2f})"
        )

        # Execute the action using the proper step() method
        namo_action = namo_rl.Action()
        namo_action.object_id = action.object_id
        namo_action.x = action.x
        namo_action.y = action.y
        namo_action.theta = action.theta
        namo_action.edge_idx = getattr(action, 'edge_idx', -1)  # Pass for direct C++ execution
        namo_action.depth = getattr(action, 'depth', -1)        # Pass for direct C++ execution
        print(f"   Action primitive: edge_idx={namo_action.edge_idx}, depth={namo_action.depth}")
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
    # Pre-parse only --config-yaml to allow YAML defaults with CLI overrides
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config-yaml", type=str, help="Path to YAML config file for defaults")
    pre_args, remaining_argv = pre_parser.parse_known_args()

    parser = argparse.ArgumentParser(description="Visual Single-Run IDFS Tester", parents=[pre_parser])
    
    # Required arguments
    parser.add_argument("--xml-file", type=str, required=False,
                        help="Path to XML environment file to test")
    
    # Algorithm selection
    available_algorithms = get_available_algorithms()
    parser.add_argument("--algorithm", type=str, default="region_opening", choices=available_algorithms,
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
    parser.add_argument("--ml-match-max-per-call", type=int, default=8,
                        help="Maximum number of ML goals to align to primitives per inference (default: 8)")
    parser.add_argument("--ml-match-position-tolerance", type=float, default=0.2,
                        help="Position tolerance for ML-primitive alignment in meters (default: 0.2)")
    parser.add_argument("--ml-match-angle-tolerance", type=float, default=0.35,
                        help="Angle tolerance for ML-primitive alignment in radians (default: 0.35)")
    parser.add_argument("--ml-match-angle-weight", type=float, default=0.5,
                        help="Weight for angle error in alignment scoring (default: 0.5)")
    parser.add_argument("--ml-k-nearest", type=int, default=1,
                        help="Vote for up to k nearest primitive slots per ML sample (default: 1)")
    parser.add_argument("--preview-ml-goal-masks", type=int, default=0,
                        help="Number of ML goal masks to preview via matplotlib before planning (0 disables)")
    parser.add_argument("--preview-aligned-primitives", action="store_true",
                        help="Save visualization of ML-to-primitive alignment showing execution order")
    
    # Planning parameters
    parser.add_argument("--max-depth", type=int, default=5,
                        help="Maximum search depth (default: 5)")
    parser.add_argument("--max-goals-per-object", type=int, default=5,
                        help="Maximum goals to sample per object (default: 5)")
    parser.add_argument("--max-terminal-checks", type=int, default=5000,
                        help="Maximum terminal checks before stopping search (default: 5000)")
    parser.add_argument("--search-timeout", type=float, default=60.0,
                        help="Search timeout in seconds (default: 60.0)")
    parser.add_argument("--goals-per-region", type=int, default=5,
                        help="Number of robot goal samples per region for validation (default: 5)")
    parser.add_argument("--region-allow-collisions", action=argparse.BooleanOptionalAction, default=True,
                        help="Allow object collisions during region opening pushes (default: True). Use --no-region-allow-collisions for strict mode (any object collision aborts the push). Robot-collisions always abort regardless.")
    parser.add_argument("--region-max-chain-depth", type=int, default=1,
                        help="Maximum chain depth for region opening: 1=single push, 2=2-push chains, 3=3-push chains (default: 1)")
    parser.add_argument("--region-max-solutions-per-neighbor", type=int, default=10,
                        help="Maximum solutions to keep per neighbor region (default: 10)")
    parser.add_argument("--region-frontier-beam-width", type=int, default=None,
                        help="Prune frontier to top N nodes by cost (None = no pruning)")
    parser.add_argument("--region-max-recorded-solutions-per-neighbor", type=int, default=2,
                        help="Maximum solutions to record per neighbor region (default: 2)")

    # Environment settings
    parser.add_argument("--config-file", type=str,
                        default="config/namo_config_complete_skill15.yaml",
                        help="NAMO configuration file (skill15 = 15 points per edge)")
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

    # Video recording settings
    video_group = parser.add_argument_group('Video Recording')
    video_group.add_argument("--record-video", action="store_true",
                        help="Record video of solution execution (requires visualize=True)")
    video_group.add_argument("--video-output", type=str, default=None,
                        help="Output video file path (default: auto-generated from XML filename)")
    video_group.add_argument("--video-width", type=int, default=640,
                        help="Video width in pixels (default: 640)")
    video_group.add_argument("--video-height", type=int, default=480,
                        help="Video height in pixels (default: 480)")
    video_group.add_argument("--video-fps", type=int, default=30,
                        help="Video output framerate (default: 30)")
    video_group.add_argument("--capture-frequency", type=int, default=100,
                        help="Capture every N physics steps (default: 100, ~250 frames per push)")
    video_group.add_argument("--camera-distance", type=float, default=None,
                        help="Camera distance/zoom (lower = more zoomed in, default: auto)")

    # Solution smoothing settings
    parser.add_argument("--smooth-solutions", action="store_true",
                        help="Apply exhaustive smoothing to find minimal subsequences")
    parser.add_argument("--max-smooth-actions", type=int, default=20,
                        help="Maximum solution length to attempt smoothing on (default: 20)")

    # General settings
    parser.add_argument("--verbose", action="store_true",
                        help="Enable verbose algorithm output")
    
    # If YAML provided, load and set parser defaults before final parse
    if pre_args.config_yaml:
        try:
            import yaml
            with open(pre_args.config_yaml, 'r') as f:
                yaml_cfg = yaml.safe_load(f) or {}
            if isinstance(yaml_cfg, dict):
                parser.set_defaults(**yaml_cfg)
        except Exception as e:
            print(f"⚠️  Warning: could not load YAML config '{pre_args.config_yaml}': {e}")

    args = parser.parse_args(remaining_argv)
    
    # Validate ML strategy requirements
    if args.object_strategy == "ml" and not args.ml_object_model:
        print("❌ Error: --ml-object-model is required when using ML object strategy")
        return 1
    
    if args.goal_strategy == "ml" and not args.ml_goal_model:
        print("❌ Error: --ml-goal-model is required when using ML goal strategy")
        return 1
    if args.preview_ml_goal_masks > 0 and args.goal_strategy != "ml":
        print("⚠️  Warning: --preview-ml-goal-masks is only used with the 'ml' goal strategy")
    
    # Ensure XML file is provided (via CLI or YAML) and exists
    if not args.xml_file:
        print("❌ Error: --xml-file is required (or provide 'xml_file' in the YAML)")
        return 1
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
        if args.smooth_solutions:
            print(f"✨ Solution Smoothing: enabled (max {args.max_smooth_actions} actions)")
        if args.preview_ml_goal_masks > 0 and args.goal_strategy == "ml":
            print(f"🖼️ Previewing first {args.preview_ml_goal_masks} ML goal masks (close figure to continue)")
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

        # For region-opening visual debugging, hide the XML scene's fixed `<site name="goal">`
        # (often a green sphere). Region-opening cares about sampled neighbour-region goals,
        # and this site is unrelated/confusing alongside the robot-goal marker square.
        if args.algorithm == "region_opening":
            try:
                planning_env.set_goal_site_visible(False)
            except Exception:
                pass
        
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
            # NOTE: RegionOpeningPlanner expects 'goal_strategy' while IDFS planners
            # historically used 'goal_selection_strategy'. Set both for compatibility.
            'goal_strategy': args.goal_strategy,
            'goal_selection_strategy': args.goal_strategy,
            'ml_samples': args.ml_samples,
            'ml_device': args.ml_device,
            'ml_match_max_per_call': args.ml_match_max_per_call,
            'ml_match_position_tolerance': args.ml_match_position_tolerance,
            'ml_match_angle_tolerance': args.ml_match_angle_tolerance,
            'ml_match_angle_weight': args.ml_match_angle_weight,
            'ml_k_nearest': args.ml_k_nearest,
            'region_allow_collisions': args.region_allow_collisions,
            'region_max_chain_depth': args.region_max_chain_depth,
            'region_max_solutions_per_neighbor': args.region_max_solutions_per_neighbor,
            'region_frontier_beam_width': args.region_frontier_beam_width,
            'region_max_recorded_solutions_per_neighbor': args.region_max_recorded_solutions_per_neighbor,
            'preview_ml_goal_masks': args.preview_ml_goal_masks,
            'preview_aligned_primitives': args.preview_aligned_primitives
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
        
        # Add XML file path for ML strategies (use absolute path directly)
        if args.object_strategy == "ml" or args.goal_strategy == "ml":
            algorithm_params['xml_file'] = args.xml_file
        
        planner_config = PlannerConfig(
            max_depth=args.max_depth,
            max_goals_per_object=args.max_goals_per_object,
            max_terminal_checks=args.max_terminal_checks,
            max_search_time_seconds=args.search_timeout,
            goals_per_region=args.goals_per_region,
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

        # Apply solution smoothing if enabled and solution found
        if args.smooth_solutions and result.solution_found and result.action_sequence:
            if len(result.action_sequence) <= args.max_smooth_actions:
                print(f"\n🎯 Applying solution smoothing (original length: {len(result.action_sequence)})...")

                # Create smoother and goal checker
                smoother = SolutionSmoother(max_search_actions=args.max_smooth_actions)
                goal_checker = create_goal_checker(robot_goal)

                # Convert action sequence to format expected by smoother
                smoother_actions = [
                    {
                        "object_name": action.object_id,
                        "target_pose": {"x": action.x, "y": action.y, "theta": action.theta}
                    }
                    for action in result.action_sequence
                ]

                # Apply smoothing using planning environment
                smooth_result = smoother.smooth_solution(planning_env, smoother_actions, goal_checker)

                # Update result if improvement found
                if smooth_result["smoothed_solution"] != smooth_result["original_solution"]:
                    # Convert back to original format
                    smoothed_actions = []
                    for act in smooth_result["smoothed_solution"]:
                        action = namo_rl.Action()
                        action.object_id = act["object_name"]
                        action.x = act["target_pose"]["x"]
                        action.y = act["target_pose"]["y"]
                        action.theta = act["target_pose"]["theta"]
                        action.edge_idx = act.get("edge_idx", -1)  # Pass for direct C++ execution
                        action.depth = act.get("depth", -1)        # Pass for direct C++ execution
                        smoothed_actions.append(action)

                    result.action_sequence = smoothed_actions
                    result.solution_depth = len(smoothed_actions)

                    print(f"✨ Solution improved! New length: {len(smoothed_actions)} (saved {len(smoother_actions) - len(smoothed_actions)} actions)")
                    if smooth_result["smoothing_stats"]:
                        print(f"📊 Smoothing stats: {smooth_result['smoothing_stats']}")
                else:
                    print("💡 No improvement found - solution is already optimal")
            else:
                print(f"⚠️  Solution too long for smoothing ({len(result.action_sequence)} > {args.max_smooth_actions}), skipping")

        # Print results
        print_solution_summary(result)
        print(f"⏱️  Total Runtime: {search_duration:.2f}s")
        
        # Visualize solution based on mode
        if result.solution_found and args.show_solution != "none":
            # Create separate visualization environment for solution
            print("🌍 Creating visualization environment for solution...")
            solution_env = namo_rl.RLEnvironment(args.xml_file, args.config_file, visualize=True)
            reset_environment_for_visualization(solution_env, robot_goal)

            if args.algorithm == "region_opening":
                try:
                    solution_env.set_goal_site_visible(False)
                except Exception:
                    pass

            # Apply collision checking settings (must match planning settings)
            if args.region_allow_collisions:
                solution_env.set_collision_checking(False)

            # Apply camera distance for visualization (only if explicitly set)
            # Uses default azimuth=0, elevation=-90 (top-down view) from mujoco_wrapper
            if args.camera_distance is not None:
                solution_env.set_camera_position(args.camera_distance, 0.0, -90.0)

            # Start video recording if requested
            if args.record_video:
                print(f"🎥 Starting video recording ({args.video_width}x{args.video_height}, every {args.capture_frequency} physics steps)...")
                solution_env.start_recording(
                    width=args.video_width,
                    height=args.video_height,
                    capture_frequency=args.capture_frequency
                )

            # Check if region_opening planner returned multiple solutions
            attempt_results = None
            if result.algorithm_stats and "attempt_results" in result.algorithm_stats:
                attempt_results = [a for a in result.algorithm_stats["attempt_results"] if a.success]

            if attempt_results and len(attempt_results) > 1:
                # Region opening found multiple solutions - visualize each one
                print(f"\n🎯 Found {len(attempt_results)} successful openings! Visualizing each one...\n")

                for i, attempt in enumerate(attempt_results, 1):
                    print(f"\n{'='*60}")
                    print(f"Solution {i}/{len(attempt_results)}: Opening to '{attempt.neighbour_region_label}' by pushing {attempt.chosen_object_id}")
                    print(f"{'='*60}")

                    # Reset environment to initial state before visualizing this solution
                    reset_environment_for_visualization(solution_env, robot_goal)

                    # For region-opening solutions, the success criterion is reachability of a sampled
                    # goal in the neighbor region. Replay using that exact goal so `robot_goal_reached`
                    # is meaningful during visualization.
                    if getattr(attempt, "region_goal_used", None):
                        rg = attempt.region_goal_used
                        print(f"🟩 Setting visualization robot goal to region_goal_used={rg}")
                        before_rg = solution_env.get_robot_goal()
                        print(f"🟩 Env robot goal before set: ({before_rg[0]:.3f}, {before_rg[1]:.3f}, {before_rg[2]:.3f})")
                        solution_env.set_robot_goal(rg[0], rg[1], rg[2] if len(rg) > 2 else 0.0)
                        after_rg = solution_env.get_robot_goal()
                        print(f"🟩 Env robot goal after set:  ({after_rg[0]:.3f}, {after_rg[1]:.3f}, {after_rg[2]:.3f})")
                        solution_env.render()

                    # Build action sequence from attempt
                    action_sequence = []
                    if attempt.goal_chain:
                        # Single or multi-push chain (goal_chain contains edge_idx/depth metadata)
                        for goal in attempt.goal_chain:
                            action = namo_rl.Action()
                            action.object_id = attempt.chosen_object_id
                            action.x = goal.x
                            action.y = goal.y
                            action.theta = goal.theta
                            action.edge_idx = getattr(goal, 'edge_idx', -1)  # Pass for direct C++ execution
                            action.depth = getattr(goal, 'depth', -1)        # Pass for direct C++ execution
                            action_sequence.append(action)
                    elif attempt.chosen_goal:
                        # Single push
                        action = namo_rl.Action()
                        action.object_id = attempt.chosen_object_id
                        action.x = attempt.chosen_goal[0]
                        action.y = attempt.chosen_goal[1]
                        action.theta = attempt.chosen_goal[2]
                        # attempt.chosen_goal is a tuple, so edge_idx/depth use defaults (-1)
                        action_sequence.append(action)

                    # Create a temporary result with this solution's actions
                    temp_result = PlannerResult(
                        success=True,
                        solution_found=True,
                        action_sequence=action_sequence,
                        solution_depth=len(action_sequence),
                        search_time_ms=result.search_time_ms,
                        algorithm_stats=result.algorithm_stats
                    )

                    if args.show_solution == "auto":
                        visualize_solution(solution_env, temp_result, step_mode=False, delay=args.solution_delay)
                    elif args.show_solution == "step":
                        visualize_solution(solution_env, temp_result, step_mode=True, delay=0)
                    elif args.show_solution == "prompt":
                        try:
                            print(f"\n🎬 Visualize this solution? (y/N): ", end="")
                            user_input = input().strip().lower()
                            if user_input in ['y', 'yes']:
                                visualize_solution(solution_env, temp_result, step_mode=False, delay=1.0)
                        except (EOFError, KeyboardInterrupt):
                            print("N")
                            break

                    # Pause between solutions (except after last one)
                    if i < len(attempt_results):
                        try:
                            input("\nPress Enter to see next solution (or Ctrl+C to stop)...")
                        except (EOFError, KeyboardInterrupt):
                            print("\n🛑 Stopping visualization")
                            break
            else:
                # Single solution visualization
                # Use structured visualization for full_namo, flat visualization for others
                use_structured = (args.algorithm == "full_namo" and
                                  result.algorithm_stats and
                                  "region_opening_sequence" in result.algorithm_stats)

                viz_func = visualize_region_opening_sequence if use_structured else visualize_solution

                # For region_opening, align the visualization robot-goal with the recorded
                # reachable region goal (if present) so `robot_goal_reached` lines up.
                if args.algorithm == "region_opening" and result.algorithm_stats and "attempt_results" in result.algorithm_stats:
                    successes = [a for a in result.algorithm_stats["attempt_results"] if getattr(a, "success", False)]
                    if successes and getattr(successes[0], "region_goal_used", None):
                        rg = successes[0].region_goal_used
                        print(f"🟩 Setting visualization robot goal to first region_goal_used={rg}")
                        before_rg = solution_env.get_robot_goal()
                        print(f"🟩 Env robot goal before set: ({before_rg[0]:.3f}, {before_rg[1]:.3f}, {before_rg[2]:.3f})")
                        solution_env.set_robot_goal(rg[0], rg[1], rg[2] if len(rg) > 2 else 0.0)
                        after_rg = solution_env.get_robot_goal()
                        print(f"🟩 Env robot goal after set:  ({after_rg[0]:.3f}, {after_rg[1]:.3f}, {after_rg[2]:.3f})")
                        solution_env.render()

                if args.show_solution == "auto":
                    print("\n🎬 Auto-visualizing solution...")
                    viz_func(solution_env, result, step_mode=False, delay=args.solution_delay)
                elif args.show_solution == "step":
                    print("\n🎬 Step-by-step solution visualization...")
                    viz_func(solution_env, result, step_mode=True, delay=0)
                elif args.show_solution == "prompt":
                    try:
                        print(f"\n🎬 Would you like to visualize the solution? (y/N): ", end="")
                        user_input = input().strip().lower()
                        if user_input in ['y', 'yes']:
                            viz_func(solution_env, result, step_mode=False, delay=1.0)
                    except (EOFError, KeyboardInterrupt):
                        print("N")  # Default to no visualization

            # Save video if recording was enabled
            if args.record_video:
                solution_env.stop_recording()
                frame_count = solution_env.get_frame_count()
                print(f"🎥 Recording stopped. Captured {frame_count} frames.")

                if frame_count > 0:
                    # Get frames as numpy array
                    frames = solution_env.get_frames()

                    # Determine output path
                    if args.video_output:
                        video_path = args.video_output
                    else:
                        # Auto-generate from XML filename
                        xml_basename = os.path.splitext(os.path.basename(args.xml_file))[0]
                        video_path = f"{xml_basename}_solution.mp4"

                    # Save video using mediapy
                    try:
                        import mediapy as mp
                        print(f"📼 Saving video to {video_path} ({args.video_fps} fps)...")
                        mp.write_video(video_path, frames, fps=args.video_fps)
                        print(f"✅ Video saved successfully: {video_path}")
                    except ImportError:
                        print("⚠️  mediapy not installed. Install with: pip install mediapy")
                        print("   Attempting to save frames as numpy array instead...")
                        import numpy as np
                        np_path = video_path.replace('.mp4', '_frames.npy')
                        np.save(np_path, frames)
                        print(f"✅ Frames saved to: {np_path}")

                    # Clear frames to free memory
                    solution_env.clear_frames()
                else:
                    print("⚠️  No frames captured - video not saved")

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
