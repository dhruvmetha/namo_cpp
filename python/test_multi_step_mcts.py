#!/usr/bin/env python3
"""
Multi-step MCTS test that continues planning until the goal is reached.

This test executes MCTS actions in sequence, updating the environment state
and re-planning from the new root state until the robot goal becomes reachable.
"""

import sys
import argparse
import namo_rl
from mcts_hierarchical import plan_with_clean_hierarchical_mcts, CleanHierarchicalMCTS
from mcts_config import MCTSConfig

def main():
    """Test multi-step MCTS planning until goal is reached."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Multi-Step Clean 2-Level Hierarchical MCTS")
    parser.add_argument("--visualize-tree", action="store_true", 
                       help="Enable live tree visualization during MCTS search")
    parser.add_argument("--budget", type=int, default=50,
                       help="MCTS simulation budget per step (default: 50)")
    parser.add_argument("--rollout-steps", type=int, default=0,
                       help="Maximum rollout steps (default: 0)")
    parser.add_argument("--k", type=float, default=2.0,
                       help="Progressive widening constant k (default: 2.0)")
    parser.add_argument("--alpha", type=float, default=0.5,
                       help="Progressive widening exponent alpha (default: 0.5)")
    parser.add_argument("--max-steps", type=int, default=10,
                       help="Maximum planning steps before giving up (default: 10)")
    parser.add_argument("--xml", type=str, default="../ml4kp_ktamp/resources/models/custom_walled_envs/aug9/easy/set2/benchmark_3/env_config_1375a.xml",
                       help="XML environment file")
    parser.add_argument("--config", type=str, default="config/namo_config_complete.yaml",
                       help="YAML config file")
    
    args = parser.parse_args()
    
    print(f"🔄 Multi-Step Clean 2-Level Hierarchical MCTS Test:")
    print(f"  Architecture: StateNode → ObjectNode → StateNode")
    print(f"  Strategy: Execute action → Update state → Re-plan until solved")
    print(f"  XML file: {args.xml}")
    print(f"  Config file: {args.config}")
    print(f"  Simulation budget per step: {args.budget}")
    print(f"  Max rollout steps: {args.rollout_steps}")
    print(f"  Progressive Widening: k={args.k}, alpha={args.alpha}")
    print(f"  Maximum planning steps: {args.max_steps}")
    print(f"  Tree visualization: {'enabled' if args.visualize_tree else 'disabled'}")
    
    # Initialize TWO separate environments
    try:
        # env1: For MCTS planning (will be reset to match env2 state each iteration)
        planning_env = namo_rl.RLEnvironment(args.xml, args.config, False)
        planning_env.reset()
        print("✓ Planning environment (env1) initialized successfully")
        
        # env2: For action execution (maintains persistent state)
        execution_env = namo_rl.RLEnvironment(args.xml, args.config, False)
        execution_env.reset()
        print("✓ Execution environment (env2) initialized successfully")
    except Exception as e:
        print(f"✗ Failed to initialize environments: {e}")
        return 1
    
    # Set robot goal in both environments
    robot_goal = (-0.4993882613295453, 1.3595015590581654, 0.0)
    planning_env.set_robot_goal(*robot_goal)
    execution_env.set_robot_goal(*robot_goal)
    print(f"🎯 Robot goal: {robot_goal}")
    
    # Check initial state in execution environment
    if execution_env.is_robot_goal_reachable():
        print("🎉 Robot goal already reachable!")
        return 0
    
    # Debug: Check initial environment state
    reachable_objects = execution_env.get_reachable_objects()
    print(f"📦 Initial reachable objects: {len(reachable_objects)} - {reachable_objects}")
    
    obs = execution_env.get_observation()
    total_objects = len([k for k in obs.keys() if k.endswith('_pose')])
    print(f"🌍 Total objects in scene: {total_objects}")
    
    # Configure clean hierarchical MCTS
    mcts_config = MCTSConfig(
        k=args.k,
        alpha=args.alpha,
        simulation_budget=args.budget,
        max_rollout_steps=args.rollout_steps,
        c_exploration=1.414,
        verbose=False  # Reduce verbosity for multi-step
    )
    
    print(f"\n🧠 Multi-Step MCTS Configuration:")
    print(f"  Simulation budget per step: {mcts_config.simulation_budget}")
    print(f"  Max rollout steps: {mcts_config.max_rollout_steps}")
    print(f"  Progressive Widening: k={mcts_config.k}, alpha={mcts_config.alpha}")
    print(f"  UCB1 exploration: {mcts_config.c_exploration}")
    
    # Multi-step planning loop
    action_sequence = []
    step = 0
    
    print(f"\n🚀 Starting Multi-Step MCTS Planning...")
    print(f"{'='*80}")
    
    while step < args.max_steps:
        step += 1
        print(f"\n📍 STEP {step}/{args.max_steps}")
        print(f"{'='*60}")
        
        # Check if goal is reachable in execution environment
        if execution_env.is_robot_goal_reachable():
            print(f"🎉 SUCCESS! Robot goal is reachable after {step-1} actions!")
            break
        
        # Get current state info from execution environment
        current_reachable = execution_env.get_reachable_objects()
        print(f"📦 Current reachable objects: {len(current_reachable)} - {current_reachable}")
        
        if not current_reachable:
            print(f"❌ FAILED! No reachable objects in step {step}")
            break
        
        # CRITICAL: Sync planning environment state to match execution environment
        print(f"🔄 Syncing planning env1 to match execution env2 state...")
        execution_state = execution_env.get_full_state()
        planning_env.set_full_state(execution_state)
        planning_env.set_robot_goal(*robot_goal)  # Ensure goal is set
        
        # Verify sync worked
        planning_reachable = planning_env.get_reachable_objects()
        if set(planning_reachable) != set(current_reachable):
            print(f"⚠️  Warning: Environment sync mismatch!")
            print(f"   Execution env reachable: {current_reachable}")
            print(f"   Planning env reachable: {planning_reachable}")
        
        # Plan with MCTS in the synced planning environment
        print(f"🧠 Planning with MCTS in env1 (budget: {args.budget})...")
        try:
            action = plan_with_clean_hierarchical_mcts(
                planning_env, robot_goal, mcts_config, visualize_tree=args.visualize_tree
            )
            
            if action:
                print(f"✅ MCTS found action for step {step}:")
                print(f"   Push object: {action.object_id}")
                print(f"   To position: ({action.x:.3f}, {action.y:.3f})")
                print(f"   With theta: {action.theta:.3f}")
                
                # Store action in sequence
                action_info = {
                    'step': step,
                    'object_id': action.object_id,
                    'target': (action.x, action.y, action.theta)
                }
                action_sequence.append(action_info)
                
                # Execute the action in the execution environment (env2)
                print(f"⚡ Executing action {step} in execution env2...")
                result = execution_env.step(action)
                print(f"   Execution result: reward={result.reward:.3f}, done={result.done}")
                if result.info:
                    print(f"   Info: {result.info}")
                
                # Check immediate success in execution environment
                if execution_env.is_robot_goal_reachable():
                    print(f"🎉 SUCCESS! Goal reachable after action {step}!")
                    break
                else:
                    print(f"🔄 Goal not yet reachable, continuing to step {step+1}...")
                    
            else:
                print(f"❌ FAILED! MCTS found no valid action in step {step}")
                break
                
        except Exception as e:
            print(f"💥 MCTS planning failed in step {step}: {e}")
            import traceback
            traceback.print_exc()
            break
    
    # Final results
    print(f"\n{'='*80}")
    print(f"🏁 Multi-Step MCTS Planning Completed!")
    print(f"{'='*80}")
    
    if execution_env.is_robot_goal_reachable():
        print(f"🎉 SUCCESS! Goal reached in {len(action_sequence)} steps")
        print(f"\n📋 Action Sequence:")
        for i, action_info in enumerate(action_sequence, 1):
            print(f"  {i}. Push {action_info['object_id']} to {action_info['target']}")
    else:
        print(f"❌ FAILED! Goal not reachable after {step} planning steps")
        if action_sequence:
            print(f"\n📋 Attempted Action Sequence ({len(action_sequence)} actions):")
            for i, action_info in enumerate(action_sequence, 1):
                print(f"  {i}. Push {action_info['object_id']} to {action_info['target']}")
    
    # Final environment state (from execution environment)
    final_reachable = execution_env.get_reachable_objects()
    print(f"\n🌍 Final State (Execution Environment):")
    print(f"  Reachable objects: {len(final_reachable)} - {final_reachable}")
    print(f"  Goal reachable: {'✅ YES' if execution_env.is_robot_goal_reachable() else '❌ NO'}")
    
    print(f"\n🎯 Multi-Step MCTS with Dual Environments Key Features:")
    print(f"1. ✅ Dual environment architecture: Planning env1 + Execution env2")
    print(f"2. ✅ State synchronization: env1 ← env2 state before each planning")
    print(f"3. ✅ Clean separation: Plan in env1, execute in env2")
    print(f"4. ✅ Fresh MCTS trees: New root state from updated env2 each step")
    print(f"5. ✅ Persistent execution state: env2 maintains world state across steps")
    print(f"6. ✅ Isolated planning: env1 doesn't affect execution during MCTS search")
    print(f"7. ✅ Goal-oriented termination: Stops when goal becomes reachable in env2")
    
    return 0 if execution_env.is_robot_goal_reachable() else 1

if __name__ == "__main__":
    sys.exit(main())