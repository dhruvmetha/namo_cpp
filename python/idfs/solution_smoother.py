"""
Solution Smoother for IDFS algorithms.

This module provides functionality to find minimal subsequences of solution actions
that still achieve the goal state. The key insight is that the final action must
always be preserved as it's what achieved the goal.
"""

from itertools import combinations
import logging
from typing import List, Dict, Any, Optional, Callable
import time

logger = logging.getLogger(__name__)

class SolutionSmoother:
    """
    Finds minimal subsequences of solution actions that still solve the task.
    
    The smoother works by:
    1. Keeping the final action (which achieved the goal)
    2. Testing combinations of prefix actions in order of increasing length
    3. Returning the first (shortest) working subsequence
    """
    
    def __init__(self, max_search_actions: int = 20):
        """
        Initialize the solution smoother.
        
        Args:
            max_search_actions: Maximum solution length to attempt smoothing on.
                               Larger solutions would have exponential search space.
        """
        self.max_search_actions = max_search_actions
        self.stats = {
            'total_smoothed': 0,
            'total_reduction': 0,
            'subsequences_tested': 0,
            'smoothing_time': 0.0
        }
    
    def smooth_solution(self, env, solution: List[Dict[str, Any]], 
                       goal_checker: Callable[[Any], bool]) -> Dict[str, Any]:
        """
        Find the minimal subsequence that still solves the task.
        
        Args:
            env: The environment instance
            solution: List of action dictionaries
            goal_checker: Function that returns True if goal is achieved
            
        Returns:
            Dictionary containing original solution, smoothed solution, and stats
        """
        start_time = time.time()
        
        print(f"[SMOOTHER] Starting smoothing for solution of length {len(solution)}")
        
        if not solution:
            print(f"[SMOOTHER] Empty solution - returning immediately")
            return {
                'original_solution': solution,
                'smoothed_solution': solution,
                'smoothing_stats': {
                    'original_length': 0,
                    'smoothed_length': 0,
                    'reduction_ratio': 0.0,
                    'subsequences_tested': 0,
                    'smoothing_time': 0.0
                }
            }
        
        # Don't attempt smoothing on very long solutions (exponential search space)
        if len(solution) > self.max_search_actions:
            print(f"[SMOOTHER] Solution too long ({len(solution)} > {self.max_search_actions}), skipping smoothing")
            return {
                'original_solution': solution,
                'smoothed_solution': solution,
                'smoothing_stats': {
                    'original_length': len(solution),
                    'smoothed_length': len(solution),
                    'reduction_ratio': 0.0,
                    'subsequences_tested': 0,
                    'smoothing_time': 0.0,
                    'skipped': True
                }
            }
        
        if len(solution) == 1:
            # Single action solution is already minimal
            print(f"[SMOOTHER] Single action solution - already minimal")
            return {
                'original_solution': solution,
                'smoothed_solution': solution,
                'smoothing_stats': {
                    'original_length': 1,
                    'smoothed_length': 1,
                    'reduction_ratio': 0.0,
                    'subsequences_tested': 0,
                    'smoothing_time': time.time() - start_time
                }
            }
        
        # Extract prefix actions and final action
        prefix_actions = solution[:-1]
        final_action = solution[-1]
        
        print(f"[SMOOTHER] Testing subsequences with {len(prefix_actions)} prefix actions + 1 final action")
        print(f"[SMOOTHER] Final action: {final_action}")
        
        subsequences_tested = 0
        
        try:
            # Test subsequences of increasing length
            for length in range(1, len(prefix_actions) + 1):
                print(f"[SMOOTHER] Testing subsequences of length {length + 1} ({length} prefix + 1 final)")
                for indices in combinations(range(len(prefix_actions)), length):
                    # Create subsequence: selected prefix actions + final action
                    subsequence = [prefix_actions[i] for i in indices] + [final_action]
                    subsequences_tested += 1
                    
                    print(f"[SMOOTHER]   Testing subsequence {subsequences_tested}: indices {indices}")
                    
                    # Test if this subsequence works
                    if self.validate_subsequence(env, subsequence, goal_checker):
                        print(f"[SMOOTHER] ✅ Found working subsequence of length {len(subsequence)}!")
                        smoothing_time = time.time() - start_time
                        
                        # Update global stats
                        self.stats['total_smoothed'] += 1
                        self.stats['total_reduction'] += len(solution) - len(subsequence)
                        self.stats['subsequences_tested'] += subsequences_tested
                        self.stats['smoothing_time'] += smoothing_time
                        
                        reduction_ratio = (len(solution) - len(subsequence)) / len(solution)
                        
                        print(f"[SMOOTHER] Smoothed solution: {len(solution)} -> {len(subsequence)} actions "
                              f"({reduction_ratio:.2%} reduction, {subsequences_tested} tests)")
                        
                        return {
                            'original_solution': solution,
                            'smoothed_solution': subsequence,
                            'smoothing_stats': {
                                'original_length': len(solution),
                                'smoothed_length': len(subsequence),
                                'reduction_ratio': reduction_ratio,
                                'subsequences_tested': subsequences_tested,
                                'smoothing_time': smoothing_time
                            }
                        }
            
            # No shorter subsequence found
            print(f"[SMOOTHER] No shorter subsequence found for solution of length {len(solution)} (tested {subsequences_tested} combinations)")
            
        except Exception as e:
            print(f"[SMOOTHER] Error during solution smoothing: {e}")
            import traceback
            traceback.print_exc()
        
        # Fallback to original solution
        smoothing_time = time.time() - start_time
        print(f"[SMOOTHER] Returning original solution (no improvement found)")
        return {
            'original_solution': solution,
            'smoothed_solution': solution,
            'smoothing_stats': {
                'original_length': len(solution),
                'smoothed_length': len(solution),
                'reduction_ratio': 0.0,
                'subsequences_tested': subsequences_tested,
                'smoothing_time': smoothing_time
            }
        }
    
    def validate_subsequence(self, env, subsequence: List[Dict[str, Any]], 
                           goal_checker: Callable[[Any], bool]) -> bool:
        """
        Test if a subsequence of actions still achieves the goal.
        
        Args:
            env: The environment instance
            subsequence: List of action dictionaries to test
            goal_checker: Function that returns True if goal is achieved
            
        Returns:
            True if the subsequence achieves the goal, False otherwise
        """
        try:
            # Reset environment to initial episode state
            print(f"[SMOOTHER]     Resetting environment to initial state")
            env.reset()
            
            print(f"[SMOOTHER]     Validating subsequence with {len(subsequence)} actions")
            
            # Execute subsequence actions  
            import namo_rl
            for i, action in enumerate(subsequence):
                print(f"[SMOOTHER]       Action {i+1}: {action['object_name']} -> ({action['target_pose']['x']:.2f}, {action['target_pose']['y']:.2f}, {action['target_pose']['theta']:.2f})")
                
                # Check if object is reachable before attempting action
                if not env.is_object_reachable(action['object_name']):
                    print(f"[SMOOTHER]       ❌ Action {i+1} failed: object '{action['object_name']}' not reachable")
                    return False
                
                # Create namo_rl.Action object
                rl_action = namo_rl.Action()
                rl_action.object_id = action['object_name']
                rl_action.x = action['target_pose']['x']
                rl_action.y = action['target_pose']['y']
                rl_action.theta = action['target_pose']['theta']
                
                step_result = env.step(rl_action)
                if not step_result.done and "error" in step_result.info:
                    # Action failed - subsequence invalid
                    print(f"[SMOOTHER]       ❌ Action {i+1} failed: {step_result.info}")
                    return False
            
            # Check if goal is achieved
            goal_achieved = goal_checker(env)
            print(f"[SMOOTHER]     Goal achieved: {goal_achieved}")
            
            # No need to restore state - each test starts fresh with env.reset()
            return goal_achieved
            
        except Exception as e:
            print(f"[SMOOTHER]     ❌ Error validating subsequence: {e}")
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get overall smoothing statistics."""
        return self.stats.copy()
    
    def reset_stats(self):
        """Reset smoothing statistics."""
        self.stats = {
            'total_smoothed': 0,
            'total_reduction': 0,
            'subsequences_tested': 0,
            'smoothing_time': 0.0
        }