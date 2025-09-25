"""
Solution Refiner for IDFS algorithms.

This module refines smoothed solutions by replacing action targets with actual achieved
positions from post-action states. The key innovation is validation: refined sequences
are only accepted if they still solve the navigation task.

The refiner addresses the gap between commanded and achieved object positions due to
physics simulation discrepancies, providing more realistic training data while
ensuring task solvability.
"""

import logging
import time
from typing import List, Dict, Any, Callable, Tuple
import math

logger = logging.getLogger(__name__)

class SolutionRefiner:
    """
    Refines smoothed solutions by using actual achieved positions as action targets.

    The refiner works by:
    1. Extracting actual positions from post-action state observations
    2. Creating a refined action sequence with these positions as targets
    3. Validating that the refined sequence still solves the task
    4. Only accepting refinements that maintain task solvability
    """

    def __init__(self):
        """Initialize the solution refiner."""
        self.stats = {
            'total_attempted': 0,
            'total_accepted': 0,
            'total_rejected': 0,
            'total_refinement_time': 0.0,
            'position_deviations': [],
            'rejection_reasons': {
                'action_execution_failed': 0,
                'goal_not_reachable': 0,
                'validation_error': 0
            }
        }

    def refine_with_validation(self, env, episode_result, goal_checker: Callable[[Any], bool]) -> Dict[str, Any]:
        """
        Refine the action sequence and validate that it still solves the task.

        Args:
            env: The environment instance
            episode_result: Episode result with smoothed data
            goal_checker: Function that returns True if goal is achieved

        Returns:
            Dictionary containing refinement results and statistics
        """
        start_time = time.time()
        self.stats['total_attempted'] += 1

        # Check if we have the required data
        if not episode_result.action_sequence or not episode_result.post_action_state_observations:
            return self._create_failure_result("Missing action sequence or post-action states", start_time)

        if len(episode_result.action_sequence) != len(episode_result.post_action_state_observations):
            return self._create_failure_result("Action and post-state length mismatch", start_time)

        try:
            # Step 1: Extract actual positions and create refined actions
            refined_actions, position_deviations = self.extract_actual_positions(
                episode_result.action_sequence,
                episode_result.post_action_state_observations
            )

            # Step 2: Validate that refined sequence still solves the task
            validation_success, failure_reason = self.validate_refined_sequence(
                env, refined_actions, goal_checker
            )

            refinement_time = time.time() - start_time
            self.stats['total_refinement_time'] += refinement_time
            self.stats['position_deviations'].extend(position_deviations)

            if validation_success:
                # Accept refinement
                self.stats['total_accepted'] += 1
                return {
                    'refinement_accepted': True,
                    'refined_action_sequence': refined_actions,
                    'refinement_stats': {
                        'position_deviations': position_deviations,
                        'avg_deviation': sum(position_deviations) / len(position_deviations) if position_deviations else 0.0,
                        'max_deviation': max(position_deviations) if position_deviations else 0.0,
                        'refinement_time': refinement_time,
                        'validation_success': True
                    }
                }
            else:
                # Reject refinement
                self.stats['total_rejected'] += 1
                self.stats['rejection_reasons'][failure_reason] += 1
                return {
                    'refinement_accepted': False,
                    'refined_action_sequence': None,
                    'refinement_stats': {
                        'position_deviations': position_deviations,
                        'avg_deviation': sum(position_deviations) / len(position_deviations) if position_deviations else 0.0,
                        'max_deviation': max(position_deviations) if position_deviations else 0.0,
                        'refinement_time': refinement_time,
                        'validation_success': False,
                        'rejection_reason': failure_reason
                    }
                }

        except Exception as e:
            logger.error(f"Error during action refinement: {e}")
            self.stats['total_rejected'] += 1
            self.stats['rejection_reasons']['validation_error'] += 1
            return self._create_failure_result(f"Validation error: {str(e)}", start_time)

    def extract_actual_positions(self, action_sequence: List[Dict[str, Any]],
                               post_action_states: List[Dict[str, List[float]]]) -> Tuple[List[Dict[str, Any]], List[float]]:
        """
        Extract actual achieved positions from post-action states and create refined actions.

        Args:
            action_sequence: Original action sequence
            post_action_states: State observations after each action

        Returns:
            Tuple of (refined_actions, position_deviations)
        """
        refined_actions = []
        position_deviations = []

        for i, action in enumerate(action_sequence):
            if i >= len(post_action_states):
                # Fallback to original action if no post-state available
                refined_actions.append(action.copy())
                continue

            object_id = action['object_id']
            original_target = action['target']
            post_state = post_action_states[i]

            # Extract actual achieved position from post-action state
            pose_key = f"{object_id}_pose"
            if pose_key not in post_state:
                # Object not found in post-state, use original target
                refined_actions.append(action.copy())
                logger.warning(f"Object {object_id} not found in post-action state")
                continue

            # Get actual achieved position (x, y, theta)
            actual_pose = post_state[pose_key]
            if len(actual_pose) < 3:
                # Invalid pose format, use original
                refined_actions.append(action.copy())
                logger.warning(f"Invalid pose format for {object_id}: {actual_pose}")
                continue

            actual_position = (actual_pose[0], actual_pose[1], actual_pose[2])

            # Calculate deviation between commanded and achieved position
            deviation = self.calculate_position_deviation(original_target, actual_position)
            position_deviations.append(deviation)

            # Create refined action with actual achieved position as target
            refined_action = {
                'object_id': object_id,
                'target': actual_position
            }
            refined_actions.append(refined_action)

        return refined_actions, position_deviations

    def validate_refined_sequence(self, env, refined_actions: List[Dict[str, Any]],
                                goal_checker: Callable[[Any], bool]) -> Tuple[bool, str]:
        """
        Validate that the refined action sequence still solves the navigation task.

        Args:
            env: The environment instance (namo_rl.RLEnvironment)
            refined_actions: Refined action sequence to validate
            goal_checker: Function that returns True if goal is achieved

        Returns:
            Tuple of (success, failure_reason)
        """
        try:
            # Reset environment to initial episode state
            env.reset()

            # Execute refined action sequence
            for action in refined_actions:
                # Check if object is reachable before attempting action
                if not env.is_object_reachable(action['object_id']):
                    return False, "action_execution_failed"

                # Create namo_rl.Action object (imported locally to avoid module-level import issues)
                import namo_rl
                rl_action = namo_rl.Action()
                rl_action.object_id = action['object_id']
                rl_action.x = action['target'][0]
                rl_action.y = action['target'][1]
                rl_action.theta = action['target'][2]

                # Execute action using the same API as standard_idfs.py
                step_result = env.step(rl_action)
                if not step_result.done and "error" in step_result.info:
                    # Action execution failed
                    return False, "action_execution_failed"

            # Check if goal is still achieved after executing refined sequence
            goal_achieved = goal_checker(env)
            if not goal_achieved:
                return False, "goal_not_reachable"

            return True, "success"

        except Exception as e:
            logger.error(f"Validation error: {e}")
            return False, "validation_error"

    def calculate_position_deviation(self, target: Tuple[float, float, float],
                                   actual: Tuple[float, float, float]) -> float:
        """
        Calculate the Euclidean distance between commanded and achieved positions.

        Args:
            target: Commanded target position (x, y, theta)
            actual: Actual achieved position (x, y, theta)

        Returns:
            Euclidean distance between positions (ignoring orientation for now)
        """
        dx = target[0] - actual[0]
        dy = target[1] - actual[1]
        return math.sqrt(dx * dx + dy * dy)

    def _create_failure_result(self, reason: str, start_time: float) -> Dict[str, Any]:
        """Helper to create failure result dictionary."""
        refinement_time = time.time() - start_time
        self.stats['total_refinement_time'] += refinement_time
        self.stats['total_rejected'] += 1
        self.stats['rejection_reasons']['validation_error'] += 1

        return {
            'refinement_accepted': False,
            'refined_action_sequence': None,
            'refinement_stats': {
                'position_deviations': [],
                'avg_deviation': 0.0,
                'max_deviation': 0.0,
                'refinement_time': refinement_time,
                'validation_success': False,
                'rejection_reason': reason
            }
        }

    def get_stats(self) -> Dict[str, Any]:
        """Get overall refinement statistics."""
        stats = self.stats.copy()
        if stats['total_attempted'] > 0:
            stats['acceptance_rate'] = stats['total_accepted'] / stats['total_attempted']
            if stats['position_deviations']:
                stats['avg_position_deviation'] = sum(stats['position_deviations']) / len(stats['position_deviations'])
                stats['max_position_deviation'] = max(stats['position_deviations'])
            else:
                stats['avg_position_deviation'] = 0.0
                stats['max_position_deviation'] = 0.0
        else:
            stats['acceptance_rate'] = 0.0
            stats['avg_position_deviation'] = 0.0
            stats['max_position_deviation'] = 0.0

        return stats

    def reset_stats(self):
        """Reset refinement statistics."""
        self.stats = {
            'total_attempted': 0,
            'total_accepted': 0,
            'total_rejected': 0,
            'total_refinement_time': 0.0,
            'position_deviations': [],
            'rejection_reasons': {
                'action_execution_failed': 0,
                'goal_not_reachable': 0,
                'validation_error': 0
            }
        }