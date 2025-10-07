"""Reachability-Expanding Iterative Deepening DFS planner with primitive-based exploration.

This planner uses precomputed motion primitives to systematically explore push actions.
Success conditions (either):
1. Robot goal is reachable
2. New objects become reachable

The planner tests primitives in ascending push_steps order, with adaptive depth limiting
based on shortest solution found so far.
"""

import time
from typing import List, Optional, Tuple, Set, Dict, Any
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from namo.strategies import ObjectSelectionStrategy, GoalSelectionStrategy

import namo_rl
from namo.core import PlannerResult
from .standard_idfs import StandardIterativeDeepeningDFS, Action, Goal


class ReachabilityExpandingIDFS(StandardIterativeDeepeningDFS):
    """IDFS planner that succeeds when new objects become reachable.

    Success Conditions (either):
    1. Robot goal is reachable (standard condition)
    2. Reachable set has expanded to include new objects

    This variant is useful for:
    - Exploration strategies
    - Opening up blocked spaces
    - Progressive reachability analysis
    """

    def __init__(self, env: namo_rl.RLEnvironment, config,
                 object_selection_strategy: Optional['ObjectSelectionStrategy'] = None,
                 goal_selection_strategy: Optional['GoalSelectionStrategy'] = None,
                 visualize_search: bool = False,
                 search_delay: float = 0.5,
                 step_mode: bool = False):
        """Initialize reachability-expanding IDFS planner.

        Args:
            env: NAMO RL environment
            config: Planner configuration
            object_selection_strategy: Strategy for ordering objects
            goal_selection_strategy: Strategy for generating goals
            visualize_search: Enable search visualization
            search_delay: Delay between visualization steps
            step_mode: Manual step-through mode
        """
        super().__init__(env, config, object_selection_strategy,
                        goal_selection_strategy, visualize_search,
                        search_delay, step_mode)

        # Track initial reachable set for comparison
        self.initial_reachable_objects: Set[str] = set()

        # Add reachability expansion tracking to stats
        if self.config.collect_stats:
            self.stats['reachability_expansions'] = 0

    @property
    def algorithm_name(self) -> str:
        """Return human-readable algorithm name."""
        return f"Reachability-Expanding IDFS ({self.object_selection_strategy.strategy_name}, {self.goal_selection_strategy.strategy_name})"

    @property
    def algorithm_version(self) -> str:
        """Return algorithm version identifier."""
        return "reachability_expanding_idfs_v1.0"

    def reset(self):
        """Reset internal algorithm state for new planning episode."""
        super().reset()
        self.initial_reachable_objects = set()

        if self.config.collect_stats:
            self.stats['reachability_expansions'] = 0

    def search(self, robot_goal: Tuple[float, float, float]) -> PlannerResult:
        """Run primitive-based reachability-expanding search.

        Systematically tests motion primitives in ascending push_steps order,
        with adaptive depth limiting based on shortest solution found.

        Args:
            robot_goal: Target robot position (x, y, theta)

        Returns:
            PlannerResult with all shortest solutions found
        """
        start_time = time.time()
        timeout_seconds = self.config.max_search_time_seconds

        # Set robot goal
        self.env.set_robot_goal(*robot_goal)

        # Capture initial reachable objects at root state
        root_state = self.env.get_full_state()
        self.env.set_full_state(root_state)
        self.initial_reachable_objects = set(self.env.get_reachable_objects())

        if self.config.verbose:
            print(f"\nStarting primitive-based expanding IDFS")
            print(f"Initial reachable objects ({len(self.initial_reachable_objects)}): {sorted(self.initial_reachable_objects)}")

        # Check if root is already terminal
        if self._is_terminal_state(root_state):
            if self.config.verbose:
                print("Terminal state at root!")

            search_time_ms = (time.time() - start_time) * 1000
            return PlannerResult(
                success=True,
                solution_found=True,
                action_sequence=[],
                solution_depth=0,
                state_observations=[],
                post_action_state_observations=[],
                search_time_ms=search_time_ms,
                nodes_expanded=0,
                terminal_checks=1,
                max_depth_reached=0,
                algorithm_stats={
                    'primitive_based': True,
                    'initial_reachable_count': len(self.initial_reachable_objects),
                    'solutions_found': 0
                }
            )

        # Get reachable objects (ordered by strategy)
        reachable_objects = self._get_reachable_objects(root_state)
        ordered_objects = self.object_selection_strategy.select_objects(
            reachable_objects, root_state, self.env
        )

        if self.config.verbose:
            print(f"Reachable objects ({len(ordered_objects)}): {ordered_objects}")

        # Track shortest solution depth and all solutions at that depth
        shortest_depth = 11  # Max push_steps is 10
        all_solutions: List[Dict[str, Any]] = []

        # For each reachable object
        for obj_idx, object_id in enumerate(ordered_objects):
            # Check timeout
            if timeout_seconds and (time.time() - start_time) > timeout_seconds:
                if self.config.verbose:
                    print(f"Search timed out after {timeout_seconds}s")
                break

            if self.config.verbose:
                print(f"\n[{obj_idx+1}/{len(ordered_objects)}] Testing object: {object_id}")

            # Generate primitive goals: List[List[Goal]] = [60 edges][10 steps]
            primitive_goals = self.goal_selection_strategy.generate_goals(
                object_id, root_state, self.env, self.config.max_goals_per_object
            )

            if not primitive_goals:
                if self.config.verbose:
                    print(f"  No primitive goals generated for {object_id}")
                continue

            # For each edge point
            for edge_idx, edge_goals in enumerate(primitive_goals):
                # Check timeout
                if timeout_seconds and (time.time() - start_time) > timeout_seconds:
                    break

                # For each push step (1-10), up to current shortest
                for step_idx in range(min(len(edge_goals), shortest_depth)):
                    push_steps = step_idx + 1

                    # Reset to root state for independent test
                    self.env.set_full_state(root_state)

                    goal = edge_goals[step_idx]
                    action = Action(object_id=object_id, goal=goal)

                    # Execute skill
                    step_result = self.env.step(action.to_namo_action())

                    if self.config.collect_stats:
                        self.stats['nodes_expanded'] += 1

                    # Check for collision (PRUNE)
                    if self._is_collision(step_result):
                        if self.config.verbose:
                            print(f"  Edge {edge_idx}: Collision at step {push_steps}, pruning")
                        break  # Stop testing longer pushes on this edge

                    # Check termination (SUCCESS)
                    new_state = self.env.get_full_state()
                    if self._is_terminal_state(new_state):
                        # Found a solution
                        solution = {
                            'object_id': object_id,
                            'edge_idx': edge_idx,
                            'push_steps': push_steps,
                            'goal': goal,
                            'action': action.to_namo_action()
                        }

                        if push_steps < shortest_depth:
                            # Found shorter solution - discard previous solutions
                            shortest_depth = push_steps
                            all_solutions = [solution]
                            if self.config.verbose:
                                print(f"  ✓ NEW SHORTEST: obj={object_id}, edge={edge_idx}, steps={push_steps}")
                        elif push_steps == shortest_depth:
                            # Found another solution at same depth
                            all_solutions.append(solution)
                            if self.config.verbose:
                                print(f"  ✓ Solution: obj={object_id}, edge={edge_idx}, steps={push_steps}")

                        break  # Stop testing longer pushes on this edge

        # Format results
        search_time_ms = (time.time() - start_time) * 1000

        if all_solutions:
            # Return the shortest solution(s)
            shortest_action = all_solutions[0]['action']

            if self.config.verbose:
                print(f"\n✓ Found {len(all_solutions)} solution(s) at depth {shortest_depth}")
                for sol in all_solutions:
                    print(f"  - {sol['object_id']}, edge {sol['edge_idx']}, steps {sol['push_steps']}")

            return PlannerResult(
                success=True,
                solution_found=True,
                action_sequence=[shortest_action],  # Return first shortest solution
                solution_depth=shortest_depth,
                state_observations=[],  # Not tracked in primitive mode
                post_action_state_observations=[],
                search_time_ms=search_time_ms,
                nodes_expanded=self.stats.get('nodes_expanded', 0),
                terminal_checks=self.stats.get('terminal_checks', 0),
                max_depth_reached=shortest_depth,
                algorithm_stats={
                    'primitive_based': True,
                    'initial_reachable_count': len(self.initial_reachable_objects),
                    'solutions_found': len(all_solutions),
                    'all_solutions': all_solutions,
                    'shortest_push_steps': shortest_depth
                }
            )
        else:
            if self.config.verbose:
                print(f"\n✗ No solution found")

            return PlannerResult(
                success=True,
                solution_found=False,
                state_observations=None,
                post_action_state_observations=None,
                search_time_ms=search_time_ms,
                nodes_expanded=self.stats.get('nodes_expanded', 0),
                terminal_checks=self.stats.get('terminal_checks', 0),
                max_depth_reached=0,
                algorithm_stats={
                    'primitive_based': True,
                    'initial_reachable_count': len(self.initial_reachable_objects),
                    'solutions_found': 0
                }
            )

    def _is_collision(self, step_result: namo_rl.StepResult) -> bool:
        """Check if action resulted in collision.

        Collision is indicated by failure_reason in step result containing
        keywords that suggest the object got stuck or hit obstacles.

        Args:
            step_result: Result from env.step()

        Returns:
            True if collision detected, False otherwise
        """
        failure_reason = step_result.info.get("failure_reason", "")

        # Keywords indicating collision/stuck
        collision_keywords = [
            "No reachable edges",
            "stuck",
            "No plan found",
            "collision"
        ]

        return any(keyword in failure_reason for keyword in collision_keywords)

    def _is_terminal_state(self, state: namo_rl.RLState) -> bool:
        """Check if state is terminal (two conditions).

        Terminal if EITHER:
        1. Robot goal is reachable
        2. New objects have become reachable

        Args:
            state: State to check

        Returns:
            True if terminal, False otherwise
        """
        # Check terminal check limit before expensive operation
        if (self.config.max_terminal_checks is not None and
            self.config.collect_stats and
            self.stats.get('terminal_checks', 0) >= self.config.max_terminal_checks):
            return False  # Treat as non-terminal to stop search

        self.env.set_full_state(state)

        # Condition 1: Check if robot goal is reachable (original condition)
        is_goal_reachable = self.env.is_robot_goal_reachable()

        if is_goal_reachable:
            if self.config.collect_stats:
                self.stats['terminal_checks'] += 1
            return True

        # Condition 2: Check if reachable set has expanded
        current_reachable = set(self.env.get_reachable_objects())
        new_objects = current_reachable - self.initial_reachable_objects

        if new_objects:
            if self.config.collect_stats:
                self.stats['terminal_checks'] += 1
                self.stats['reachability_expansions'] += 1
            return True

        # Neither condition met - not terminal
        if self.config.collect_stats:
            self.stats['terminal_checks'] += 1

        return False


# Register the planner with the factory
from namo.core import PlannerFactory
PlannerFactory.register_planner("expanding_idfs", ReachabilityExpandingIDFS)


# Convenience function
def plan_with_reachability_expanding_idfs(env: namo_rl.RLEnvironment,
                                          robot_goal: Tuple[float, float, float],
                                          max_depth: int = 5,
                                          max_goals_per_object: int = 5,
                                          random_seed: Optional[int] = None,
                                          verbose: bool = False,
                                          collect_stats: bool = True,
                                          object_selection_strategy: Optional['ObjectSelectionStrategy'] = None,
                                          goal_selection_strategy: Optional['GoalSelectionStrategy'] = None) -> Optional[List[namo_rl.Action]]:
    """Plan action sequence using reachability-expanding IDFS.

    Args:
        env: NAMO RL environment
        robot_goal: Target robot position (x, y, theta)
        max_depth: Maximum search depth
        max_goals_per_object: Maximum goals to try per object
        random_seed: Random seed for reproducibility
        verbose: Enable verbose output
        collect_stats: Collect search statistics
        object_selection_strategy: Strategy for ordering objects
        goal_selection_strategy: Strategy for generating goals

    Returns:
        List of actions if solution found, None otherwise
    """
    from namo.core import PlannerConfig

    config = PlannerConfig(
        max_depth=max_depth,
        max_goals_per_object=max_goals_per_object,
        random_seed=random_seed,
        verbose=verbose,
        collect_stats=collect_stats
    )

    planner = ReachabilityExpandingIDFS(env, config, object_selection_strategy, goal_selection_strategy)
    result = planner.search(robot_goal)

    return result.action_sequence if result.solution_found else None
