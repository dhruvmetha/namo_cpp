"""ML-Driven Async Search for N-Push NAMO Problems.

This module implements an event-driven search algorithm that:
- Submits ML inference immediately when discovering new states
- Prioritizes ML results when they arrive (they "jump the queue")
- Fills wait time with fallback work (CPU never idle)
- Finds ML-guided solutions first (better solutions found faster)

See docs/ML_DRIVEN_ASYNC_ALGORITHM.md for detailed algorithm specification.
"""

import heapq
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple, Any, TYPE_CHECKING

import namo_rl

from namo.strategies import Goal, PrimitiveGoalStrategy, MLPrimitiveAsyncStrategy

if TYPE_CHECKING:
    from namo.strategies import AsyncGoalResult


@dataclass
class WorkEntry:
    """Single entry per state with ML and fallback candidates.

    Each state being explored has exactly ONE WorkEntry in the WORK_QUEUE.
    Contains both ML candidates (when available) and fallback primitives.

    Fallback ordering: (depth, edge) to match ml_fallback strategy - tries
    shallowest depths first across all edges before moving to deeper pushes.
    """
    state: namo_rl.RLState
    state_id: int
    object_id: str

    # ML candidates (empty until ML arrives, then filled with scored goals)
    ml_candidates: List[Goal] = field(default_factory=list)
    ml_idx: int = 0

    # Fallback candidates (all primitives, available immediately)
    # Stored as List[List[Goal]] - [edge_idx][depth_idx]
    fallback_goals_per_edge: List[List[Goal]] = field(default_factory=list)

    # Fallback iteration: sorted list of (depth_idx, edge_idx) for (depth, edge) ordering
    # Built lazily on first access to match ml_fallback ordering
    _fallback_order: Optional[List[Tuple[int, int]]] = field(default=None, repr=False)
    _fallback_idx: int = 0

    # Track which primitives have been tried (edge_idx, depth_idx)
    tried_primitives: Set[Tuple[int, int]] = field(default_factory=set)

    # Path info
    parent_chain: List[Goal] = field(default_factory=list)
    path_ml_score: float = 0.0

    # Reachable edges at this state (for filtering)
    reachable_edges: Set[int] = field(default_factory=set)

    def _build_fallback_order(self) -> None:
        """Build sorted fallback order: (depth, edge) to match ml_fallback."""
        if self._fallback_order is not None:
            return

        # Collect all valid (depth, edge) pairs for reachable edges
        candidates = []
        for edge_idx, edge_goals in enumerate(self.fallback_goals_per_edge):
            if edge_idx not in self.reachable_edges:
                continue
            for depth_idx in range(len(edge_goals)):
                candidates.append((depth_idx, edge_idx))

        # Sort by (depth, edge) - shallowest depths first, then by edge
        candidates.sort(key=lambda x: (x[0], x[1]))
        self._fallback_order = candidates
        self._fallback_idx = 0

    def has_ml_work(self) -> bool:
        """Check if there are untried ML candidates."""
        return self.ml_idx < len(self.ml_candidates)

    def has_fallback_work(self) -> bool:
        """Check if there are untried fallback candidates."""
        self._build_fallback_order()

        # Find next untried primitive in (depth, edge) order
        while self._fallback_idx < len(self._fallback_order):
            depth_idx, edge_idx = self._fallback_order[self._fallback_idx]
            key = (edge_idx, depth_idx)
            if key not in self.tried_primitives:
                return True
            self._fallback_idx += 1

        return False

    def has_any_work(self) -> bool:
        """Check if there's any work left."""
        return self.has_ml_work() or self.has_fallback_work()

    def priority(self) -> Tuple[int, float]:
        """Priority for queue sorting. Higher = process first.

        Returns:
            Tuple of (has_ml_work, path_ml_score) where:
            - has_ml_work: 1 if ML candidates available, 0 otherwise
            - path_ml_score: cumulative ML score along path (higher = better)
        """
        return (
            1 if self.has_ml_work() else 0,  # ML work before fallback
            self.path_ml_score,               # Higher path score first
        )

    def get_next_candidate(self) -> Optional[Tuple[Goal, int, int, bool]]:
        """Get next candidate to try (ML first, then fallback).

        Fallback order: (depth, edge) - tries shallowest depths first across
        all edges, matching ml_fallback strategy ordering.

        Returns:
            Tuple of (goal, edge_idx, depth_idx, is_ml_candidate) or None if exhausted.
        """
        # Try ML candidates first
        if self.has_ml_work():
            goal = self.ml_candidates[self.ml_idx]
            self.ml_idx += 1
            # ML candidates have edge_idx and depth_idx encoded in score or as attributes
            # For now, return -1, -1 since ML candidates are already filtered
            return (goal, -1, -1, True)

        # Try fallback primitives in (depth, edge) order
        self._build_fallback_order()

        while self._fallback_idx < len(self._fallback_order):
            depth_idx, edge_idx = self._fallback_order[self._fallback_idx]
            key = (edge_idx, depth_idx)

            if key not in self.tried_primitives:
                goal = self.fallback_goals_per_edge[edge_idx][depth_idx]
                self.tried_primitives.add(key)
                self._fallback_idx += 1
                return (goal, edge_idx, depth_idx, False)

            self._fallback_idx += 1

        return None

    def update_ml_candidates(self, ml_scores: Dict[Tuple[int, int], float]) -> None:
        """Update entry with ML scores, creating ordered candidate list.

        Args:
            ml_scores: Dict mapping (edge_idx, depth_idx) to vote count.
        """
        if not ml_scores:
            return

        # Build ML candidates from scored slots, excluding already-tried primitives
        candidates = []
        skipped_tried = 0
        skipped_unreachable = 0
        skipped_invalid = 0

        for (edge_idx, depth_idx), score in ml_scores.items():
            # Skip if already tried via fallback
            if (edge_idx, depth_idx) in self.tried_primitives:
                skipped_tried += 1
                continue

            # Skip unreachable edges
            if edge_idx not in self.reachable_edges:
                skipped_unreachable += 1
                continue

            # Get the actual goal from fallback grid
            if edge_idx < len(self.fallback_goals_per_edge):
                edge_goals = self.fallback_goals_per_edge[edge_idx]
                if depth_idx < len(edge_goals):
                    goal = edge_goals[depth_idx]
                    # Create new goal with ML score
                    ml_goal = Goal(
                        x=goal.x,
                        y=goal.y,
                        theta=goal.theta,
                        score=score
                    )
                    candidates.append((score, depth_idx, edge_idx, ml_goal))
                else:
                    skipped_invalid += 1
            else:
                skipped_invalid += 1

        # Sort by (-score, depth, edge) to match ml_fallback ordering
        # Higher scores first, then shallowest depths, then by edge
        candidates.sort(key=lambda x: (-x[0], x[1], x[2]))

        # Extract goals (mark primitives as will-be-tried-via-ML)
        self.ml_candidates = []
        for score, depth_idx, edge_idx, goal in candidates:
            self.ml_candidates.append(goal)
            # Mark as tried so fallback skips them
            self.tried_primitives.add((edge_idx, depth_idx))

        self.ml_idx = 0


class WorkQueue:
    """Priority queue with O(1) lookup and O(log n) priority update.

    Uses lazy heap pattern: when priority changes, push new entry with
    incremented version. On pop, skip entries with stale versions.

    Key insight: We never actually remove from heap - we just mark entries
    as stale by removing from `entries` dict. The heap cleans itself lazily.
    """

    def __init__(self):
        # The heap: (neg_priority_tuple, version, state_id)
        # neg_priority because heapq is min-heap, we want max-priority first
        self.heap: List[Tuple[Tuple[int, float], int, int]] = []

        # state_id -> WorkEntry (single source of truth)
        self.entries: Dict[int, WorkEntry] = {}

        # state_id -> current version (for staleness check)
        self.versions: Dict[int, int] = {}

        # Global version counter
        self.counter: int = 0

    def add(self, entry: WorkEntry) -> None:
        """Add new entry. O(log n)."""
        state_id = entry.state_id
        self.entries[state_id] = entry
        self._push_to_heap(state_id)

    def get(self, state_id: int) -> Optional[WorkEntry]:
        """Get entry by state_id. O(1)."""
        return self.entries.get(state_id)

    def update_priority(self, state_id: int) -> None:
        """Call when entry's priority changes (ML arrived). O(log n).

        Pushes new heap entry with fresh version. Old entry becomes stale
        and will be skipped on future pops.
        """
        if state_id in self.entries:
            self._push_to_heap(state_id)

    def peek_best(self) -> Optional[WorkEntry]:
        """Get highest priority entry without removing. O(log n) amortized.

        Cleans stale entries from heap top until valid entry found.
        """
        self._clean_stale()
        if not self.heap:
            return None
        _, _, state_id = self.heap[0]
        return self.entries.get(state_id)

    def pop_best(self) -> Optional[WorkEntry]:
        """Remove and return highest priority entry. O(log n) amortized."""
        self._clean_stale()
        if not self.heap:
            return None
        _, _, state_id = heapq.heappop(self.heap)
        entry = self.entries.pop(state_id, None)
        self.versions.pop(state_id, None)
        return entry

    def remove(self, state_id: int) -> None:
        """Remove entry by state_id. O(1) - lazy removal.

        Just removes from entries dict. Heap entry becomes stale and
        will be skipped on future pops.
        """
        self.entries.pop(state_id, None)
        self.versions.pop(state_id, None)

    def __len__(self) -> int:
        """Number of valid entries. O(1)."""
        return len(self.entries)

    def __bool__(self) -> bool:
        """True if queue has valid entries. O(1)."""
        return bool(self.entries)

    def _push_to_heap(self, state_id: int) -> None:
        """Push entry to heap with fresh version."""
        self.counter += 1
        self.versions[state_id] = self.counter

        entry = self.entries[state_id]
        priority = entry.priority()

        # Negate priority for min-heap -> max-priority behavior
        neg_priority = (-priority[0], -priority[1])
        heapq.heappush(self.heap, (neg_priority, self.counter, state_id))

    def _clean_stale(self) -> None:
        """Remove stale entries from heap top."""
        while self.heap:
            _, version, state_id = self.heap[0]

            # Entry removed?
            if state_id not in self.versions:
                heapq.heappop(self.heap)
                continue

            # Entry has newer version? (priority was updated)
            if version != self.versions[state_id]:
                heapq.heappop(self.heap)
                continue

            # Valid entry found
            break


@dataclass
class SearchSolution:
    """A successful solution found during search."""
    chain: List[Goal]              # Full sequence of pushes
    path_ml_score: float           # How ML-guided this solution is
    num_pushes: int                # Length of chain
    object_id: str
    neighbor_label: str            # Target region that was opened
    resulting_state: namo_rl.RLState  # State after executing chain
    state_observations: List[Dict[str, Any]] = field(default_factory=list)
    post_action_observations: List[Dict[str, Any]] = field(default_factory=list)
    timing_ms: float = 0.0
    # Collision tracking for hardness metrics
    any_wall_collision: bool = False  # Did any push hit a wall?
    unique_movable_collision_count: int = 0  # Unique movable objects hit across chain


class MLDrivenAsyncSearch:
    """ML-driven async search for N-push NAMO problems.

    This search algorithm ensures:
    1. Zero idle time - every frontier added to WORK_QUEUE with fallback ready
    2. ML results always get priority - entries with ML candidates processed first
    3. No blocking on ML - check for ML results after every push
    4. Works for any N pushes - depth is just chain length
    5. Best solutions first - ML-guided paths found before fallback paths
    """

    def __init__(
        self,
        env: namo_rl.RLEnvironment,
        primitive_strategy: PrimitiveGoalStrategy,
        ml_strategy: Optional[MLPrimitiveAsyncStrategy] = None,
        max_chain_depth: int = 3,
        max_solutions: int = 1,
        verbose: bool = False,
        terminate_on_collision: bool = True,
    ):
        """Initialize ML-driven async search.

        Args:
            env: NAMO RL environment.
            primitive_strategy: Strategy for generating primitive goals.
            ml_strategy: Optional async ML strategy. If None, fallback-only mode.
            max_chain_depth: Maximum number of pushes in a chain.
            max_solutions: Maximum solutions to find before stopping.
            verbose: Enable debug output.
            terminate_on_collision: Whether to terminate push on collision.
        """
        self.env = env
        self.primitive_strategy = primitive_strategy
        self.ml_strategy = ml_strategy
        self.max_chain_depth = max_chain_depth
        self.max_solutions = max_solutions
        self.verbose = verbose
        self.terminate_on_collision = terminate_on_collision

        # Search state
        self.work_queue = WorkQueue()
        self.pending_ml: Dict[int, 'AsyncGoalResult'] = {}  # state_id -> async result
        self.solutions: List[SearchSolution] = []

        # Stats
        self.total_pushes = 0
        self.ml_arrivals = 0

    def _debug(self, message: str):
        """Print debug message if verbose enabled."""
        if self.verbose:
            print(message)

    def search(
        self,
        object_id: str,
        baseline_state: namo_rl.RLState,
        neighbor_label: str,
        validate_opening_fn,
    ) -> List[SearchSolution]:
        """Run ML-driven async search to find openings.

        Args:
            object_id: Object to push.
            baseline_state: Starting environment state.
            neighbor_label: Target neighbor region.
            validate_opening_fn: Function(env) -> (is_open, count, goal, all_goals)

        Returns:
            List of SearchSolution found.
        """
        # Cancel and wait for any previous ML inference to complete
        # This happens BEFORE timing starts so it doesn't inflate search time
        from namo.strategies import MLPrimitiveAsyncStrategy
        MLPrimitiveAsyncStrategy.cancel_all_pending()
        self._wait_for_pending_ml()

        # Clear cancellation signal BEFORE submitting new ML
        MLPrimitiveAsyncStrategy.clear_cancellation()

        # Start timing AFTER previous ML cleanup
        start_time = time.time()

        # Reset search state
        self.work_queue = WorkQueue()
        self.pending_ml = {}
        self.solutions = []
        self.total_pushes = 0
        self.ml_arrivals = 0
        self._baseline_state = baseline_state  # Store for observation collection

        # Initialize: add initial state to both buckets
        self._initialize_search(object_id, baseline_state)

        self._debug(f"🚀 ML-Driven Async Search started for {object_id}")
        self._debug(f"   Max chain depth: {self.max_chain_depth}, Max solutions: {self.max_solutions}")

        # Main loop
        iteration = 0
        while self._should_continue():
            iteration += 1
            # Step 1: Harvest ML results (non-blocking)
            self._harvest_ml_results()

            # Step 2: Process highest priority entry
            entry = self.work_queue.peek_best()

            if entry is None:
                # Work queue empty, check if ML pending
                if self.pending_ml:
                    # Wait briefly for ML
                    time.sleep(0.01)
                    continue
                else:
                    # Nothing left to do
                    break

            # Check depth limit
            current_depth = len(entry.parent_chain) + 1
            if current_depth > self.max_chain_depth:
                self.work_queue.remove(entry.state_id)
                continue

            # Check if entry has work
            # Don't remove if ML is still pending for this entry
            if not entry.has_any_work():
                if entry.state_id not in self.pending_ml:
                    self.work_queue.remove(entry.state_id)
                continue

            # Get next candidate
            candidate = entry.get_next_candidate()
            if candidate is None:
                self.work_queue.remove(entry.state_id)
                continue

            goal, edge_idx, depth_idx, is_ml = candidate

            # Execute push
            self.env.set_full_state(entry.state)

            # Capture pre-action state
            pre_obs = self.env.get_observation()

            # Execute action
            action = namo_rl.Action()
            action.object_id = object_id
            action.x = goal.x
            action.y = goal.y
            action.theta = goal.theta

            self.total_pushes += 1

            try:
                step_result = self.env.step(action)
            except Exception as e:
                self._debug(f"      ❌ Push failed: {e}")
                continue

            # Capture post-action state
            post_obs = self.env.get_observation()

            # Check for collision/stuck
            collision = False
            if self.terminate_on_collision and "collision_object" in step_result.info:
                collision = True
            if "stuck" in step_result.info and step_result.info["stuck"] == "true":
                collision = True

            # Check if opening created
            is_open, reachable_count, region_goal, all_goals = validate_opening_fn(self.env)

            if is_open:
                # Solution found!
                new_chain = entry.parent_chain + [goal]
                new_score = entry.path_ml_score + (goal.score if is_ml else 0.0)
                resulting_state = self.env.get_full_state()

                # For multi-push chains, re-execute to collect all observations
                if len(new_chain) > 1:
                    state_obs, post_obs_list, any_wall_collision, unique_movable_collision_count = self._collect_chain_observations(
                        object_id, new_chain, self._baseline_state
                    )
                else:
                    # Single push - extract collision info from current step_result
                    state_obs = [pre_obs]
                    post_obs_list = [post_obs]
                    any_wall_collision = step_result.info.get("wall_collision", "false") == "true"
                    movable_str = step_result.info.get("movable_collisions", "")
                    unique_movable_collision_count = len([s for s in movable_str.split(",") if s.strip()]) if movable_str else 0

                solution = SearchSolution(
                    chain=new_chain,
                    path_ml_score=new_score,
                    num_pushes=len(new_chain),
                    object_id=object_id,
                    neighbor_label=neighbor_label,
                    resulting_state=resulting_state,
                    state_observations=state_obs,
                    post_action_observations=post_obs_list,
                    timing_ms=(time.time() - start_time) * 1000,
                    any_wall_collision=any_wall_collision,
                    unique_movable_collision_count=unique_movable_collision_count,
                )
                self.solutions.append(solution)

                ml_type = "ML" if is_ml else "FB"
                self._debug(f"      ✅ SOLUTION! {ml_type} push, chain={len(new_chain)}, score={new_score:.1f}")

                if len(self.solutions) >= self.max_solutions:
                    break

            elif not collision:
                # Valid push but no solution yet - create new frontier
                if current_depth < self.max_chain_depth:
                    new_chain = entry.parent_chain + [goal]
                    new_score = entry.path_ml_score + (goal.score if is_ml else 0.0)

                    self._add_frontier(
                        object_id=object_id,
                        state=self.env.get_full_state(),
                        parent_chain=new_chain,
                        path_ml_score=new_score,
                    )

            # Note: priority update happens in _harvest_ml_results when ML arrives
            # No need to update here - entry stays at same priority

        # Record search time BEFORE waiting for pending ML
        # This way the wait doesn't inflate the recorded timing
        search_time_ms = (time.time() - start_time) * 1000

        # Wait for any running ML inference to complete before returning
        # This ensures the next env doesn't start until DDIM is done
        # NOTE: This wait is NOT included in solution timing_ms
        from namo.strategies import MLPrimitiveAsyncStrategy
        MLPrimitiveAsyncStrategy.cancel_all_pending()

        for state_id, async_result in list(self.pending_ml.items()):
            if async_result.ml_future is not None:
                if not async_result.ml_future.done():
                    self._debug(f"      ⏳ Waiting for ML {state_id} to complete...")
                    try:
                        async_result.ml_future.result(timeout=30.0)
                        self._debug(f"      ✓ ML {state_id} completed")
                    except Exception as e:
                        self._debug(f"      ⚠️ ML {state_id} wait failed: {e}")
                elif async_result.cancel_if_pending():
                    self._debug(f"      🛑 Cancelled pending ML for state {state_id}")
        self.pending_ml.clear()

        total_time_ms = (time.time() - start_time) * 1000
        wait_time_ms = total_time_ms - search_time_ms
        self._debug(f"🏁 Search complete: {len(self.solutions)} solutions, "
                   f"{self.total_pushes} pushes, {self.ml_arrivals} ML arrivals, "
                   f"search={search_time_ms:.0f}ms, wait={wait_time_ms:.0f}ms")

        return self.solutions

    def _wait_for_pending_ml(self) -> None:
        """Wait for any pending ML inference to complete (from previous search).

        This ensures the GPU is free before starting new inference.
        Only waits if there are entries in pending_ml from a previous search.
        """
        if not self.pending_ml:
            return

        self._debug(f"      ⏳ Waiting for {len(self.pending_ml)} pending ML to complete...")
        for state_id, async_result in list(self.pending_ml.items()):
            if async_result.ml_future is not None and not async_result.ml_future.done():
                try:
                    # Wait with timeout to avoid infinite hang
                    async_result.ml_future.result(timeout=30.0)
                except Exception as e:
                    self._debug(f"      ⚠️ ML wait failed: {e}")
        self.pending_ml.clear()
        self._debug(f"      ✓ Previous ML cleared")

    def _should_continue(self) -> bool:
        """Check if search should continue."""
        if len(self.solutions) >= self.max_solutions:
            return False
        if not self.work_queue and not self.pending_ml:
            return False
        return True

    def _initialize_search(self, object_id: str, baseline_state: namo_rl.RLState) -> None:
        """Initialize search with initial state in both buckets."""
        self.env.set_full_state(baseline_state)

        # Generate all primitives (sync, fast)
        primitives = self.primitive_strategy.generate_goals(
            object_id, baseline_state, self.env, max_goals=0
        )

        if not primitives:
            self._debug("   ⚠️ No primitives generated")
            return

        # Get reachable edges
        reachable_edges = set(self.env.get_reachable_edges(object_id))

        state_id = id(baseline_state)

        # Create work entry with fallback
        entry = WorkEntry(
            state=baseline_state,
            state_id=state_id,
            object_id=object_id,
            ml_candidates=[],
            ml_idx=0,
            fallback_goals_per_edge=primitives,
            tried_primitives=set(),
            parent_chain=[],
            path_ml_score=0.0,
            reachable_edges=reachable_edges,
        )

        # Add to work queue
        self.work_queue.add(entry)

        num_edges = len(primitives)
        num_depths = len(primitives[0]) if primitives else 0
        self._debug(f"   Initial state: {num_edges} edges × {num_depths} depths, "
                   f"{len(reachable_edges)} reachable edges")

        # Submit ML inference (async)
        if self.ml_strategy:
            self._submit_ml_async(state_id, object_id, baseline_state)

    def _add_frontier(
        self,
        object_id: str,
        state: namo_rl.RLState,
        parent_chain: List[Goal],
        path_ml_score: float,
    ) -> None:
        """Add new frontier state to both buckets."""
        state_id = id(state)

        # Generate primitives for new state
        primitives = self.primitive_strategy.generate_goals(
            object_id, state, self.env, max_goals=0
        )

        if not primitives:
            return

        # Get reachable edges
        self.env.set_full_state(state)
        reachable_edges = set(self.env.get_reachable_edges(object_id))

        # Create work entry
        entry = WorkEntry(
            state=state,
            state_id=state_id,
            object_id=object_id,
            ml_candidates=[],
            ml_idx=0,
            fallback_goals_per_edge=primitives,
            tried_primitives=set(),
            parent_chain=parent_chain,
            path_ml_score=path_ml_score,
            reachable_edges=reachable_edges,
        )

        # Add to work queue
        self.work_queue.add(entry)

        # Submit ML inference (async)
        if self.ml_strategy:
            self._submit_ml_async(state_id, object_id, state)

    def _submit_ml_async(self, state_id: int, object_id: str, state: namo_rl.RLState) -> None:
        """Submit ML inference request (async)."""
        try:
            # Use async strategy to get AsyncGoalResult
            async_result = self.ml_strategy.generate_goals(
                object_id, state, self.env, max_goals=0
            )

            if async_result.ml_future is not None:
                self.pending_ml[state_id] = async_result
                self._debug(f"      📡 ML submitted for state {state_id}")
        except Exception as e:
            self._debug(f"      ⚠️ ML submission failed: {e}")

    def _harvest_ml_results(self) -> None:
        """Check pending ML futures and update work entries (non-blocking)."""
        completed = []

        for state_id, async_result in self.pending_ml.items():
            if async_result.poll_ml_ready():
                completed.append(state_id)

                # Get ML scores
                ml_scores = async_result.get_ml_scores()

                if ml_scores:
                    self.ml_arrivals += 1
                    self._debug(f"      🎯 ML arrived for state {state_id}: {len(ml_scores)} slots")

                    # Update work entry
                    entry = self.work_queue.get(state_id)
                    if entry:
                        entry.update_ml_candidates(ml_scores)
                        self.work_queue.update_priority(state_id)

        # Remove completed from pending
        for state_id in completed:
            del self.pending_ml[state_id]

    def _collect_chain_observations(
        self,
        object_id: str,
        chain: List[Goal],
        baseline_state: namo_rl.RLState,
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], bool, int]:
        """Re-execute chain to collect all observations and collision info.

        Args:
            object_id: Object being pushed.
            chain: Full chain of goals.
            baseline_state: Starting state before first push.

        Returns:
            Tuple of (state_observations, post_action_observations,
                     any_wall_collision, unique_movable_collision_count).
        """
        state_observations = []
        post_action_observations = []

        # Collision tracking - accumulate across all pushes
        any_wall_collision = False
        all_movable_collisions: Set[str] = set()

        # Start from baseline
        self.env.set_full_state(baseline_state)

        for goal in chain:
            # Capture pre-action observation
            pre_obs = self.env.get_observation()
            state_observations.append(pre_obs)

            # Execute push
            action = namo_rl.Action()
            action.object_id = object_id
            action.x = goal.x
            action.y = goal.y
            action.theta = goal.theta

            try:
                step_result = self.env.step(action)

                # Extract collision info from step result
                if step_result.info.get("wall_collision", "false") == "true":
                    any_wall_collision = True
                movable_str = step_result.info.get("movable_collisions", "")
                if movable_str:
                    for obj_name in movable_str.split(","):
                        obj_name = obj_name.strip()
                        if obj_name:
                            all_movable_collisions.add(obj_name)
            except Exception:
                # Push failed during replay - shouldn't happen for valid chains
                pass

            # Capture post-action observation
            post_obs = self.env.get_observation()
            post_action_observations.append(post_obs)

        unique_movable_collision_count = len(all_movable_collisions)
        return state_observations, post_action_observations, any_wall_collision, unique_movable_collision_count
