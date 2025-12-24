# ML-Driven Async N-Push Search Algorithm

This document describes the ML-driven asynchronous search algorithm for solving N-push NAMO problems where ML inference guides the search while keeping both GPU and CPU fully utilized.

## Table of Contents

1. [Overview](#overview)
2. [Key Concepts](#key-concepts)
3. [Data Structures](#data-structures)
4. [The Algorithm](#the-algorithm)
5. [Why This Works for N Pushes](#why-this-works-for-n-pushes)
6. [Example Trace](#example-trace)
7. [Implementation Plan](#implementation-plan)

---

## Overview

### The Problem

In multi-push NAMO problems, we need to find a sequence of pushes that opens a path to a goal region. Traditional approaches either:

1. **Wait for ML** at each step (wastes CPU time)
2. **Ignore ML priority** and mix ML with fallback (finds suboptimal solutions first)

### The Solution

An event-driven search that:

- **Submits ML immediately** when discovering new states
- **Prioritizes ML results** when they arrive (they "jump the queue")
- **Fills wait time** with fallback work (CPU never idle)
- **Finds ML-guided solutions first** (better solutions found faster)

### Key Principles

```
RULE 1: Always ask ML immediately
   When you discover a new state, don't wait - ask ML right away

RULE 2: ML results always cut in line
   When ML comes back with suggestions, try them before anything else

RULE 3: Best ML paths first
   If multiple states have ML ready, pick the one with the
   most ML-guided history (path with highest total ML scores)

RULE 4: Fallback is last resort
   Only try non-ML primitives when there's zero ML-ready work

RULE 5: Even fallback can lead to ML
   If a fallback push creates a new state, still ask ML about it

RULE 6: Every frontier gets added to WORK_QUEUE
   When any push creates a new frontier:
   - Submit ML request (track in PENDING_ML)
   - Add entry to WORK_QUEUE with all primitives as fallback
   This guarantees CPU always has work to do
```

---

## Key Concepts

### Path ML Score

The "path ML score" measures how ML-guided a path is:

```
path_ml_score = sum of ML vote scores for each push in the path

Example:
  Push 1: Edge 47 (ML gave 8 votes)
  Push 2: Edge 15 (ML gave 6 votes)
  Push 3: Edge 8  (ML gave 5 votes)

  path_ml_score = 8 + 6 + 5 = 19
```

Higher path_ml_score = more ML-guided = higher priority.

### Solution Priority Order

Solutions are found in this order (best to worst):

```
Priority 1: ML → ML → ML        (fully ML-guided, all pushes suggested by ML)
Priority 2: ML → ML → Fallback  (ML guided most of the path)
Priority 3: ML → Fallback → ML  (mixed)
Priority 4: Fallback → ML → ML  (fallback start, ML finish)
Priority 5: Fallback → Fallback (pure exhaustive search, last resort)
```

### No Waiting (Ever)

The CPU never sits idle - not even at the start:

```
WITH WAITING (bad):
   T=0: Submit ML → Wait... wait... wait... → ML ready! → Do ML work
                    └─── CPU idle 1-2s ───┘

NO WAITING (good):
   T=0: Submit ML AND start fallback immediately
        → Do fallback push → Check ML → Still pending?
        → Do fallback push → Check ML → ML ready! → Switch to ML work
          └─── CPU productive from T=0 ───┘
```

**Key insight**: At initialization, we:
- Submit ML request (track future in PENDING_ML)
- Add initial state to WORK_QUEUE with all primitives as fallback

This way CPU can start fallback work immediately while ML runs on GPU.

### Handling Already-Tried Primitives

When ML arrives for a state that was already being explored via fallback, we need to track which primitives were already tried:

```
T=0:     Start fallback on initial: [E0, E1, E2, E3, ...]
T=50:    Tried E0 → created frontier
T=100:   Tried E1 → collision
T=150:   Tried E2 → created frontier
...
T=1500:  ML arrives! Says: [E47(8), E23(4)]

Now we have:
  - ML candidates: [E47, E23] (try these first!)
  - Already tried: [E0, E1, E2, ..., E30] (skip these in fallback)
  - Remaining fallback: [E31, E32, ...] minus [E47, E23]
```

**Implementation**: Track `tried_primitives: Set[int]` per state entry, or simply track `fallback_idx` and exclude ML candidates from fallback list.

### Zero Idle Time Guarantee

To ensure CPU is NEVER idle, every new frontier is added to WORK_QUEUE immediately:

```
When a push creates a valid new frontier:

1. Generate all primitives for new state (sync, ~1ms)

2. Submit ML inference (async)
   PENDING_ML[state_id] = future

3. Add to WORK_QUEUE with primitives as fallback:
   WORK_QUEUE.add({state, ml_candidates=[], fallback=all_primitives, ...})
```

This guarantees:
- WORK_QUEUE is NEVER empty while there are states to explore
- CPU can always do fallback work while waiting for ML
- When ML arrives, we update the entry's ml_candidates and re-prioritize

```
Timeline with zero idle:

GPU:  [░░░ ML_A ░░░][░░░ ML_B ░░░][░░░ ML_C ░░░]
CPU:  [fallback A][fallback B][ML_A ready→try ML][fallback C][ML_B ready→try ML]...
      └── always working, never waiting ──────────────────────────────────────┘
```

---

## Data Structures

We use a **simplified 2-bucket design** where each state has exactly ONE entry:

### PENDING_ML: Dict[state_id → Future]

Lightweight map tracking which states have pending ML inference. No duplicate state info.

```python
# Just tracks futures - the actual state info is in WORK_QUEUE
pending_ml: Dict[int, Future] = {}  # state_id → future
```

### WORK_QUEUE: Priority Queue of WorkEntry

Single source of truth for all states being explored. Each state has exactly ONE entry.

```python
@dataclass
class WorkEntry:
    state: RLState
    state_id: int                   # Unique identifier (id(state) or hash)

    # ML candidates (empty until ML arrives, then filled)
    ml_candidates: List[Goal]       # Sorted by score descending
    ml_idx: int                     # Current position in ML list

    # Fallback candidates (all primitives, available immediately)
    fallback_candidates: List[Goal] # Sorted by depth
    fallback_idx: int               # Current position in fallback list

    # Path info
    parent_chain: List[Goal]        # Pushes that led to this state
    path_ml_score: float            # Sum of ML scores along path

    object_id: str

    def has_ml_work(self) -> bool:
        """Check if there are untried ML candidates."""
        return self.ml_idx < len(self.ml_candidates)

    def has_fallback_work(self) -> bool:
        """Check if there are untried fallback candidates."""
        return self.fallback_idx < len(self.fallback_candidates)

    def has_any_work(self) -> bool:
        """Check if there's any work left."""
        return self.has_ml_work() or self.has_fallback_work()

    def priority(self) -> tuple:
        """Priority for queue sorting. Higher = process first."""
        return (
            self.has_ml_work(),     # ML work before fallback (True > False)
            self.path_ml_score,     # Higher path score first
        )

    def get_next_candidate(self) -> Optional[Goal]:
        """Get next candidate to try (ML first, then fallback)."""
        if self.has_ml_work():
            goal = self.ml_candidates[self.ml_idx]
            self.ml_idx += 1
            return goal
        elif self.has_fallback_work():
            goal = self.fallback_candidates[self.fallback_idx]
            self.fallback_idx += 1
            return goal
        return None
```

### WorkQueue Implementation: Lazy Heap

For large frontier sizes (n > 2000), we need efficient priority queue operations. A **Lazy Heap** approach provides:

- **O(log n)** add, pop, and priority update
- **O(1)** lookup by state_id
- Simple implementation using Python's `heapq`

#### Why Lazy Heap?

```
PROBLEM: Priorities change when ML arrives
  - Standard heap doesn't support priority updates
  - Indexed heap is complex to implement
  - Dict + max() is O(n) per lookup

SOLUTION: Lazy Heap with versioning
  - When priority changes, push NEW entry with new version
  - On pop, skip entries with stale versions
  - Stale entries are "lazy deleted" - cleaned up on pop
```

#### Complexity Analysis

| Operation           | Lazy Heap  | Dict + max() | Indexed Heap |
|---------------------|------------|--------------|--------------|
| add()               | O(log n)   | O(1)         | O(log n)     |
| peek_best()         | O(log n)*  | O(n)         | O(1)         |
| update_priority()   | O(log n)   | O(1)         | O(log n)     |
| remove()            | O(1)       | O(1)         | O(log n)     |
| get() by state_id   | O(1)       | O(1)         | O(1)         |

*Amortized - includes cleanup of stale entries

**For n > 2000**: Lazy Heap is ideal. O(log 2000) ≈ 11 comparisons vs O(2000) linear scan.

#### Complete Implementation

```python
import heapq
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

@dataclass
class WorkEntry:
    """Single entry per state with ML and fallback candidates."""
    state: 'RLState'
    state_id: int

    # ML candidates (empty until ML arrives)
    ml_candidates: List['Goal'] = field(default_factory=list)
    ml_idx: int = 0

    # Fallback candidates (all primitives, available immediately)
    fallback_candidates: List['Goal'] = field(default_factory=list)
    fallback_idx: int = 0

    # Path info
    parent_chain: List['Goal'] = field(default_factory=list)
    path_ml_score: float = 0.0

    object_id: str = ""

    def has_ml_work(self) -> bool:
        return self.ml_idx < len(self.ml_candidates)

    def has_fallback_work(self) -> bool:
        return self.fallback_idx < len(self.fallback_candidates)

    def has_any_work(self) -> bool:
        return self.has_ml_work() or self.has_fallback_work()

    def priority(self) -> Tuple[int, float]:
        """Priority for queue sorting. Higher = process first."""
        return (
            1 if self.has_ml_work() else 0,  # ML work before fallback
            self.path_ml_score,               # Higher path score first
        )

    def get_next_candidate(self) -> Optional['Goal']:
        """Get next candidate (ML first, then fallback)."""
        if self.has_ml_work():
            goal = self.ml_candidates[self.ml_idx]
            self.ml_idx += 1
            return goal
        elif self.has_fallback_work():
            goal = self.fallback_candidates[self.fallback_idx]
            self.fallback_idx += 1
            return goal
        return None


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

        # state_id → WorkEntry (single source of truth)
        self.entries: Dict[int, WorkEntry] = {}

        # state_id → current version (for staleness check)
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

        # Negate priority for min-heap → max-priority behavior
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
```

#### Usage Example

```python
# Create queue
queue = WorkQueue()

# Add initial state with fallback
entry = WorkEntry(
    state=initial_state,
    state_id=id(initial_state),
    ml_candidates=[],           # Empty - ML not ready yet
    fallback_candidates=all_primitives,
    path_ml_score=0.0
)
queue.add(entry)

# Get best entry (fallback only for now)
best = queue.peek_best()        # O(log n)
goal = best.get_next_candidate()  # Gets fallback

# ... ML arrives later ...

# Update entry with ML candidates
entry = queue.get(state_id)     # O(1)
entry.ml_candidates = ml_results
entry.ml_idx = 0

# Notify queue that priority changed
queue.update_priority(state_id)  # O(log n) - pushes new heap entry

# Now peek_best() returns this entry (has ML work = higher priority)
best = queue.peek_best()        # O(log n) amortized
assert best.has_ml_work()       # True!
```

### Why 2 Buckets Instead of 3?

```
PROBLEM WITH 3 BUCKETS:
  - Same state exists in multiple places (pending ML + fallback queue)
  - When ML arrives, must find and update/remove from fallback queue
  - Complex cross-bucket state management

SOLUTION WITH 2 BUCKETS:
  - Each state has ONE entry in WORK_QUEUE
  - PENDING_ML just tracks futures (lightweight, no state duplication)
  - When ML arrives, update the entry in place and re-sort
  - Single source of truth - much cleaner!
```

### Solutions

```python
@dataclass
class SearchSolution:
    chain: List[Goal]           # Full sequence of pushes
    path_ml_score: float        # How ML-guided this solution is
    num_pushes: int             # Length of chain
    object_id: str
    neighbor_label: str         # Target region that was opened
```

---

## The Algorithm

### Initialization

```
1. Start with the initial environment state

2. Generate ALL primitives for the blocking object
   (600 goals: 60 edges × 10 depths) - ready immediately, no ML needed

3. Submit ML inference request to GPU for initial state
   PENDING_ML[initial_state_id] = ml_future

4. Add initial state to WORK_QUEUE:
   WorkEntry(
     state: initial_state,
     state_id: id(initial_state),
     ml_candidates: [],              ← Empty, ML not ready yet
     ml_idx: 0,
     fallback_candidates: all_primitives,  ← All 600 primitives ready
     fallback_idx: 0,
     parent_chain: [],               ← Empty, this is the root
     path_ml_score: 0                ← No ML decisions yet
   )

5. Initialize:
   - SOLUTIONS: empty

Now CPU can start fallback work immediately (WORK_QUEUE has entry with fallback)
while GPU runs ML inference (PENDING_ML has future).
```

### Main Loop

```
WHILE (want more solutions) AND (WORK_QUEUE not empty OR PENDING_ML not empty):

    ┌─────────────────────────────────────────────────────────────────┐
    │ STEP 1: HARVEST ML RESULTS                                      │
    │                                                                 │
    │ Check PENDING_ML for completed futures, update WORK_QUEUE       │
    └─────────────────────────────────────────────────────────────────┘

    For each (state_id, future) in PENDING_ML:

        If future.done():

            ml_goals = future.get_result()

            # Get ML candidates (score > 0), sorted by score descending
            ml_candidates = sorted([g for g in ml_goals if g.score > 0],
                                   key=lambda g: -g.score)

            # Find the entry in WORK_QUEUE for this state
            entry = WORK_QUEUE.find(state_id)

            # Update entry with ML candidates
            # Insert ML candidates, excluding any already tried via fallback
            entry.ml_candidates = [g for g in ml_candidates
                                   if g not in entry.tried_fallback()]
            entry.ml_idx = 0

            # Re-sort WORK_QUEUE (entry's priority changed - now has ML work!)
            WORK_QUEUE.re_sort()

            # Remove from pending
            Remove state_id from PENDING_ML


    ┌─────────────────────────────────────────────────────────────────┐
    │ STEP 2: PROCESS HIGHEST PRIORITY ENTRY                          │
    │                                                                 │
    │ Get best entry, try next candidate (ML if available, else FB)   │
    └─────────────────────────────────────────────────────────────────┘

    If WORK_QUEUE is not empty:

        # Get highest priority entry (has_ml_work=True first, then by path_score)
        entry = WORK_QUEUE.peek_best()

        # Check depth limit
        current_depth = len(entry.parent_chain) + 1
        If current_depth > MAX_CHAIN_DEPTH:
            WORK_QUEUE.remove(entry)
            CONTINUE

        # Check if entry has any work left
        If not entry.has_any_work():
            WORK_QUEUE.remove(entry)
            CONTINUE

        # Get next candidate (ML first if available, then fallback)
        goal = entry.get_next_candidate()
        is_ml_candidate = (goal in entry.ml_candidates)

        # Execute push
        Set environment to entry.state
        result = Execute push(entry.object_id, goal)

        If SOLVED:
            solution_chain = entry.parent_chain + [goal]
            new_score = entry.path_ml_score + (goal.score if is_ml_candidate else 0)

            SOLUTIONS.add(SearchSolution(
                chain: solution_chain,
                path_ml_score: new_score,
                num_pushes: len(solution_chain)
            ))

            If len(SOLUTIONS) >= MAX_SOLUTIONS:
                RETURN SOLUTIONS

        Else If VALID (push worked, didn't solve):
            # New frontier!
            new_state = current environment state
            new_chain = entry.parent_chain + [goal]
            new_score = entry.path_ml_score + (goal.score if is_ml_candidate else 0)

            If len(new_chain) < MAX_CHAIN_DEPTH:
                # Generate primitives for new state (sync, fast)
                new_primitives = generate_all_primitives(new_state)

                # Submit ML inference (async)
                new_state_id = id(new_state)
                PENDING_ML[new_state_id] = submit_ml_async(new_state)

                # Add new entry to WORK_QUEUE
                WORK_QUEUE.add(WorkEntry(
                    state: new_state,
                    state_id: new_state_id,
                    ml_candidates: [],           # Empty until ML arrives
                    ml_idx: 0,
                    fallback_candidates: new_primitives,
                    fallback_idx: 0,
                    parent_chain: new_chain,
                    path_ml_score: new_score
                ))

        # Re-sort queue (entry's priority may have changed)
        WORK_QUEUE.re_sort()

        CONTINUE  # Go back to STEP 1


    ┌─────────────────────────────────────────────────────────────────┐
    │ STEP 3: CHECK IF DONE                                           │
    └─────────────────────────────────────────────────────────────────┘

    If WORK_QUEUE is empty AND PENDING_ML is empty:
        BREAK

    # Edge case: WORK_QUEUE empty but PENDING_ML has futures
    # This is rare (only if all entries exhausted before ML returned)
    If WORK_QUEUE is empty AND PENDING_ML is not empty:
        # Wait briefly for any ML to complete
        Sleep(10ms)
        CONTINUE

RETURN SOLUTIONS
```

---

## Why This Works for N Pushes

The algorithm doesn't use "depth levels" - it tracks state and chain length:

```
parent_chain = list of pushes that led to current state
current_depth = len(parent_chain) + 1

For MAX_CHAIN_DEPTH = 5:

    Push 1: parent_chain = []           → depth 1 ✓
    Push 2: parent_chain = [p1]         → depth 2 ✓
    Push 3: parent_chain = [p1, p2]     → depth 3 ✓
    Push 4: parent_chain = [p1, p2, p3] → depth 4 ✓
    Push 5: parent_chain = [p1...p4]    → depth 5 ✓
    Push 6: parent_chain = [p1...p5]    → depth 6 ✗ (exceeds limit)
```

Every valid push creates a new frontier, which gets ML inference. The chain grows naturally regardless of the maximum depth.

---

## Example Trace

### Setup

```
MAX_CHAIN_DEPTH = 3
Robot is blocked by Box1
Goal: Open path to region_C
```

### Execution

```
T=0ms   INITIALIZATION
        Generate all 600 primitives (sync, ~1ms)
        Submit ML for initial_state
        PENDING_ML: {initial_id: future}
        WORK_QUEUE: [{initial, ml=[], fallback=[E0..E599], score=0}]

        CPU starts fallback immediately (no waiting!):

T=50    PROCESS WORK_QUEUE (initial has no ML yet, do fallback)
        Try E0 (fallback, score=0)
        → Valid but not solved, creates state_X
        → Submit ML for state_X, add to WORK_QUEUE
        PENDING_ML: {initial_id, state_X_id}
        WORK_QUEUE: [{initial, fb_idx=1}, {state_X, ml=[], fb=[E0..E599], score=0}]

T=100   PROCESS WORK_QUEUE (pick by priority - both have no ML, equal score)
        Try E1 on initial (fallback)
        → Collision, skip

T=150   PROCESS WORK_QUEUE
        Try E2 on initial (fallback)
        → Valid, creates state_Y
        → Submit ML for state_Y, add to WORK_QUEUE
        PENDING_ML: {initial_id, state_X_id, state_Y_id}

        ...CPU keeps doing fallback, ~30 pushes done before ML returns...

T=1500  ★ ML READY for initial!
        ML suggests: E47(8 votes), E23(4 votes)

        Update initial's entry in WORK_QUEUE:
        - Set ml_candidates = [E47, E23]
        - Re-sort queue (initial now has ML work → higher priority!)

        PENDING_ML: {state_X_id, state_Y_id}  ← initial removed
        WORK_QUEUE: [{initial, ml=[E47,E23], score=0},  ← NOW HAS ML, PRIORITY!
                     {state_X, ml=[], score=0},
                     {state_Y, ml=[], score=0}]

T=1510  PROCESS WORK_QUEUE (initial has ML work → process first!)
        Try E47 (ML, 8 votes) on initial
        → Valid but not solved, creates state_A
        → Submit ML for state_A, add to WORK_QUEUE with score=8

        WORK_QUEUE: [{initial, ml=[E23], score=0},      ← still has ML
                     {state_A, ml=[], score=8},         ← high path score!
                     {state_X, ml=[], score=0},
                     {state_Y, ml=[], score=0}]

T=1520  PROCESS WORK_QUEUE (initial still has ML → process)
        Try E23 (ML, 4 votes) on initial
        → Valid but not solved, creates state_B
        → Submit ML, add to WORK_QUEUE with score=4

        WORK_QUEUE: [{initial, ml=[], score=0},         ← ML exhausted
                     {state_A, ml=[], score=8},
                     {state_B, ml=[], score=4},
                     ...]

T=1530  PROCESS WORK_QUEUE
        initial has no ML work, but has fallback
        Others have no ML either
        Pick state_A (highest path_score=8), but it has no ML yet
        → Do fallback on state_A... OR continue fallback on initial
        (Priority: has_ml > path_score, so all equal, pick any)

T=3000  ★ ML READY for state_A! (path_score=8)
        ML suggests: E15(6 votes), E30(2 votes)

        Update state_A's entry:
        - Set ml_candidates = [E15, E30]
        - Re-sort queue (state_A now has ML → HIGHEST PRIORITY!)

        WORK_QUEUE: [{state_A, ml=[E15,E30], score=8},  ← HAS ML + HIGH SCORE!
                     {initial, ml=[], score=0},
                     ...]

T=3010  PROCESS WORK_QUEUE (state_A has ML → process first!)
        Try E15 (ML, 6 votes) on state_A
        → Valid but not solved, creates state_D
        → Submit ML, add to WORK_QUEUE with score=8+6=14

        WORK_QUEUE: [{state_A, ml=[E30], score=8},
                     {state_D, ml=[], score=14},        ← highest path score!
                     ...]

T=4500  ★ ML READY for state_D! (path_score=14)
        ML suggests: E8(5 votes)

        Update state_D, re-sort:
        WORK_QUEUE: [{state_D, ml=[E8], score=14},      ← HAS ML + HIGHEST SCORE!
                     {state_A, ml=[E30], score=8},
                     ...]

T=4510  PROCESS WORK_QUEUE (state_D has ML + highest score → process!)
        Try E8 (ML, 5 votes) on state_D
        → SOLVED! Path to region_C is now open!

        SOLUTION FOUND:
          Chain: [E47(8), E15(6), E8(5)]
          Path type: ML → ML → ML (fully ML-guided)
          Total score: 14 + 5 = 19
          Pushes: 3
```

### Timeline Visualization

```
Time ────────────────────────────────────────────────────────────────────────────────►

GPU:    [░░░░░░░░░ ML initial ░░░░░░░░░]  [░░ ML state_A ░░]  [░░ ML state_D ░░]
                                       ↓           ↓                ↓
                                    ready       ready            ready

CPU:    [E0][E1][E2]...[E20]    [E47][E23]  [E15]    ...    [E8] ✓ SOLVED!
        └─ fallback while ─┘    └─ ML! ─┘   └ML─┘           └ML─┘
           waiting for ML        (priority)

        ↓     ↓                    ↓          ↓               ↓
     state_X state_Y           state_A    state_D         SOLUTION
     (score 0)(score 0)        (score 8)  (score 14)      (score 19)

What happened:
  T=0-1500ms:   CPU did ~30 fallback pushes while waiting for ML
                Created state_X, state_Y, etc. (all with score=0)
                GPU was running ML inference

  T=1500ms:     ML ready! CPU switches to ML work immediately
                Tries E47 (8 votes), E23 (4 votes)
                Creates state_A (score=8), state_B (score=4)

  T=3000ms:     ML ready for state_A (highest score!)
                Tries E15 (6 votes) → state_D (score=14)

  T=4500ms:     ML ready for state_D (highest score!)
                Tries E8 (5 votes) → SOLVED!

Key: ML-derived states (score > 0) always processed before fallback states (score = 0)
```

---

## Implementation Plan

### Files to Create/Modify

```
1. python/namo/planners/opening/ml_driven_search.py  (NEW)
   - WorkEntry dataclass
   - SearchSolution dataclass
   - MLDrivenAsyncSearch class with main loop
   - PENDING_ML dict and WORK_QUEUE priority queue

2. python/namo/planners/opening/region_opening.py
   - Add integration point for new search
   - New goal_sampler option: "ml_driven_async"

3. python/namo/data_collection/region_opening_collection.yaml
   - Add config options for ml_driven_async mode
```

### Implementation Steps

```
STEP 1: Data Structures
   - Define WorkEntry dataclass with priority() and get_next_candidate()
   - Define SearchSolution dataclass
   - Implement WorkQueue using Lazy Heap pattern (heapq + versioning)
   - Key: O(log n) for add/pop/update_priority, O(1) for get/remove

STEP 2: ML Submission (0.5 days)
   - Reuse MLPrimitiveAsyncStrategy for async ML inference
   - Create _submit_ml_async() wrapper
   - Create _harvest_ml_results() to poll PENDING_ML futures

STEP 3: Main Loop (1.5 days)
   - Implement main loop with STEP 1 (harvest) and STEP 2 (process)
   - Handle ML arrival: update entry, re-sort queue
   - Implement solution recording and chain building
   - Handle depth limits and edge cases

STEP 4: Integration (0.5 days)
   - Add "ml_driven_async" option to goal_sampler
   - Wire up to RegionOpeningPlanner
   - Add config parameters

STEP 5: Testing (1 day)
   - Unit tests for WorkEntry priority and candidate selection
   - Integration test with MAX_CHAIN_DEPTH = 2, 3, 4
   - Verify ML solutions found before fallback
   - Benchmark vs sync hybrid
```

### Configuration

```yaml
# region_opening_collection.yaml

goal_sampler: ml_driven_async    # Enable ML-driven async search

# ML settings
ml_goal_model: /path/to/model
ml_samples: 32
ml_device: cuda

# Search settings
region_max_chain_depth: 3        # Max pushes (works for 2, 3, 4, 5, ...)
max_solutions_per_neighbor: 1    # Stop after finding N solutions

# Tolerance for ML-to-primitive alignment
ml_match_position_tolerance: 0.2
ml_match_angle_tolerance: 0.2
```

---

## Comparison with Other Approaches

| Aspect | Sequential | Sync Hybrid | Async (Current) | ML-Driven Async |
|--------|-----------|-------------|-----------------|-----------------|
| Initial wait | Yes (ML) | Yes (ML) | Yes (ML) | **No (fallback starts T=0)** |
| ML wait | Blocks | Blocks | Within-frontier | **Never blocks** |
| CPU usage | Low | Medium | Medium | **High (always busy)** |
| GPU usage | Sparse | Sparse | Better | **Near-continuous** |
| Solution order | Depth-first | ML then fallback | Mixed | **ML paths first** |
| N-push support | Yes | Yes | Yes | Yes |
| Fallback timing | Per depth | Per depth | Per frontier | **Only when no ML ready** |

---

## Summary

The ML-Driven Async algorithm ensures:

1. **Zero idle time guarantee** - every frontier added to WORK_QUEUE with fallback ready, CPU always has work
2. **ML results always get priority** - entries with ML candidates processed before fallback-only entries
3. **No blocking on ML** - check for ML results after every single push
4. **Works for any N pushes** - depth is just chain length, no special handling
5. **Best solutions first** - ML→ML→ML paths found before any fallback paths
6. **Complete coverage** - fallback ensures we find solutions even if ML is wrong
7. **GPU fully utilized** - ML inference submitted immediately for every new frontier
8. **Simple implementation** - 2 buckets (PENDING_ML + WORK_QUEUE) instead of 3
9. **Scalable data structures** - Lazy Heap provides O(log n) operations for n > 2000 entries

### The Key Insight

```
Every new frontier:
  1. Submit ML request → track future in PENDING_ML
  2. Add entry to WORK_QUEUE with all primitives as fallback

Priority in WORK_QUEUE:
  - has_ml_work() = True  → process first (ML candidates available)
  - has_ml_work() = False → process later (fallback only)
  - Within same ML status, sort by path_ml_score (higher = better)

When ML arrives:
  - Find entry in WORK_QUEUE
  - Set ml_candidates = ML results
  - Re-sort queue (entry now has higher priority!)

This means:
  - CPU is NEVER waiting for GPU (always has fallback work)
  - GPU is NEVER waiting for CPU (ML submitted immediately)
  - Single source of truth (each state has ONE entry in WORK_QUEUE)
```
