"""Modular IDFS Planning System for NAMO.

This package provides clean, modular implementations of iterative deepening DFS
algorithms for Navigation Among Movable Obstacles (NAMO) planning.

Available Algorithms:
- Standard IDFS: Restart-based iterative deepening (traditional approach)
- Tree-IDFS: Tree-maintained iterative deepening (optimized, no re-exploration)

Main modules:
- base_planner: Abstract planner interface and factory
- standard_idfs: Standard restart-based IDFS implementation  
- tree_idfs: Tree-maintained IDFS implementation
- modular_parallel_collection: Algorithm-agnostic parallel data collector
- test_tree_idfs: Testing and benchmarking framework
"""

from .base_planner import (
    BasePlanner,
    PlannerConfig, 
    PlannerResult,
    PlannerFactory
)

from .standard_idfs import (
    StandardIterativeDeepeningDFS,
    plan_with_idfs
)

from .tree_idfs import (
    TreeIterativeDeepeningDFS,
    plan_with_tree_idfs
)

__all__ = [
    # Base classes
    'BasePlanner',
    'PlannerConfig',
    'PlannerResult', 
    'PlannerFactory',
    
    # Algorithm implementations
    'StandardIterativeDeepeningDFS',
    'TreeIterativeDeepeningDFS',
    
    # Convenience functions
    'plan_with_idfs',
    'plan_with_tree_idfs'
]

# Available algorithm names for factory
AVAILABLE_ALGORITHMS = ['idfs', 'standard_idfs', 'tree_idfs']