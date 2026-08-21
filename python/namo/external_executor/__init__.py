"""External Executor Bridge for NAMO.

This package provides a MuJoCo-Python executor that:
1. Runs the environment with real physics (no teleportation)
2. Generates temporary planner XMLs from executor state
3. Calls the NAMO planner oracle for push goal_chains
4. Executes pushes with navigation + continuous control

MJCF Naming Invariants (must hold for planner compatibility):
- Robot: body named "robot", geom named "robot" (sphere type, radius in size[0])
- Goal: site named "goal" with pos="x y z"
- Movables: bodies with "_movable" suffix, e.g. "obstacle_1_movable"
- Statics/Walls: geoms with "wall" prefix, e.g. "wall_1", "wall_2"

Usage:
    from namo.external_executor import ExternalExecutorBridge

    bridge = ExternalExecutorBridge(
        executor_xml="/path/to/env.xml",
        planner_template_xml="/path/to/planner_template.xml",
        config_yaml="/path/to/namo_config_complete_skill15_car_1x.yaml"
    )
    
    # Run until goal reached or budget exhausted
    success = bridge.run_to_goal(goal=(x, y, theta), max_replans=20)
"""

from .config import ExecutorConfig, SKILL15_DEFAULTS
from .executor import MuJoCoExecutor, SE2Pose, ExecutorSnapshot
from .xml_builder import TmpPlannerXmlBuilder
from .edge_mapping import EdgeContactMapper, EdgeContact
from .navigation import WavefrontNavigator, NavigationResult, OccupancyGrid
from .push_executor import PushExecutor, PushResult, ChainExecutionResult
from .planner_oracle import PlannerOracle, OracleResult, ChainLink
from .bridge import ExternalExecutorBridge, BridgeRunResult, create_bridge_from_env
from .verify import run_verification, VerificationResults

__all__ = [
    # Config
    "ExecutorConfig",
    "SKILL15_DEFAULTS",
    # Executor
    "MuJoCoExecutor",
    "SE2Pose",
    "ExecutorSnapshot",
    # XML Builder
    "TmpPlannerXmlBuilder",
    # Edge Mapping
    "EdgeContactMapper",
    "EdgeContact",
    # Navigation
    "WavefrontNavigator",
    "NavigationResult",
    "OccupancyGrid",
    # Push Execution
    "PushExecutor",
    "PushResult",
    "ChainExecutionResult",
    # Planner Oracle
    "PlannerOracle",
    "OracleResult",
    "ChainLink",
    # Bridge
    "ExternalExecutorBridge",
    "BridgeRunResult",
    "create_bridge_from_env",
    # Verification
    "run_verification",
    "VerificationResults",
]
