"""External Executor Bridge - Main integration class.

This is the top-level interface for the external executor workflow:
1. Step physics in MuJoCo Python executor
2. Export snapshot (robot + movable poses)
3. Generate tmp_planner.xml from template + snapshot
4. Call NAMO planner oracle to get push goal_chain
5. Execute goal_chain with navigation + non-teleport pushing
6. Repeat until goal reached or budget exhausted

Usage:
    from namo.external_executor import ExternalExecutorBridge
    
    bridge = ExternalExecutorBridge(
        executor_xml="/path/to/env.xml",
        planner_template_xml="/path/to/planner_template.xml",
        config_yaml="/path/to/namo_config_complete_skill15.yaml"
    )
    
    success = bridge.run_to_goal(goal=(x, y, theta), max_replans=20)
"""

import math
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

from .config import ExecutorConfig, DEFAULT_PLANNER_TEMPLATE, DEFAULT_CONFIG_YAML
from .executor import MuJoCoExecutor, SE2Pose, ExecutorSnapshot
from .xml_builder import TmpPlannerXmlBuilder
from .navigation import WavefrontNavigator, NavigationResult
from .edge_mapping import EdgeContactMapper
from .push_executor import PushExecutor, ChainExecutionResult
from .planner_oracle import PlannerOracle, OracleResult, check_goal_reachable


@dataclass
class BridgeRunResult:
    """Result of running the bridge to goal."""
    success: bool
    goal_reached: bool
    replans: int
    total_chains_executed: int
    total_pushes: int
    final_robot_pose: SE2Pose
    error_message: Optional[str] = None


class ExternalExecutorBridge:
    """Main integration class for external executor + NAMO planner.
    
    Workflow per replan cycle:
    1. Export executor snapshot
    2. Build tmp_planner.xml
    3. Query planner oracle for goal_chain
    4. Execute goal_chain (nav + push for each link)
    5. Check if goal reachable; if not, replan
    
    The bridge maintains the executor state across replans.
    """
    
    def __init__(
        self,
        executor_xml: str,
        planner_template_xml: Optional[str] = None,
        config_yaml: Optional[str] = None,
        verbose: bool = False
    ):
        """Initialize the external executor bridge.
        
        Args:
            executor_xml: Path to MuJoCo XML for executor environment
            planner_template_xml: Path to planner template XML (uses default if None)
            config_yaml: Path to NAMO config YAML (uses skill15 default if None)
            verbose: Enable verbose output
        """
        # Resolve paths
        executor_xml_path = Path(executor_xml)
        template_path = Path(planner_template_xml) if planner_template_xml else DEFAULT_PLANNER_TEMPLATE
        config_path = Path(config_yaml) if config_yaml else DEFAULT_CONFIG_YAML
        
        # Create config
        self.config = ExecutorConfig(
            executor_xml=executor_xml_path,
            planner_template_xml=template_path,
            config_yaml=config_path
        )
        
        self.verbose = verbose
        
        # Initialize components
        self.executor = MuJoCoExecutor(self.config)
        self.xml_builder = TmpPlannerXmlBuilder(self.config)
        self.navigator = WavefrontNavigator(self.executor, self.config)
        self.push_executor = PushExecutor(self.executor, self.navigator, self.config)
        self.planner_oracle = PlannerOracle(self.config.config_yaml, verbose=verbose)
        
        # Statistics
        self._replans = 0
        self._chains_executed = 0
        self._total_pushes = 0
    
    def run_to_goal(
        self,
        goal: Tuple[float, float, float],
        max_replans: int = 20
    ) -> BridgeRunResult:
        """Run the executor until goal is reached or budget exhausted.
        
        Args:
            goal: Target robot position (x, y, theta)
            max_replans: Maximum number of replan cycles
            
        Returns:
            BridgeRunResult with success status and statistics
        """
        self._replans = 0
        self._chains_executed = 0
        self._total_pushes = 0
        
        goal_x, goal_y, goal_theta = goal
        
        for replan_idx in range(max_replans):
            self._replans = replan_idx + 1
            
            if self.verbose:
                print(f"\n{'='*60}")
                print(f"Replan cycle {replan_idx + 1}/{max_replans}")
                print(f"{'='*60}")
            
            # Step 1: Export snapshot
            snapshot = self.executor.get_snapshot()
            
            if self.verbose:
                print(f"Robot at ({snapshot.robot_pose.x:.2f}, {snapshot.robot_pose.y:.2f})")
                print(f"Goal at ({goal_x:.2f}, {goal_y:.2f})")
            
            # Step 2: Build tmp_planner.xml
            tmp_xml = self.xml_builder.build(snapshot)
            
            # Step 3: Check if goal already reachable
            if check_goal_reachable(tmp_xml, self.config.config_yaml, goal):
                if self.verbose:
                    print("Goal is reachable! Navigating to goal...")
                
                # Navigate to goal
                nav_result = self.navigator.navigate_to(goal_x, goal_y)
                
                if nav_result.success:
                    return BridgeRunResult(
                        success=True,
                        goal_reached=True,
                        replans=self._replans,
                        total_chains_executed=self._chains_executed,
                        total_pushes=self._total_pushes,
                        final_robot_pose=self.executor.get_robot_pose()
                    )
            
            # Step 4: Query planner oracle
            oracle_result = self.planner_oracle.get_next_push_chain(tmp_xml, goal)
            
            if not oracle_result.success:
                if self.verbose:
                    print(f"Oracle failed: {oracle_result.error_message}")
                
                # No path found - try to continue anyway
                if oracle_result.error_message == "Goal already reachable":
                    # This shouldn't happen since we checked above, but handle it
                    nav_result = self.navigator.navigate_to(goal_x, goal_y)
                    if nav_result.success:
                        return BridgeRunResult(
                            success=True,
                            goal_reached=True,
                            replans=self._replans,
                            total_chains_executed=self._chains_executed,
                            total_pushes=self._total_pushes,
                            final_robot_pose=self.executor.get_robot_pose()
                        )
                
                # Continue to next replan cycle
                continue
            
            if not oracle_result.goal_chain:
                if self.verbose:
                    print("Empty goal chain returned (goal may be reachable)")
                continue
            
            if self.verbose:
                print(f"Got goal chain with {len(oracle_result.goal_chain)} links:")
                for i, link in enumerate(oracle_result.goal_chain):
                    print(f"  [{i}] {link.object_id} edge={link.edge_idx} depth={link.depth}")
            
            # Step 5: Execute goal_chain
            chain_result = self.push_executor.execute_chain(oracle_result.goal_chain)
            self._chains_executed += 1
            self._total_pushes += chain_result.links_completed
            
            if self.verbose:
                print(f"Chain execution: {chain_result.links_completed}/{chain_result.total_links} links")
                if not chain_result.success:
                    print(f"  Failed: {chain_result.error_message}")
            
            # Chain executed (success or partial), continue to next replan
        
        # Budget exhausted
        return BridgeRunResult(
            success=False,
            goal_reached=False,
            replans=self._replans,
            total_chains_executed=self._chains_executed,
            total_pushes=self._total_pushes,
            final_robot_pose=self.executor.get_robot_pose(),
            error_message="Max replans reached"
        )
    
    def step_physics(self, n: int = 1):
        """Manually step the physics simulation.
        
        Useful for letting the system settle after pushes.
        
        Args:
            n: Number of simulation steps
        """
        self.executor.step_n(n)
    
    def get_robot_pose(self) -> SE2Pose:
        """Get current robot pose."""
        return self.executor.get_robot_pose()
    
    def get_snapshot(self) -> ExecutorSnapshot:
        """Get current executor snapshot."""
        return self.executor.get_snapshot()
    
    def navigate_to(self, x: float, y: float) -> NavigationResult:
        """Navigate robot to a position.
        
        Args:
            x: Target X coordinate
            y: Target Y coordinate
            
        Returns:
            NavigationResult with success status
        """
        return self.navigator.navigate_to(x, y)
    
    def cleanup(self):
        """Cleanup temporary files."""
        self.xml_builder.cleanup()
    
    def reset(self):
        """Reset executor to initial state."""
        self.executor.reset()
        self._replans = 0
        self._chains_executed = 0
        self._total_pushes = 0


def create_bridge_from_env(
    env_xml: str,
    verbose: bool = False
) -> ExternalExecutorBridge:
    """Convenience function to create a bridge from just an environment XML.
    
    Uses default planner template and skill15 config.
    
    Args:
        env_xml: Path to environment XML
        verbose: Enable verbose output
        
    Returns:
        Configured ExternalExecutorBridge
    """
    return ExternalExecutorBridge(
        executor_xml=env_xml,
        planner_template_xml=None,  # Use default
        config_yaml=None,           # Use default
        verbose=verbose
    )
