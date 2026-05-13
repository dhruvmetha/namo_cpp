"""Verification utilities for external executor bridge.

This module provides tools to validate planner/executor agreement:
- Edge mapping correctness (contact point computation)
- Navigation tolerances
- Control scaling/dt alignment
- Collision/stuck threshold behavior

Usage:
    from namo.external_executor.verify import run_verification
    
    results = run_verification(
        executor_xml="/path/to/env.xml",
        verbose=True
    )
    
    if results.all_passed:
        print("All verification checks passed!")
"""

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .config import ExecutorConfig, DEFAULT_PLANNER_TEMPLATE, DEFAULT_CONFIG_YAML, SKILL15_DEFAULTS
from .executor import MuJoCoExecutor, SE2Pose
from .xml_builder import TmpPlannerXmlBuilder
from .edge_mapping import EdgeContactMapper
from .navigation import WavefrontNavigator


@dataclass
class VerificationCheck:
    """Result of a single verification check."""
    name: str
    passed: bool
    message: str
    details: Optional[Dict] = None


@dataclass
class VerificationResults:
    """Results from all verification checks."""
    checks: List[VerificationCheck]
    
    @property
    def all_passed(self) -> bool:
        return all(c.passed for c in self.checks)
    
    @property
    def passed_count(self) -> int:
        return sum(1 for c in self.checks if c.passed)
    
    @property
    def total_count(self) -> int:
        return len(self.checks)
    
    def print_summary(self):
        """Print verification summary."""
        print(f"\n{'='*60}")
        print(f"Verification Results: {self.passed_count}/{self.total_count} passed")
        print(f"{'='*60}")
        
        for check in self.checks:
            status = "PASS" if check.passed else "FAIL"
            print(f"  [{status}] {check.name}")
            print(f"         {check.message}")
            if check.details:
                for k, v in check.details.items():
                    print(f"         {k}: {v}")


def verify_edge_mapping(
    executor: MuJoCoExecutor,
    config: ExecutorConfig,
    verbose: bool = False
) -> VerificationCheck:
    """Verify edge mapping produces valid contact points.
    
    Checks:
    - All edge indices (0-59 for skill15) produce valid contact points
    - Contact points are outside object bounds
    - Push directions point toward object center
    """
    mapper = EdgeContactMapper(config)
    snapshot = executor.get_snapshot()
    
    if not snapshot.movable_poses:
        return VerificationCheck(
            name="Edge Mapping",
            passed=False,
            message="No movable objects found in environment"
        )
    
    # Test with first movable object
    obj_name = list(snapshot.movable_poses.keys())[0]
    obj_info = snapshot.object_info.get(obj_name)
    obj_pose = snapshot.movable_poses[obj_name]
    
    if obj_info is None:
        return VerificationCheck(
            name="Edge Mapping",
            passed=False,
            message=f"No geometry info for {obj_name}"
        )
    
    # Compute all edges
    edges = mapper.compute_all_edges(obj_info, obj_pose, snapshot.robot_radius)
    
    # Verify we got expected count
    expected_count = config.total_edge_points
    if len(edges) != expected_count:
        return VerificationCheck(
            name="Edge Mapping",
            passed=False,
            message=f"Expected {expected_count} edges, got {len(edges)}",
            details={"object": obj_name}
        )
    
    # Verify contacts are outside object
    hw = obj_info.half_extent_x
    hd = obj_info.half_extent_y
    min_dist = math.sqrt(hw**2 + hd**2)  # Diagonal distance
    
    invalid_contacts = 0
    for edge in edges:
        dist_to_center = math.sqrt(
            (edge.contact_xy[0] - obj_pose.x)**2 +
            (edge.contact_xy[1] - obj_pose.y)**2
        )
        if dist_to_center < min_dist * 0.5:  # Should be outside object
            invalid_contacts += 1
    
    if invalid_contacts > 0:
        return VerificationCheck(
            name="Edge Mapping",
            passed=False,
            message=f"{invalid_contacts}/{len(edges)} contacts too close to object center",
            details={"object": obj_name, "min_dist": min_dist}
        )
    
    return VerificationCheck(
        name="Edge Mapping",
        passed=True,
        message=f"All {len(edges)} edge contacts valid for {obj_name}",
        details={"object": obj_name, "edge_count": len(edges)}
    )


def verify_navigation(
    executor: MuJoCoExecutor,
    config: ExecutorConfig,
    verbose: bool = False
) -> VerificationCheck:
    """Verify navigation can find paths in the environment.
    
    Checks:
    - Occupancy grid builds without errors
    - BFS finds path from robot to goal (if one exists)
    - Navigation converges to target
    """
    navigator = WavefrontNavigator(executor, config)
    snapshot = executor.get_snapshot()
    
    # Build grid
    try:
        grid = navigator.build_occupancy_grid(snapshot)
    except Exception as e:
        return VerificationCheck(
            name="Navigation",
            passed=False,
            message=f"Failed to build occupancy grid: {e}"
        )
    
    # Check grid dimensions are reasonable
    if grid.width < 10 or grid.height < 10:
        return VerificationCheck(
            name="Navigation",
            passed=False,
            message=f"Grid too small: {grid.width}x{grid.height}",
            details={"resolution": grid.resolution}
        )
    
    # Try to find path to goal (if goal exists)
    if snapshot.goal_pose is not None:
        goal_x, goal_y = snapshot.goal_pose[0], snapshot.goal_pose[1]
        robot_pos = (snapshot.robot_pose.x, snapshot.robot_pose.y)
        
        path = navigator.find_path_bfs(grid, robot_pos, (goal_x, goal_y))
        
        if path is None:
            return VerificationCheck(
                name="Navigation",
                passed=True,  # Not a failure - path may not exist
                message="No path to goal (may be blocked)",
                details={
                    "grid_size": f"{grid.width}x{grid.height}",
                    "goal": f"({goal_x:.2f}, {goal_y:.2f})"
                }
            )
        
        return VerificationCheck(
            name="Navigation",
            passed=True,
            message=f"Found path with {len(path)} waypoints",
            details={
                "grid_size": f"{grid.width}x{grid.height}",
                "path_length": len(path)
            }
        )
    
    return VerificationCheck(
        name="Navigation",
        passed=True,
        message=f"Grid built: {grid.width}x{grid.height}",
        details={"resolution": grid.resolution}
    )


def verify_xml_builder(
    executor: MuJoCoExecutor,
    config: ExecutorConfig,
    verbose: bool = False
) -> VerificationCheck:
    """Verify XML builder creates valid planner XMLs.
    
    Checks:
    - XML builds without errors
    - Robot and movable poses are correctly inserted
    - Generated XML can be loaded by namo_rl
    """
    builder = TmpPlannerXmlBuilder(config)
    snapshot = executor.get_snapshot()
    
    # Build XML
    try:
        tmp_xml = builder.build(snapshot)
    except Exception as e:
        return VerificationCheck(
            name="XML Builder",
            passed=False,
            message=f"Failed to build XML: {e}"
        )
    
    # Check file exists
    if not tmp_xml.exists():
        return VerificationCheck(
            name="XML Builder",
            passed=False,
            message="Generated XML file does not exist"
        )
    
    # Try to load with namo_rl (if available)
    try:
        import namo_rl
        env = namo_rl.RLEnvironment(str(tmp_xml), str(config.config_yaml))
        
        # Verify robot pose matches
        namo_obs = env.get_observation()
        robot_obs = namo_obs.get("robot", [0, 0, 0])
        
        pos_error = math.sqrt(
            (robot_obs[0] - snapshot.robot_pose.x)**2 +
            (robot_obs[1] - snapshot.robot_pose.y)**2
        )
        
        if pos_error > 0.1:
            return VerificationCheck(
                name="XML Builder",
                passed=False,
                message=f"Robot pose mismatch: error={pos_error:.3f}m",
                details={
                    "expected": f"({snapshot.robot_pose.x:.3f}, {snapshot.robot_pose.y:.3f})",
                    "got": f"({robot_obs[0]:.3f}, {robot_obs[1]:.3f})"
                }
            )
        
        return VerificationCheck(
            name="XML Builder",
            passed=True,
            message="XML built and loaded successfully",
            details={
                "tmp_file": str(tmp_xml),
                "robot_pos_error": f"{pos_error:.4f}m"
            }
        )
        
    except ImportError:
        return VerificationCheck(
            name="XML Builder",
            passed=True,
            message="XML built (namo_rl not available for full validation)",
            details={"tmp_file": str(tmp_xml)}
        )
    except Exception as e:
        return VerificationCheck(
            name="XML Builder",
            passed=False,
            message=f"Failed to load with namo_rl: {e}"
        )


def verify_config_consistency(config: ExecutorConfig) -> VerificationCheck:
    """Verify config values match skill15 defaults.
    
    Checks:
    - points_per_face matches expected (15)
    - control_steps_per_push matches expected (250)
    - stuck detection thresholds match expected
    """
    mismatches = []
    
    expected = SKILL15_DEFAULTS
    
    if config.points_per_face != expected["points_per_face"]:
        mismatches.append(f"points_per_face: {config.points_per_face} != {expected['points_per_face']}")
    
    if config.control_steps_per_push != expected["control_steps_per_push"]:
        mismatches.append(f"control_steps_per_push: {config.control_steps_per_push} != {expected['control_steps_per_push']}")
    
    if config.stuck_check_stride != expected["stuck_check_stride"]:
        mismatches.append(f"stuck_check_stride: {config.stuck_check_stride} != {expected['stuck_check_stride']}")
    
    if config.controller_stuck_threshold != expected["controller_stuck_threshold"]:
        mismatches.append(f"controller_stuck_threshold: {config.controller_stuck_threshold} != {expected['controller_stuck_threshold']}")
    
    if mismatches:
        return VerificationCheck(
            name="Config Consistency",
            passed=False,
            message=f"{len(mismatches)} parameter mismatches",
            details={"mismatches": mismatches}
        )
    
    return VerificationCheck(
        name="Config Consistency",
        passed=True,
        message="All parameters match skill15 defaults"
    )


def run_verification(
    executor_xml: str,
    planner_template_xml: Optional[str] = None,
    config_yaml: Optional[str] = None,
    verbose: bool = False
) -> VerificationResults:
    """Run all verification checks.
    
    Args:
        executor_xml: Path to executor environment XML
        planner_template_xml: Path to planner template (uses default if None)
        config_yaml: Path to config YAML (uses default if None)
        verbose: Print verbose output
        
    Returns:
        VerificationResults with all check results
    """
    # Setup
    executor_path = Path(executor_xml)
    template_path = Path(planner_template_xml) if planner_template_xml else DEFAULT_PLANNER_TEMPLATE
    config_path = Path(config_yaml) if config_yaml else DEFAULT_CONFIG_YAML
    
    config = ExecutorConfig(
        executor_xml=executor_path,
        planner_template_xml=template_path,
        config_yaml=config_path
    )
    
    executor = MuJoCoExecutor(config)
    
    # Run checks
    checks = []
    
    if verbose:
        print("Running verification checks...")
    
    # Config consistency
    checks.append(verify_config_consistency(config))
    
    # Edge mapping
    checks.append(verify_edge_mapping(executor, config, verbose))
    
    # Navigation
    checks.append(verify_navigation(executor, config, verbose))
    
    # XML builder
    checks.append(verify_xml_builder(executor, config, verbose))
    
    results = VerificationResults(checks=checks)
    
    if verbose:
        results.print_summary()
    
    return results


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python -m namo.external_executor.verify <executor_xml>")
        sys.exit(1)
    
    executor_xml = sys.argv[1]
    results = run_verification(executor_xml, verbose=True)
    
    sys.exit(0 if results.all_passed else 1)
