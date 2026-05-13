"""Temporary planner XML builder.

Generates tmp_planner.xml from planner_template.xml by injecting
executor state (robot pose, movable poses, optionally goal pose).

Key considerations:
- Angle units: aug9 templates typically use degrees in euler attributes
  (no <compiler angle="radian"> specified means degrees is default)
- Stateful fields: robot pos, movable pos/euler, goal site pos
- Non-stateful: physics parameters, geom sizes, materials (keep unchanged)
"""

import math
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, Optional, Tuple

from .config import (
    ExecutorConfig,
    ROBOT_BODY_NAME,
    ROBOT_GEOM_NAME,
    GOAL_SITE_NAME,
    MOVABLE_SUFFIX,
)
from .executor import ExecutorSnapshot, SE2Pose


class TmpPlannerXmlBuilder:
    """Builds temporary planner XMLs from template + executor state.
    
    Usage:
        builder = TmpPlannerXmlBuilder(config)
        tmp_xml_path = builder.build(snapshot)
        # tmp_xml_path can now be passed to NAMO planner
    """
    
    def __init__(self, config: ExecutorConfig):
        """Initialize builder.
        
        Args:
            config: Executor configuration with template path and tmp directory
        """
        self.config = config
        self.template_path = config.planner_template_xml
        self.tmp_dir = config.tmp_planner_dir
        self.name_mapping = config.name_mapping or {}
        
        # Parse template once to determine angle convention
        self._uses_radians = self._detect_angle_convention()
        
        # Counter for unique tmp file names
        self._build_counter = 0
    
    def _detect_angle_convention(self) -> bool:
        """Detect whether template uses radians or degrees.
        
        Returns:
            True if radians, False if degrees
        """
        tree = ET.parse(self.template_path)
        root = tree.getroot()
        
        # Look for <compiler angle="radian">
        compiler = root.find("compiler")
        if compiler is not None:
            angle_attr = compiler.get("angle", "degree")
            return angle_attr.lower() == "radian"
        
        # Default: degrees (most aug9 templates)
        return False
    
    def _rad_to_template_angle(self, rad: float) -> float:
        """Convert radians to template angle convention."""
        if self._uses_radians:
            return rad
        return math.degrees(rad)
    
    def _map_name(self, executor_name: str) -> str:
        """Map executor object name to planner template name."""
        return self.name_mapping.get(executor_name, executor_name)
    
    def build(
        self,
        snapshot: ExecutorSnapshot,
        update_goal: bool = False,
        goal_override: Optional[Tuple[float, float, float]] = None
    ) -> Path:
        """Build a temporary planner XML from template + snapshot.
        
        Args:
            snapshot: Current executor state
            update_goal: Whether to update goal site position
            goal_override: Override goal position (x, y, z)
            
        Returns:
            Path to generated tmp_planner.xml
        """
        # Parse template
        tree = ET.parse(self.template_path)
        root = tree.getroot()
        
        # Update robot pose
        self._update_robot_pose(root, snapshot.robot_pose)
        
        # Update movable poses
        self._update_movable_poses(root, snapshot.movable_poses)
        
        # Optionally update goal
        if update_goal and goal_override is not None:
            self._update_goal_site(root, goal_override)
        
        # Write to tmp file
        self._build_counter += 1
        tmp_path = self.tmp_dir / f"tmp_planner_{self._build_counter}.xml"
        
        tree.write(tmp_path, encoding="unicode", xml_declaration=True)
        
        return tmp_path
    
    def _update_robot_pose(self, root: ET.Element, pose: SE2Pose):
        """Update robot geom position in XML.
        
        The robot geom has a pos attribute that includes z-height.
        We only update x, y, keeping z unchanged.
        """
        worldbody = root.find("worldbody")
        if worldbody is None:
            return
        
        robot_body = worldbody.find(f".//body[@name='{ROBOT_BODY_NAME}']")
        if robot_body is None:
            return
        
        robot_geom = robot_body.find(f".//geom[@name='{ROBOT_GEOM_NAME}']")
        if robot_geom is None:
            return
        
        # Parse existing pos to get z
        pos_str = robot_geom.get("pos", "0 0 0.15")
        pos_parts = pos_str.split()
        z = float(pos_parts[2]) if len(pos_parts) >= 3 else 0.15
        
        # Update with new x, y
        robot_geom.set("pos", f"{pose.x} {pose.y} {z}")
    
    def _update_movable_poses(self, root: ET.Element, poses: Dict[str, SE2Pose]):
        """Update all movable object poses in XML."""
        worldbody = root.find("worldbody")
        if worldbody is None:
            return
        
        for executor_name, pose in poses.items():
            planner_name = self._map_name(executor_name)
            
            # Find body and geom
            body = worldbody.find(f".//body[@name='{planner_name}']")
            if body is None:
                continue
            
            geom = body.find(f".//geom[@name='{planner_name}']")
            if geom is None:
                continue
            
            # Parse existing pos to get z
            pos_str = geom.get("pos", "0 0 0.3")
            pos_parts = pos_str.split()
            z = float(pos_parts[2]) if len(pos_parts) >= 3 else 0.3
            
            # Update pos
            geom.set("pos", f"{pose.x} {pose.y} {z}")
            
            # Update euler (roll, pitch, yaw) - only yaw changes
            euler_str = geom.get("euler", "0 0 0")
            euler_parts = euler_str.split()
            roll = float(euler_parts[0]) if len(euler_parts) >= 1 else 0.0
            pitch = float(euler_parts[1]) if len(euler_parts) >= 2 else 0.0
            
            yaw_in_template_units = self._rad_to_template_angle(pose.theta)
            geom.set("euler", f"{roll} {pitch} {yaw_in_template_units}")
    
    def _update_goal_site(self, root: ET.Element, goal: Tuple[float, float, float]):
        """Update goal site position in XML."""
        worldbody = root.find("worldbody")
        if worldbody is None:
            return
        
        goal_site = worldbody.find(f".//site[@name='{GOAL_SITE_NAME}']")
        if goal_site is None:
            return
        
        goal_site.set("pos", f"{goal[0]} {goal[1]} {goal[2]}")
    
    def cleanup(self):
        """Remove all generated tmp files."""
        for f in self.tmp_dir.glob("tmp_planner_*.xml"):
            f.unlink()


def yaw_to_quat(yaw: float) -> Tuple[float, float, float, float]:
    """Convert yaw angle to quaternion (w, x, y, z).
    
    Args:
        yaw: Yaw angle in radians
        
    Returns:
        Quaternion as (w, x, y, z)
    """
    half_yaw = yaw / 2.0
    w = math.cos(half_yaw)
    x = 0.0
    y = 0.0
    z = math.sin(half_yaw)
    return (w, x, y, z)


def quat_to_euler_degrees(quat: Tuple[float, float, float, float]) -> Tuple[float, float, float]:
    """Convert quaternion (w, x, y, z) to euler angles (roll, pitch, yaw) in degrees.
    
    This matches MuJoCo's expected euler format when angle="degree" (default).
    """
    w, x, y, z = quat
    
    # Roll (x-axis rotation)
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)
    
    # Pitch (y-axis rotation)
    sinp = 2.0 * (w * y - z * x)
    if abs(sinp) >= 1:
        pitch = math.copysign(math.pi / 2, sinp)
    else:
        pitch = math.asin(sinp)
    
    # Yaw (z-axis rotation)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    
    return (math.degrees(roll), math.degrees(pitch), math.degrees(yaw))
