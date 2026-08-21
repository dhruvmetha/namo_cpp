"""MuJoCo Python Executor with snapshot export.

This module provides the executor-side simulation and state extraction.
It uses the `mujoco` Python package for physics simulation.

Key responsibilities:
- Load and step the MuJoCo environment
- Export snapshot: robot_pose, movable_poses, robot geometry
- Apply controls (for navigation and pushing)
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import math
import numpy as np

try:
    import mujoco
except ImportError:
    raise ImportError("mujoco package required. Install with: pip install mujoco")

from .config import (
    ExecutorConfig,
    ROBOT_BODY_NAME,
    ROBOT_GEOM_NAME,
    GOAL_SITE_NAME,
    MOVABLE_SUFFIX,
    WALL_PREFIX,
)


@dataclass
class SE2Pose:
    """2D pose (x, y, theta)."""
    x: float
    y: float
    theta: float
    
    def to_tuple(self) -> Tuple[float, float, float]:
        return (self.x, self.y, self.theta)
    
    def distance_to(self, other: "SE2Pose") -> float:
        """Euclidean distance to another pose (ignores theta)."""
        return math.sqrt((self.x - other.x)**2 + (self.y - other.y)**2)


@dataclass
class ObjectInfo:
    """Geometric information for an object."""
    name: str
    half_extent_x: float  # half-width
    half_extent_y: float  # half-depth
    is_movable: bool


@dataclass
class ExecutorSnapshot:
    """Snapshot of executor state for planner synchronization.
    
    This is exported to build tmp_planner.xml.
    """
    robot_pose: SE2Pose
    robot_radius: float
    movable_poses: Dict[str, SE2Pose]  # name -> pose
    object_info: Dict[str, ObjectInfo]  # name -> geometry info
    goal_pose: Optional[Tuple[float, float, float]]  # (x, y, z) from goal site
    world_bounds: Tuple[float, float, float, float]  # (xmin, xmax, ymin, ymax)


class MuJoCoExecutor:
    """MuJoCo Python executor for NAMO environments.
    
    Provides:
    - Physics simulation via mujoco.mj_step
    - State extraction for planner synchronization
    - Control application for navigation and pushing
    """
    
    def __init__(self, config: ExecutorConfig):
        """Initialize executor with MuJoCo model.
        
        Args:
            config: Executor configuration with paths and parameters
        """
        self.config = config
        
        # Load MuJoCo model and data
        self.model = mujoco.MjModel.from_xml_path(str(config.executor_xml))
        self.data = mujoco.MjData(self.model)
        
        # Initialize simulation
        mujoco.mj_forward(self.model, self.data)
        
        # Cache object info (geometry doesn't change)
        self._object_info: Dict[str, ObjectInfo] = {}
        self._movable_body_ids: Dict[str, int] = {}
        self._robot_body_id: Optional[int] = None
        self._robot_geom_id: Optional[int] = None
        self._robot_radius: float = 0.15  # default
        self._goal_site_id: Optional[int] = None
        
        self._cache_object_info()
    
    def _cache_object_info(self):
        """Cache object geometry and IDs at initialization."""
        # Find robot body and geom
        for i in range(self.model.nbody):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, i)
            if name == ROBOT_BODY_NAME:
                self._robot_body_id = i
                break
        
        for i in range(self.model.ngeom):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, i)
            if name == ROBOT_GEOM_NAME:
                self._robot_geom_id = i
                # Get robot radius from geom size
                self._robot_radius = float(self.model.geom_size[i, 0])
                break
        
        # Find goal site
        for i in range(self.model.nsite):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_SITE, i)
            if name == GOAL_SITE_NAME:
                self._goal_site_id = i
                break
        
        # Find movable objects
        for i in range(self.model.nbody):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, i)
            if name is None:
                continue
            
            is_movable = MOVABLE_SUFFIX in name
            
            if is_movable:
                self._movable_body_ids[name] = i
                
                # Get geometry info from geom with same name
                geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, name)
                if geom_id >= 0:
                    # For box geoms, size is (half_x, half_y, half_z)
                    size = self.model.geom_size[geom_id]
                    self._object_info[name] = ObjectInfo(
                        name=name,
                        half_extent_x=float(size[0]),
                        half_extent_y=float(size[1]),
                        is_movable=True
                    )
    
    @property
    def robot_radius(self) -> float:
        """Robot radius (for contact offset calculation)."""
        return self._robot_radius
    
    @property
    def timestep(self) -> float:
        """Simulation timestep."""
        return float(self.model.opt.timestep)
    
    def get_robot_pose(self) -> SE2Pose:
        """Get current robot pose (x, y, theta)."""
        if self._robot_geom_id is None:
            raise RuntimeError("Robot geom not found in model")
        
        # Get position from geom (which includes the offset in pos attribute)
        pos = self.data.geom_xpos[self._robot_geom_id]
        
        # For a sphere robot on slide joints, theta is always 0
        # (no rotation joint)
        return SE2Pose(x=float(pos[0]), y=float(pos[1]), theta=0.0)
    
    def get_movable_pose(self, name: str) -> SE2Pose:
        """Get pose of a movable object.
        
        Args:
            name: Object name (e.g., "obstacle_1_movable")
            
        Returns:
            SE2Pose with x, y, theta (theta from quaternion yaw)
        """
        if name not in self._movable_body_ids:
            raise KeyError(f"Movable object '{name}' not found")
        
        body_id = self._movable_body_ids[name]
        
        # Get position from body xpos
        pos = self.data.xpos[body_id]
        
        # Get orientation from body xquat and extract yaw
        quat = self.data.xquat[body_id]  # (w, x, y, z) format in MuJoCo
        theta = self._quat_to_yaw(quat)
        
        return SE2Pose(x=float(pos[0]), y=float(pos[1]), theta=theta)
    
    def get_all_movable_poses(self) -> Dict[str, SE2Pose]:
        """Get poses of all movable objects."""
        return {name: self.get_movable_pose(name) for name in self._movable_body_ids}
    
    def get_goal_position(self) -> Optional[Tuple[float, float, float]]:
        """Get goal site position (x, y, z)."""
        if self._goal_site_id is None:
            return None
        pos = self.data.site_xpos[self._goal_site_id]
        return (float(pos[0]), float(pos[1]), float(pos[2]))
    
    def get_world_bounds(self) -> Tuple[float, float, float, float]:
        """Get world bounds from wall positions.
        
        Returns:
            (xmin, xmax, ymin, ymax)
        """
        # Find boundary walls (wall_1 through wall_4 are typically the outer walls)
        xmin, xmax = -8.0, 8.0  # defaults from template
        ymin, ymax = -8.0, 8.0
        
        for i in range(self.model.ngeom):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, i)
            if name and name.startswith(WALL_PREFIX):
                pos = self.model.geom_pos[i]
                size = self.model.geom_size[i]
                
                # Check if this is a boundary wall
                if abs(pos[0]) > 5:  # vertical boundary
                    if pos[0] < 0:
                        xmin = max(xmin, pos[0] + size[0])
                    else:
                        xmax = min(xmax, pos[0] - size[0])
                if abs(pos[1]) > 5:  # horizontal boundary
                    if pos[1] < 0:
                        ymin = max(ymin, pos[1] + size[1])
                    else:
                        ymax = min(ymax, pos[1] - size[1])
        
        return (xmin, xmax, ymin, ymax)
    
    def get_snapshot(self) -> ExecutorSnapshot:
        """Export current state snapshot for planner synchronization."""
        return ExecutorSnapshot(
            robot_pose=self.get_robot_pose(),
            robot_radius=self._robot_radius,
            movable_poses=self.get_all_movable_poses(),
            object_info=dict(self._object_info),
            goal_pose=self.get_goal_position(),
            world_bounds=self.get_world_bounds()
        )
    
    def step(self, ctrl: Optional[np.ndarray] = None):
        """Step simulation by one timestep.
        
        Args:
            ctrl: Control input array (actuator values). If None, uses current.
        """
        if ctrl is not None:
            np.copyto(self.data.ctrl, ctrl[:len(self.data.ctrl)])
        mujoco.mj_step(self.model, self.data)
    
    def step_n(self, n: int, ctrl: Optional[np.ndarray] = None):
        """Step simulation by n timesteps.
        
        Args:
            n: Number of steps
            ctrl: Control input (held constant across steps)
        """
        for _ in range(n):
            self.step(ctrl)
    
    def set_robot_control(self, vx: float, vy: float):
        """Set robot velocity control.
        
        Args:
            vx: X velocity (actuator_x)
            vy: Y velocity (actuator_y)
        """
        # Assuming actuator order: actuator_x, actuator_y
        ctrl = np.array([vx, vy], dtype=np.float64)
        np.copyto(self.data.ctrl[:2], ctrl)
    
    def get_qpos(self) -> np.ndarray:
        """Get full qpos array (for state save/restore)."""
        return self.data.qpos.copy()
    
    def get_qvel(self) -> np.ndarray:
        """Get full qvel array."""
        return self.data.qvel.copy()
    
    def set_state(self, qpos: np.ndarray, qvel: Optional[np.ndarray] = None):
        """Set simulation state.
        
        Args:
            qpos: Position state
            qvel: Velocity state (zeros if None)
        """
        np.copyto(self.data.qpos, qpos)
        if qvel is not None:
            np.copyto(self.data.qvel, qvel)
        else:
            self.data.qvel[:] = 0
        mujoco.mj_forward(self.model, self.data)
    
    def reset(self):
        """Reset simulation to initial state."""
        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)
    
    def check_collision(self, body1_name: str, body2_name: str) -> bool:
        """Check if two bodies are in collision.
        
        Args:
            body1_name: First body name
            body2_name: Second body name
            
        Returns:
            True if bodies have an active contact
        """
        body1_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body1_name)
        body2_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body2_name)
        
        if body1_id < 0 or body2_id < 0:
            return False
        
        # Check contacts
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            geom1_body = self.model.geom_bodyid[contact.geom1]
            geom2_body = self.model.geom_bodyid[contact.geom2]
            
            if (geom1_body == body1_id and geom2_body == body2_id) or \
               (geom1_body == body2_id and geom2_body == body1_id):
                return True
        
        return False
    
    def check_robot_collision_with_movables(self, exclude: Optional[str] = None) -> Optional[str]:
        """Check if robot collides with any movable object.
        
        Args:
            exclude: Object name to exclude from check (e.g., the object being pushed)
            
        Returns:
            Name of colliding object, or None if no collision
        """
        for name in self._movable_body_ids:
            if name == exclude:
                continue
            if self.check_collision(ROBOT_BODY_NAME, name):
                return name
        return None
    
    def contacts_with_object(self, object_name: str) -> List[str]:
        """Names of movables and walls the object is currently touching.

        Reported, never fatal: object-object and object-wall contact is a normal
        part of a push. Mirrors what the C++ controller accumulates into
        wall_collision_during_push_ / movable_collisions_during_push_.
        """
        touching = []
        for name in self._movable_body_ids:
            if name == object_name:
                continue
            if self.check_collision(object_name, name):
                touching.append(name)
        if self.check_collision(object_name, "walls"):
            touching.append("walls")
        return touching

    @staticmethod
    def _quat_to_yaw(quat: np.ndarray) -> float:
        """Convert quaternion to yaw angle.
        
        MuJoCo uses (w, x, y, z) quaternion format.
        
        Args:
            quat: Quaternion array [w, x, y, z]
            
        Returns:
            Yaw angle in radians
        """
        w, x, y, z = quat
        # Yaw from quaternion: atan2(2*(w*z + x*y), 1 - 2*(y^2 + z^2))
        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        return math.atan2(siny_cosp, cosy_cosp)
