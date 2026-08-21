"""Configuration for external executor bridge.

Contains skill15 defaults and naming invariants derived from:
- config/namo_config_complete_skill15_car_1x.yaml
- src/config/config_manager.cpp
- src/planning/namo_push_controller.cpp
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Tuple

from namo.runtime_profile import require_canonical_runtime_config


# MJCF Naming Invariants (from plan document)
ROBOT_BODY_NAME = "robot"
ROBOT_GEOM_NAME = "robot"
GOAL_SITE_NAME = "goal"
MOVABLE_SUFFIX = "_movable"
WALL_PREFIX = "wall"
STATIC_KEYWORDS = ("wall", "static")


# car 1x d5 defaults from config/namo_config_complete_skill15_car_1x.yaml
SKILL15_DEFAULTS = {
    # Edge/contact mapping
    "points_per_face": 15,           # 4 faces * 15 = 60 total edge_idx values
    "total_edge_points": 60,
    "robot_contact_offset": 0.02,    # offset = robot_radius + 0.02
    
    # Push execution timing
    "control_steps_per_push": 250,   # micro-steps per push_step
    "force_scaling": 1.0,            # control force multiplier
    
    # Collision/termination
    
    # Controller stuck detection
    "stuck_check_stride": 30,         # steps between checks
    "controller_stuck_threshold": 3,  # checks before abort
    "controller_min_position_change": 0.001,  # meters
    "controller_min_angle_change": 0.05,      # radians
    
    # Navigation
    "grid_resolution": 0.01,          # meters per cell (from wavefront_planner config)
    "robot_inflation": 0.05,          # inflation for occupancy grid
    "nav_goal_tolerance": 0.05,       # position tolerance for navigation
    
    # Pure pursuit
    "lookahead_distance": 0.2,        # meters
    "max_linear_velocity": 0.5,       # m/s
    "max_angular_velocity": 1.0,      # rad/s
    
    # Approach to contact
    "approach_distance": 0.1,         # back off from contact by this much
}


@dataclass
class ExecutorConfig:
    """Configuration for external executor bridge.
    
    Attributes:
        executor_xml: Path to MuJoCo XML for the executor environment
        planner_template_xml: Path to template XML for planner (frozen aug9 template)
        config_yaml: Path to NAMO config YAML (skill15)
        tmp_planner_dir: Directory to write temporary planner XMLs
        name_mapping: Optional mapping {executor_name -> planner_name} for movables
        
        # From skill15 defaults
        points_per_face: Edge points per object face
        control_steps_per_push: Micro-steps per push_step
        force_scaling: Force multiplier
        stuck_check_stride: Steps between stuck checks
        controller_stuck_threshold: Stuck checks before abort
        controller_min_position_change: Min position delta for stuck check
        controller_min_angle_change: Min angle delta for stuck check
        grid_resolution: Occupancy grid resolution
        robot_inflation: Robot inflation for grid
        nav_goal_tolerance: Navigation goal tolerance
        lookahead_distance: Pure pursuit lookahead
        approach_distance: Distance to back off from contact point
    """
    
    executor_xml: Path
    planner_template_xml: Path
    config_yaml: Path
    tmp_planner_dir: Path = field(default_factory=lambda: Path("/tmp/namo_planner"))
    name_mapping: Optional[Dict[str, str]] = None
    
    # skill15 defaults
    points_per_face: int = SKILL15_DEFAULTS["points_per_face"]
    control_steps_per_push: int = SKILL15_DEFAULTS["control_steps_per_push"]
    force_scaling: float = SKILL15_DEFAULTS["force_scaling"]
    stuck_check_stride: int = SKILL15_DEFAULTS["stuck_check_stride"]
    controller_stuck_threshold: int = SKILL15_DEFAULTS["controller_stuck_threshold"]
    controller_min_position_change: float = SKILL15_DEFAULTS["controller_min_position_change"]
    controller_min_angle_change: float = SKILL15_DEFAULTS["controller_min_angle_change"]
    grid_resolution: float = SKILL15_DEFAULTS["grid_resolution"]
    robot_inflation: float = SKILL15_DEFAULTS["robot_inflation"]
    nav_goal_tolerance: float = SKILL15_DEFAULTS["nav_goal_tolerance"]
    lookahead_distance: float = SKILL15_DEFAULTS["lookahead_distance"]
    approach_distance: float = SKILL15_DEFAULTS["approach_distance"]
    
    @property
    def total_edge_points(self) -> int:
        """Total edge points (4 faces * points_per_face)."""
        return 4 * self.points_per_face
    
    @property
    def robot_contact_offset(self) -> float:
        """Offset from object edge for contact point (robot_radius + 0.02)."""
        return SKILL15_DEFAULTS["robot_contact_offset"]
    
    def __post_init__(self):
        """Ensure paths are Path objects and create tmp dir."""
        self.executor_xml = Path(self.executor_xml)
        self.planner_template_xml = Path(self.planner_template_xml)
        self.config_yaml = Path(self.config_yaml)
        require_canonical_runtime_config(self.config_yaml)
        self.tmp_planner_dir = Path(self.tmp_planner_dir)
        self.tmp_planner_dir.mkdir(parents=True, exist_ok=True)


# Default paths for project
DEFAULT_PLANNER_TEMPLATE = Path(__file__).parent.parent.parent.parent.parent / "data" / "planner_templates" / "planner_template.xml"
DEFAULT_CONFIG_YAML = (
    Path(__file__).parent.parent.parent.parent.parent
    / "config"
    / "namo_config_complete_skill15_car_1x.yaml"
)
