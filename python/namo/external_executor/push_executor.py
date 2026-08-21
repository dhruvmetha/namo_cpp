"""Non-teleporting push execution.

This module executes pushes with actual physics simulation rather than
teleporting the robot to contact points. It implements the push primitive
semantics from NAMO's internal controller:

- push_steps = depth + 1
- Each push_step runs control_steps_per_push=250 micro-steps
- Each micro-step recomputes push_dir from current object pose
- Control is scaled by force_scaling=1.0
- Object contact recorded, never terminating
- Stuck ends a push; the pose is kept if the object moved, dropped if not

Key differences from internal NAMO controller:
- Robot physically navigates to pre-contact position (no teleport)
- Push control applied through simulation (not direct force application)
- Continuous feedback during push execution
"""

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Deque, List, Optional, Tuple

from collections import deque

from .config import ExecutorConfig, SKILL15_DEFAULTS
from .executor import MuJoCoExecutor, SE2Pose, ObjectInfo
from .edge_mapping import EdgeContactMapper, EdgeContact, compute_push_direction_from_poses
from .navigation import WavefrontNavigator, NavigationResult
from .planner_oracle import ChainLink


# Mirrors kMinUsefulPushDisplacementM / kMinUsefulPushYawRad in
# namo_push_controller.cpp. A push that stopped on stuck detection keeps its
# result once the object has travelled this far or turned this much; below both
# it produced nothing and the push failed. Two executors, one rule.
MIN_USEFUL_PUSH_DISPLACEMENT_M = 0.01
MIN_USEFUL_PUSH_YAW_RAD = math.radians(5.0)

# Ticks to let physics settle once control stops, matching the C++ controller's
# post-push settle so a preserved pose is a settled one.
SETTLE_STEPS = 100


class PushTermination(Enum):
    """Reason for push termination."""
    COMPLETED = "completed"
    STUCK = "stuck"
    # Stuck fired, but the object had already moved enough to be worth keeping.
    STOPPED_EARLY = "stopped_early"
    MAX_STEPS = "max_steps"
    NAV_FAILED = "nav_failed"


@dataclass
class PushStepResult:
    """Result of a single push step."""
    success: bool
    termination: PushTermination
    steps_taken: int
    object_moved: float  # Distance object moved
    robot_moved: float   # Distance robot moved
    # Walls and movables the pushed object touched. Telemetry, not failure.
    contacts: List[str] = field(default_factory=list)
    # The object moved and then jammed. The pose is real and the caller should
    # replan rather than run the remaining push steps.
    stopped_early: bool = False


@dataclass
class PushResult:
    """Result of executing a full push (depth+1 push_steps)."""
    success: bool
    termination: PushTermination
    push_steps_completed: int
    total_micro_steps: int
    object_total_moved: float
    object_final_pose: SE2Pose
    robot_final_pose: SE2Pose
    collided_with: Optional[str] = None
    contacts: List[str] = field(default_factory=list)
    stopped_early: bool = False


@dataclass
class ChainExecutionResult:
    """Result of executing a full goal_chain."""
    success: bool
    links_completed: int
    total_links: int
    push_results: List[PushResult]
    final_robot_pose: SE2Pose
    error_message: Optional[str] = None


class PushExecutor:
    """Executes pushes with non-teleporting physics simulation.
    
    Usage:
        push_exec = PushExecutor(executor, navigator, config)
        result = push_exec.execute_chain(goal_chain)
    """
    
    def __init__(
        self,
        executor: MuJoCoExecutor,
        navigator: WavefrontNavigator,
        config: ExecutorConfig
    ):
        """Initialize push executor.
        
        Args:
            executor: MuJoCo executor for simulation
            navigator: Navigator for getting to contact
            config: Configuration with push parameters
        """
        self.executor = executor
        self.navigator = navigator
        self.config = config
        
        # Edge mapper for contact computation
        self.edge_mapper = EdgeContactMapper(config)
        
        # Push parameters from skill15
        self.control_steps_per_push = config.control_steps_per_push
        self.force_scaling = config.force_scaling
        
        # Stuck detection parameters
        self.stuck_check_stride = config.stuck_check_stride
        self.stuck_threshold = config.controller_stuck_threshold
        self.min_position_change = config.controller_min_position_change
        self.min_angle_change = config.controller_min_angle_change
        
        # Approach distance for pre-contact
        self.approach_distance = config.approach_distance
    
    def execute_chain(self, chain: List[ChainLink]) -> ChainExecutionResult:
        """Execute a full goal_chain (multiple push links).
        
        Args:
            chain: List of ChainLink (object_id, edge_idx, depth)
            
        Returns:
            ChainExecutionResult with success status and push results
        """
        if not chain:
            return ChainExecutionResult(
                success=True,
                links_completed=0,
                total_links=0,
                push_results=[],
                final_robot_pose=self.executor.get_robot_pose()
            )
        
        push_results = []
        
        for i, link in enumerate(chain):
            result = self.execute_push(link)
            push_results.append(result)
            
            if not result.success:
                return ChainExecutionResult(
                    success=False,
                    links_completed=i,
                    total_links=len(chain),
                    push_results=push_results,
                    final_robot_pose=self.executor.get_robot_pose(),
                    error_message=f"Link {i} failed: {result.termination.value}"
                )
        
        return ChainExecutionResult(
            success=True,
            links_completed=len(chain),
            total_links=len(chain),
            push_results=push_results,
            final_robot_pose=self.executor.get_robot_pose()
        )
    
    def execute_push(self, link: ChainLink) -> PushResult:
        """Execute a single push link.
        
        Workflow:
        1. Get object info and pose
        2. Compute contact point from edge_idx
        3. Navigate to pre-contact position
        4. Execute depth+1 push_steps
        
        Args:
            link: ChainLink with object_id, edge_idx, depth
            
        Returns:
            PushResult with success status and details
        """
        object_id = link.object_id
        edge_idx = link.edge_idx
        push_steps = link.depth + 1  # depth=0 -> 1 push, depth=1 -> 2 pushes, etc.
        
        # Get object info
        snapshot = self.executor.get_snapshot()
        obj_info = snapshot.object_info.get(object_id)
        if obj_info is None:
            return PushResult(
                success=False,
                termination=PushTermination.NAV_FAILED,
                push_steps_completed=0,
                total_micro_steps=0,
                object_total_moved=0.0,
                object_final_pose=SE2Pose(0, 0, 0),
                robot_final_pose=self.executor.get_robot_pose(),
                collided_with=f"Object {object_id} not found"
            )
        
        # Get object pose
        obj_pose = snapshot.movable_poses.get(object_id)
        if obj_pose is None:
            return PushResult(
                success=False,
                termination=PushTermination.NAV_FAILED,
                push_steps_completed=0,
                total_micro_steps=0,
                object_total_moved=0.0,
                object_final_pose=SE2Pose(0, 0, 0),
                robot_final_pose=self.executor.get_robot_pose(),
                collided_with=f"Object {object_id} pose not found"
            )
        
        # Compute contact point
        contact = self.edge_mapper.compute_contact(
            obj_info, obj_pose, edge_idx, snapshot.robot_radius
        )
        
        # Compute pre-contact position
        precontact_xy = self.edge_mapper.get_precontact_position(
            contact, self.approach_distance
        )
        
        # Navigate to pre-contact
        nav_result = self.navigator.navigate_to(
            precontact_xy[0], precontact_xy[1],
            exclude_object=object_id
        )
        
        if not nav_result.success:
            return PushResult(
                success=False,
                termination=PushTermination.NAV_FAILED,
                push_steps_completed=0,
                total_micro_steps=0,
                object_total_moved=0.0,
                object_final_pose=obj_pose,
                robot_final_pose=self.executor.get_robot_pose(),
                collided_with=f"Navigation failed: {nav_result.reason}"
            )
        
        # Execute push_steps
        initial_obj_pose = self.executor.get_movable_pose(object_id)
        total_micro_steps = 0
        object_total_moved = 0.0
        
        contacts: List[str] = []

        for step in range(push_steps):
            step_result = self._execute_push_step(object_id)
            total_micro_steps += step_result.steps_taken
            object_total_moved += step_result.object_moved
            for name in step_result.contacts:
                if name not in contacts:
                    contacts.append(name)

            if not step_result.success:
                final_obj_pose = self.executor.get_movable_pose(object_id)
                return PushResult(
                    success=False,
                    termination=step_result.termination,
                    push_steps_completed=step,
                    total_micro_steps=total_micro_steps,
                    object_total_moved=object_total_moved,
                    object_final_pose=final_obj_pose,
                    robot_final_pose=self.executor.get_robot_pose(),
                    contacts=contacts,
                )

            if step_result.stopped_early:
                # The object moved and jammed. Every later push step was chosen
                # against a world that no longer exists, so stop here and let
                # the caller replan from the pose it actually reached.
                return PushResult(
                    success=True,
                    termination=PushTermination.STOPPED_EARLY,
                    push_steps_completed=step + 1,
                    total_micro_steps=total_micro_steps,
                    object_total_moved=object_total_moved,
                    object_final_pose=self.executor.get_movable_pose(object_id),
                    robot_final_pose=self.executor.get_robot_pose(),
                    contacts=contacts,
                    stopped_early=True,
                )
        
        final_obj_pose = self.executor.get_movable_pose(object_id)
        
        return PushResult(
            success=True,
            termination=PushTermination.COMPLETED,
            push_steps_completed=push_steps,
            total_micro_steps=total_micro_steps,
            object_total_moved=object_total_moved,
            object_final_pose=final_obj_pose,
            robot_final_pose=self.executor.get_robot_pose(),
            contacts=contacts,
        )
    
    def _settle(self) -> None:
        """Stop driving and let physics come to rest."""
        self.executor.set_robot_control(0.0, 0.0)
        for _ in range(SETTLE_STEPS):
            self.executor.step()

    def _execute_push_step(self, object_id: str) -> PushStepResult:
        """Execute a single push_step (control_steps_per_push micro-steps).
        
        Args:
            object_id: ID of object being pushed
            
        Returns:
            PushStepResult with success status and details
        """
        # Stuck detection state
        stuck_check_count = 0
        prev_obj_pose: Optional[SE2Pose] = None
        
        initial_robot_pose = self.executor.get_robot_pose()
        initial_obj_pose = self.executor.get_movable_pose(object_id)
        
        contacts: List[str] = []

        for step in range(self.control_steps_per_push):
            # Get current poses
            robot_pose = self.executor.get_robot_pose()
            obj_pose = self.executor.get_movable_pose(object_id)
            
            # Compute push direction (from robot toward object)
            push_dir = compute_push_direction_from_poses(robot_pose, obj_pose)
            
            # Apply control (velocity toward object)
            vx = push_dir[0] * self.force_scaling
            vy = push_dir[1] * self.force_scaling
            self.executor.set_robot_control(vx, vy)
            
            # Step simulation
            self.executor.step()
            
            # Stuck check (every stuck_check_stride steps)
            if (step + 1) % self.stuck_check_stride == 0:
                new_obj_pose = self.executor.get_movable_pose(object_id)
                # Record contact before deciding anything, so a push that stops
                # because the object wedged still names what it wedged against.
                for name in self.executor.contacts_with_object(object_id):
                    if name not in contacts:
                        contacts.append(name)
                
                if prev_obj_pose is not None:
                    pos_change = self._pose_distance(prev_obj_pose, new_obj_pose)
                    angle_change = abs(self._angle_diff(prev_obj_pose.theta, new_obj_pose.theta))
                    
                    if pos_change < self.min_position_change and angle_change < self.min_angle_change:
                        stuck_check_count += 1
                        
                        if stuck_check_count >= self.stuck_threshold:
                            moved = self._pose_distance(initial_obj_pose, new_obj_pose)
                            turned = abs(
                                self._angle_diff(initial_obj_pose.theta, new_obj_pose.theta)
                            )
                            useful = (
                                moved >= MIN_USEFUL_PUSH_DISPLACEMENT_M
                                or turned >= MIN_USEFUL_PUSH_YAW_RAD
                            )
                            if useful:
                                # Keep what the object did. Stop driving, let it
                                # settle, and report it so the caller replans
                                # instead of running the rest of the push.
                                self._settle()
                                settled_pose = self.executor.get_movable_pose(object_id)
                                return PushStepResult(
                                    success=True,
                                    termination=PushTermination.STOPPED_EARLY,
                                    steps_taken=step + 1,
                                    object_moved=self._pose_distance(
                                        initial_obj_pose, settled_pose
                                    ),
                                    robot_moved=self._pose_distance(
                                        initial_robot_pose, self.executor.get_robot_pose()
                                    ),
                                    contacts=contacts,
                                    stopped_early=True,
                                )
                            return PushStepResult(
                                success=False,
                                termination=PushTermination.STUCK,
                                steps_taken=step + 1,
                                object_moved=moved,
                                robot_moved=self._pose_distance(initial_robot_pose, robot_pose),
                                contacts=contacts,
                            )
                    else:
                        stuck_check_count = 0
                
                prev_obj_pose = new_obj_pose
        
        # Completed all micro-steps
        final_obj_pose = self.executor.get_movable_pose(object_id)
        final_robot_pose = self.executor.get_robot_pose()
        
        return PushStepResult(
            success=True,
            termination=PushTermination.COMPLETED,
            steps_taken=self.control_steps_per_push,
            object_moved=self._pose_distance(initial_obj_pose, final_obj_pose),
            robot_moved=self._pose_distance(initial_robot_pose, final_robot_pose),
            contacts=contacts,
        )
    
    def _pose_distance(self, p1: SE2Pose, p2: SE2Pose) -> float:
        """Euclidean distance between two poses."""
        return math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)
    
    def _angle_diff(self, a1: float, a2: float) -> float:
        """Smallest angle difference (handles wraparound)."""
        diff = a2 - a1
        while diff > math.pi:
            diff -= 2 * math.pi
        while diff < -math.pi:
            diff += 2 * math.pi
        return diff
