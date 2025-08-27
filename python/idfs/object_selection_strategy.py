"""Object selection strategies for IDFS planners.

This module provides different strategies for selecting which objects to explore
during IDFS search, allowing for different heuristics while keeping the core
search algorithm unchanged.
"""

import math
from abc import ABC, abstractmethod
from typing import List, Dict, Any
import namo_rl


class ObjectSelectionStrategy(ABC):
    """Abstract base class for object selection strategies."""
    
    @abstractmethod
    def select_objects(self, 
                      reachable_objects: List[str], 
                      state: namo_rl.RLState,
                      env: namo_rl.RLEnvironment) -> List[str]:
        """Select and order objects for exploration.
        
        Args:
            reachable_objects: List of object IDs that are currently reachable
            state: Current environment state
            env: Environment instance for querying object positions
            
        Returns:
            Ordered list of object IDs to explore (can be same as input or reordered)
        """
        pass
    
    @property
    @abstractmethod
    def strategy_name(self) -> str:
        """Return human-readable name of this strategy."""
        pass


class NoHeuristicStrategy(ObjectSelectionStrategy):
    """Default strategy that maintains original object order without modification.
    
    This preserves the original IDFS behavior where objects are explored
    in the order returned by get_reachable_objects().
    """
    
    def select_objects(self, 
                      reachable_objects: List[str], 
                      state: namo_rl.RLState,
                      env: namo_rl.RLEnvironment) -> List[str]:
        """Return objects in original order without modification."""
        return reachable_objects.copy()
    
    @property
    def strategy_name(self) -> str:
        return "No Heuristic"


class NearestFirstStrategy(ObjectSelectionStrategy):
    """Strategy that orders objects by distance from robot, nearest first.
    
    This implements the "Nearest Standard IDFS" heuristic where objects
    closest to the robot are explored first at each depth level.
    """
    
    def select_objects(self, 
                      reachable_objects: List[str], 
                      state: namo_rl.RLState,
                      env: namo_rl.RLEnvironment) -> List[str]:
        """Return objects ordered by distance from robot, nearest first."""
        if not reachable_objects:
            return []
        
        # Save current environment state to restore later
        original_state = env.get_full_state()
        
        try:
            # Set state to get current positions
            env.set_full_state(state)
            obs = env.get_observation()
        finally:
            # Always restore original state to avoid corrupting search
            env.set_full_state(original_state)
        
        # Get robot position
        robot_pose = obs.get('robot_pose')
        if robot_pose is None:
            # Fallback to original order if robot pose unavailable
            return reachable_objects.copy()
        
        robot_x, robot_y = robot_pose[0], robot_pose[1]
        
        # Calculate distances and sort objects
        object_distances = []
        for obj_id in reachable_objects:
            pose_key = f"{obj_id}_pose"
            if pose_key in obs:
                obj_x, obj_y = obs[pose_key][0], obs[pose_key][1]
                distance = math.sqrt((obj_x - robot_x)**2 + (obj_y - robot_y)**2)
                object_distances.append((distance, obj_id))
            else:
                # If object pose unavailable, put it at the end with large distance
                object_distances.append((float('inf'), obj_id))
        
        # Sort by distance (nearest first)
        object_distances.sort(key=lambda x: x[0])
        
        # Return sorted object IDs
        return [obj_id for _, obj_id in object_distances]
    
    @property
    def strategy_name(self) -> str:
        return "Nearest First"


class GoalProximityStrategy(ObjectSelectionStrategy):
    """Strategy that orders objects by distance from robot goal, closest first.
    
    This implements goal-directed object selection where objects closest to 
    the robot's target goal position are explored first. This can be more
    strategic than nearest-to-robot since it considers the ultimate objective.
    """
    
    def select_objects(self, 
                      reachable_objects: List[str], 
                      state: namo_rl.RLState,
                      env: namo_rl.RLEnvironment) -> List[str]:
        """Return objects ordered by distance from robot goal, closest first."""
        if not reachable_objects:
            return []
        
        # Save current environment state to restore later
        original_state = env.get_full_state()
        
        try:
            # Set state to get current positions and goal
            env.set_full_state(state)
            obs = env.get_observation()
            
            # Get robot goal position - try multiple ways to access it
            goal_pose = None
            
            # Method 1: Check if environment has _robot_goal attribute
            if hasattr(env, '_robot_goal'):
                goal_pose = getattr(env, '_robot_goal')
            
            # Method 2: Check if it's in observations (some environments expose it)
            elif 'robot_goal' in obs:
                goal_pose = obs['robot_goal']
            
            # Method 3: Try to get it via a method if available
            elif hasattr(env, 'get_robot_goal'):
                try:
                    goal_pose = env.get_robot_goal()
                except:
                    pass
            
            if goal_pose is None or len(goal_pose) < 2:
                # Fallback to original order if goal unavailable or invalid
                return reachable_objects.copy()
            
            goal_x, goal_y = goal_pose[0], goal_pose[1]
            
            # Calculate distances from objects to robot goal (while state is still set)
            object_distances = []
            for obj_id in reachable_objects:
                pose_key = f"{obj_id}_pose"
                if pose_key in obs:
                    obj_x, obj_y = obs[pose_key][0], obs[pose_key][1]
                    distance = math.sqrt((obj_x - goal_x)**2 + (obj_y - goal_y)**2)
                    object_distances.append((distance, obj_id))
                else:
                    # If object pose unavailable, put it at the end with infinite distance
                    object_distances.append((float('inf'), obj_id))
            
        finally:
            # Always restore original state to avoid corrupting search
            env.set_full_state(original_state)
        
        # Sort by distance from goal (closest first)
        object_distances.sort(key=lambda x: x[0])
        
        # Return sorted object IDs
        return [obj_id for _, obj_id in object_distances]
    
    @property
    def strategy_name(self) -> str:
        return "Goal Proximity"


class FarthestFirstStrategy(ObjectSelectionStrategy):
    """Strategy that orders objects by distance from robot, farthest first.
    
    This is the opposite of NearestFirstStrategy - objects farthest from
    the robot are explored first. Useful for comparison and testing.
    """
    
    def select_objects(self, 
                      reachable_objects: List[str], 
                      state: namo_rl.RLState,
                      env: namo_rl.RLEnvironment) -> List[str]:
        """Return objects ordered by distance from robot, farthest first."""
        if not reachable_objects:
            return []
        
        # Save current environment state to restore later
        original_state = env.get_full_state()
        
        try:
            # Set state to get current positions
            env.set_full_state(state)
            obs = env.get_observation()
        finally:
            # Always restore original state to avoid corrupting search
            env.set_full_state(original_state)
        
        # Get robot position
        robot_pose = obs.get('robot_pose')
        if robot_pose is None:
            # Fallback to original order if robot pose unavailable
            return reachable_objects.copy()
        
        robot_x, robot_y = robot_pose[0], robot_pose[1]
        
        # Calculate distances and sort objects
        object_distances = []
        for obj_id in reachable_objects:
            pose_key = f"{obj_id}_pose"
            if pose_key in obs:
                obj_x, obj_y = obs[pose_key][0], obs[pose_key][1]
                distance = math.sqrt((obj_x - robot_x)**2 + (obj_y - robot_y)**2)
                object_distances.append((distance, obj_id))
            else:
                # If object pose unavailable, put it at the end with infinite distance
                # (this way unavailable objects are handled consistently across strategies)
                object_distances.append((float('inf'), obj_id))
        
        # Sort by distance (farthest first)
        object_distances.sort(key=lambda x: x[0], reverse=True)
        
        # Return sorted object IDs
        return [obj_id for _, obj_id in object_distances]
    
    @property
    def strategy_name(self) -> str:
        return "Farthest First"