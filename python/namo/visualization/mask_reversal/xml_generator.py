"""
XML Generator - Generate MuJoCo XML files from reconstructed scenes.

This module creates valid MuJoCo XML environment files from world-space scene data.
"""

from xml.etree.ElementTree import Element, SubElement, tostring
from xml.dom import minidom
from typing import List, Tuple
from pathlib import Path
import numpy as np

from .scene_reconstructor import ReconstructedScene, WorldObject


class XMLGenerator:
    """Generates MuJoCo XML from reconstructed scenes."""
    
    def __init__(self, wall_height: float = 0.3, robot_radius: float = 0.15):
        """Initialize generator.
        
        Args:
            wall_height: Height of walls and obstacles (default 0.3m)
            robot_radius: Robot radius (default 0.15m)
        """
        self.wall_height = wall_height
        self.robot_radius = robot_radius
        self.wall_thickness = 0.05
        self.robot_z_pos = 0.2
        self.goal_z_pos = 0.0
    
    def create_base_xml(self, model_name: str = "reconstructed_environment") -> Element:
        """Create base MuJoCo XML structure.
        
        Args:
            model_name: Name of the model
            
        Returns:
            Root XML element
        """
        root = Element("mujoco", model=model_name)
        
        # Options
        SubElement(root, "option", timestep="0.01", integrator="RK4", cone="elliptic")
        
        # Default settings
        default = SubElement(root, "default")
        SubElement(default, "geom", density="1")
        
        # Assets
        asset = SubElement(root, "asset")
        SubElement(asset, "texture",
                  builtin="gradient",
                  height="3072",
                  rgb1="0.3 0.5 0.7",
                  rgb2="0 0 0",
                  type="skybox",
                  width="512")
        SubElement(asset, "texture",
                  builtin="checker",
                  height="300",
                  mark="edge",
                  markrgb="0.8 0.8 0.8",
                  name="groundplane",
                  rgb1="0.2 0.3 0.4",
                  rgb2="0.1 0.2 0.3",
                  type="2d",
                  width="300")
        SubElement(asset, "material",
                  name="groundplane",
                  reflectance="0.2",
                  texrepeat="5 5",
                  texture="groundplane",
                  texuniform="true")
        SubElement(asset, "material",
                  name="robot",
                  rgba="0.0 1.0 0.0 1")
        
        return root
    
    def add_worldbody(self, root: Element, scene: ReconstructedScene) -> None:
        """Add worldbody with all scene objects.
        
        Args:
            root: Root XML element
            scene: Reconstructed scene
        """
        worldbody = SubElement(root, "worldbody")
        
        # Lighting
        SubElement(worldbody, "light",
                  dir="0 0 -1",
                  directional="true",
                  pos="0 0 1.5")
        
        # Ground plane
        SubElement(worldbody, "geom",
                  condim="4",
                  friction="0.5 0.005 0.001",
                  material="groundplane",
                  name="floor",
                  size="0 0 0.05",
                  type="plane")
        
        # Origin markers
        SubElement(worldbody, "site",
                  name="origin_x",
                  pos="0.1 0 0.0",
                  rgba="1 0 0 1",
                  size="0.05",
                  type="sphere")
        SubElement(worldbody, "site",
                  name="origin_y",
                  pos="0 0.1 0",
                  rgba="0 1 0 1",
                  size="0.05",
                  type="sphere")
        SubElement(worldbody, "site",
                  name="origin_z",
                  pos="0 0 0.1",
                  rgba="0 0 1 1",
                  size="0.05",
                  type="sphere")
        
        # Add static walls
        self.add_static_objects(worldbody, scene.static_objects)
        
        # Add movable objects
        self.add_movable_objects(worldbody, scene.movable_objects)
        
        # Add robot
        self.add_robot(worldbody, scene.robot_position)
        
        # Add goal
        self.add_goal(worldbody, scene.goal_position)
    
    def add_static_objects(self, worldbody: Element, static_objects: List[WorldObject]) -> None:
        """Add static walls and obstacles.
        
        Args:
            worldbody: Worldbody XML element
            static_objects: List of static objects
        """
        walls_body = SubElement(worldbody, "body", name="walls")
        
        for i, obj in enumerate(static_objects):
            # Convert theta to quaternion (rotation around z-axis)
            qw, qx, qy, qz = self.theta_to_quaternion(obj.theta)
            
            SubElement(walls_body, "geom",
                      name=f"wall_{i}",
                      condim="4",
                      pos=f"{obj.x:.6f} {obj.y:.6f} {self.wall_height}",
                      quat=f"{qw:.6f} {qx:.6f} {qy:.6f} {qz:.6f}",
                      rgba="0.8 0.8 0.8 1",
                      size=f"{obj.half_width:.6f} {obj.half_height:.6f} {self.wall_height}",
                      type="box")
    
    def add_movable_objects(self, worldbody: Element, movable_objects: List[WorldObject]) -> None:
        """Add movable obstacles.
        
        Args:
            worldbody: Worldbody XML element
            movable_objects: List of movable objects
        """
        for i, obj in enumerate(movable_objects):
            body = SubElement(worldbody, "body", name=f"obstacle_{i}_movable")
            
            # Add geom with position and euler angles (matching original XML format)
            # Convert theta (radians) to euler angles (degrees) around z-axis
            euler_z_deg = np.degrees(obj.theta)
            
            SubElement(body, "geom",
                      name=f"obstacle_{i}_movable",
                      condim="4",
                      pos=f"{obj.x:.6f} {obj.y:.6f} {self.wall_height}",
                      euler=f"0 0 {euler_z_deg:.6f}",
                      friction="0.0 0.005 0.001",
                      rgba="1 1 0 1",
                      size=f"{obj.half_width:.6f} {obj.half_height:.6f} {self.wall_height}",
                      type="box",
                      mass="0.1")
            
            # Add free joint (matching original XML format)
            SubElement(body, "joint", type="free")
    
    def add_robot(self, worldbody: Element, robot_position: Tuple[float, float, float]) -> None:
        """Add robot body.
        
        Args:
            worldbody: Worldbody XML element
            robot_position: Robot position (x, y, theta)
        """
        robot_x, robot_y, robot_theta = robot_position
        
        # Create robot body at origin (like original XML)
        robot_body = SubElement(worldbody, "body", name="robot")
        
        # Add slide joints for X and Y movement (like original XML)
        SubElement(robot_body, "joint",
                  name="joint_x",
                  type="slide",
                  pos="0 0 0",
                  axis="1 0 0")
        
        SubElement(robot_body, "joint",
                  name="joint_y",
                  type="slide",
                  pos="0 0 0",
                  axis="0 1 0")
        
        # Add geom with actual position (like original XML)
        # This places the robot at the correct location in the scene
        SubElement(robot_body, "geom",
                  name="robot",
                  type="sphere",
                  pos=f"{robot_x:.6f} {robot_y:.6f} {self.robot_z_pos}",
                  size=f"{self.robot_radius} {self.robot_radius} {self.robot_radius}",
                  mass="5.0",
                  friction="1.0 0.005 0.0001",
                  condim="4")
    
    def add_goal(self, worldbody: Element, goal_position: Tuple[float, float, float]) -> None:
        """Add goal site.
        
        Args:
            worldbody: Worldbody XML element
            goal_position: Goal position (x, y, theta)
        """
        goal_x, goal_y, goal_theta = goal_position
        
        SubElement(worldbody, "site",
                  name="goal",
                  pos=f"{goal_x:.6f} {goal_y:.6f} {self.goal_z_pos}",
                  rgba="1 0 0 0.3",
                  size="0.3",
                  type="cylinder")
    
    def theta_to_quaternion(self, theta: float) -> Tuple[float, float, float, float]:
        """Convert rotation angle to quaternion (rotation around z-axis).
        
        Args:
            theta: Rotation angle in radians
            
        Returns:
            (qw, qx, qy, qz) quaternion
        """
        import math
        half_theta = theta / 2
        qw = math.cos(half_theta)
        qx = 0.0
        qy = 0.0
        qz = math.sin(half_theta)
        return (qw, qx, qy, qz)
    
    def generate_xml(self, scene: ReconstructedScene, 
                    model_name: str = "reconstructed_environment") -> str:
        """Generate complete MuJoCo XML from reconstructed scene.
        
        Args:
            scene: Reconstructed scene
            model_name: Name for the model
            
        Returns:
            Pretty-printed XML string
        """
        # Create base structure
        root = self.create_base_xml(model_name)
        
        # Add worldbody with all objects
        self.add_worldbody(root, scene)
        
        # Convert to pretty-printed string
        rough_string = tostring(root, encoding='unicode')
        reparsed = minidom.parseString(rough_string)
        pretty_xml = reparsed.toprettyxml(indent="  ")
        
        # Remove extra blank lines
        lines = [line for line in pretty_xml.split('\n') if line.strip()]
        return '\n'.join(lines)
    
    def save_xml(self, scene: ReconstructedScene, output_path: str,
                model_name: str = "reconstructed_environment") -> None:
        """Generate and save MuJoCo XML file.
        
        Args:
            scene: Reconstructed scene
            output_path: Path to save XML file
            model_name: Name for the model
        """
        xml_string = self.generate_xml(scene, model_name)
        
        # Create directory if needed
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Write to file
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(xml_string)
        
        print(f"XML saved to: {output_path}")
