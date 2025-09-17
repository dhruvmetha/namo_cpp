#!/usr/bin/env python3
"""
XML Goal Parser for NAMO Environment Files

This module extracts robot goal positions from MuJoCo XML environment files.
Each XML file contains a <site name="goal" ... pos="x y z"/> element that
defines the target location for the robot to reach.

Usage:
    from namo.core.xml_goal_parser import extract_goal_from_xml
    goal = extract_goal_from_xml("path/to/env_config.xml")  # Returns (x, y, theta)
"""

import xml.etree.ElementTree as ET
from typing import Tuple, Optional
from pathlib import Path
import logging


class GoalExtractionError(Exception):
    """Exception raised when goal extraction fails."""
    pass


def extract_goal_from_xml(xml_file_path: str) -> Tuple[float, float, float]:
    """
    Extract robot goal position from MuJoCo XML environment file.
    
    Args:
        xml_file_path: Path to the XML environment file
        
    Returns:
        Tuple of (x, y, theta) coordinates. Theta is always 0.0 for these environments.
        
    Raises:
        GoalExtractionError: If goal extraction fails for any reason
    """
    try:
        # Validate file exists
        xml_path = Path(xml_file_path)
        if not xml_path.exists():
            raise GoalExtractionError(f"XML file not found: {xml_file_path}")
        
        if not xml_path.is_file():
            raise GoalExtractionError(f"Path is not a file: {xml_file_path}")
        
        # Parse XML
        try:
            tree = ET.parse(xml_file_path)
            root = tree.getroot()
        except ET.ParseError as e:
            raise GoalExtractionError(f"Invalid XML format: {e}")
        except Exception as e:
            raise GoalExtractionError(f"Failed to read XML file: {e}")
        
        # Find goal site element
        goal_site = None
        
        # Search for <site name="goal" ...>
        for site in root.iter('site'):
            if site.get('name') == 'goal':
                goal_site = site
                break
        
        if goal_site is None:
            raise GoalExtractionError("No <site name=\"goal\"> element found in XML")
        
        # Extract position attribute
        pos_attr = goal_site.get('pos')
        if pos_attr is None:
            raise GoalExtractionError("Goal site element missing 'pos' attribute")
        
        # Parse position string "x y z"
        try:
            pos_parts = pos_attr.strip().split()
            if len(pos_parts) != 3:
                raise GoalExtractionError(f"Expected 3 position values, got {len(pos_parts)}: {pos_attr}")
            
            x = float(pos_parts[0])
            y = float(pos_parts[1])
            z = float(pos_parts[2])  # Usually 0.0, but we'll extract it anyway
            
            # For NAMO, theta is typically 0.0 (no rotation constraint)
            theta = 0.0
            
            return (x, y, theta)
            
        except ValueError as e:
            raise GoalExtractionError(f"Invalid position values: {pos_attr}, error: {e}")
    
    except GoalExtractionError:
        # Re-raise goal extraction errors
        raise
    except Exception as e:
        # Catch any unexpected errors
        raise GoalExtractionError(f"Unexpected error during goal extraction: {e}")


def extract_goal_with_fallback(xml_file_path: str, 
                              fallback_goal: Tuple[float, float, float] = (-0.5, 1.3, 0.0)) -> Tuple[float, float, float]:
    """
    Extract goal from XML with fallback to default goal if extraction fails.
    
    Args:
        xml_file_path: Path to the XML environment file
        fallback_goal: Default goal to use if extraction fails
        
    Returns:
        Tuple of (x, y, theta) coordinates
    """
    try:
        return extract_goal_from_xml(xml_file_path)
    except GoalExtractionError as e:
        logging.warning(f"Goal extraction failed for {xml_file_path}: {e}. Using fallback goal {fallback_goal}")
        return fallback_goal


def validate_goal_coordinates(goal: Tuple[float, float, float], 
                             bounds: Tuple[Tuple[float, float], Tuple[float, float]] = ((-5.0, 5.0), (-5.0, 5.0))) -> bool:
    """
    Validate that goal coordinates are within reasonable bounds.
    
    Args:
        goal: Goal coordinates (x, y, theta)
        bounds: ((x_min, x_max), (y_min, y_max)) bounds for validation
        
    Returns:
        True if goal is valid, False otherwise
    """
    x, y, theta = goal
    (x_min, x_max), (y_min, y_max) = bounds
    
    # Check if coordinates are within bounds
    if not (x_min <= x <= x_max):
        return False
    if not (y_min <= y <= y_max):
        return False
    
    # Check for NaN or infinite values
    if not all(isinstance(coord, (int, float)) and not (coord != coord or abs(coord) == float('inf')) 
               for coord in [x, y, theta]):
        return False
    
    return True


def batch_extract_goals(xml_file_paths: list) -> dict:
    """
    Extract goals from multiple XML files efficiently.
    
    Args:
        xml_file_paths: List of XML file paths
        
    Returns:
        Dictionary mapping file paths to extracted goals or error messages
    """
    results = {}
    
    for xml_path in xml_file_paths:
        try:
            goal = extract_goal_from_xml(xml_path)
            results[xml_path] = {
                'success': True,
                'goal': goal,
                'error': None
            }
        except GoalExtractionError as e:
            results[xml_path] = {
                'success': False, 
                'goal': None,
                'error': str(e)
            }
    
    return results


if __name__ == "__main__":
    """Test goal extraction with sample files."""
    import sys
    
    # Test with sample XML files
    test_files = [
        "../ml4kp_ktamp/resources/models/custom_walled_envs/aug9/easy/set1/benchmark_1/env_config_100a.xml",
        "../ml4kp_ktamp/resources/models/custom_walled_envs/aug9/easy/set1/benchmark_1/env_config_100b.xml", 
        "../ml4kp_ktamp/resources/models/custom_walled_envs/aug9/easy/set1/benchmark_1/env_config_100c.xml"
    ]
    
    print("🎯 Testing XML Goal Extraction")
    print("=" * 50)
    
    for xml_file in test_files:
        try:
            goal = extract_goal_from_xml(xml_file)
            valid = validate_goal_coordinates(goal)
            print(f"✅ {Path(xml_file).name}: {goal} {'(valid)' if valid else '(INVALID)'}")
        except GoalExtractionError as e:
            print(f"❌ {Path(xml_file).name}: ERROR - {e}")
        except Exception as e:
            print(f"💥 {Path(xml_file).name}: UNEXPECTED ERROR - {e}")
    
    # Test batch extraction
    print(f"\n📦 Batch extraction test:")
    batch_results = batch_extract_goals(test_files)
    successful = sum(1 for r in batch_results.values() if r['success'])
    print(f"   Successful: {successful}/{len(test_files)}")
    
    if successful == len(test_files):
        print("🎉 All tests passed!")
        sys.exit(0)
    else:
        print("❌ Some tests failed!")
        sys.exit(1)