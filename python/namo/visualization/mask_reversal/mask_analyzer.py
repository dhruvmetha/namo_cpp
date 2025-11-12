"""
Mask Analyzer - Detect objects in binary masks using computer vision.

This module analyzes mask images to extract object positions, shapes, and orientations.
"""

import numpy as np
import cv2
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass


@dataclass
class DetectedObject:
    """Represents an object detected in a mask."""
    
    # Center position in pixel coordinates
    center_px: Tuple[int, int]
    
    # Oriented bounding box (center, size, angle)
    # size is (width, height) in pixels
    # angle is rotation in degrees (0-180)
    width_px: float
    height_px: float
    angle_deg: float
    
    # Bounding box corners (4 points)
    corners_px: np.ndarray  # Shape (4, 2)
    
    # Area in pixels
    area_px: int
    
    # Object type
    obj_type: str  # 'robot', 'goal', 'movable', 'static'
    
    # Optional: contour points
    contour: Optional[np.ndarray] = None


@dataclass
class DetectedWall:
    """Represents a wall detected as a line segment."""
    
    # Line endpoints in pixel coordinates
    start_px: Tuple[int, int]
    end_px: Tuple[int, int]
    
    # Center position
    center_px: Tuple[int, int]
    
    # Length in pixels
    length_px: float
    
    # Angle in degrees (0 = horizontal, 90 = vertical)
    angle_deg: float
    
    # Thickness estimate in pixels
    thickness_px: float
    
    # Wall type
    wall_type: str  # 'horizontal' or 'vertical'


class MaskAnalyzer:
    """Analyzes mask images to detect objects."""
    
    IMG_SIZE = 224
    
    def __init__(self, min_area: int = 10, max_area: int = 20000):
        """Initialize analyzer.
        
        Args:
            min_area: Minimum object area in pixels to consider
            max_area: Maximum object area in pixels to consider
        """
        self.min_area = min_area
        self.max_area = max_area
    
    def detect_circles(self, mask: np.ndarray, obj_type: str) -> List[DetectedObject]:
        """Detect circular objects (robot, goal) in mask.
        
        Args:
            mask: Binary mask (224x224)
            obj_type: Type of object ('robot' or 'goal')
            
        Returns:
            List of detected circular objects
        """
        # Convert to uint8
        mask_uint8 = (mask * 255).astype(np.uint8)
        
        # Threshold
        _, binary = cv2.threshold(mask_uint8, 127, 255, cv2.THRESH_BINARY)
        
        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        detected = []
        for contour in contours:
            area = cv2.contourArea(contour)
            
            if area < self.min_area or area > self.max_area:
                continue
            
            # Get minimum enclosing circle
            (cx, cy), radius = cv2.minEnclosingCircle(contour)
            
            # For circles, treat as square bounding box for consistency
            diameter = 2 * radius
            
            # Create pseudo-corners for circle
            corners = np.array([
                [cx - radius, cy - radius],
                [cx + radius, cy - radius],
                [cx + radius, cy + radius],
                [cx - radius, cy + radius]
            ], dtype=np.float32)
            
            detected.append(DetectedObject(
                center_px=(int(cx), int(cy)),
                width_px=diameter,
                height_px=diameter,
                angle_deg=0.0,  # Circles have no orientation
                corners_px=corners,
                area_px=int(area),
                obj_type=obj_type,
                contour=contour
            ))
        
        return detected
    
    def detect_rectangles(self, mask: np.ndarray, obj_type: str) -> List[DetectedObject]:
        """Detect rectangular objects (movable, static) in mask.
        
        Args:
            mask: Binary mask (224x224)
            obj_type: Type of object ('movable' or 'static')
            
        Returns:
            List of detected rectangular objects
        """
        # Convert to uint8
        mask_uint8 = (mask * 255).astype(np.uint8)
        
        # Threshold
        _, binary = cv2.threshold(mask_uint8, 127, 255, cv2.THRESH_BINARY)
        
        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        detected = []
        for contour in contours:
            area = cv2.contourArea(contour)
            
            if area < self.min_area or area > self.max_area:
                continue
            
            # Get minimum area rotated rectangle
            if len(contour) < 5:
                # Need at least 5 points for minAreaRect
                continue
            
            rect = cv2.minAreaRect(contour)
            # rect = ((center_x, center_y), (width, height), angle)
            center, size, angle = rect
            
            # Get box corners
            box = cv2.boxPoints(rect)
            box = np.intp(box)  # Use np.intp instead of deprecated np.int0
            
            detected.append(DetectedObject(
                center_px=(int(center[0]), int(center[1])),
                width_px=size[0],
                height_px=size[1],
                angle_deg=angle,
                corners_px=box,
                area_px=int(area),
                obj_type=obj_type,
                contour=contour
            ))
        
        return detected
    
    def detect_walls(self, mask: np.ndarray) -> List[DetectedWall]:
        """Detect wall segments in static mask using Hough line detection.
        
        Args:
            mask: Binary mask (224x224) containing walls
            
        Returns:
            List of detected wall segments
        """
        # Convert to uint8
        mask_uint8 = (mask * 255).astype(np.uint8)
        
        # Apply Canny edge detection
        edges = cv2.Canny(mask_uint8, 50, 150)
        
        # Detect lines using Hough transform
        # Parameters tuned for wall detection
        # Note: maxLineGap=50 allows bridging gaps caused by movable objects
        lines = cv2.HoughLinesP(
            edges, 
            rho=1,                # Distance resolution in pixels
            theta=np.pi/180,      # Angle resolution in radians
            threshold=40,         # Minimum votes
            minLineLength=20,     # Minimum line length
            maxLineGap=50         # Maximum gap between line segments (allows object gaps)
        )
        
        if lines is None:
            return []
        
        detected_walls = []
        
        for line in lines:
            x1, y1, x2, y2 = line[0]
            
            # Calculate line properties
            length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
            angle = np.abs(np.arctan2(y2-y1, x2-x1) * 180 / np.pi)
            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)
            
            # Classify as horizontal or vertical
            # Horizontal: angle close to 0 or 180
            # Vertical: angle close to 90
            if angle < 20 or angle > 160:
                wall_type = 'horizontal'
                thickness = self._estimate_wall_thickness(mask_uint8, center_x, center_y, 'horizontal')
            elif 70 < angle < 110:
                wall_type = 'vertical'
                thickness = self._estimate_wall_thickness(mask_uint8, center_x, center_y, 'vertical')
            else:
                # Skip diagonal lines (shouldn't be walls)
                continue
            
            detected_walls.append(DetectedWall(
                start_px=(x1, y1),
                end_px=(x2, y2),
                center_px=(center_x, center_y),
                length_px=length,
                angle_deg=angle,
                thickness_px=thickness,
                wall_type=wall_type
            ))
        
        # Merge nearby parallel wall segments
        merged_walls = self._merge_wall_segments(detected_walls)
        
        # Split long walls that have gaps (indicating separate wall segments)
        split_walls = self._split_walls_at_gaps(merged_walls, mask_uint8)
        
        return split_walls
    
    def _estimate_wall_thickness(self, mask: np.ndarray, cx: int, cy: int, 
                                 wall_type: str, search_range: int = 10) -> float:
        """Estimate wall thickness by sampling perpendicular to the wall.
        
        Note: Due to anti-aliasing and rendering artifacts, walls in masks appear
        thicker than their actual geometry. We use a fixed minimum thickness that
        corresponds to typical thin walls in MuJoCo (0.05-0.1 world units).
        
        Args:
            mask: Binary mask  
            cx, cy: Center point of wall
            wall_type: 'horizontal' or 'vertical'
            search_range: How many pixels to search in each direction
            
        Returns:
            Estimated thickness in pixels (fixed at 2.0 for thin walls)
        """
        # Use fixed thin wall thickness to avoid mask rendering artifacts
        # 2 pixels ≈ 0.1 world units at 224px / 11 units scale (≈20 px/unit)
        # This matches typical MuJoCo wall thickness of 0.05 half-width (0.1 total)
        return 2.0
    
    def _merge_wall_segments(self, walls: List[DetectedWall], 
                            distance_threshold: int = 15,
                            angle_threshold: float = 10.0) -> List[DetectedWall]:
        """Merge nearby parallel wall segments into longer walls.
        
        Args:
            walls: List of detected wall segments
            distance_threshold: Maximum distance between segments to merge (pixels)
            angle_threshold: Maximum angle difference to merge (degrees)
            
        Returns:
            List of merged wall segments
        """
        if not walls:
            return []
        
        # Group walls by type
        h_walls = [w for w in walls if w.wall_type == 'horizontal']
        v_walls = [w for w in walls if w.wall_type == 'vertical']
        
        # Merge each group
        merged_h = self._merge_aligned_walls(h_walls, 'horizontal', distance_threshold)
        merged_v = self._merge_aligned_walls(v_walls, 'vertical', distance_threshold)
        
        return merged_h + merged_v
    
    def _merge_aligned_walls(self, walls: List[DetectedWall], wall_type: str,
                            distance_threshold: int) -> List[DetectedWall]:
        """Merge walls that are aligned (on same axis).
        
        This uses a more sophisticated approach that checks if walls are collinear
        (on the same line) even if they have large gaps between them (e.g., due to
        movable objects blocking the wall).
        
        Args:
            walls: List of walls of same type
            wall_type: 'horizontal' or 'vertical'
            distance_threshold: Maximum perpendicular distance to consider aligned
            
        Returns:
            Merged walls
        """
        if not walls:
            return []
        
        # Use Union-Find to group collinear walls
        groups = []  # List of lists of wall indices
        used = [False] * len(walls)
        
        for i, wall_i in enumerate(walls):
            if used[i]:
                continue
            
            # Start a new group with this wall
            group = [i]
            used[i] = True
            
            # Find all walls that are collinear with this wall
            for j, wall_j in enumerate(walls):
                if used[j] or i == j:
                    continue
                
                # Check if walls are collinear
                if self._are_walls_collinear(wall_i, wall_j, wall_type, distance_threshold):
                    group.append(j)
                    used[j] = True
            
            groups.append(group)
        
        # Combine each group
        merged = []
        for group in groups:
            group_walls = [walls[idx] for idx in group]
            merged.append(self._combine_wall_group(group_walls, wall_type))
        
        return merged
    
    def _are_walls_collinear(self, wall1: DetectedWall, wall2: DetectedWall, 
                             wall_type: str, threshold: int) -> bool:
        """Check if two walls are collinear (on the same line).
        
        Args:
            wall1, wall2: Two walls to check
            wall_type: 'horizontal' or 'vertical'
            threshold: Maximum perpendicular distance to consider aligned
            
        Returns:
            True if walls are collinear
        """
        if wall_type == 'horizontal':
            # Check if y-coordinates are close (walls are on same horizontal line)
            y1 = (wall1.start_px[1] + wall1.end_px[1]) / 2
            y2 = (wall2.start_px[1] + wall2.end_px[1]) / 2
            return abs(y1 - y2) < threshold
        else:  # vertical
            # Check if x-coordinates are close (walls are on same vertical line)
            x1 = (wall1.start_px[0] + wall1.end_px[0]) / 2
            x2 = (wall2.start_px[0] + wall2.end_px[0]) / 2
            return abs(x1 - x2) < threshold
    
    def _combine_wall_group(self, walls: List[DetectedWall], wall_type: str) -> DetectedWall:
        """Combine a group of wall segments into one.
        
        Args:
            walls: List of walls to combine
            wall_type: 'horizontal' or 'vertical'
            
        Returns:
            Combined wall
        """
        if len(walls) == 1:
            return walls[0]
        
        # Find extent of all walls
        if wall_type == 'horizontal':
            min_x = min(min(w.start_px[0], w.end_px[0]) for w in walls)
            max_x = max(max(w.start_px[0], w.end_px[0]) for w in walls)
            avg_y = int(np.mean([w.center_px[1] for w in walls]))
            
            start = (min_x, avg_y)
            end = (max_x, avg_y)
        else:  # vertical
            avg_x = int(np.mean([w.center_px[0] for w in walls]))
            min_y = min(min(w.start_px[1], w.end_px[1]) for w in walls)
            max_y = max(max(w.start_px[1], w.end_px[1]) for w in walls)
            
            start = (avg_x, min_y)
            end = (avg_x, max_y)
        
        length = np.sqrt((end[0]-start[0])**2 + (end[1]-start[1])**2)
        center = ((start[0]+end[0])//2, (start[1]+end[1])//2)
        avg_thickness = np.mean([w.thickness_px for w in walls])
        avg_angle = np.mean([w.angle_deg for w in walls])
        
        return DetectedWall(
            start_px=start,
            end_px=end,
            center_px=center,
            length_px=length,
            angle_deg=avg_angle,
            thickness_px=avg_thickness,
            wall_type=wall_type
        )
    
    def _split_walls_at_gaps(self, walls: List[DetectedWall], mask: np.ndarray,
                            min_gap_size: int = 15) -> List[DetectedWall]:
        """Split long walls that have significant gaps in the mask.
        
        This handles cases where two separate walls are detected as one because
        they're perfectly aligned (e.g., two vertical walls at x=0 but different y ranges).
        
        Args:
            walls: List of detected walls
            mask: Binary mask to check for gaps
            min_gap_size: Minimum gap size (in pixels) to trigger a split
            
        Returns:
            List of walls, with long walls split at gaps
        """
        split_walls = []
        
        for wall in walls:
            # Only check walls longer than 100 pixels (likely candidates for splitting)
            if wall.length_px < 100:
                split_walls.append(wall)
                continue
            
            # Sample the mask along the wall to find gaps
            if wall.wall_type == 'horizontal':
                # Check horizontal line
                y = wall.center_px[1]
                x_start = min(wall.start_px[0], wall.end_px[0])
                x_end = max(wall.start_px[0], wall.end_px[0])
                
                # Get profile along the wall
                if 0 <= y < mask.shape[0]:
                    x_start = max(0, x_start)
                    x_end = min(mask.shape[1], x_end)
                    profile = mask[int(y), int(x_start):int(x_end)+1]
                else:
                    split_walls.append(wall)
                    continue
                    
            else:  # vertical
                # Check vertical line
                x = wall.center_px[0]
                y_start = min(wall.start_px[1], wall.end_px[1])
                y_end = max(wall.start_px[1], wall.end_px[1])
                
                # Get profile along the wall
                if 0 <= x < mask.shape[1]:
                    y_start = max(0, y_start)
                    y_end = min(mask.shape[0], y_end)
                    profile = mask[int(y_start):int(y_end)+1, int(x)]
                else:
                    split_walls.append(wall)
                    continue
            
            # Find gaps in the profile
            gaps = []
            in_gap = False
            gap_start = 0
            
            for i, val in enumerate(profile):
                if val < 128:  # Black pixel (gap)
                    if not in_gap:
                        gap_start = i
                        in_gap = True
                else:  # White pixel (wall)
                    if in_gap:
                        gap_size = i - gap_start
                        if gap_size >= min_gap_size:
                            gaps.append((gap_start, i-1))
                        in_gap = False
            
            # If no significant gaps, keep the wall as is
            if not gaps:
                split_walls.append(wall)
                continue
            
            # Split the wall at gaps
            if wall.wall_type == 'horizontal':
                current_x = x_start
                for gap_start_idx, gap_end_idx in gaps:
                    segment_end = x_start + gap_start_idx - 1
                    if segment_end > current_x:
                        # Create wall segment before gap
                        split_walls.append(DetectedWall(
                            start_px=(int(current_x), int(y)),
                            end_px=(int(segment_end), int(y)),
                            center_px=(int((current_x + segment_end) / 2), int(y)),
                            length_px=float(segment_end - current_x),
                            angle_deg=wall.angle_deg,
                            thickness_px=wall.thickness_px,
                            wall_type='horizontal'
                        ))
                    current_x = x_start + gap_end_idx + 1
                
                # Add final segment
                if current_x < x_end:
                    split_walls.append(DetectedWall(
                        start_px=(int(current_x), int(y)),
                        end_px=(int(x_end), int(y)),
                        center_px=(int((current_x + x_end) / 2), int(y)),
                        length_px=float(x_end - current_x),
                        angle_deg=wall.angle_deg,
                        thickness_px=wall.thickness_px,
                        wall_type='horizontal'
                    ))
                    
            else:  # vertical
                current_y = y_start
                for gap_start_idx, gap_end_idx in gaps:
                    segment_end = y_start + gap_start_idx - 1
                    if segment_end > current_y:
                        # Create wall segment before gap
                        split_walls.append(DetectedWall(
                            start_px=(int(x), int(current_y)),
                            end_px=(int(x), int(segment_end)),
                            center_px=(int(x), int((current_y + segment_end) / 2)),
                            length_px=float(segment_end - current_y),
                            angle_deg=wall.angle_deg,
                            thickness_px=wall.thickness_px,
                            wall_type='vertical'
                        ))
                    current_y = y_start + gap_end_idx + 1
                
                # Add final segment
                if current_y < y_end:
                    split_walls.append(DetectedWall(
                        start_px=(int(x), int(current_y)),
                        end_px=(int(x), int(y_end)),
                        center_px=(int(x), int((current_y + y_end) / 2)),
                        length_px=float(y_end - current_y),
                        angle_deg=wall.angle_deg,
                        thickness_px=wall.thickness_px,
                        wall_type='vertical'
                    ))
        
        # Filter out very short wall segments (likely noise from splitting)
        split_walls = [w for w in split_walls if w.length_px >= 20]
        
        return split_walls
    
    def analyze_all_masks(self, robot_mask: np.ndarray, goal_mask: np.ndarray,
                         movable_mask: np.ndarray, static_mask: np.ndarray) -> Dict[str, List]:
        """Analyze all masks and detect objects.
        
        Args:
            robot_mask: Robot position mask
            goal_mask: Goal position mask
            movable_mask: Movable objects mask
            static_mask: Static objects mask
            
        Returns:
            Dictionary with keys 'robot', 'goal', 'movable', 'static' 
            mapping to lists of detected objects
        """
        results = {}
        
        # Detect robot (circle)
        results['robot'] = self.detect_circles(robot_mask, 'robot')
        
        # Detect goal (circle)
        results['goal'] = self.detect_circles(goal_mask, 'goal')
        
        # Detect movable objects (rectangles)
        results['movable'] = self.detect_rectangles(movable_mask, 'movable')
        
        # Detect static walls using line detection
        results['static'] = self.detect_walls(static_mask)
        
        return results
    
    def visualize_detections(self, detections: Dict[str, List[DetectedObject]], 
                           output_path: str, img_size: int = 224) -> None:
        """Visualize detected objects.
        
        Args:
            detections: Dictionary of detected objects
            output_path: Path to save visualization
            img_size: Image size (default 224)
        """
        # Create blank image
        vis = np.zeros((img_size, img_size, 3), dtype=np.uint8)
        
        # Colors for each type
        colors = {
            'robot': (0, 255, 0),      # Green
            'goal': (255, 0, 0),       # Blue
            'movable': (0, 0, 255),    # Red
            'static': (128, 128, 128)  # Gray
        }
        
        # Draw all objects
        for obj_type, objects in detections.items():
            color = colors.get(obj_type, (255, 255, 255))
            
            for obj in objects:
                if isinstance(obj, DetectedWall):
                    # Draw wall as line
                    cv2.line(vis, obj.start_px, obj.end_px, color, max(2, int(obj.thickness_px)))
                    cv2.circle(vis, obj.center_px, 3, (255, 255, 0), -1)  # Yellow center
                elif hasattr(obj, 'obj_type') and obj.obj_type in ['robot', 'goal']:
                    # Draw circle
                    cv2.circle(vis, obj.center_px, int(obj.width_px / 2), color, 2)
                    cv2.circle(vis, obj.center_px, 2, (255, 255, 255), -1)
                else:
                    # Draw rotated rectangle
                    cv2.drawContours(vis, [obj.corners_px], 0, color, 2)
                    cv2.circle(vis, obj.center_px, 2, (255, 255, 255), -1)
        
        # Save
        cv2.imwrite(output_path, vis)
        print(f"Visualization saved to: {output_path}")
