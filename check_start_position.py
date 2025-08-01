#!/usr/bin/env python3
"""
Check if the robot start positions are valid in the reachability grid.
"""

import numpy as np

def world_to_grid(world_x, world_y, bounds, resolution):
    """Convert world coordinates to grid coordinates."""
    grid_x = int(np.floor((world_x - bounds[0]) / resolution))
    grid_y = int(np.floor((world_y - bounds[2]) / resolution))
    return grid_x, grid_y

def check_start_position(filename, start_pos, bounds=[-5.525, 5.525, -5.525, 5.525], resolution=0.1):
    """Check if start position is valid in the grid."""
    data = np.loadtxt(filename)
    x_coords = data[:, 0]
    y_coords = data[:, 1] 
    values = data[:, 2]
    
    # Convert start position to grid coordinates
    grid_x, grid_y = world_to_grid(start_pos[0], start_pos[1], bounds, resolution)
    grid_width = int((bounds[1] - bounds[0]) / resolution)
    grid_height = int((bounds[3] - bounds[2]) / resolution)
    
    print(f"\n=== {filename} ===")
    print(f"Start position: ({start_pos[0]:.6f}, {start_pos[1]:.6f})")
    print(f"Grid coordinates: ({grid_x}, {grid_y})")
    print(f"Grid bounds: 0 <= x < {grid_width}, 0 <= y < {grid_height}")
    
    # Check bounds
    if grid_x < 0 or grid_x >= grid_width or grid_y < 0 or grid_y >= grid_height:
        print("❌ START POSITION IS OUTSIDE GRID BOUNDS!")
        return False
    
    # Find the corresponding value in the data
    # Find closest grid cell
    target_world_x = bounds[0] + grid_x * resolution
    target_world_y = bounds[2] + grid_y * resolution
    
    # Find matching cell in data
    tolerance = resolution / 10  # Small tolerance for floating point comparison
    matching_mask = (np.abs(x_coords - target_world_x) < tolerance) & (np.abs(y_coords - target_world_y) < tolerance)
    
    if np.sum(matching_mask) == 0:
        print("❌ COULD NOT FIND MATCHING CELL IN DATA!")
        return False
    
    cell_value = values[matching_mask][0]
    print(f"Cell value at start position: {cell_value}")
    
    if cell_value == -2:
        print("❌ START POSITION IS INSIDE AN OBSTACLE!")
        return False
    elif cell_value == 0:
        print("⚠️  START POSITION IS UNREACHABLE (this is the problem!)")
        return False  
    elif cell_value == 1:
        print("✅ START POSITION IS REACHABLE")
        return True
    else:
        print(f"❓ UNKNOWN CELL VALUE: {cell_value}")
        return False

def main():
    """Check all start positions."""
    
    # Start positions from the test output
    start_positions = [
        ("debug_wavefront.txt", [-2.29203, -4.22824]),      # Initial robot position
        ("debug_wavefront_1.txt", [0.543022, 0.718257]),    # After Push 1
        ("debug_wavefront_2.txt", [0.872323, -0.037757]),   # After Push 2  
        ("debug_wavefront_3.txt", [0.892341, 0.879361])     # After Push 3
    ]
    
    print("Checking Robot Start Positions in Reachability Grid")
    print("=" * 60)
    
    for filename, start_pos in start_positions:
        try:
            is_valid = check_start_position(filename, start_pos)
        except Exception as e:
            print(f"Error checking {filename}: {e}")

if __name__ == "__main__":
    main()