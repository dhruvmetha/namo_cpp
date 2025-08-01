#!/usr/bin/env python3
"""
Debug the suspicious wavefront behavior where most cells become unreachable.
"""

import numpy as np

def load_and_analyze(filename):
    """Load wavefront and analyze start position."""
    data = np.loadtxt(filename)
    x_coords = data[:, 0]
    y_coords = data[:, 1] 
    values = data[:, 2]
    
    # Find reachable cells (value = 1)
    reachable_mask = values == 1
    reachable_x = x_coords[reachable_mask]
    reachable_y = y_coords[reachable_mask]
    
    # Find obstacles
    obstacle_mask = values == -2
    obstacle_x = x_coords[obstacle_mask]
    obstacle_y = y_coords[obstacle_mask]
    
    print(f"\n=== {filename} ===")
    print(f"Total cells: {len(values)}")
    print(f"Obstacles (-2): {np.sum(values == -2)}")
    print(f"Unreachable (0): {np.sum(values == 0)}")
    print(f"Reachable (1): {np.sum(values == 1)}")
    
    if len(reachable_x) > 0:
        print(f"Reachable region bounds:")
        print(f"  X: [{np.min(reachable_x):.3f}, {np.max(reachable_x):.3f}]")
        print(f"  Y: [{np.min(reachable_y):.3f}, {np.max(reachable_y):.3f}]")
        
        # Find center of reachable region (likely start position)
        center_x = (np.min(reachable_x) + np.max(reachable_x)) / 2
        center_y = (np.min(reachable_y) + np.max(reachable_y)) / 2
        print(f"Reachable center: ({center_x:.3f}, {center_y:.3f})")
    else:
        print("NO REACHABLE CELLS!")
    
    return reachable_x, reachable_y

def main():
    """Analyze the wavefront start position issue."""
    
    files = [
        "debug_wavefront.txt",
        "debug_wavefront_1.txt", 
        "debug_wavefront_2.txt",
        "debug_wavefront_3.txt"
    ]
    
    print("Debugging Wavefront Start Position Issue")
    print("=" * 50)
    
    for filename in files:
        try:
            reachable_x, reachable_y = load_and_analyze(filename)
        except Exception as e:
            print(f"Error loading {filename}: {e}")
    
    print("\n" + "="*50)
    print("ANALYSIS:")
    print("After Push 1, almost all cells become unreachable (0).")
    print("This suggests the BFS start position is invalid or blocked.")
    print("\nPossible causes:")
    print("1. Robot position after push is inside an obstacle")
    print("2. Start position is outside grid bounds")
    print("3. Coordinate system mismatch")
    print("4. Object moved to block the robot's current position")

if __name__ == "__main__":
    main()