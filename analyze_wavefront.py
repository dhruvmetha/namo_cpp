#!/usr/bin/env python3
"""
Analyze wavefront debug files to check if reachability changes after object pushes.
"""

import numpy as np
import matplotlib.pyplot as plt

def load_wavefront(filename):
    """Load wavefront file and return grid data."""
    data = np.loadtxt(filename)
    x_coords = data[:, 0]
    y_coords = data[:, 1] 
    values = data[:, 2]
    
    # Find unique coordinates to determine grid size
    unique_x = np.unique(x_coords)
    unique_y = np.unique(y_coords)
    
    # Create grid
    grid = np.zeros((len(unique_x), len(unique_y)))
    
    # Fill grid
    for i, (x, y, val) in enumerate(data):
        x_idx = np.where(unique_x == x)[0][0]
        y_idx = np.where(unique_y == y)[0][0] 
        grid[x_idx, y_idx] = val
        
    return grid, unique_x, unique_y

def analyze_differences(grid1, grid2, name1, name2):
    """Analyze differences between two grids."""
    diff = grid2 - grid1
    
    # Count changes
    changed_cells = np.sum(diff != 0)
    total_cells = grid1.size
    
    # Count by change type
    became_obstacle = np.sum(diff == -2)  # 0 -> -2 or 1 -> -2
    became_free = np.sum(diff == 2)       # -2 -> 0 
    became_reachable = np.sum((grid1 == 0) & (grid2 == 1))  # 0 -> 1
    became_unreachable = np.sum((grid1 == 1) & (grid2 == 0))  # 1 -> 0
    
    print(f"\n=== {name1} vs {name2} ===")
    print(f"Total cells: {total_cells}")
    print(f"Changed cells: {changed_cells} ({100*changed_cells/total_cells:.1f}%)")
    
    if changed_cells > 0:
        print(f"  Became obstacles: {became_obstacle}")
        print(f"  Became free: {became_free}")
        print(f"  Became reachable: {became_reachable}")
        print(f"  Became unreachable: {became_unreachable}")
        
        # Show some example changes
        if changed_cells < 50:  # Only show if not too many
            changes = np.where(diff != 0)
            print(f"  Example changes:")
            for i in range(min(10, len(changes[0]))):
                x_idx, y_idx = changes[0][i], changes[1][i]
                old_val = grid1[x_idx, y_idx]
                new_val = grid2[x_idx, y_idx]
                print(f"    ({x_idx}, {y_idx}): {old_val} -> {new_val}")
    
    return changed_cells > 0

def count_values(grid, name):
    """Count different values in grid."""
    unique, counts = np.unique(grid, return_counts=True)
    print(f"\n{name} value counts:")
    for val, count in zip(unique, counts):
        if val == -2:
            print(f"  Obstacles (-2): {count}")
        elif val == 0:
            print(f"  Unreachable (0): {count}")
        elif val == 1:
            print(f"  Reachable (1): {count}")
        else:
            print(f"  Unknown ({val}): {count}")

def main():
    """Main analysis function."""
    print("Analyzing Wavefront Changes After Object Pushes")
    print("=" * 50)
    
    # Load all wavefront files
    files = [
        ("debug_wavefront.txt", "Initial"),
        ("debug_wavefront_1.txt", "After Push 1"),
        ("debug_wavefront_2.txt", "After Push 2"), 
        ("debug_wavefront_3.txt", "After Push 3")
    ]
    
    grids = {}
    coords = {}
    
    for filename, name in files:
        try:
            grid, x_coords, y_coords = load_wavefront(filename)
            grids[name] = grid
            coords[name] = (x_coords, y_coords)
            count_values(grid, name)
        except Exception as e:
            print(f"Error loading {filename}: {e}")
            return
    
    # Compare consecutive wavefronts
    names = [name for _, name in files]
    any_changes = False
    
    for i in range(len(names) - 1):
        name1, name2 = names[i], names[i+1]
        if name1 in grids and name2 in grids:
            has_changes = analyze_differences(grids[name1], grids[name2], name1, name2)
            any_changes = any_changes or has_changes
    
    if not any_changes:
        print("\n" + "="*50)
        print("🚨 WARNING: NO CHANGES DETECTED BETWEEN WAVEFRONTS!")
        print("This suggests the reachability grid is not updating properly.")
        print("Possible issues:")
        print("  1. Object movements are too small to affect grid")
        print("  2. Object change detection is not working")
        print("  3. Grid update is not being called")
        print("  4. All reachable regions remain the same")
    else:
        print("\n" + "="*50)
        print("✅ Changes detected - reachability grid is updating correctly!")

if __name__ == "__main__":
    main()