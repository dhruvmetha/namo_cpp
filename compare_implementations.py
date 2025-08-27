#!/usr/bin/env python3
"""
Compare our implementation vs reference implementation approach.
"""

import sys
import numpy as np
sys.path.append('python')

from mask_generation.visualizer import NAMODataVisualizer

class ReferenceStyleConverter:
    """Reference implementation approach for comparison."""
    IMG_SIZE = 224
    
    def __init__(self, world_bounds):
        # Calculate dynamic world size based on bounds with some padding
        self.x_min, self.x_max, self.y_min, self.y_max = world_bounds
        self.WORLD_WIDTH = self.x_max - self.x_min
        self.WORLD_HEIGHT = self.y_max - self.y_min
        # Use the larger dimension to maintain square images
        self.WORLD_SIZE = max(self.WORLD_WIDTH, self.WORLD_HEIGHT)
        self.SCALE = self.IMG_SIZE / self.WORLD_SIZE  # pixels per world unit
    
    def _world_to_pixel(self, x, y):
        # Convert from world coordinates to pixel coordinates
        # Center the world bounds in the image
        world_center_x = (self.x_min + self.x_max) / 2
        world_center_y = (self.y_min + self.y_max) / 2
        
        # For rectangular environments, we need to center properly within the square image
        # The image is always square (IMG_SIZE x IMG_SIZE) but world might be rectangular
        img_center = self.IMG_SIZE / 2
        
        # Translate to center and scale
        px = int((x - world_center_x) * self.SCALE + img_center)
        py = int((y - world_center_y) * self.SCALE + img_center)
        
        return px, py

def compare_coordinate_transforms():
    """Compare our implementation vs reference style."""
    print("🔄 Comparing coordinate transformation implementations...")
    
    # Test bounds (what our debug showed)
    world_bounds = (-2.3, 2.3, -2.3, 2.3)
    
    # Create both implementations
    our_visualizer = NAMODataVisualizer()
    ref_converter = ReferenceStyleConverter(world_bounds)
    
    print(f"📐 World bounds: {world_bounds}")
    print(f"   Reference SCALE: {ref_converter.SCALE:.3f}")
    
    # Compare coordinate transformations
    test_points = [
        (-2.3, -2.3, "bottom-left corner"),
        (2.3, -2.3, "bottom-right corner"),
        (-2.3, 2.3, "top-left corner"), 
        (2.3, 2.3, "top-right corner"),
        (0.0, 0.0, "center"),
        (-2.1, 0.0, "left wall left edge"),
        (-1.9, 0.0, "left wall right edge"),
        (1.9, 0.0, "right wall left edge"),
        (2.1, 0.0, "right wall right edge"),
    ]
    
    print(f"\n🎯 Coordinate transformation comparison:")
    print(f"{'Point':<25} {'Our Impl':<12} {'Reference':<12} {'Match':<6}")
    print("-" * 60)
    
    for x, y, label in test_points:
        our_px, our_py = our_visualizer._world_to_pixel(x, y, world_bounds)
        ref_px, ref_py = ref_converter._world_to_pixel(x, y)
        
        match = "✓" if (our_px == ref_px and our_py == ref_py) else "✗"
        
        print(f"{label:<25} ({our_px:>3},{our_py:>3})     ({ref_px:>3},{ref_py:>3})     {match}")
        
        if match == "✗":
            print(f"  -> Difference: our=({our_px},{our_py}) vs ref=({ref_px},{ref_py})")
    
    # Test if walls should be at image edges with no padding
    print(f"\n🧱 Wall edge analysis:")
    print(f"   Left wall at x=-2.1 -> pixels {our_visualizer._world_to_pixel(-2.1, 0, world_bounds)[0]} (our) vs {ref_converter._world_to_pixel(-2.1, 0)[0]} (ref)")
    print(f"   Right wall at x=2.1 -> pixels {our_visualizer._world_to_pixel(2.1, 0, world_bounds)[0]} (our) vs {ref_converter._world_to_pixel(2.1, 0)[0]} (ref)")
    print(f"   Expected if at edges: left=0, right=223")
    print(f"   Actual distance from edges: left={our_visualizer._world_to_pixel(-2.1, 0, world_bounds)[0]}, right={223-our_visualizer._world_to_pixel(2.1, 0, world_bounds)[0]}")

if __name__ == "__main__":
    compare_coordinate_transforms()