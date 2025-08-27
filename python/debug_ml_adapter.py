#!/usr/bin/env python3
"""Debug script to check what the ML adapter is actually loading from XML."""

import sys
sys.path.append('/common/home/dm1487/robotics_research/ktamp/namo/python')

from ml_image_converter_adapter import MLImageConverterAdapter

def debug_ml_adapter():
    """Debug what the ML adapter loads from XML."""
    print("=== Debugging ML Adapter XML Loading ===")
    
    xml_path = "custom_walled_envs/aug9/easy/set1/benchmark_2/env_config_1275a.xml"
    adapter = MLImageConverterAdapter(xml_path)
    
    print(f"XML path: {xml_path}")
    print(f"World bounds: {adapter.world_bounds}")
    print(f"Scale: {adapter.SCALE}")
    
    print(f"\nObject sizes loaded from XML:")
    for obj_name, size_array in adapter.object_sizes.items():
        print(f"  {obj_name}: {size_array}")
    
    return adapter

if __name__ == "__main__":
    debug_ml_adapter()