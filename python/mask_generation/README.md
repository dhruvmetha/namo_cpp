# NAMO Mask Generation

This module provides tools for generating visualization masks and datasets from NAMO (Navigation Among Movable Obstacles) planning data.

## Overview

The mask generation system processes collected NAMO planning data (`.pkl` files) and creates standardized 224x224 mask images for machine learning applications. Each successful planning episode generates 9 different mask types that represent various aspects of the environment and planning state.

## Generated Masks (9 total per episode)

### Binary Masks (0/1 values):
- **robot**: Robot position as filled circle
- **goal**: Goal position as filled circle  
- **movable**: All movable objects as filled rectangles
- **static**: Static walls and obstacles
- **reachable**: Objects reachable by robot (within connectivity)
- **target_object**: The specific object being manipulated
- **target_goal**: Target object at its goal position

### Distance Field Masks:
- **robot_distance**: Wavefront distance field from robot position
- **goal_distance**: Wavefront distance field from goal position

Distance fields use a cost model:
- Free space traversal cost: 1
- Movable object traversal cost: 4  
- Static obstacles: impassable (-1)
- Values normalized to [0, 1] range (excluding -1 for obstacles)

## Directory Structure

```
mask_generation/
├── __init__.py              # Package initialization and exports
├── README.md               # This documentation
├── visualizer.py           # Core visualization and mask generation
├── batch_collection.py     # Batch processing pipeline
└── examples/
    ├── __init__.py
    └── example_visualization.py  # Usage examples
```

## Usage

### Batch Processing (Recommended)

Process entire directories of planning data with parallel workers:

```bash
# Using the convenience runner (parallel processing)
python run_mask_generation.py batch \
    --input-dir /path/to/pkl/files \
    --output-dir /path/to/output \
    --workers 8

# Using module directly with auto-detected workers
python -m mask_generation.batch_collection \
    --input-dir /path/to/pkl/files \
    --output-dir /path/to/output \
    --pattern "*_results.pkl"

# Serial processing for debugging
python -m mask_generation.batch_collection \
    --input-dir /path/to/pkl/files \
    --output-dir /path/to/output \
    --serial
```

### Programmatic Usage

```python
from mask_generation import NAMODataVisualizer
import pickle

# Load episode data
with open('episode.pkl', 'rb') as f:
    data = pickle.load(f)

episode = data['episode_results'][0]

# Generate masks
visualizer = NAMODataVisualizer()
masks = visualizer.generate_episode_masks_batch(episode)

# Masks is now a dict with 9 numpy arrays (224x224 each)
print(f"Generated {len(masks)} masks:")
for name, mask in masks.items():
    print(f"  {name}: {mask.shape}, max={mask.max():.3f}")
```

### Running Examples

```bash
python run_mask_generation.py example
```

## Output Format

### Directory Structure
```
output_dir/
├── task_id_1/
│   ├── episode_1.npz      # Compressed masks + metadata
│   ├── episode_2.npz
│   └── ...
├── task_id_2/
│   └── ...
```

### NPZ File Contents
Each `.npz` file contains:
- 9 mask arrays (224x224 float32)
- Metadata: episode_id, algorithm, search_time, action_sequence, etc.

```python
import numpy as np

# Load generated data
data = np.load('episode.npz')
print(f"Available masks: {list(data.keys())}")

# Access specific masks
robot_mask = data['robot']
goal_distance = data['goal_distance']
metadata = data['episode_id'][0]  # String metadata
```

## Data Requirements

Input episodes must have:
- `solution_found = True` 
- `len(action_sequence) > 0` (non-trivial solutions)
- Valid `state_observations` with object poses
- Required object info in `static_object_info`

## Performance

- **Processing rate**: 
  - Serial: ~3 files/second  
  - Parallel (8 workers): ~20+ files/second (varies by system)
- **Storage efficiency**: ~20KB per episode (9 masks + metadata, compressed)
- **Memory usage**: Minimal per worker due to efficient numpy operations
- **Scalability**: Tested on 2000+ file datasets with parallel processing
- **Auto-scaling**: Automatically detects CPU count and limits workers appropriately

## Parallel Processing

The batch collection pipeline supports parallel processing for significantly improved performance:

### Worker Architecture
- **File-level parallelization**: Each worker processes one `.pkl` file completely
- **Independent workers**: Each worker has its own `NAMODataVisualizer` instance
- **Thread-safe**: No shared state between workers, safe file writing
- **Auto-detection**: Automatically uses `cpu_count()` workers if not specified
- **Smart limiting**: Never uses more workers than files available

### Configuration Options
```bash
# Auto-detect workers (recommended)
python -m mask_generation.batch_collection --input-dir /data --output-dir /output

# Specify worker count
python -m mask_generation.batch_collection --input-dir /data --output-dir /output --workers 16

# Force single-threaded (debugging)
python -m mask_generation.batch_collection --input-dir /data --output-dir /output --serial
```

### Performance Scaling
- **2-4 workers**: 2-3x speedup on typical systems
- **8+ workers**: 5-7x speedup on high-core systems  
- **Diminishing returns**: Beyond 16 workers, I/O becomes bottleneck
- **Memory scaling**: Linear increase with worker count (~200MB per worker)

## Technical Details

### Coordinate System
- World coordinates: Standard MuJoCo coordinate system
- Pixel coordinates: 224x224 grid with configurable world bounds
- Transformations handle SE(2) poses with proper rotation

### Distance Field Computation
- Algorithm: Dijkstra's shortest path with cost model
- Robot radius inflation: 0.15m (morphological dilation)
- Normalization: Values ≥ 0 scaled to [0,1], obstacles remain -1

### Mask Generation Pipeline
1. Extract environment info from episode data
2. Create binary masks for objects and regions
3. Compute distance fields with wavefront propagation  
4. Apply coordinate transformations and normalization
5. Compress and save with metadata

## Dependencies

- numpy: Array operations and mask generation
- matplotlib: Visualization and polygon rendering
- pickle: Data loading
- tqdm: Progress tracking
- heapq: Priority queue for distance field computation