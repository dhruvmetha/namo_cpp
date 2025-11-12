# Mask Reversal System

Convert NPZ mask files back to MuJoCo XML environment files.

## Overview

This module reverses the mask generation process, reconstructing 3D MuJoCo scene descriptions from 2D mask images. It uses computer vision techniques to detect objects in masks, converts pixel coordinates back to world coordinates, and generates valid MuJoCo XML files.

## Architecture

```
NPZ File (masks) → Object Detection → World Reconstruction → XML Generation
```

### Pipeline Steps

1. **NPZ Loader** (`npz_loader.py`)
   - Loads and validates `.npz` files
   - Extracts required masks: robot, goal, movable, static
   - Reads optional metadata (episode_id, robot_goal, etc.)

2. **Mask Analyzer** (`mask_analyzer.py`)
   - Detects objects in binary masks using OpenCV
   - Finds circles (robot, goal) and rectangles (movable, static)
   - Extracts positions, sizes, and orientations

3. **Scene Reconstructor** (`scene_reconstructor.py`)
   - Converts pixel coordinates to world coordinates
   - Applies inverse transform from mask generation
   - Reconstructs complete scene with all objects

4. **XML Generator** (`xml_generator.py`)
   - Generates valid MuJoCo XML structure
   - Creates worldbody with walls, obstacles, robot, and goal
   - Handles coordinate conversions and quaternions

## Installation

No additional installation required beyond the main `namo_rl` package dependencies:
- numpy
- opencv-python (cv2)

## Usage

### Command Line

#### Single File

```bash
python -m namo.visualization.mask_reversal.reverse_masks \
    --input path/to/episode.npz \
    --output path/to/reconstructed.xml
```

#### Batch Processing

```bash
python -m namo.visualization.mask_reversal.reverse_masks \
    --batch \
    --input-dir path/to/npz_files \
    --output-dir path/to/xml_output
```

#### With Known World Bounds

If you know the world bounds used during mask generation:

```bash
python -m namo.visualization.mask_reversal.reverse_masks \
    --input episode.npz \
    --output reconstructed.xml \
    --world-bounds -5.5 5.5 -5.5 5.5
```

#### With Visualization

Save visualization of detected objects:

```bash
python -m namo.visualization.mask_reversal.reverse_masks \
    --input episode.npz \
    --output reconstructed.xml \
    --visualize
```

### Python API

```python
from namo.visualization.mask_reversal import reverse_masks_to_xml, batch_reverse_masks

# Single file
success = reverse_masks_to_xml(
    npz_path='episode.npz',
    output_path='reconstructed.xml',
    world_bounds=(-5.5, 5.5, -5.5, 5.5),  # Optional
    visualize=True  # Optional
)

# Batch processing
successful, failed = batch_reverse_masks(
    input_dir='./masks',
    output_dir='./xml_files',
    pattern='*.npz',
    world_bounds=(-5.5, 5.5, -5.5, 5.5),
    visualize=False
)
print(f"Processed: {successful} successful, {failed} failed")
```

### Programmatic Usage

```python
from namo.visualization.mask_reversal import (
    NPZLoader, MaskAnalyzer, SceneReconstructor, XMLGenerator
)

# Load NPZ file
mask_data = NPZLoader.load('episode.npz')

# Analyze masks
analyzer = MaskAnalyzer()
detections = analyzer.analyze_all_masks(
    mask_data.robot,
    mask_data.goal,
    mask_data.movable,
    mask_data.static
)

# Reconstruct scene
reconstructor = SceneReconstructor(world_bounds=(-5.5, 5.5, -5.5, 5.5))
scene = reconstructor.reconstruct_scene(detections, mask_data.robot_goal)

# Generate XML
generator = XMLGenerator()
generator.save_xml(scene, 'reconstructed.xml')
```

## Input Format

NPZ files must contain the following arrays:

### Required Masks (224×224 float32)
- `robot`: Robot position mask (binary, circle)
- `goal`: Goal position mask (binary, circle)
- `movable`: Movable objects mask (binary, rectangles)
- `static`: Static walls/obstacles mask (binary, rectangles)

### Optional Metadata
- `episode_id`: Episode identifier
- `task_id`: Task identifier
- `robot_goal`: Goal position [x, y, theta]
- `xml_file`: Original XML file path

### Not Required
- `reachable`, `target_object`, `target_goal`: Not needed for reconstruction
- `robot_distance`, `goal_distance`: Distance fields not needed

## Output Format

Generated XML files are valid MuJoCo environments with:

- **Worldbody structure**
  - Ground plane with material
  - Lighting
  - Origin markers

- **Static objects** (walls)
  - Box geoms with positions and orientations
  - Proper quaternion rotations
  - Named as `wall_0`, `wall_1`, etc.

- **Movable objects**
  - Bodies with freejoint
  - Box geoms with positions
  - Named as `obstacle_0_movable`, etc.

- **Robot**
  - Body with freejoint at detected position
  - Sphere geom with radius 0.15m
  - Named `robot`

- **Goal**
  - Site marker at detected position
  - Cylinder visualization
  - Named `goal`

## Coordinate Transform

The reversal uses the inverse transform from `NAMOImageConverter`:

**Forward (Generation):**
```
world (x,y) → center in image → scale → flip Y → pixel (px, py)
```

**Reverse (Reconstruction):**
```
pixel (px, py) → unflip Y → unscale → uncenter → world (x, y)
```

**Key parameters:**
- Image size: 224×224 pixels
- World size: max(world_width, world_height)
- Scale: 224 / world_size (pixels per world unit)

## Limitations

1. **Object Separation**: Merged objects in masks may be detected as single objects
2. **Small Objects**: Objects smaller than `min_area` (default 10 pixels) are ignored
3. **Overlapping Objects**: Heavily overlapping objects may not be distinguished
4. **World Bounds**: If unknown, uses default ±5.5 units (may need adjustment)
5. **Object Types**: Cannot distinguish between different movable object types

## Troubleshooting

### No objects detected
- Check mask threshold values (should be 0-1 range)
- Verify `min_area` and `max_area` parameters
- Use `--visualize` to inspect detected objects

### Incorrect world coordinates
- Provide correct `--world-bounds` matching original environment
- Check that world bounds are symmetric if using default

### XML validation errors
- Verify generated XML with MuJoCo validator
- Check for negative sizes or invalid quaternions

## Examples

See `example_reversal.py` for complete working examples.

## Performance

- **Single file**: ~0.1-0.5 seconds per file
- **Batch processing**: Parallel processing possible (future enhancement)
- **Memory usage**: ~10MB per file (mask loading + processing)

## Future Enhancements

- [ ] Parallel batch processing
- [ ] Better object type classification
- [ ] Support for more complex geometries
- [ ] Automatic world bounds estimation
- [ ] Object merging/splitting controls
- [ ] XML validation and error checking
