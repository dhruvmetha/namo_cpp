# IDFS Data Pipeline Documentation

This document provides complete instructions for running the IDFS (Iterative Deepening First Search) data collection and mask generation pipeline for neural network training.

## Overview

The IDFS pipeline consists of three main components:

1. **Modular Parallel Collection**: Parallel IDFS episode data collection with configurable planners
2. **Sequential ML Collection**: Sequential data collection for ML training
3. **Mask Generation**: Convert collected episodes to mask-based training datasets

## Component 1: Modular Parallel Collection

### Purpose
Collect IDFS episode data in parallel across multiple environments using configurable planning algorithms.

### Script Location
```
python/namo/data_collection/modular_parallel_collection.py
```

### Basic Usage
```bash
PYTHONPATH=/common/home/dm1487/robotics_research/ktamp/namo/build_python_mjxrl_westeros \
/common/users/dm1487/envs/mjxrl/bin/python python/namo/data_collection/modular_parallel_collection.py \
--output-dir ./idfs_data \
--start-idx 0 \
--end-idx 50 \
--workers 8 \
--episodes-per-env 3 \
--planner standard_idfs \
--max-depth 5
```

### Key Parameters

| Parameter | Description | Default | Options |
|-----------|-------------|---------|---------|
| `--output-dir` | Output directory for collected data | Required | Any valid path |
| `--start-idx` | Starting environment index | Required | 0+ |
| `--end-idx` | Ending environment index | Required | > start-idx |
| `--workers` | Number of parallel worker processes | 4 | 1-32 |
| `--episodes-per-env` | Episodes to collect per environment | 1 | 1-10 |
| `--planner` | Planning algorithm to use | standard_idfs | See below |
| `--max-depth` | Maximum search depth | 5 | 1-15 |
| `--max-goals-per-object` | Max goals to try per object | 5 | 1-20 |
| `--object-strategy` | Object selection strategy | no_heuristic | See below |
| `--goal-strategy` | Goal selection strategy | random | See below |

### Available Planners
- `standard_idfs` - Standard iterative deepening DFS
- `tree_idfs` - Tree-based iterative deepening DFS
- `optimal_idfs` - Optimal iterative deepening DFS
- `random_sampling` - Random sampling baseline

### Object Selection Strategies
- `no_heuristic` - Random object selection
- `nearest_first` - Select nearest objects first
- `goal_proximity` - Select objects closest to robot goal
- `farthest_first` - Select farthest objects first
- `ml` - ML-based object selection (requires model)

### Goal Selection Strategies
- `random` - Random goal sampling
- `grid` - Grid-based goal sampling
- `adaptive` - Adaptive goal sampling
- `ml` - ML-based goal selection (requires model)

### Example Commands

**Small Test Collection:**
```bash
PYTHONPATH=/common/home/dm1487/robotics_research/ktamp/namo/build_python_mjxrl_westeros \
/common/users/dm1487/envs/mjxrl/bin/python python/namo/data_collection/modular_parallel_collection.py \
--output-dir ./test_idfs_data \
--start-idx 0 \
--end-idx 10 \
--workers 4 \
--episodes-per-env 1 \
--planner standard_idfs \
--max-depth 3
```

**Production Collection with ML Strategies:**
```bash
PYTHONPATH=/common/home/dm1487/robotics_research/ktamp/namo/build_python_mjxrl_westeros \
/common/users/dm1487/envs/mjxrl/bin/python python/namo/data_collection/modular_parallel_collection.py \
--output-dir ./idfs_ml_data \
--start-idx 0 \
--end-idx 100 \
--workers 12 \
--episodes-per-env 5 \
--planner optimal_idfs \
--max-depth 8 \
--object-strategy ml \
--goal-strategy ml \
--ml-object-model-path /path/to/object_model.pth \
--ml-goal-model-path /path/to/goal_model.pth
```

**Comparison Study (Multiple Planners):**
```bash
# Standard IDFS
PYTHONPATH=/common/home/dm1487/robotics_research/ktamp/namo/build_python_mjxrl_westeros \
/common/users/dm1487/envs/mjxrl/bin/python python/namo/data_collection/modular_parallel_collection.py \
--output-dir ./comparison_standard \
--start-idx 0 --end-idx 50 --workers 8 \
--planner standard_idfs --max-depth 5

# Optimal IDFS
PYTHONPATH=/common/home/dm1487/robotics_research/ktamp/namo/build_python_mjxrl_westeros \
/common/users/dm1487/envs/mjxrl/bin/python python/namo/data_collection/modular_parallel_collection.py \
--output-dir ./comparison_optimal \
--start-idx 0 --end-idx 50 --workers 8 \
--planner optimal_idfs --max-depth 5

# Random Sampling Baseline
PYTHONPATH=/common/home/dm1487/robotics_research/ktamp/namo/build_python_mjxrl_westeros \
/common/users/dm1487/envs/mjxrl/bin/python python/namo/data_collection/modular_parallel_collection.py \
--output-dir ./comparison_random \
--start-idx 0 --end-idx 50 --workers 8 \
--planner random_sampling --max-depth 5
```

## Component 2: Sequential ML Collection

### Purpose
Sequential data collection specifically designed for ML training with careful episode management.

### Script Location
```
python/namo/data_collection/sequential_ml_collection.py
```

### Basic Usage
```bash
PYTHONPATH=/common/home/dm1487/robotics_research/ktamp/namo/build_python_mjxrl_westeros \
/common/users/dm1487/envs/mjxrl/bin/python python/namo/data_collection/sequential_ml_collection.py \
--output-dir ./ml_training_data \
--start-idx 0 \
--end-idx 100 \
--episodes-per-env 3 \
--planner standard_idfs
```

### Key Parameters

| Parameter | Description | Default | Notes |
|-----------|-------------|---------|-------|
| `--output-dir` | Output directory | Required | For ML training data |
| `--start-idx` | Starting environment index | Required | |
| `--end-idx` | Ending environment index | Required | |
| `--episodes-per-env` | Episodes per environment | 1 | For data diversity |
| `--planner` | Planning algorithm | standard_idfs | Same options as modular |
| `--max-depth` | Maximum search depth | 5 | |
| `--collect-failures` | Include failed episodes | False | For failure analysis |
| `--ml-ready` | Format for ML training | True | Structured for neural nets |

### Example Commands

**ML Training Data Collection:**
```bash
PYTHONPATH=/common/home/dm1487/robotics_research/ktamp/namo/build_python_mjxrl_westeros \
/common/users/dm1487/envs/mjxrl/bin/python python/namo/data_collection/sequential_ml_collection.py \
--output-dir ./ml_dataset \
--start-idx 0 \
--end-idx 200 \
--episodes-per-env 5 \
--planner optimal_idfs \
--max-depth 6 \
--ml-ready \
--collect-failures
```

**Failure Analysis Collection:**
```bash
PYTHONPATH=/common/home/dm1487/robotics_research/ktamp/namo/build_python_mjxrl_westeros \
/common/users/dm1487/envs/mjxrl/bin/python python/namo/data_collection/sequential_ml_collection.py \
--output-dir ./failure_analysis \
--start-idx 0 \
--end-idx 50 \
--episodes-per-env 10 \
--planner standard_idfs \
--max-depth 3 \
--collect-failures
```

## Component 3: Mask Generation

### Purpose
Convert collected IDFS episode data to mask-based training datasets for neural networks.

### Script Location
```
python/namo/visualization/run_mask_generation.py
```

### Basic Usage
```bash
python python/namo/visualization/run_mask_generation.py batch \
--input-dir ./idfs_data \
--output-dir ./idfs_training_masks \
--workers 8
```

### Key Parameters

| Parameter | Description | Options |
|-----------|-------------|---------|
| `batch` | Batch processing mode | Required first argument |
| `--input-dir` | Directory with episode .pkl files | Required |
| `--output-dir` | Output directory for .npz mask files | Required |
| `--workers` | Number of parallel workers | Default: 4 |
| `--episode-filter` | Filter specific episodes | Optional |

### Example Commands

**Basic Mask Generation:**
```bash
python python/namo/visualization/run_mask_generation.py batch \
--input-dir ./idfs_data/data_westeros \
--output-dir ./idfs_masks \
--workers 8
```

**Large Scale Processing:**
```bash
python python/namo/visualization/run_mask_generation.py batch \
--input-dir ./large_idfs_dataset/data_westeros \
--output-dir ./large_idfs_masks \
--workers 16
```

## Complete Pipeline Examples

### Example 1: Basic IDFS Pipeline
```bash
# Step 1: Collect IDFS data
PYTHONPATH=/common/home/dm1487/robotics_research/ktamp/namo/build_python_mjxrl_westeros \
/common/users/dm1487/envs/mjxrl/bin/python python/namo/data_collection/modular_parallel_collection.py \
--output-dir ./basic_idfs \
--start-idx 0 --end-idx 50 --workers 8 \
--planner standard_idfs --max-depth 5

# Step 2: Generate training masks
python python/namo/visualization/run_mask_generation.py batch \
--input-dir ./basic_idfs/data_westeros \
--output-dir ./basic_idfs_masks \
--workers 8
```

### Example 2: ML Training Pipeline
```bash
# Step 1: Sequential ML collection
PYTHONPATH=/common/home/dm1487/robotics_research/ktamp/namo/build_python_mjxrl_westeros \
/common/users/dm1487/envs/mjxrl/bin/python python/namo/data_collection/sequential_ml_collection.py \
--output-dir ./ml_training \
--start-idx 0 --end-idx 100 --episodes-per-env 5 \
--planner optimal_idfs --ml-ready

# Step 2: Generate training masks
python python/namo/visualization/run_mask_generation.py batch \
--input-dir ./ml_training \
--output-dir ./ml_training_masks \
--workers 12
```

### Example 3: Comparative Study Pipeline
```bash
# Collect data for different planners
for planner in standard_idfs optimal_idfs random_sampling; do
    PYTHONPATH=/common/home/dm1487/robotics_research/ktamp/namo/build_python_mjxrl_westeros \
    /common/users/dm1487/envs/mjxrl/bin/python python/namo/data_collection/modular_parallel_collection.py \
    --output-dir ./study_${planner} \
    --start-idx 0 --end-idx 100 --workers 10 \
    --planner ${planner} --max-depth 5

    # Generate masks for each
    python python/namo/visualization/run_mask_generation.py batch \
    --input-dir ./study_${planner}/data_westeros \
    --output-dir ./study_${planner}_masks \
    --workers 8
done
```

## Output Data Formats

### Episode Data (.pkl files)
Each episode file contains:
```python
{
    'episode_id': 'westeros_env_000001_episode_0',
    'success': True,
    'planner': 'standard_idfs',
    'total_steps': 4,
    'xml_file': '../ml4kp_ktamp/resources/models/...',
    'search_statistics': {
        'nodes_expanded': 45,
        'terminal_checks': 120,
        'search_time_ms': 1500.0,
        'max_depth_reached': 4
    },
    'step_data': [
        {
            'scene_observation': {...},  # Object poses
            'robot_goal': [x, y, theta],
            'selected_object': 'obstacle_1_movable',
            'selected_goal': [x, y, theta],
            'reachable_objects': [...],
            'search_info': {...}
        },
        # ... more steps
    ],
    'final_action_sequence': [...],
    'failure_info': {...}  # If failed
}
```

### Mask Data (.npz files)
Generated mask files contain:
```python
# Scene masks (224x224 each)
'robot': robot_position_mask
'goal': robot_goal_mask
'movable': movable_objects_mask
'static': static_objects_mask
'reachable': reachable_objects_mask

# Action masks
'selected_object': target_object_mask
'selected_goal': object_at_goal_mask

# Metadata
'success': episode_success
'planner': algorithm_used
'step_info': {...}
```

## Performance Tips

1. **Worker Count**: Use `CPU cores - 4` for optimal performance
2. **Batch Size**: Process 50-100 environments per batch
3. **Memory**: Monitor memory usage with many workers
4. **Storage**: Ensure sufficient disk space for masks (~10MB per episode)
5. **Network**: Avoid network storage for temporary files

## Troubleshooting

### Common Issues

**Import Errors:**
```bash
# Ensure PYTHONPATH is set correctly
export PYTHONPATH=/common/home/dm1487/robotics_research/ktamp/namo/build_python_mjxrl_westeros:$PYTHONPATH

# Check namo_rl module
python -c "import namo_rl; print('OK')"
```

**Environment Errors:**
```bash
# Check environment file paths
ls ../ml4kp_ktamp/resources/models/custom_walled_envs/

# Verify XML files exist
find ../ml4kp_ktamp -name "*.xml" | head -5
```

**Worker Issues:**
```bash
# Reduce workers if memory issues
--workers 4

# Use single worker for debugging
--workers 1
```

## Integration with Neural Network Training

### Data Loading Example
```python
import numpy as np
from pathlib import Path

def load_idfs_training_data(mask_dir):
    """Load IDFS mask data for neural network training."""
    mask_files = list(Path(mask_dir).glob("*.npz"))

    samples = []
    for file_path in mask_files:
        data = np.load(file_path)

        # Stack scene masks as input channels
        scene_input = np.stack([
            data['robot'],
            data['movable'],
            data['static'],
            data['goal']
        ])  # Shape: (4, 224, 224)

        # Action masks as supervision
        action_target = np.stack([
            data['selected_object'],
            data['selected_goal']
        ])  # Shape: (2, 224, 224)

        sample = {
            'scene': scene_input,
            'action': action_target,
            'success': data['success'],
            'planner': str(data['planner'])
        }
        samples.append(sample)

    return samples

# Usage
training_data = load_idfs_training_data('./idfs_masks')
print(f"Loaded {len(training_data)} IDFS training samples")
```

This pipeline provides a complete workflow from IDFS planning to neural network training data, enabling you to train models on high-quality planning demonstrations.