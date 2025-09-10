# MCTS Data Pipeline Documentation

This document provides complete instructions for the MCTS data collection and mask generation pipeline for neural network training.

## Overview

The pipeline consists of two main stages:

1. **MCTS Data Collection**: Generate raw MCTS episode data with tree search statistics
2. **Mask Generation**: Convert MCTS episodes to mask-based training datasets for neural networks

## Stage 1: MCTS Data Collection

### Purpose
Collect MCTS episode data containing:
- Scene observations per planning step
- MCTS tree statistics (visit counts, Q-values)  
- Object proposals and goal proposals
- Episode outcomes and action sequences

### Commands

**Small Test Collection (10 environments):**
```bash
PYTHONPATH=/common/home/dm1487/robotics_research/ktamp/namo/build_python_mjxrl_westeros \
/common/users/dm1487/envs/mjxrl/bin/python python/parallel_data_collection.py \
--output-dir ./test_mcts_data \
--start-idx 0 \
--end-idx 10 \
--workers 4 \
--episodes-per-env 1 \
--mcts-budget 30 \
--max-steps 5
```

**Medium Production Collection (100 environments):**
```bash
PYTHONPATH=/common/home/dm1487/robotics_research/ktamp/namo/build_python_mjxrl_westeros \
/common/users/dm1487/envs/mjxrl/bin/python python/parallel_data_collection.py \
--output-dir ./mcts_episodes \
--start-idx 0 \
--end-idx 100 \
--workers 12 \
--episodes-per-env 3 \
--mcts-budget 100 \
--max-steps 10
```

**Large Scale Collection (1000+ environments):**
```bash
PYTHONPATH=/common/home/dm1487/robotics_research/ktamp/namo/build_python_mjxrl_westeros \
/common/users/dm1487/envs/mjxrl/bin/python python/parallel_data_collection.py \
--output-dir ./large_mcts_dataset \
--start-idx 0 \
--end-idx 1000 \
--workers 20 \
--episodes-per-env 5 \
--mcts-budget 200 \
--max-steps 15
```

### Key Parameters

| Parameter | Description | Recommendation |
|-----------|-------------|----------------|
| `--mcts-budget` | Simulations per planning step | 100-200 for production |
| `--episodes-per-env` | Episodes per environment | 3-5 for good coverage |
| `--max-steps` | Max planning steps per episode | 10-15 |
| `--workers` | Parallel processes | CPU cores - 4 |

### Output Structure
```
output_dir/
├── data_westeros/                           # Hostname-based subdirectory
│   ├── episode_westeros_env_000001.pkl     # MCTS episode data
│   ├── episode_westeros_env_000002.pkl
│   ├── ...
│   ├── collection_progress.txt              # Live progress
│   └── collection_summary_westeros.pkl     # Final statistics
```

### Episode Data Format
Each `.pkl` file contains:
```python
{
    'episode_id': 'westeros_env_000001_episode_0',
    'success': True,
    'total_steps': 3,
    'xml_file': '../ml4kp_ktamp/resources/models/...',
    'step_data': [                           # One per planning step
        {
            'scene_observation': {           # Object poses
                'robot_pose': [x, y, theta],
                'obstacle_1_movable_pose': [x, y, theta],
                'obstacle_2_movable_pose': [x, y, theta],
                # ...
            },
            'robot_goal': [x, y, theta],
            'object_proposals': [            # Object selection proposals
                {
                    'object_id': 'obstacle_1_movable',
                    'probability': 0.65,
                    'visit_count': 35,
                    'q_value': 0.82
                },
                # ...
            ],
            'goal_proposals': {              # Goal proposals per object
                'obstacle_1_movable': [
                    {
                        'goal_position': [x, y, theta],
                        'probability': 0.4,
                        'visit_count': 20,
                        'q_value': 0.85
                    },
                    # ...
                ]
            },
            'state_value': 0.42,             # V(s) from MCTS root
            'object_q_values': {             # Q(s,object) values
                'obstacle_1_movable': 0.85,
                'obstacle_2_movable': -0.23,
                # ...
            },
            'reachable_objects': ['obstacle_1_movable', ...],
        },
        # ... more planning steps
    ],
    'final_action_sequence': [               # Actions executed
        {
            'step': 0,
            'object_id': 'obstacle_1_movable',
            'target': [x, y, theta],
            'reward': 1.0,
            'post_action_poses': {               # SE(2) poses after action
                'robot_pose': [x, y, theta],
                'obstacle_1_movable_pose': [x, y, theta],
                'obstacle_2_movable_pose': [x, y, theta],
                # ... only movable objects and robot
            }
        },
        # ...
    ]
}
```

## Stage 2: Mask Generation for Neural Networks

### Purpose
Convert MCTS episode data to mask-based training datasets:
- **Goal Proposal Data**: For diffusion models learning P(goal|object,state)
- **Value Network Data**: For networks learning V(s) and Q(s,a)

### Commands

**Generate All Mask Types:**
```bash
python run_mcts_mask_generation.py batch \
  --input-dir ./mcts_episodes/data_westeros \
  --output-dir ./mcts_training_masks \
  --workers 8
```

**Goal Proposal Data Only (for diffusion models):**
```bash
python run_mcts_mask_generation.py batch \
  --input-dir ./mcts_episodes/data_westeros \
  --output-dir ./diffusion_training_data \
  --goal-proposal-only \
  --workers 8
```

**Value Network Data Only:**
```bash
python run_mcts_mask_generation.py batch \
  --input-dir ./mcts_episodes/data_westeros \
  --output-dir ./value_network_data \
  --value-network-only \
  --workers 8
```

**Debug Mode (single-threaded):**
```bash
python run_mcts_mask_generation.py batch \
  --input-dir ./mcts_episodes/data_westeros \
  --output-dir ./debug_masks \
  --serial
```

### Generated Training Data

#### Goal Proposal Files (for Diffusion Models)
**File Pattern**: `episode_X_step_Y_goal_OBJECT_proposal_Z.npz`

Each file contains one goal proposal training sample:
```python
# Input features (224x224 masks)
'robot': robot_mask                  # Robot position
'goal': robot_goal_mask             # Target robot position  
'movable': movable_objects_mask     # All movable objects
'static': static_objects_mask       # Static obstacles
'reachable': reachable_objects_mask # Reachable objects
'target_object': selected_object_mask # Object being pushed
'target_goal': object_at_goal_mask  # Object at proposed goal
'robot_distance': distance_field    # Distance from robot
'goal_distance': distance_field     # Distance to robot goal

# Target mask (what diffusion model learns to generate)
'target_goal': target_goal_mask     # Object at proposed goal position (224x224)

# MCTS quality signals (for conditional diffusion)
'goal_probability': 0.4             # MCTS visit probability
'goal_q_value': 0.85               # MCTS Q-value
'goal_visit_count': 20             # Raw visit count

# Metadata
'object_id': "obstacle_1_movable"
'robot_goal': [x, y, theta]
'world_bounds': [xmin, ymin, xmax, ymax]
'proposal_index': 0
```

#### Value Network Files  
**File Pattern**: `episode_X_step_Y_value.npz`

Each file contains value training data for one planning step:
```python
# Input features (224x224 masks) 
'robot': robot_mask                 # Robot position
'goal': robot_goal_mask            # Target robot position
'movable': movable_objects_mask    # All movable objects  
'static': static_objects_mask      # Static obstacles
'reachable': reachable_objects_mask # Reachable objects
'robot_distance': distance_field   # Distance from robot
'goal_distance': distance_field    # Distance to robot goal

# Action-specific masks (when available)
'target_object': selected_object_mask # Selected object
'target_goal': object_at_goal_mask    # Object at selected goal

# Value targets
'state_value': 0.42                # V(s) target from MCTS root
'object_q_values': {               # Q(s,object) targets
    'obstacle_1_movable': 0.85,
    'obstacle_2_movable': -0.23,
    # ...
}

# Action info (when available)
'selected_object_id': "obstacle_1_movable"
'selected_goal': [x, y, theta]

# Metadata  
'robot_goal': [x, y, theta]
'world_bounds': [xmin, ymin, xmax, ymax]
```

### Output Statistics Example
```
🎉 MCTS mask generation complete!
✅ Successfully processed: 45
❌ Failed: 5

📊 Generated datasets:
   Goal proposal files: 432        # Multiple proposals per step
   Value network files: 124        # One per planning step  
   Total .npz files: 556
```

## Data Loading for Training

### Goal Proposal Diffusion Model
```python
import numpy as np
from pathlib import Path

def load_goal_proposal_data(data_dir):
    """Load goal proposal data for diffusion training."""
    data_files = list(Path(data_dir).glob("*_goal_*_proposal_*.npz"))
    
    samples = []
    for file_path in data_files:
        data = np.load(file_path)
        
        sample = {
            # Input masks (stack as channels)
            'input_masks': np.stack([
                data['robot'],
                data['target_object'], 
                data['movable'],
                data['static'],
                data['goal']
            ]),  # Shape: (5, 224, 224)
            
            # Target mask (what diffusion learns to generate)
            'target_goal': data['target_goal'],  # Shape: (224, 224)
            
            # Optional: conditioning signals
            'conditioning': np.array([
                data['goal_q_value'],
                data['goal_probability']
            ])  # Shape: (2,)
        }
        samples.append(sample)
    
    return samples

# Usage
training_data = load_goal_proposal_data('./diffusion_training_data')
print(f"Loaded {len(training_data)} goal proposal samples")
```

### Value Network Training
```python
def load_value_network_data(data_dir):
    """Load value network data for V(s) and Q(s,a) training.""" 
    data_files = list(Path(data_dir).glob("*_value.npz"))
    
    samples = []
    for file_path in data_files:
        data = np.load(file_path)
        
        sample = {
            # Input masks
            'scene_masks': np.stack([
                data['robot'],
                data['movable'],
                data['static'],
                data['goal']
            ]),  # Shape: (4, 224, 224)
            
            # Value targets
            'state_value': data['state_value'],  # V(s) target
            'object_q_values': dict(data['object_q_values'].item()),  # Q(s,obj) targets
        }
        
        # Add action-specific data if available
        if 'target_object' in data:
            sample['action_masks'] = np.stack([
                data['target_object'],
                data['target_goal']
            ])  # Shape: (2, 224, 224)
            
        samples.append(sample)
    
    return samples

# Usage  
value_data = load_value_network_data('./value_network_data')
print(f"Loaded {len(value_data)} value network samples")
```

## Complete Pipeline Example

```bash
# Step 1: Generate MCTS episodes (100 environments)
PYTHONPATH=/common/home/dm1487/robotics_research/ktamp/namo/build_python_mjxrl_westeros \
/common/users/dm1487/envs/mjxrl/bin/python python/parallel_data_collection.py \
--output-dir ./my_mcts_dataset \
--start-idx 0 \
--end-idx 100 \
--workers 12 \
--episodes-per-env 3 \
--mcts-budget 150

# Step 2: Generate training masks for diffusion model
python run_mcts_mask_generation.py batch \
  --input-dir ./my_mcts_dataset/data_westeros \
  --output-dir ./diffusion_training_masks \
  --goal-proposal-only \
  --workers 8

# Step 3: Train diffusion model
python train_diffusion_model.py \
  --data-dir ./diffusion_training_masks \
  --output-dir ./trained_models
```

## Troubleshooting

**Common Issues:**

1. **"No files found matching pattern"**
   - Check `--input-dir` path is correct
   - Ensure `.pkl` files exist in the directory

2. **"XML file not found"** 
   - Check XML paths in episode data are valid
   - Verify `../ml4kp_ktamp/resources/models/` exists

3. **Memory issues**
   - Reduce `--workers` 
   - Use `--serial` flag for debugging

4. **Import errors**
   - Ensure `PYTHONPATH` is set correctly
   - Check that `namo_rl` module is built

**Performance Tips:**

- Use `--workers` = CPU cores - 4 for optimal performance
- Set `--mcts-budget` higher (200+) for better quality data
- Use `--goal-proposal-only` if only training diffusion models
- Monitor disk space - mask files can be large

## File Structure Summary

```
project_root/
├── MCTS Episodes (Stage 1 output)
│   └── data_westeros/
│       ├── episode_*.pkl              # Raw MCTS data
│       └── collection_summary_*.pkl   # Statistics
│
├── Training Masks (Stage 2 output)  
│   ├── *_goal_*_proposal_*.npz        # Goal proposal training data
│   ├── *_value.npz                    # Value network training data
│   └── generation_stats.txt           # Generation statistics
│
└── Scripts
    ├── python/parallel_data_collection.py  # Stage 1: MCTS data collection
    ├── python/run_mcts_mask_generation.py  # Stage 2: Mask generation
    └── MCTS_DATA_PIPELINE.md               # This documentation
```

This pipeline provides a complete path from MCTS tree search to neural network training data, enabling you to train diffusion models and value networks on high-quality planning data.

## Single Trial with Tree Visualization

For debugging, understanding MCTS behavior, or demonstrating the system, you can run single trials with live tree visualization.

### Single-Step MCTS Planning

**Basic visualization:**
```bash
PYTHONPATH=/common/home/dm1487/robotics_research/ktamp/namo/build_python_mjxrl_westeros \
/common/users/dm1487/envs/mjxrl/bin/python python/test_clean_mcts.py \
--visualize-tree \
--budget 50
```

**With custom parameters:**
```bash
PYTHONPATH=/common/home/dm1487/robotics_research/ktamp/namo/build_python_mjxrl_westeros \
/common/users/dm1487/envs/mjxrl/bin/python python/test_clean_mcts.py \
--visualize-tree \
--budget 100 \
--k 2.0 \
--alpha 0.5 \
--xml /path/to/custom/environment.xml
```

### Multi-Step MCTS Planning

**Complete episode with visualization:**
```bash
PYTHONPATH=/common/home/dm1487/robotics_research/ktamp/namo/build_python_mjxrl_westeros \
/common/users/dm1487/envs/mjxrl/bin/python python/test_multi_step_mcts.py \
--visualize-tree \
--budget 50 \
--max-steps 10
```

### Tree Visualization Features

The live tree visualization shows:

**🏠 ROOT STATE**: The current planning state
- **V**: Visit count for this state
- **Q**: Q-value (expected reward)
- **Objects**: Number of reachable objects

**🥇 OBJ object_name**: Object selection nodes
- **V**: Visit count (how often MCTS selected this object)
- **Q**: Q-value (expected reward from pushing this object)
- **Goals**: Number of goal proposals for this object

**🎯 GOAL (x, y)**: Goal proposal nodes
- **V**: Visit count (how often MCTS tried this goal)
- **Q**: Q-value (expected reward from this goal)

### Example Output

```
🌳 Clean 2-Level MCTS - Iteration 20
└── 🏠 ROOT STATE (V:20, Q:-0.020, Objects:3)
    ├── 🥇 OBJ obstacle_1_movable (V:16, Q:0.291, Goals:11)
    │   ├── 🎯 GOAL (0.65, 1.72) (V:2, Q:1.000)
    │   ├── 📍 GOAL (0.71, 2.24) (V:2, Q:1.000) 
    │   └── 📍 GOAL (0.63, 2.22) (V:2, Q:1.000)
    ├── 🥈 OBJ obstacle_2_movable (V:2, Q:-1.267, Goals:2)
    │   ├── 🎯 GOAL (-0.23, 0.86) (V:1, Q:-1.267)
    │   └── 📍 GOAL (-1.80, 0.43) (V:1, Q:-1.267)
    └── 🥉 OBJ obstacle_3_movable (V:2, Q:-1.267, Goals:2)
        ├── 🎯 GOAL (-0.48, 1.20) (V:1, Q:-1.267)
        └── 📍 GOAL (1.13, 0.36) (V:1, Q:-1.267)
```

**Interpretation:**
- **obstacle_1_movable** is the most promising object (highest Q-value: 0.291)
- **GOAL (0.65, 1.72)** is the best goal for obstacle_1 (Q-value: 1.000) 
- Goals with Q-value 1.000 lead to successful robot goal reaching
- Goals with Q-value -1.267 lead to failure

### Available Parameters

| Parameter | Description | Default | Range |
|-----------|-------------|---------|--------|
| `--budget` | MCTS simulations per planning step | 30-50 | 10-500 |
| `--k` | Progressive widening constant | 2.0-3.0 | 1.0-5.0 |
| `--alpha` | Progressive widening exponent | 0.5 | 0.0-1.0 |
| `--max-steps` | Max planning steps (multi-step only) | 10 | 1-20 |
| `--rollout-steps` | Max random rollout steps | 3-5 | 0-10 |

### Use Cases

**🔧 Debugging**: Understand why MCTS makes certain decisions
**🎓 Learning**: See how progressive widening expands the tree
**🎥 Demos**: Show MCTS decision-making process to others
**⚙️ Tuning**: Find optimal k, alpha, budget parameters
**🧪 Testing**: Verify MCTS behavior on specific environments

The tree visualization provides deep insights into the MCTS decision-making process and is invaluable for understanding how the system works.