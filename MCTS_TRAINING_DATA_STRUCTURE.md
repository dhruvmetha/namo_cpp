# MCTS Training Data Structure

## Overview
The MCTS data collection pipeline generates two types of training datasets for neural networks:
1. **Goal Proposal Dataset** - For training diffusion models to generate object goal positions
2. **Value Network Dataset** - For training state/action value estimators

## File Organization

```
{episode_id}_step_{step}_goal_{object_id}_proposal_{i}.npz  # Goal proposal samples
{episode_id}_step_{step}_value.npz                         # Value network samples
```

## 🎯 Goal Proposal Dataset (Diffusion Model Training)

**Purpose**: Train P(goal_position | state, object) using diffusion models

### Input Features (11 channels, 224×224 each):
```python
# Pre-Action Scene State
'robot'           : (224, 224) float32  # Robot position mask
'goal'            : (224, 224) float32  # Robot goal position mask  
'movable'         : (224, 224) float32  # All movable objects mask
'static'          : (224, 224) float32  # Static walls/obstacles mask
'reachable'       : (224, 224) float32  # Reachable objects mask
'robot_distance'  : (224, 224) float32  # Distance field from robot (wavefront)
'goal_distance'   : (224, 224) float32  # Distance field from goal
'combined_distance': (224, 224) float32  # Combined distance field

# Post-Action State (NEW!)
'post_action_robot'    : (224, 224) float32  # Robot position after push action
'post_action_movable'  : (224, 224) float32  # Objects after push action  
'post_action_robot_distance': (224, 224) float32  # Distance field from new robot position
```

### Target Output:
```python
'target_goal'     : (224, 224) float32  # Object at proposed goal position
```

### MCTS Supervision Signals:
```python
'goal_probability': float32             # P(goal) from MCTS visit distribution  
'goal_q_value'    : float32             # Q(state,goal) from MCTS tree evaluation
'goal_visit_count': int32               # Raw MCTS visit count
```

### Metadata:
```python
'object_id'       : str                 # Object being manipulated
'proposal_index'  : int                 # Goal proposal index (0,1,2,...)
'goal_coordinates': (3,) float32        # Goal pose (x,y,theta) for reference
```

## 💰 Value Network Dataset

**Purpose**: Train V(state) and Q(state,object) estimators

### Input Features:
Same 11 channels as goal proposal dataset (includes post-action masks)

### Target Outputs:
```python
'state_value'     : float32             # V(state) from MCTS root node
'object_q_values' : dict                # Q(state,object) per object from MCTS
```

## Neural Network Architectures

### 🌊 Diffusion Model (Goal Proposal)
- **Input**: 8 × 224 × 224 scene representation
- **Output**: 1 × 224 × 224 goal mask
- **Training**: MSE loss with MCTS probability weighting
- **Use Case**: Generate spatial goal positions for object manipulation

### 🧠 Value Networks
- **Input**: 8 × 224 × 224 scene representation  
- **Output**: Scalar state value + per-object Q-values
- **Training**: MSE against MCTS estimates
- **Use Case**: State evaluation and object selection

## Training Data Statistics

From our test collection:
- **Total files generated**: 10 per episode step
- **Goal proposals**: 9 files (3 objects × 3 proposals each)
- **Value samples**: 1 file per step
- **Image resolution**: 224×224 pixels
- **Data type**: float32 (memory efficient)
- **MCTS statistics**: Probabilities, Q-values, visit counts included

## Loading Data Example

```python
import numpy as np

# Load goal proposal sample
data = np.load('episode_step_00_goal_obstacle_1_movable_proposal_0.npz')

# Input features for neural network
input_masks = np.stack([
    data['robot'], data['goal'], data['movable'], data['static'],
    data['reachable'], data['robot_distance'], data['goal_distance'], 
    data['combined_distance'], data['post_action_robot'],
    data['post_action_movable'], data['post_action_robot_distance']
])  # Shape: (11, 224, 224)

# Target for diffusion model
target_goal = data['target_goal']  # Shape: (224, 224)

# MCTS supervision
probability = data['goal_probability']  # Scalar
q_value = data['goal_q_value']         # Scalar
```

## Key Features

✅ **No XML Dependency**: All environment information captured in `static_object_info`  
✅ **MCTS Statistics**: Rich supervision signals from tree search  
✅ **Spatial Representation**: 224×224 masks compatible with vision models  
✅ **Multiple Proposals**: Captures MCTS exploration across different goals  
✅ **Complete Pipeline**: From MCTS tree → neural network training data