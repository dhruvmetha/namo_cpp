# Q-Function Training Plan with Hydra + Lightning

## Overview
Train Q(state, action) function using MCTS data with post-action masks, following the existing learning framework pattern in `../learning`.

## Data Pipeline

### Input Format
- **State**: 11-channel 224×224 masks (pre-action + post-action)
- **Action**: Goal proposal (object_id, target_pose)  
- **Target**: Q-value from MCTS tree search

### Training Data Structure
```python
# From MCTS mask generation
{
    # Input state (11 channels, 224×224)
    'robot': robot_mask,
    'goal': robot_goal_mask,
    'movable': movable_objects_mask,
    'static': static_obstacles_mask,
    'reachable': reachable_objects_mask,
    'robot_distance': distance_field,
    'goal_distance': distance_field,
    'combined_distance': distance_field,
    'post_action_robot': post_action_robot_mask,      # NEW!
    'post_action_movable': post_action_movable_mask,  # NEW!
    'post_action_robot_distance': post_action_distance, # NEW!
    
    # Action representation
    'target_object': selected_object_mask,
    'target_goal': object_at_goal_mask,
    
    # Target Q-value
    'goal_q_value': float32,  # From MCTS tree
    
    # Metadata
    'object_id': str,
    'goal_coordinates': [x, y, theta]
}
```

## Architecture

### Q-Function Network
```python
class QFunctionNet(pl.LightningModule):
    """
    Q(s,a) network that takes state masks + action and predicts Q-value.
    Similar to existing DiT models but outputs scalar Q-value instead of masks.
    """
    def __init__(self, 
                 state_channels=11,  # 8 original + 3 post-action
                 action_channels=2,  # target_object + target_goal
                 hidden_dim=256):
        # Encoder: CNN to process state + action masks
        # Decoder: MLP to output Q-value
        
    def forward(self, state_masks, action_masks):
        # state_masks: (B, 11, 224, 224) 
        # action_masks: (B, 2, 224, 224)
        # returns: (B, 1) Q-values
```

### Loss Function
```python
loss = F.mse_loss(predicted_q, target_q_from_mcts)
```

## Implementation Plan

### Step 1: Create Data Module
```python
# namo/learning_qfunction/data/q_function_data.py
class QFunctionDataModule(pl.LightningDataModule):
    """Load MCTS mask data for Q-function training."""
    
    def __init__(self, data_dir, batch_size=32, num_workers=4):
        # Load .npz files from MCTS mask generation
        # Create train/val/test splits
        
    def prepare_data(self):
        # Scan data_dir for goal proposal .npz files
        # Extract state masks + action masks + Q-values
        
    def setup(self, stage=None):
        # Create PyTorch datasets
        
    def train_dataloader(self):
        # Return DataLoader for training
```

### Step 2: Create Q-Function Model  
```python
# namo/learning_qfunction/model/q_function_net.py
class QFunctionNet(pl.LightningModule):
    """Q(s,a) network using CNN encoder + MLP decoder."""
    
    def __init__(self, cfg):
        self.encoder = CNNEncoder(input_channels=13)  # 11 state + 2 action
        self.decoder = MLP(hidden_dim=cfg.hidden_dim, output_dim=1)
        self.lr = cfg.learning_rate
        
    def forward(self, x):
        # x: (B, 13, 224, 224) concatenated state+action masks
        features = self.encoder(x)  # (B, feature_dim)
        q_value = self.decoder(features)  # (B, 1)
        return q_value
        
    def training_step(self, batch, batch_idx):
        state_masks = batch['state_masks']  # (B, 11, 224, 224)
        action_masks = batch['action_masks']  # (B, 2, 224, 224)
        target_q = batch['target_q']  # (B, 1)
        
        # Concatenate inputs
        inputs = torch.cat([state_masks, action_masks], dim=1)  # (B, 13, 224, 224)
        
        pred_q = self.forward(inputs)
        loss = F.mse_loss(pred_q, target_q)
        
        self.log('train_loss', loss)
        return loss
        
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
```

### Step 3: Configuration Files
```yaml
# namo/learning_qfunction/config/train_q_function.yaml
defaults:
    - _self_
    - data: q_function_data
    - model: q_function_net
    - trainer: gpu

model_name: q_function_net
loss_type: mse

seed: 42
batch_size: 64
max_epochs: 100

base_lr: 0.001
gpu_id: 0

# Data paths
data_dir: ../mcts_training_masks  # Generated from MCTS pipeline

hydra:
  run:
    dir: outputs/q_function/${loss_type}/${now:%Y-%m-%d_%H-%M-%S}
```

```yaml
# namo/learning_qfunction/config/data/q_function_data.yaml
_target_: learning_qfunction.data.q_function_data.QFunctionDataModule
data_dir: ${data_dir}
batch_size: ${batch_size}
num_workers: ${num_workers}
train_split: 0.8
val_split: 0.1
test_split: 0.1
```

```yaml
# namo/learning_qfunction/config/model/q_function_net.yaml
_target_: learning_qfunction.model.q_function_net.QFunctionNet
input_channels: 13  # 11 state + 2 action
hidden_dim: 512
learning_rate: ${base_lr}
```

### Step 4: Training Script
```python
# namo/learning_qfunction/train_q_function.py
import hydra
import lightning.pytorch as pl
from pathlib import Path

@hydra.main(config_path="config", config_name="train_q_function.yaml")
def main(cfg):
    if cfg.get("seed"):
        pl.seed_everything(cfg.seed)
        
    # Data module
    data_module = hydra.utils.instantiate(cfg.data)
    
    # Model
    model = hydra.utils.instantiate(cfg.model)
    
    # Trainer
    trainer = hydra.utils.instantiate(cfg.trainer)
    
    # Train
    trainer.fit(model=model, datamodule=data_module)
    
if __name__ == "__main__":
    main()
```

## Directory Structure
```
namo/
├── learning_qfunction/           # New Q-function training module
│   ├── __init__.py
│   ├── config/
│   │   ├── train_q_function.yaml
│   │   ├── data/
│   │   │   └── q_function_data.yaml
│   │   ├── model/  
│   │   │   └── q_function_net.yaml
│   │   └── trainer/
│   │       └── default.yaml
│   ├── data/
│   │   ├── __init__.py
│   │   └── q_function_data.py
│   ├── model/
│   │   ├── __init__.py
│   │   └── q_function_net.py
│   └── train_q_function.py
│
├── mcts_training_masks/          # Generated by MCTS pipeline
│   ├── episode_*_goal_*_proposal_*.npz
│   └── ...
│
└── post_action_mask_test/        # Test images (can be removed)
    └── *.png
```

## Usage

### Step 1: Generate MCTS Training Data
```bash
# Collect MCTS episodes
python python/parallel_data_collection.py \
  --output-dir ./mcts_episodes \
  --start-idx 0 --end-idx 100 \
  --mcts-budget 100

# Generate training masks  
python python/run_mcts_mask_generation.py batch \
  --input-dir ./mcts_episodes/data_westeros \
  --output-dir ./mcts_training_masks
```

### Step 2: Train Q-Function
```bash
cd learning_qfunction
python train_q_function.py \
  data_dir=../mcts_training_masks \
  gpu_id=0 \
  batch_size=64 \
  max_epochs=100
```

### Step 3: Evaluate/Use Trained Model
```python
# Load trained Q-function
model = QFunctionNet.load_from_checkpoint("path/to/checkpoint.ckpt")

# Predict Q-values for new state-action pairs
q_value = model(state_masks, action_masks)
```

## Expected Outcomes

1. **Q-Function Learning**: Network learns to predict MCTS Q-values from visual state+action
2. **Action Selection**: Use trained Q-function for object/goal selection without MCTS
3. **Fast Planning**: Replace expensive MCTS search with learned value function
4. **Transfer Learning**: Pre-trained Q-function can bootstrap RL training

## Integration with Existing Learning Framework

This follows the exact same pattern as the existing `../learning` directory:
- **Hydra** configuration management
- **Lightning** training framework  
- **Modular** data/model/trainer structure
- **GPU** support and logging
- **Checkpoint** saving and loading

The Q-function training will integrate seamlessly with the existing ML pipeline for NAMO planning.