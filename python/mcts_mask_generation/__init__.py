"""
MCTS Mask Generation for Neural Network Training

This module generates mask-based datasets from MCTS/AlphaZero data collection
for training neural networks on NAMO planning tasks.

Key Features:
- Goal Proposal Networks: P(goal|object,state) as spatial heatmaps
- Value Networks: V(s) and Q(s,a) using masks as input features  
- Compatible with existing mask_generation infrastructure
- TODO: Object Selection P(o|s) - mask approach needs design

Usage:
    from mcts_mask_generation.batch_collection import process_mcts_episode_file
    from mcts_mask_generation.mcts_visualizer import MCTSMaskGenerator
"""

__version__ = "1.0.0"
__author__ = "NAMO Research Team"

# Import main classes for convenience
from .mcts_visualizer import MCTSMaskGenerator
from .batch_collection import process_mcts_episode_batch

__all__ = [
    "MCTSMaskGenerator",
    "process_mcts_episode_batch"
]