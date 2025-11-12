"""
Mask Reversal System - Convert NPZ mask files back to MuJoCo XML environments.

This module provides tools to reverse the mask generation process, reconstructing
3D scene descriptions from 2D mask images.
"""

from .npz_loader import NPZLoader, MaskData
from .mask_analyzer import MaskAnalyzer, DetectedObject
from .scene_reconstructor import SceneReconstructor, ReconstructedScene
from .xml_generator import XMLGenerator
from .reverse_masks import reverse_masks_to_xml, batch_reverse_masks

__all__ = [
    'NPZLoader',
    'MaskData',
    'MaskAnalyzer', 
    'DetectedObject',
    'SceneReconstructor',
    'ReconstructedScene',
    'XMLGenerator',
    'reverse_masks_to_xml',
    'batch_reverse_masks'
]
