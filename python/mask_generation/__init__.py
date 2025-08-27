"""
NAMO Mask Generation Package

This package provides tools for generating visualization masks and datasets from NAMO planning data.

Main components:
- visualizer: Core visualization and mask generation functionality
- batch_collection: Batch processing pipeline for large datasets
- examples: Example usage scripts

Usage:
    from mask_generation import NAMODataVisualizer
    from mask_generation.batch_collection import main as run_batch_collection
"""

from .visualizer import NAMODataVisualizer, NAMOXMLParser

__all__ = ['NAMODataVisualizer', 'NAMOXMLParser']
__version__ = '1.0.0'