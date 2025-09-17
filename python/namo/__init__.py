"""NAMO: Navigation Among Movable Obstacles

A comprehensive planning and data collection framework for robotic navigation
among movable rectangular objects.
"""

from . import core
from . import config
from . import strategies
from . import planners
from . import data_collection
from . import visualization

__version__ = "1.0.0"
__all__ = [
    "core",
    "config",
    "strategies",
    "planners",
    "data_collection",
    "visualization"
]