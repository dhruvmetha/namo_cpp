"""NAMO: Navigation Among Movable Obstacles

A comprehensive planning and data collection framework for robotic navigation
among movable rectangular objects.
"""

# Load machine-local config (<parent>/.env) so os.environ-based paths resolve no
# matter how python was launched. override=False -> explicit shell exports win.
# Silent no-op if python-dotenv isn't installed (pip install python-dotenv).
from pathlib import Path as _Path

try:
    from dotenv import load_dotenv as _load_dotenv

    _load_dotenv(_Path(__file__).resolve().parents[3] / ".env", override=False)
except ModuleNotFoundError:
    pass

from importlib import import_module
from typing import TYPE_CHECKING, Any

__version__ = "1.0.0"

__all__ = (
    "core",
    "config",
    "strategies",
    "planners",
    "data_collection",
    "visualization",
)

_LAZY_SUBMODULES = {name: f"namo.{name}" for name in __all__}


def __getattr__(name: str) -> Any:
    if name in _LAZY_SUBMODULES:
        module = import_module(_LAZY_SUBMODULES[name])
        globals()[name] = module
        return module
    raise AttributeError(f"module 'namo' has no attribute '{name}'")


def __dir__() -> list[str]:
    return sorted(list(globals().keys()) + list(__all__))


if TYPE_CHECKING:
    from . import core as core
    from . import config as config
    from . import strategies as strategies
    from . import planners as planners
    from . import data_collection as data_collection
    from . import visualization as visualization