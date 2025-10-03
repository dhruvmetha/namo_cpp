"""Planning algorithms and utilities for NAMO."""

from typing import Any, Dict, Set, Tuple

try:  # pragma: no cover - optional during static analysis
	import namo_rl  # type: ignore[import]
except ImportError:  # pragma: no cover
	namo_rl = None  # type: ignore[assignment]

from . import idfs
from . import mcts
from . import sampling
from .connectivity_snapshot import snapshot_region_connectivity


def get_region_connectivity(
	env: Any,
) -> Tuple[
	Dict[str, Set[str]],
	Dict[str, Dict[str, Set[str]]],
	Dict[int, str],
]:
	"""Return region adjacency, boundary objects, and region labels from the RL environment.

	Args:
		env: Active NAMO RL environment instance. The environment state is used directly;
			callers should ensure it reflects the desired snapshot (e.g., robot/object poses).

	Returns:
		A tuple ``(adjacency, edge_objects, region_labels)`` where:

		* ``adjacency`` maps region labels to the set of neighboring region labels.
		* ``edge_objects`` maps each region to its neighbors and the set of movable objects that
		  connect them when removed.
		* ``region_labels`` maps numeric region identifiers to their human-readable labels.

	Notes:
		The helper prefers the Python snapshot exporter (``snapshot_region_connectivity``) when the
		environment exposes XML and config getters, which avoids rebuilding wavefront grids on the C++
		side. If those getters are unavailable, it falls back to the legacy C++ binding to preserve
		behavioural parity.
	"""

	if namo_rl is None:  # pragma: no cover - defensive fallback
		raise RuntimeError("namo_rl bindings are not available on the PYTHONPATH")

	xml_path = getattr(env, "get_xml_path", None)
	config_path = getattr(env, "get_config_path", None)
	try:
		xml_value = xml_path() if callable(xml_path) else None
		config_value = config_path() if callable(config_path) else None
	except Exception:  # pragma: no cover - defensive
		xml_value = None
		config_value = None

	if xml_value and config_value:
		xml_str = str(xml_value)
		config_str = str(config_value)
		adjacency, edge_objects, region_labels, _ = snapshot_region_connectivity(
			env,
			xml_str,
			config_str,
			include_snapshot=False,
		)
		return adjacency, edge_objects, region_labels

	adjacency, edge_objects, region_labels = env.get_region_connectivity()
	adjacency_py = {region: set(neighbors) for region, neighbors in adjacency.items()}
	edge_objects_py = {
		region: {neighbor: set(objs) for neighbor, objs in neighbor_map.items()}
		for region, neighbor_map in edge_objects.items()
	}
	return adjacency_py, edge_objects_py, dict(region_labels)


__all__ = [
	"idfs",
	"mcts",
	"sampling",
	"get_region_connectivity",
	"snapshot_region_connectivity",
]