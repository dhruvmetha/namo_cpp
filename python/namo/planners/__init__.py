"""Planning algorithms and utilities for NAMO."""

from typing import Any, Dict, Optional, Set, Tuple

try:  # pragma: no cover - optional during static analysis
	import namo_rl  # type: ignore[import]
except ImportError:  # pragma: no cover
	namo_rl = None  # type: ignore[assignment]

from . import sampling
from . import full_namo
from .connectivity_snapshot import (
	RegionGoalBundle,
	RegionGoalSamples,
	RegionGoalSample,
	snapshot_region_connectivity,
	find_robot_label,
	restrict_to_local_regions,
)


def _clone_goal_bundle(bundle: Any) -> RegionGoalBundle:
	return RegionGoalBundle(
		goals=[RegionGoalSample(sample.x, sample.y, sample.theta) for sample in bundle.goals],
		blocking_objects=set(bundle.blocking_objects),
	)


def get_region_snapshot(
	env: Any,
	*,
	goals_per_region: int = 0,
	goal_radius: Optional[float] = None,
	local_info_only: bool = False,
	seed: int = 42,
	use_cpp_unified: bool = True,
	use_xml_goal: bool = True,
) -> Dict[str, Any]:
	"""Return unified region/connectivity snapshot for the current environment state.

	When available, this uses the C++ binding `env.get_region_snapshot(...)` to avoid
	duplicating high-level wavefront logic in Python. The Python snapshot exporter is
	kept as fallback for compatibility/debug.
	"""

	if namo_rl is None:  # pragma: no cover - defensive fallback
		raise RuntimeError("namo_rl bindings are not available on the PYTHONPATH")

	if use_cpp_unified and hasattr(env, "get_region_snapshot"):
		raw = env.get_region_snapshot(
			goals_per_region,
			(-1.0 if goal_radius is None else float(goal_radius)),
			local_info_only,
			int(seed),
			bool(use_xml_goal),
		)
		adjacency = {
			str(region): set(neighbors)
			for region, neighbors in dict(raw.get("adjacency", {})).items()
		}
		edge_objects = {
			str(region): {
				str(neighbor): set(objs)
				for neighbor, objs in dict(neighbor_map).items()
			}
			for region, neighbor_map in dict(raw.get("edge_objects", {})).items()
		}
		region_labels = {
			int(idx): str(label)
			for idx, label in dict(raw.get("region_labels", {})).items()
		}
		region_goals = {
			str(region): _clone_goal_bundle(bundle)
			for region, bundle in dict(raw.get("region_goals", {})).items()
		}
		# Boundaries no single object opens. This wrapper rebuilds the dict key by key, so a new
		# binding field is invisible here until it is named, which is how it went missing once.
		multi_object_edges = {
			str(region): set(neighbors)
			for region, neighbors in dict(raw.get("multi_object_edges", {})).items()
		}
		return {
			"adjacency": adjacency,
			"edge_objects": edge_objects,
			"multi_object_edges": multi_object_edges,
			"region_labels": region_labels,
			"region_goals": region_goals,
			"robot_label": str(raw.get("robot_label", "")),
			"goal_label": str(raw.get("goal_label", "")),
			"goal_reachable": bool(raw.get("goal_reachable", False)),
			"goal_in_free_space": bool(raw.get("goal_in_free_space", False)),
			"source": "cpp",
		}

	# Fallback to legacy Python snapshot path.
	xml_path = getattr(env, "get_xml_path", None)
	config_path = getattr(env, "get_config_path", None)
	try:
		xml_value = xml_path() if callable(xml_path) else None
		config_value = config_path() if callable(config_path) else None
	except Exception:  # pragma: no cover - defensive
		xml_value = None
		config_value = None

	if not (xml_value and config_value):
		adjacency, edge_objects, region_labels = env.get_region_connectivity()
		adjacency_py = {region: set(neighbors) for region, neighbors in adjacency.items()}
		edge_objects_py = {
			region: {neighbor: set(objs) for neighbor, objs in neighbor_map.items()}
			for region, neighbor_map in edge_objects.items()
		}
		region_labels_py = dict(region_labels)
		robot_label = find_robot_label(region_labels_py) or ""
		goal_label = ""
		for label in region_labels_py.values():
			if label == "goal":
				goal_label = label
				break
		if not goal_label:
			for label in region_labels_py.values():
				if "goal" in str(label):
					goal_label = str(label)
					break
		return {
			"adjacency": adjacency_py,
			"edge_objects": edge_objects_py,
			"region_labels": region_labels_py,
			"region_goals": {},
			"robot_label": robot_label,
			"goal_label": goal_label,
			"goal_reachable": ("goal" in robot_label) if robot_label else False,
			"goal_in_free_space": bool(goal_label) or ("goal" in robot_label if robot_label else False),
			"source": "legacy_cpp",
		}

	adjacency, edge_objects, region_labels, region_goals, _ = snapshot_region_connectivity(
		env,
		str(xml_value),
		str(config_value),
		include_snapshot=False,
		local_info_only=local_info_only,
		goals_per_region=goals_per_region,
		generate_training_data=(goals_per_region > 0),
		use_current_state=True,
	)
	robot_label = find_robot_label(region_labels) or ""
	goal_label = ""
	for label in region_labels.values():
		if label == "goal":
			goal_label = label
			break
	if not goal_label:
		for label in region_labels.values():
			if "goal" in str(label):
				goal_label = str(label)
				break
	return {
		"adjacency": adjacency,
		"edge_objects": edge_objects,
		"region_labels": region_labels,
		"region_goals": {region: _clone_goal_bundle(bundle) for region, bundle in region_goals.items()},
		"robot_label": robot_label,
		"goal_label": goal_label,
		"goal_reachable": ("goal" in robot_label) if robot_label else False,
		"goal_in_free_space": bool(goal_label) or ("goal" in robot_label if robot_label else False),
		"source": "python_snapshot",
	}


def get_region_connectivity(
	env: Any,
	*,
	use_cpp_unified: bool = True,
	local_info_only: bool = False,
) -> Tuple[
	Dict[str, Set[str]],
	Dict[str, Dict[str, Set[str]]],
	Dict[int, str],
]:
	"""Return region adjacency, boundary objects, and region labels."""
	snapshot = get_region_snapshot(
		env,
		goals_per_region=0,
		local_info_only=local_info_only,
		use_cpp_unified=use_cpp_unified,
	)
	return snapshot["adjacency"], snapshot["edge_objects"], snapshot["region_labels"]


def get_region_goal_samples(
	env: Any,
	goals_per_region: int,
	*,
	seed: int = 42,
	use_cpp_unified: bool = True,
) -> RegionGoalSamples:
	"""Sample goal poses for each non-robot region, including blocking objects to clear."""

	if goals_per_region <= 0:
		return {}

	snapshot = get_region_snapshot(
		env,
		goals_per_region=goals_per_region,
		seed=seed,
		use_cpp_unified=use_cpp_unified,
	)
	return {
		region: _clone_goal_bundle(bundle)
		for region, bundle in snapshot["region_goals"].items()
		if "robot" not in str(region).lower()
	}


__all__ = [
	"sampling",
	"full_namo",
	"get_region_snapshot",
	"get_region_connectivity",
	"get_region_goal_samples",
	"snapshot_region_connectivity",
]
