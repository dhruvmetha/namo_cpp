"""Serializable success criteria carried across simulated and physical pushes."""

from typing import Any, Mapping


def keyhole_is_open(env: Any, target: Mapping[str, Any]) -> bool:
    """Grade a frozen target in simulator coordinates; the final goal also ends it.

    Region labels may change after a push. Coordinates and object identities
    define the target instead. Callers translate object IDs when rebuilding XML.
    """
    if env.is_robot_goal_reachable():
        return True
    kind = target["kind"]
    if kind == "goal":
        return False
    if kind == "goal_clearance":
        return not env.object_occupies_point(
            target["blocking_objects"][0], tuple(target["witness_xy"])
        )
    if kind == "region":
        count, _ = env.count_reachable_points([tuple(p) for p in target["target_points"]])
        return int(count) >= int(target["min_reachable"])
    raise ValueError(f"Unknown keyhole target kind: {kind!r}")
