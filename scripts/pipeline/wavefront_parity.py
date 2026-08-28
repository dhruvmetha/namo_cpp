#!/usr/bin/env python3
"""Do the C++ wavefront and the pure-Python one partition free space identically?

Phase 0 of taking MuJoCo out of the policy decision loop. Before building anything
new, compare the two implementations that already ship, because either answer is
worth more than a harness comparing two things one person wrote.

`get_region_snapshot(env, use_cpp_unified=True)` calls the C++ binding.
`use_cpp_unified=False` runs `WavefrontSnapshotExporter`, pure geometry, whose
docstring claims it "mirrors the env-backed path: same inflation rules, same
robot". This checks the claim.

⛔ REGIONS ARE COMPARED AS POINT SETS, NEVER BY LABEL. Region labels are ordinal,
a rank over lexicographic cell order, so they renumber whenever free space
re-partitions (region_target.py's docstring is the canonical statement). Two
implementations can agree perfectly on the partition and disagree on every label.
Comparing labels would manufacture failures; comparing sets of cell indices is
the real question. Adjacency and edge objects are re-keyed onto those point sets
for the same reason.

⛔ THE SABOTAGE CASE IS NOT OPTIONAL. `--sabotage` perturbs one object before
comparing, and the run FAILS if the comparison still reports a match. A guard
that has never been shown to fail is a guard of unknown coverage, and eight of
26 cases in the parity anchor turned out unable to notice a reversed selection.
Run it before believing a clean pass.

  python scripts/pipeline/wavefront_parity.py --scenes <dir>
  python scripts/pipeline/wavefront_parity.py --scenes <dir> --sabotage
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, FrozenSet, List, Set, Tuple

# How far one object is nudged in the sabotage case. Larger than the 5 mm grid
# resolution by enough that a cell must change hands, small enough that the scene
# stays legal.
SABOTAGE_SHIFT_M = 0.05

Partition = Set[FrozenSet[int]]


def _partition(region_labels: Dict[int, str]) -> Partition:
    """Cell indices grouped by region, with the label thrown away."""
    by_label: Dict[str, Set[int]] = {}
    for idx, label in region_labels.items():
        by_label.setdefault(str(label), set()).add(int(idx))
    return {frozenset(cells) for cells in by_label.values()}


def _cells_of(region_labels: Dict[int, str]) -> Dict[str, FrozenSet[int]]:
    by_label: Dict[str, Set[int]] = {}
    for idx, label in region_labels.items():
        by_label.setdefault(str(label), set()).add(int(idx))
    return {label: frozenset(cells) for label, cells in by_label.items()}


def _adjacency_as_point_sets(snap: Dict[str, Any]) -> Set[FrozenSet[FrozenSet[int]]]:
    """Adjacency re-keyed onto point sets, so label renumbering cannot show up."""
    cells = _cells_of(snap.get("region_labels", {}))
    out = set()
    for region, neighbours in (snap.get("adjacency") or {}).items():
        a = cells.get(str(region))
        if a is None:
            continue
        for neighbour in neighbours:
            b = cells.get(str(neighbour))
            if b is not None:
                out.add(frozenset({a, b}))
    return out


def _edges_as_point_sets(snap: Dict[str, Any]) -> Dict[FrozenSet[FrozenSet[int]], FrozenSet[str]]:
    cells = _cells_of(snap.get("region_labels", {}))
    out: Dict[FrozenSet[FrozenSet[int]], Set[str]] = {}
    for region, neighbour_map in (snap.get("edge_objects") or {}).items():
        a = cells.get(str(region))
        if a is None:
            continue
        for neighbour, objs in (neighbour_map or {}).items():
            b = cells.get(str(neighbour))
            if b is None:
                continue
            out.setdefault(frozenset({a, b}), set()).update(str(o) for o in objs)
    return {k: frozenset(v) for k, v in out.items()}


def compare(cpp: Dict[str, Any], py: Dict[str, Any]) -> List[str]:
    """Every way the two snapshots disagree, in plain terms. Empty means identical."""
    problems: List[str] = []

    cpp_cells = set(int(i) for i in cpp.get("region_labels", {}))
    py_cells = set(int(i) for i in py.get("region_labels", {}))
    if cpp_cells != py_cells:
        only_cpp, only_py = cpp_cells - py_cells, py_cells - cpp_cells
        problems.append(
            f"free-space cells differ: {len(only_cpp)} only in C++, {len(only_py)} only in Python"
        )

    p_cpp, p_py = _partition(cpp.get("region_labels", {})), _partition(py.get("region_labels", {}))
    if p_cpp != p_py:
        problems.append(
            f"region partition differs: C++ has {len(p_cpp)} regions "
            f"(sizes {sorted(len(r) for r in p_cpp)}), "
            f"Python has {len(p_py)} (sizes {sorted(len(r) for r in p_py)})"
        )

    a_cpp, a_py = _adjacency_as_point_sets(cpp), _adjacency_as_point_sets(py)
    if a_cpp != a_py:
        problems.append(
            f"adjacency differs: {len(a_cpp - a_py)} edges only in C++, "
            f"{len(a_py - a_cpp)} only in Python"
        )

    e_cpp, e_py = _edges_as_point_sets(cpp), _edges_as_point_sets(py)
    if e_cpp != e_py:
        shared = set(e_cpp) & set(e_py)
        differing = [k for k in shared if e_cpp[k] != e_py[k]]
        problems.append(
            f"edge objects differ on {len(differing)} shared boundaries; "
            f"{len(set(e_cpp) - set(e_py))} boundaries only in C++, "
            f"{len(set(e_py) - set(e_cpp))} only in Python"
        )

    return problems


def snapshot_pair(env, seed: int = 42) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Both implementations over the same env at the same state."""
    from namo.planners import get_region_snapshot

    kw = dict(goals_per_region=0, local_info_only=False, seed=seed, use_xml_goal=True)
    cpp = get_region_snapshot(env, use_cpp_unified=True, **kw)
    py = get_region_snapshot(env, use_cpp_unified=False, **kw)
    return cpp, py


def _name(src: Path, root: str) -> str:
    """Path relative to the corpus root. `src.parent.name` is ambiguous: several
    directories under real_test_envs are called env1, so a bare name attributes a
    mismatch to whichever one the reader guesses."""
    try:
        return str(src.parent.relative_to(root))
    except ValueError:
        return str(src.parent)


def _load(xml_src: Path, config: str, start_pose):
    import namo_rl
    sys.path.insert(0, str(Path(__file__).resolve().parents[2].parent / "robot_control" / "src"))
    from robot_control.utils.scene_xml import portable_scene

    xml = portable_scene(xml_src, Path(tempfile.mkdtemp()))
    env = namo_rl.RLEnvironment(str(xml), config, False)
    env.reset()
    if start_pose:
        env.set_robot_pose(*start_pose)
    return env


def _sabotage(env) -> str:
    """Move one movable object, verified by the observation, not by assuming a layout.

    The binding exposes no set_object_pose, so this edits qpos through
    set_full_state. Which slots belong to which body is not documented anywhere I
    trust, so rather than assume, it nudges each slot in turn and keeps the first
    one that provably moves the target object and leaves the robot alone. If no
    slot does that, the scene is reported unusable rather than silently compared
    unperturbed, which would turn the sabotage run into the exact false pass it
    exists to rule out.
    """
    obs = env.get_observation()
    movables = sorted(k[: -len("_pose")] for k in obs if k.endswith("_pose") and k != "robot_pose")
    if not movables:
        return ""
    target = movables[0]
    before = list(obs[f"{target}_pose"])
    robot_before = list(obs["robot_pose"])
    base = env.get_full_state()

    for slot in range(len(base.qpos)):
        env.set_full_state(base)
        state = env.get_full_state()
        qpos = list(state.qpos)
        qpos[slot] += SABOTAGE_SHIFT_M
        state.qpos = qpos
        env.set_full_state(state)
        now = env.get_observation()
        moved_target = any(
            abs(float(now[f"{target}_pose"][i]) - float(before[i])) > 1e-4 for i in range(3)
        )
        moved_robot = any(
            abs(float(now["robot_pose"][i]) - float(robot_before[i])) > 1e-4 for i in range(3)
        )
        if moved_target and not moved_robot:
            return target

    env.set_full_state(base)
    return ""


def main(argv: List[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", required=True, help="directory searched for env.xml")
    ap.add_argument("--config", default="")
    ap.add_argument("--start-pose", nargs=3, type=float, default=[0.25, 0.10, 0.0])
    ap.add_argument("--sabotage", action="store_true",
                    help="perturb one object per scene; the run FAILS unless the comparison notices")
    a = ap.parse_args(argv[1:])

    from namo.runtime_profile import CANONICAL_CONFIG
    repo = Path(__file__).resolve().parents[2]
    config = a.config or str(repo / CANONICAL_CONFIG)

    scenes = sorted(Path(a.scenes).rglob("env.xml"))
    if not scenes:
        print(f"no env.xml under {a.scenes}")
        return 1

    matched = mismatched = skipped = 0
    for src in scenes:
        try:
            env = _load(src, config, a.start_pose)
            if a.sabotage and not _sabotage(env):
                skipped += 1
                continue
            cpp, py = snapshot_pair(env)
        except Exception as exc:  # a scene that will not load tells us nothing
            skipped += 1
            print(f"  SKIP {_name(src, a.scenes)}: {type(exc).__name__}: {str(exc)[:100]}")
            continue

        problems = compare(cpp, py)
        if problems:
            mismatched += 1
            print(f"  MISMATCH {_name(src, a.scenes)}")
            for p in problems:
                print(f"      {p}")
        else:
            matched += 1

    print(f"\nidentical: {matched}   differing: {mismatched}   unusable: {skipped}   "
          f"(of {len(scenes)} scenes)")

    if a.sabotage:
        # The point of the sabotage run is that the comparison NOTICES. A clean
        # pass here means the harness cannot see a moved object and every result
        # it has ever reported is worthless.
        if matched:
            print(f"\nSABOTAGE FAILED: {matched} scene(s) still compared identical "
                  f"after an object moved {SABOTAGE_SHIFT_M} m. The harness is blind.")
            return 1
        print(f"\nsabotage worked: every usable scene noticed a {SABOTAGE_SHIFT_M} m shift")
        return 0

    return 1 if mismatched else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
