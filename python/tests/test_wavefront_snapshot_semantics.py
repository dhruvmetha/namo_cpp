import numpy as np

from namo.visualization.wavefront_snapshot import (
    ObjectInstance,
    ObjectTemplate,
    WavefrontSnapshotExporter,
)


class _GeometryEnv:
    def __init__(self, bounds=(0.0, 0.20, 0.0, 0.20)):
        self._bounds = bounds

    def get_world_bounds(self):
        return self._bounds

    def get_object_info(self):
        return {"robot": {"size_x": 0.001, "size_y": 0.001}}

    def get_config_path(self):
        return ""


def _exporter(bounds=(0.0, 0.20, 0.0, 0.20)):
    exporter = WavefrontSnapshotExporter(
        _GeometryEnv(bounds),
        resolution=0.01,
        robot_half_extent_override=(0.001, 0.001),
    )
    exporter.tier1_inflation_margin_m = 0.0
    return exporter


def _object(name, *, center=(0.105, 0.105), half_extent=(0.02, 0.02), is_static=False):
    return ObjectInstance(
        ObjectTemplate(name, half_extent, is_static),
        center,
        (1.0, 0.0, 0.0, 0.0),
    )


def test_rasterise_and_remove_use_the_same_cell_centres():
    exporter = _exporter()
    obj = _object("box")
    grid = np.zeros((exporter.grid_width, exporter.grid_height), dtype=np.int16)

    exporter._rasterise_object(obj, obj.half_extent, grid)

    rasterised = {tuple(cell) for cell in np.argwhere(grid == -1)}
    collected = set(exporter._collect_footprint_cells(obj, obj.half_extent))
    assert collected == rasterised


def test_goal_radius_selects_a_circle_not_a_square():
    exporter = _exporter()

    cells = exporter._goal_cells((0.105, 0.105, 0.0), radius=0.02)

    assert (12, 10) in cells
    assert (12, 12) not in cells
    assert len(cells) == 13


def test_removing_a_movable_keeps_overlapping_static_occupancy():
    exporter = _exporter(bounds=(0.0, 0.05, 0.0, 0.05))
    wall = _object(
        "wall",
        center=(0.025, 0.025),
        half_extent=(0.006, 0.03),
        is_static=True,
    )
    movable = _object(
        "box",
        center=(0.025, 0.025),
        half_extent=(0.006, 0.03),
    )
    grids = exporter._build_grids([wall], [movable])
    region_map = np.zeros_like(grids["dynamic"], dtype=np.int32)
    region_map[:2, :] = 1
    region_map[3:, :] = 2

    adjacency, edge_objects = exporter._build_connectivity(
        grids["dynamic"].copy(),
        grids["occupancy_count"],
        region_map,
        {1: "robot", 2: "goal"},
        [movable],
    )

    assert np.all(grids["occupancy_count"][2, :] == 2)
    assert adjacency == {"robot": set(), "goal": set()}
    assert edge_objects == {"robot": {}, "goal": {}}


def test_removing_the_only_occupant_connects_the_regions():
    exporter = _exporter(bounds=(0.0, 0.05, 0.0, 0.05))
    movable = _object(
        "box",
        center=(0.025, 0.025),
        half_extent=(0.006, 0.03),
    )
    grids = exporter._build_grids([], [movable])
    region_map = np.zeros_like(grids["dynamic"], dtype=np.int32)
    region_map[:2, :] = 1
    region_map[3:, :] = 2

    adjacency, edge_objects = exporter._build_connectivity(
        grids["dynamic"].copy(),
        grids["occupancy_count"],
        region_map,
        {1: "robot", 2: "goal"},
        [movable],
    )

    assert adjacency == {"robot": {"goal"}, "goal": {"robot"}}
    assert edge_objects["robot"]["goal"] == {"box"}
    assert edge_objects["goal"]["robot"] == {"box"}
