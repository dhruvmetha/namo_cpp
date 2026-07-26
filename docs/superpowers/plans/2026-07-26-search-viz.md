# 2-Push Search Visualization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A local static web page that replays the best-first search per episode on the 2-push test set, showing the model's push ranking against exhaustive ground truth and where the truly-good pushes land in the priority queue.

**Architecture:** Three layers, each independently testable. (1) Pure-Python schema/metric modules under `scripts/viz/` with no simulator dependency. (2) A flag-gated `--trace-out` addition to `scripts/sandbox/eval_bestfirst.py` that writes one JSON per episode, plus an offline joiner that reads `testset_gt.h5` into per-episode green/not-green truth. (3) A dependency-free static page in `viz/search/` served by `python -m http.server`, fetching those JSON files lazily.

**Tech Stack:** Python 3 (numpy, h5py, pytest), vanilla HTML/CSS/JS with inline SVG. No build step, no npm, no CDN.

## Global Constraints

- Spec: `docs/superpowers/specs/2026-07-26-search-viz-design.md`. Read it before starting.
- Run everything from the repo root with the box env sourced: `source env.ilab.sh` then `PYTHONPATH="$PWD/build_python:$PWD/python"`.
- Tests live in `python/tests/`, run as `PYTHONPATH="$PWD/build_python:$PWD/python" python -m pytest python/tests/<file>.py -v`. There is no pytest config file; `python/tests/conftest.py` stubs `namo_rl` so pure-Python tests run without the compiled bindings.
- Never hardcode data paths. Use `namo.paths` (`SCRATCH`, `DATASETS`, `MANIFESTS`) and `namo.eval_sets`. Guard: `check_no_hardcoded_paths.sh`.
- Markdown prose is one line per paragraph. No hard-wrapping sentences across source lines.
- With `--trace-out` unset, `eval_bestfirst.py` must behave byte-identically to today. This is a gate, not an aspiration.
- Episode key is `(xml_realpath, object_id)`. On disk: `<xml basename without extension>__<object_id>.json`.
- Ground truth is a badge, never a number: `value_target == 1.0` is an opener, `== 0.9` is a setup, everything else is not green.

---

## File Structure

| path | responsibility |
| --- | --- |
| `scripts/pipeline/add_contact_px.py` | MODIFY — factor out the world-frame contact offset formula so both the pixel path and the viz path share one implementation |
| `scripts/viz/trace_schema.py` | CREATE — pure functions building the trace dict and the on-disk episode filename |
| `scripts/viz/index_metrics.py` | CREATE — pure functions computing rank-of-best-green and top-1 truth from a candidate ordering plus a green set |
| `scripts/sandbox/eval_bestfirst.py` | MODIFY — `--trace-out DIR`, board pool/grid/parent capture, pop recording, scene dump |
| `scripts/viz/build_gt_json.py` | CREATE — join `testset_gt.h5` + `pure2push.json` into per-episode green sets |
| `scripts/viz/build_manifest.py` | CREATE — walk trace and gt dirs, emit `manifest.json` with the index table rows |
| `viz/search/index.html` | CREATE — dropdowns and the sortable episode table |
| `viz/search/episode.html` | CREATE — the four-zone replay view |
| `viz/search/app.js` | CREATE — data loading, scrub clock, cross-highlighting |
| `viz/search/style.css` | CREATE — layout and the green/red/grey badge palette |
| `python/tests/test_viz_trace_schema.py` | CREATE — trace dict and filename tests |
| `python/tests/test_viz_index_metrics.py` | CREATE — rank-of-best-green tests |
| `python/tests/test_viz_gt_join.py` | CREATE — GT join against a synthetic H5 |
| `python/tests/test_contact_offsets_world.py` | CREATE — refactor equivalence test |
| `docs/experiments/horizon_q_model_registry.md` | MODIFY — consistent `train_h5:` per entry |
| `docs/experiments/eval_set_registry.md` | MODIFY — `testset_gt.h5` schema block |

---

### Task 1: World-frame contact offsets

The 60 contact points are currently computed only in crop-pixel space, inside `contact_px` at `scripts/pipeline/add_contact_px.py:15`. The SVG needs them in meters. Factor the meter-space half out so there is exactly one copy of the edge ordering, and have the pixel function call it.

**Files:**
- Modify: `scripts/pipeline/add_contact_px.py:15`
- Test: `python/tests/test_contact_offsets_world.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `contact_offsets_world(hw: float, hd: float, theta: float) -> np.ndarray` of shape `(60, 2)`, world-frame XY offsets in meters from the object center. `contact_px(edge, hw, hd, theta, crop_m, S=64)` keeps its existing signature and return value.

- [ ] **Step 1: Read the existing function**

Read `scripts/pipeline/add_contact_px.py` lines 1-40. Note the exact edge ordering (4 faces × 15 points, matching `generate_rectangular_edge_points` in `src/skills/namo_push_controller.cpp`) and where it rotates the local point by `theta` to get `(wx, wy)` before dividing by resolution. That `(wx, wy)` is the value being factored out.

- [ ] **Step 2: Write the failing test**

Create `python/tests/test_contact_offsets_world.py`:

```python
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "pipeline"))

from add_contact_px import contact_offsets_world, contact_px  # noqa: E402


def test_shape_and_dtype():
    off = contact_offsets_world(0.15, 0.10, 0.0)
    assert off.shape == (60, 2)
    assert off.dtype == np.float32


def test_zero_rotation_offsets_lie_on_the_object_rectangle():
    hw, hd = 0.15, 0.10
    off = contact_offsets_world(hw, hd, 0.0)
    on_edge = (np.isclose(np.abs(off[:, 0]), hw, atol=1e-6) |
               np.isclose(np.abs(off[:, 1]), hd, atol=1e-6))
    assert on_edge.all()


def test_rotation_is_a_rigid_transform():
    hw, hd, th = 0.15, 0.10, 0.7
    a = contact_offsets_world(hw, hd, 0.0)
    b = contact_offsets_world(hw, hd, th)
    c, s = np.cos(th), np.sin(th)
    expected = np.stack([a[:, 0] * c - a[:, 1] * s, a[:, 0] * s + a[:, 1] * c], axis=1)
    assert np.allclose(b, expected, atol=1e-6)


@pytest.mark.parametrize("edge", [0, 7, 15, 31, 44, 59])
def test_pixel_path_matches_the_factored_offsets(edge):
    hw, hd, th, crop_m, S = 0.15, 0.10, 0.7, 1.0, 64
    px = contact_px(edge, hw, hd, th, crop_m, S)
    wx, wy = contact_offsets_world(hw, hd, th)[edge]
    res = crop_m / S
    assert np.allclose(px, (S / 2.0 + wx / res, S / 2.0 + wy / res), atol=1e-4)
```

The last test encodes the current pixel convention. If reading Step 1 shows a different center or sign convention, fix the assertion to match the existing code — the point of this test is that the refactor changes nothing, so the expectation must mirror what `contact_px` does today, not what you would prefer.

- [ ] **Step 3: Run the test to verify it fails**

```bash
source env.ilab.sh
PYTHONPATH="$PWD/build_python:$PWD/python" python -m pytest python/tests/test_contact_offsets_world.py -v
```

Expected: FAIL with `ImportError: cannot import name 'contact_offsets_world'`.

- [ ] **Step 4: Implement the refactor**

In `scripts/pipeline/add_contact_px.py`, add `contact_offsets_world` above `contact_px`, moving the local-point construction and rotation into it verbatim, then rewrite `contact_px` to call it:

```python
def contact_offsets_world(hw, hd, theta):
    """World-frame XY offsets (meters) from the object center for all 60 contact points.

    Edge ordering is 4 faces x 15 points, matching generate_rectangular_edge_points in
    src/skills/namo_push_controller.cpp. This is the single source of that ordering; the
    pixel-space helper below is a thin wrapper over it."""
    out = np.zeros((60, 2), np.float32)
    c, s = np.cos(theta), np.sin(theta)
    for e in range(60):
        lx, ly = _local_edge_point(e, hw, hd)   # the existing local-point logic, extracted
        out[e, 0] = lx * c - ly * s
        out[e, 1] = lx * s + ly * c
    return out


def contact_px(edge, hw, hd, theta, crop_m, S=64):
    wx, wy = contact_offsets_world(hw, hd, theta)[edge]
    res = crop_m / S
    return (S / 2.0 + wx / res, S / 2.0 + wy / res)
```

`_local_edge_point(e, hw, hd)` is the existing per-edge local coordinate computation lifted into its own helper. Do not change any number in it.

- [ ] **Step 5: Run the test to verify it passes**

```bash
PYTHONPATH="$PWD/build_python:$PWD/python" python -m pytest python/tests/test_contact_offsets_world.py -v
```

Expected: 9 passed.

- [ ] **Step 6: Verify no downstream regression**

`scripts/eval_scorer.py:45` holds an identical copy of the formula, and `scripts/sandbox/live_scorer.py:150` calls one of them. Confirm both still produce the same numbers:

```bash
PYTHONPATH="$PWD/build_python:$PWD/python" python -c "
import sys; sys.path.insert(0, 'scripts/pipeline'); sys.path.insert(0, 'scripts')
from add_contact_px import contact_px as a
from eval_scorer import contact_px as b
import numpy as np
assert all(np.allclose(a(e,0.15,0.10,0.7,1.0,64), b(e,0.15,0.10,0.7,1.0,64)) for e in range(60))
print('OK')"
```

Expected: `OK`.

- [ ] **Step 7: Commit**

```bash
git add scripts/pipeline/add_contact_px.py python/tests/test_contact_offsets_world.py
git commit -m "refactor: factor world-frame contact offsets out of contact_px"
```

---

### Task 2: Trace schema

Pure functions describing what a trace file contains. No simulator, no model — so this task is fully testable on its own and pins the contract every later task reads.

**Files:**
- Create: `scripts/viz/trace_schema.py`
- Create: `scripts/viz/__init__.py` (empty)
- Test: `python/tests/test_viz_trace_schema.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `episode_filename(xml_path: str, object_id: str) -> str` — `"<xml stem>__<object_id>.json"`.
  - `make_board(board_id: int, depth: int, parent_edge: int, parent_depth: int, pool: list[dict], grid: list[list[float]] | None, w0: float, free_strikes: int) -> dict`. Each `pool` entry is `{"obj": str, "edge": int, "depth": int, "q": float}`. Root boards use `parent_edge = parent_depth = -1`.
  - `make_pop(t: int, board_id: int, obj: str, edge: int, depth: int, q: float, bp: float, w: float, opened: bool) -> dict` — includes the derived `"se": bp * w`.
  - `build_trace(meta: dict, scene: dict, boards: list[dict], pops: list[dict], result: dict) -> dict` — the top-level document, with a `"schema_version": 1` field.

- [ ] **Step 1: Write the failing test**

Create `python/tests/test_viz_trace_schema.py`:

```python
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from viz.trace_schema import build_trace, episode_filename, make_board, make_pop  # noqa: E402


def test_episode_filename_uses_stem_and_object_id():
    name = episode_filename("/scratch/x/run_0056/env_0056_pair_001.xml", "obstacle_7_movable")
    assert name == "env_0056_pair_001__obstacle_7_movable.json"


def test_root_board_has_sentinel_parent():
    b = make_board(0, 0, -1, -1, [], None, 1.0, 0)
    assert b["board_id"] == 0 and b["depth"] == 0
    assert b["parent_edge"] == -1 and b["parent_depth"] == -1


def test_child_board_records_the_setup_push_that_spawned_it():
    pool = [{"obj": "o", "edge": 12, "depth": 1, "q": 0.4}]
    b = make_board(3, 1, 54, 2, pool, None, 1.0, 1)
    assert (b["parent_edge"], b["parent_depth"]) == (54, 2)
    assert b["n_candidates"] == 1
    assert b["pool"] == pool


def test_pop_carries_the_effective_priority():
    p = make_pop(7, 3, "o", 12, 1, 0.4, 0.5, 0.2, False)
    assert p["t"] == 7 and p["board_id"] == 3
    assert p["se"] == 0.5 * 0.2
    assert p["opened"] is False


def test_build_trace_is_json_serializable_and_versioned():
    import json
    doc = build_trace(
        meta={"xml": "/x/a.xml", "object_id": "o", "model": "ceiling", "strategy": "off"},
        scene={"bounds": [0, 1, 0, 1], "static": [], "movable": [], "robot": [0, 0, 0],
               "goal": [0.5, 0.5, 0.0], "contacts": []},
        boards=[make_board(0, 0, -1, -1, [], None, 1.0, 0)],
        pops=[make_pop(1, 0, "o", 5, 0, 0.9, 0.9, 1.0, True)],
        result={"solved": True, "sims": 1, "plan_len": 1, "end": "solved"},
    )
    assert doc["schema_version"] == 1
    assert doc["result"]["solved"] is True
    json.dumps(doc)
```

- [ ] **Step 2: Run to verify it fails**

```bash
PYTHONPATH="$PWD/build_python:$PWD/python" python -m pytest python/tests/test_viz_trace_schema.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'viz'`.

- [ ] **Step 3: Implement**

Create `scripts/viz/__init__.py` empty, and `scripts/viz/trace_schema.py`:

```python
"""On-disk contract for one episode's search trace. Pure data, no simulator imports.

Consumed by the static page in viz/search/. Bump schema_version if any field changes meaning."""
import os

SCHEMA_VERSION = 1


def episode_filename(xml_path, object_id):
    return f"{os.path.splitext(os.path.basename(xml_path))[0]}__{object_id}.json"


def make_board(board_id, depth, parent_edge, parent_depth, pool, grid, w0, free_strikes):
    return {"board_id": board_id, "depth": depth,
            "parent_edge": parent_edge, "parent_depth": parent_depth,
            "n_candidates": len(pool), "pool": pool, "grid": grid,
            "w0": w0, "free_strikes": free_strikes}


def make_pop(t, board_id, obj, edge, depth, q, bp, w, opened):
    return {"t": t, "board_id": board_id, "obj": obj, "edge": edge, "depth": depth,
            "q": q, "bp": bp, "w": w, "se": bp * w, "opened": bool(opened)}


def build_trace(meta, scene, boards, pops, result):
    return {"schema_version": SCHEMA_VERSION, "meta": meta, "scene": scene,
            "boards": boards, "pops": pops, "result": result}
```

- [ ] **Step 4: Run to verify it passes**

```bash
PYTHONPATH="$PWD/build_python:$PWD/python" python -m pytest python/tests/test_viz_trace_schema.py -v
```

Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add scripts/viz/__init__.py scripts/viz/trace_schema.py python/tests/test_viz_trace_schema.py
git commit -m "feat: trace schema for the 2-push search visualization"
```

---

### Task 3: Index metrics

The page's headline number. Pure functions over an ordering plus a green set, so no data files are needed to test them.

**Files:**
- Create: `scripts/viz/index_metrics.py`
- Test: `python/tests/test_viz_index_metrics.py`

**Interfaces:**
- Consumes: `scripts/viz/trace_schema.py` board dicts (uses the `pool` field only).
- Produces:
  - `rank_of_best_green(pool: list[dict], green: set[tuple[int, int]]) -> int | None` — 1-based rank of the highest-scoring pool entry whose `(edge, depth)` is green, sorting the pool by descending `q` with `(edge, depth)` as a deterministic tiebreak. `None` when the pool contains no green.
  - `top1_truth(pool: list[dict], openers: set[tuple[int, int]], setups: set[tuple[int, int]]) -> str` — one of `"opener"`, `"setup"`, `"dead"`, or `"empty"` for an empty pool.

- [ ] **Step 1: Write the failing test**

Create `python/tests/test_viz_index_metrics.py`:

```python
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from viz.index_metrics import rank_of_best_green, top1_truth  # noqa: E402


def _pool(*triples):
    return [{"obj": "o", "edge": e, "depth": d, "q": q} for (e, d, q) in triples]


def test_rank_one_when_the_model_tops_a_green():
    pool = _pool((5, 0, 0.9), (7, 1, 0.4))
    assert rank_of_best_green(pool, {(5, 0)}) == 1


def test_rank_counts_position_in_descending_q_order():
    pool = _pool((5, 0, 0.9), (7, 1, 0.8), (9, 2, 0.7))
    assert rank_of_best_green(pool, {(9, 2)}) == 3


def test_rank_is_none_when_nothing_is_green():
    assert rank_of_best_green(_pool((5, 0, 0.9)), set()) is None


def test_ties_break_deterministically_on_edge_then_depth():
    pool = _pool((9, 0, 0.5), (2, 0, 0.5))
    assert rank_of_best_green(pool, {(9, 0)}) == 2
    assert rank_of_best_green(pool, {(2, 0)}) == 1


def test_empty_pool():
    assert rank_of_best_green([], {(1, 1)}) is None
    assert top1_truth([], set(), set()) == "empty"


def test_top1_truth_labels_the_highest_scoring_candidate():
    pool = _pool((5, 0, 0.9), (7, 1, 0.4))
    assert top1_truth(pool, {(5, 0)}, set()) == "opener"
    assert top1_truth(pool, set(), {(5, 0)}) == "setup"
    assert top1_truth(pool, {(7, 1)}, set()) == "dead"
```

- [ ] **Step 2: Run to verify it fails**

```bash
PYTHONPATH="$PWD/build_python:$PWD/python" python -m pytest python/tests/test_viz_index_metrics.py -v
```

Expected: FAIL with `ModuleNotFoundError` or `ImportError` on `viz.index_metrics`.

- [ ] **Step 3: Implement**

Create `scripts/viz/index_metrics.py`:

```python
"""Index-table metrics. The headline quantity is the queue position of the best truly-good push."""


def _ordered(pool):
    return sorted(pool, key=lambda c: (-c["q"], c["edge"], c["depth"]))


def rank_of_best_green(pool, green):
    for i, c in enumerate(_ordered(pool), start=1):
        if (c["edge"], c["depth"]) in green:
            return i
    return None


def top1_truth(pool, openers, setups):
    if not pool:
        return "empty"
    c = _ordered(pool)[0]
    k = (c["edge"], c["depth"])
    if k in openers:
        return "opener"
    if k in setups:
        return "setup"
    return "dead"
```

- [ ] **Step 4: Run to verify it passes**

```bash
PYTHONPATH="$PWD/build_python:$PWD/python" python -m pytest python/tests/test_viz_index_metrics.py -v
```

Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
git add scripts/viz/index_metrics.py python/tests/test_viz_index_metrics.py
git commit -m "feat: rank-of-best-green index metrics"
```

---

### Task 4: GT joiner

Turn `testset_gt.h5` into one small JSON per episode holding green sets. This is where the 981-of-1018 coverage gap gets handled honestly.

**Files:**
- Create: `scripts/viz/build_gt_json.py`
- Test: `python/tests/test_viz_gt_join.py`

**Interfaces:**
- Consumes: `scripts/viz/trace_schema.episode_filename`.
- Produces:
  - `green_sets_from_grid(value_target: np.ndarray) -> tuple[set, set]` — `(openers, setups)` as `(edge, depth)` sets, from cells equal to `1.0` and `0.9` respectively.
  - `build_episode_gt(h5, xml: str, object_id: str) -> dict | None` — `None` when this episode has no GT root row. Otherwise `{"root": {"openers": [[e, d], ...], "setups": [...]}, "finish": {"<parent_edge>_<parent_depth>": {"openers": [...], "setups": [...]}}}`.
  - CLI: `python scripts/viz/build_gt_json.py --out-dir <dir>` writing one file per covered episode plus `_coverage.json` recording which manifest episodes had no GT root.

- [ ] **Step 1: Write the failing test**

Create `python/tests/test_viz_gt_join.py`. It builds a synthetic H5 with the real field names and shapes, so it needs no access to the 214 MB file:

```python
import sys
from pathlib import Path

import numpy as np
import pytest

h5py = pytest.importorskip("h5py")

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from viz.build_gt_json import build_episode_gt, green_sets_from_grid  # noqa: E402


def test_green_sets_split_openers_from_setups():
    vt = np.zeros((60, 5), np.float32)
    vt[5, 0] = 1.0
    vt[12, 3] = 0.9
    vt[7, 1] = 0.0
    vt[9, 2] = -1.0
    openers, setups = green_sets_from_grid(vt)
    assert openers == {(5, 0)}
    assert setups == {(12, 3)}


@pytest.fixture
def synthetic_h5(tmp_path):
    p = tmp_path / "gt.h5"
    root_vt = np.zeros((60, 5), np.float32); root_vt[5, 0] = 1.0; root_vt[54, 2] = 0.9
    kid_vt = np.zeros((60, 5), np.float32); kid_vt[12, 1] = 1.0
    other_vt = np.zeros((60, 5), np.float32); other_vt[1, 1] = 1.0
    with h5py.File(p, "w") as f:
        f.create_dataset("value_target", data=np.stack([root_vt, kid_vt, other_vt]))
        f.create_dataset("node_kind", data=np.array([b"root", b"depth2", b"root"]))
        f.create_dataset("xml", data=np.array([b"/x/a.xml", b"/x/a.xml", b"/x/b.xml"]))
        f.create_dataset("object_id", data=np.array([b"obj1", b"obj1", b"obj2"]))
        f.create_dataset("parent_edge", data=np.array([-1, 54, -1], np.int16))
        f.create_dataset("parent_depth", data=np.array([-1, 2, -1], np.int16))
    return p


def test_join_returns_root_and_finish_grids(synthetic_h5):
    with h5py.File(synthetic_h5, "r") as f:
        gt = build_episode_gt(f, "/x/a.xml", "obj1")
    assert gt["root"]["openers"] == [[5, 0]]
    assert gt["root"]["setups"] == [[54, 2]]
    assert gt["finish"]["54_2"]["openers"] == [[12, 1]]


def test_missing_root_returns_none(synthetic_h5):
    with h5py.File(synthetic_h5, "r") as f:
        assert build_episode_gt(f, "/x/missing.xml", "obj1") is None
```

- [ ] **Step 2: Run to verify it fails**

```bash
PYTHONPATH="$PWD/build_python:$PWD/python" python -m pytest python/tests/test_viz_gt_join.py -v
```

Expected: FAIL on importing `viz.build_gt_json`.

- [ ] **Step 3: Implement**

Create `scripts/viz/build_gt_json.py`:

```python
#!/usr/bin/env python3
"""Join testset_gt.h5 into per-episode green sets for the search visualization.

GT is used as a BADGE, never as a number: value_target 1.0 = opener, 0.9 = setup whose subtree
contained a verified win. Everything else is not green. Label semantics: build_rung2_h5.py:95-113.

testset_gt.h5 roots 981 of the 1018 manifest episodes (build-version drift recorded in
docs/experiments/eval_set_registry.md). Uncovered episodes are listed in _coverage.json, never faked."""
import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np

REPO = Path(__file__).resolve().parents[2]
for _p in (f"{REPO}/python", f"{REPO}/scripts"):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from namo import eval_sets  # noqa: E402
from viz.trace_schema import episode_filename  # noqa: E402


def _s(v):
    return v.decode() if isinstance(v, bytes) else str(v)


def green_sets_from_grid(value_target):
    openers = set(zip(*np.where(value_target == 1.0)))
    setups = set(zip(*np.where(value_target == 0.9)))
    return {(int(e), int(d)) for e, d in openers}, {(int(e), int(d)) for e, d in setups}


def _pairs(s):
    return sorted([int(e), int(d)] for e, d in s)


def build_episode_gt(h5, xml, object_id):
    xmls = [_s(v) for v in h5["xml"][:]]
    objs = [_s(v) for v in h5["object_id"][:]]
    kinds = [_s(v) for v in h5["node_kind"][:]]
    rows = [i for i in range(len(xmls))
            if os.path.realpath(xmls[i]) == os.path.realpath(xml) and objs[i] == object_id]
    root_rows = [i for i in rows if kinds[i] == "root"]
    if not root_rows:
        return None
    vt = h5["value_target"]
    o, s = green_sets_from_grid(vt[root_rows[0]])
    out = {"root": {"openers": _pairs(o), "setups": _pairs(s)}, "finish": {}}
    pe, pd = h5["parent_edge"][:], h5["parent_depth"][:]
    for i in rows:
        if kinds[i] == "root":
            continue
        fo, fs = green_sets_from_grid(vt[i])
        out["finish"][f"{int(pe[i])}_{int(pd[i])}"] = {"openers": _pairs(fo), "setups": _pairs(fs)}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    key = json.load(open(eval_sets.PURE2PUSH))
    covered, uncovered = 0, []
    with h5py.File(eval_sets.TWOPUSH_GT_H5, "r") as f:
        # index once: (realpath, object_id) -> row ids, so the pass is linear not quadratic
        xmls = [os.path.realpath(_s(v)) for v in f["xml"][:]]
        objs = [_s(v) for v in f["object_id"][:]]
        kinds = [_s(v) for v in f["node_kind"][:]]
        vt, pe, pd = f["value_target"], f["parent_edge"][:], f["parent_depth"][:]
        by_ep = defaultdict(list)
        for i in range(len(xmls)):
            by_ep[(xmls[i], objs[i])].append(i)
        for xml, recs in key.items():
            rp = os.path.realpath(xml)
            for rec in recs:
                oid = rec["object_id"]
                rows = by_ep.get((rp, oid), [])
                root_rows = [i for i in rows if kinds[i] == "root"]
                if not root_rows:
                    uncovered.append([xml, oid])
                    continue
                o, s = green_sets_from_grid(vt[root_rows[0]])
                doc = {"root": {"openers": _pairs(o), "setups": _pairs(s)}, "finish": {}}
                for i in rows:
                    if kinds[i] == "root":
                        continue
                    fo, fs = green_sets_from_grid(vt[i])
                    doc["finish"][f"{int(pe[i])}_{int(pd[i])}"] = {
                        "openers": _pairs(fo), "setups": _pairs(fs)}
                json.dump(doc, open(os.path.join(a.out_dir, episode_filename(xml, oid)), "w"))
                covered += 1
    json.dump({"covered": covered, "uncovered": uncovered},
              open(os.path.join(a.out_dir, "_coverage.json"), "w"))
    print(f"covered={covered} uncovered={len(uncovered)}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run to verify it passes**

```bash
PYTHONPATH="$PWD/build_python:$PWD/python" python -m pytest python/tests/test_viz_gt_join.py -v
```

Expected: 3 passed.

- [ ] **Step 5: Run it for real and check the coverage number**

```bash
source env.ilab.sh
PYTHONPATH="$PWD/build_python:$PWD/python" python scripts/viz/build_gt_json.py \
  --out-dir "$NAMO_SCRATCH/viz/search/gt"
```

Expected: `covered=981 uncovered=37`. If the numbers differ, stop and report — the spec's coverage claim comes from `docs/experiments/eval_set_registry.md:36-39` and a mismatch means the data moved, not that the code is wrong.

- [ ] **Step 6: Commit**

```bash
git add scripts/viz/build_gt_json.py python/tests/test_viz_gt_join.py
git commit -m "feat: join testset GT into per-episode green sets"
```

---

### Task 5: Trace output from the search

Wire trace capture into `eval_bestfirst.py` behind `--trace-out`. The hard requirement is that the flag being absent changes nothing.

**Files:**
- Modify: `scripts/sandbox/eval_bestfirst.py:83-139` (`solve_scene`), `:97-101` (`new_board`), `:192-240` (arg parsing and the episode loop)
- Test: manual gate below (this task's correctness is an equality check against the unpatched script, which a unit test cannot express)

**Interfaces:**
- Consumes: `scripts/viz/trace_schema.py`.
- Produces: one `<episode_filename>` JSON per episode under `--trace-out`, matching `build_trace`'s document shape.

- [ ] **Step 1: Capture a pre-patch baseline**

```bash
source env.ilab.sh
CKPT="$NAMO_SCRATCH/round3/models/d20_plus_setup_only_splitloss/checkpoints/epoch011-val_loss1.6952.ckpt"
PYTHONPATH="$PWD/build_python:$PWD/python" python scripts/sandbox/eval_bestfirst.py \
  --ckpt "$CKPT" --hmax 2 --sim-budget 30 --start 0 --end 5 \
  --discount conf --tau 0.15 \
  --out /tmp/base.json --leaf-out /tmp/base.jsonl
cp /tmp/base.json /tmp/base.keep.json; cp /tmp/base.jsonl /tmp/base.keep.jsonl
```

Keep those two files. They are the gate for Step 5.

- [ ] **Step 2: Add the flag and the recording hooks**

In `scripts/sandbox/eval_bestfirst.py`:

Add the import near the other repo imports (after line 37):

```python
sys.path.insert(0, f"{REPO}/scripts")
from viz.trace_schema import build_trace, episode_filename, make_board, make_pop  # noqa: E402
```

Extend `new_board` (`:97`) to carry the parent link, the pool and the grid. Change its signature to `new_board(depth, npool, w0=1.0, free_strikes=0, parent_edge=-1, parent_depth=-1, pool_rows=None, grid=None)` and append those four keys to the dict it stores.

Extend `solve_scene`'s signature with `trace_out=None` (a list; when not `None`, pops are appended to it). Inside the pop loop, immediately after the `opened = bool(is_open(env))` line (`:120`), add:

```python
if trace_out is not None:
    trace_out.append(make_pop(sims, it["board_id"], it["obj"], int(it["obj_edge"]),
                              int(it["obj_depth"]), float(it["q"]), float(it["bp"]),
                              float(board["w"]), opened))
```

For `it["obj_edge"]` / `it["obj_depth"]` to exist, extend the two `push({...})` call sites (`:111` and `:135`) with `"obj_edge": int(g.edge_idx), "obj_depth": int(g.depth)` (and `g2` at the second site). These are plain ints copied off the `Goal`, so they cost nothing when tracing is off.

At each `new_board` call site, pass the pool rows and, when tracing, the model grid:

```python
pool_rows = [{"obj": o, "edge": int(gg.edge_idx), "depth": int(gg.depth), "q": float(qq)}
             for (o, gg, qq) in pool2]
grid = None
if trace_out is not None and prior != "uniform":
    grid = planner.scorer.score_state(env, restrict_obj, goal, xml, h=h, raw=raw).tolist()
    env.set_full_state(s_new)          # score_state may move the state; restore, as eval_m3.py:73 does
```

The root board (`:109`) gets the same treatment using `pool`, `V0`, `hmax` and `s0`.

- [ ] **Step 3: Add `--trace-out` and the scene dump**

Add the argument next to `--lifetime-out` (`:192`):

```python
ap.add_argument("--trace-out", default="", help="per-episode search trace JSON dir (for viz/search)")
ap.add_argument("--trace-model", default="", help="model label written into each trace's meta")
```

In the episode loop, build the scene once per XML (after `s0 = env.get_full_state()` at `:224`):

```python
scene = None
if a.trace_out:
    info = env.get_object_info()
    obs = env.get_observation()
    bx = list(env.get_world_bounds())
    static = [{"name": k, "x": v["pos_x"], "y": v["pos_y"],
               "hw": v["size_x"], "hd": v["size_y"],
               "qw": v.get("quat_w", 1.0), "qz": v.get("quat_z", 0.0)}
              for k, v in info.items() if "pos_x" in v]
    movable = [{"name": k, "x": obs[f"{k}_pose"][0], "y": obs[f"{k}_pose"][1],
                "theta": obs[f"{k}_pose"][2], "hw": v["size_x"], "hd": v["size_y"]}
               for k, v in info.items() if f"{k}_pose" in obs and "pos_x" not in v]
    scene = {"bounds": bx, "static": static, "movable": movable,
             "robot": list(obs["robot_pose"]), "goal": list(goal)}
```

Inside the per-record loop, pass `trace_out=pops` and write the file:

```python
pops = [] if a.trace_out else None
solved, sims, plen, boards, end = solve_scene(..., trace_out=pops)
if a.trace_out:
    from add_contact_px import contact_offsets_world
    oi = env.get_object_info()[obj]
    ctheta = env.get_observation()[f"{obj}_pose"]
    off = contact_offsets_world(oi["size_x"], oi["size_y"], ctheta[2])
    ep_scene = dict(scene)
    ep_scene["contacts"] = [[float(ctheta[0] + dx), float(ctheta[1] + dy)] for dx, dy in off]
    doc = build_trace(
        meta={"xml": xml, "object_id": obj, "region": rec.get("region"),
              "model": a.trace_model or os.path.basename(a.ckpt),
              "strategy": (f"{a.discount}" if a.discount == "off" else f"{a.discount}_tau{a.tau}")},
        scene=ep_scene,
        boards=[{k: v for k, v in b.items() if k != "tries"} for b in boards],
        pops=pops, result={"solved": solved, "sims": sims, "plan_len": plen, "end": end})
    os.makedirs(a.trace_out, exist_ok=True)
    json.dump(doc, open(os.path.join(a.trace_out, episode_filename(xml, obj)), "w"))
```

`scripts/pipeline` must be on `sys.path` for the `add_contact_px` import; add it to the bootstrap list at `:30`.

- [ ] **Step 4: Smoke the trace output**

```bash
PYTHONPATH="$PWD/build_python:$PWD/python" python scripts/sandbox/eval_bestfirst.py \
  --ckpt "$CKPT" --hmax 2 --sim-budget 30 --start 0 --end 5 \
  --discount conf --tau 0.15 --trace-model ceiling \
  --out /tmp/tr.json --leaf-out /tmp/tr.jsonl --trace-out /tmp/trace
PYTHONPATH="$PWD/build_python:$PWD/python" python -c "
import glob, json
fs = sorted(glob.glob('/tmp/trace/*.json'))
print(len(fs), 'traces')
d = json.load(open(fs[0]))
assert d['schema_version'] == 1
assert len(d['pops']) == d['result']['sims'], (len(d['pops']), d['result']['sims'])
b0 = d['boards'][0]
assert b0['depth'] == 0 and b0['parent_edge'] == -1
assert b0['n_candidates'] == len(b0['pool'])
assert b0['grid'] is None or (len(b0['grid']) == 60 and len(b0['grid'][0]) == 5)
popped = {(p['board_id'], p['edge'], p['depth']) for p in d['pops']}
pool = {(b['board_id'], c['edge'], c['depth']) for b in d['boards'] for c in b['pool']}
assert popped <= pool, popped - pool
print('trace OK')"
```

Expected: a trace count above zero and `trace OK`. The `popped <= pool` assertion is the important one — every simulated push must have been a listed candidate of its board.

- [ ] **Step 5: Gate — the flag off must change nothing**

```bash
PYTHONPATH="$PWD/build_python:$PWD/python" python scripts/sandbox/eval_bestfirst.py \
  --ckpt "$CKPT" --hmax 2 --sim-budget 30 --start 0 --end 5 \
  --discount conf --tau 0.15 \
  --out /tmp/after.json --leaf-out /tmp/after.jsonl
diff /tmp/base.keep.json /tmp/after.json && diff /tmp/base.keep.jsonl /tmp/after.jsonl && echo IDENTICAL
```

Expected: `IDENTICAL`. If it differs, the patch leaked into the untraced path — fix before committing.

- [ ] **Step 6: Commit**

```bash
git add scripts/sandbox/eval_bestfirst.py
git commit -m "feat: --trace-out per-episode search traces for the visualization"
```

---

### Task 6: Manifest builder

One small file the page loads first, holding the arm list, the episode list with tiers, and the precomputed index rows.

**Files:**
- Create: `scripts/viz/build_manifest.py`
- Test: extends `python/tests/test_viz_index_metrics.py`

**Interfaces:**
- Consumes: `viz.index_metrics.rank_of_best_green`, `viz.index_metrics.top1_truth`, `viz.trace_schema.episode_filename`.
- Produces: `index_row(trace: dict, gt: dict | None, tier: str) -> dict` with keys `key, xml, object_id, tier, solved, sims, rank_best_green, top1, has_gt`. CLI writes `manifest.json` with `{"arms": [...], "episodes": [...], "index": {"<model>|<strategy>": [row, ...]}}`.

- [ ] **Step 1: Write the failing test**

Append to `python/tests/test_viz_index_metrics.py`:

```python
from viz.build_manifest import index_row  # noqa: E402


def _trace(pool, solved=True, sims=3):
    return {"meta": {"xml": "/x/a.xml", "object_id": "o"},
            "boards": [{"board_id": 0, "depth": 0, "pool": pool}],
            "result": {"solved": solved, "sims": sims}}


def test_index_row_uses_the_root_board_ordering():
    pool = [{"obj": "o", "edge": 5, "depth": 0, "q": 0.9},
            {"obj": "o", "edge": 7, "depth": 1, "q": 0.4}]
    gt = {"root": {"openers": [[7, 1]], "setups": []}}
    row = index_row(_trace(pool), gt, "hard")
    assert row["rank_best_green"] == 2
    assert row["top1"] == "dead"
    assert row["tier"] == "hard" and row["has_gt"] is True
    assert row["key"] == "a__o"


def test_index_row_without_gt_is_marked_and_has_no_rank():
    pool = [{"obj": "o", "edge": 5, "depth": 0, "q": 0.9}]
    row = index_row(_trace(pool), None, "easy")
    assert row["has_gt"] is False
    assert row["rank_best_green"] is None
    assert row["top1"] == "unknown"
```

- [ ] **Step 2: Run to verify it fails**

```bash
PYTHONPATH="$PWD/build_python:$PWD/python" python -m pytest python/tests/test_viz_index_metrics.py -v
```

Expected: FAIL importing `viz.build_manifest`.

- [ ] **Step 3: Implement**

Create `scripts/viz/build_manifest.py`:

```python
#!/usr/bin/env python3
"""Build manifest.json for viz/search: arms, episodes with difficulty tiers, and index rows."""
import argparse
import glob
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
for _p in (f"{REPO}/python", f"{REPO}/scripts"):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from namo import eval_sets  # noqa: E402
from viz.index_metrics import rank_of_best_green, top1_truth  # noqa: E402
from viz.trace_schema import episode_filename  # noqa: E402


def index_row(trace, gt, tier):
    xml = trace["meta"]["xml"]
    oid = trace["meta"]["object_id"]
    root = next(b for b in trace["boards"] if b["depth"] == 0)
    pool = root["pool"]
    if gt is None:
        rank, top1 = None, "unknown"
    else:
        openers = {(e, d) for e, d in gt["root"]["openers"]}
        setups = {(e, d) for e, d in gt["root"]["setups"]}
        rank = rank_of_best_green(pool, openers | setups)
        top1 = top1_truth(pool, openers, setups)
    return {"key": episode_filename(xml, oid)[:-len(".json")],
            "xml": xml, "object_id": oid, "tier": tier,
            "solved": trace["result"]["solved"], "sims": trace["result"]["sims"],
            "rank_best_green": rank, "top1": top1, "has_gt": gt is not None}


def _tiers():
    div = json.load(open(eval_sets.DIVISIONS))
    out = {}
    for tier, entries in div.items():
        for e in entries:
            out[(os.path.realpath(e["xml"]), e["object_id"])] = tier
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True, help="dir holding trace/<model>/<strategy>/ and gt/")
    a = ap.parse_args()
    tiers = _tiers()
    arms, index, episodes = [], {}, {}
    for tdir in sorted(glob.glob(os.path.join(a.data_root, "trace", "*", "*"))):
        strategy = os.path.basename(tdir)
        model = os.path.basename(os.path.dirname(tdir))
        arm = f"{model}|{strategy}"
        arms.append({"model": model, "strategy": strategy, "dir": f"trace/{model}/{strategy}"})
        rows = []
        for tf in sorted(glob.glob(os.path.join(tdir, "*.json"))):
            trace = json.load(open(tf))
            gtf = os.path.join(a.data_root, "gt", os.path.basename(tf))
            gt = json.load(open(gtf)) if os.path.exists(gtf) else None
            tier = tiers.get((os.path.realpath(trace["meta"]["xml"]), trace["meta"]["object_id"]), "unknown")
            row = index_row(trace, gt, tier)
            rows.append(row)
            episodes[row["key"]] = {"xml": row["xml"], "object_id": row["object_id"],
                                    "tier": tier, "has_gt": row["has_gt"]}
        index[arm] = rows
        print(f"{arm}: {len(rows)} episodes")
    json.dump({"arms": arms, "episodes": episodes, "index": index},
              open(os.path.join(a.data_root, "manifest.json"), "w"))


if __name__ == "__main__":
    main()
```

If `_tiers()` raises because `pure2push_divisions.json` has a different shape than `{tier: [{xml, object_id}]}`, open the file first and adapt the two lines that read it — do not guess. Its path is `eval_sets.DIVISIONS` and its counts must come to easy 238 / medium 409 / hard 371.

- [ ] **Step 4: Run to verify it passes**

```bash
PYTHONPATH="$PWD/build_python:$PWD/python" python -m pytest python/tests/test_viz_index_metrics.py -v
```

Expected: 8 passed.

- [ ] **Step 5: Commit**

```bash
git add scripts/viz/build_manifest.py python/tests/test_viz_index_metrics.py
git commit -m "feat: manifest builder for the search visualization index"
```

---

### Task 7: Index page

**Files:**
- Create: `viz/search/index.html`, `viz/search/style.css`
- Modify: none

**Interfaces:**
- Consumes: `manifest.json` produced by Task 6.
- Produces: a page linking to `episode.html?arm=<model>|<strategy>&key=<episode key>`.

- [ ] **Step 1: Write the page**

Create `viz/search/index.html`. It fetches `manifest.json` from the same directory, fills three `<select>` elements, and renders a table sorted by `rank_best_green` descending with nulls first (worst first, as the spec requires). Columns: scene, object, tier, solved, sims, rank of best green, top-1 truth, GT flag. Each row links to the episode view. Keep all logic inline in one `<script>` block — this page is small enough that a separate module would only add indirection.

Behaviour to implement, stated so a reviewer can check it:

- The model and strategy dropdowns are populated from `manifest.arms`, the difficulty dropdown from the fixed list `all, easy, medium, hard`.
- Changing any dropdown re-renders the table from `manifest.index[model + "|" + strategy]`, filtered by tier.
- Clicking a column header sorts by it, toggling direction. `rank_best_green` nulls always sort to the "worst" end.
- A header line shows counts: total episodes shown, how many solved, and how many lack GT.

- [ ] **Step 2: Write the stylesheet**

Create `viz/search/style.css` with the palette used across both pages: solid green `#2e7d32` for openers, hollow green (green border, transparent fill) for setups, red `#c62828` for dead, grey `#9e9e9e` for masked or unknown. Monospace for numeric columns so ranks line up. Keep it under 150 lines.

- [ ] **Step 3: Verify against real data**

```bash
cd "$NAMO_SCRATCH/viz/search" && ln -sfn "$REPO/viz/search"/*.html "$REPO/viz/search"/*.css "$REPO/viz/search"/*.js .
python -m http.server 8765
```

Open `http://localhost:8765/index.html`. Confirm: the table renders, the tier counts sum to the episodes present, sorting by rank of best green puts the worst episodes on top, and the browser console is clean.

- [ ] **Step 4: Commit**

```bash
git add viz/search/index.html viz/search/style.css
git commit -m "feat: episode index page for the search visualization"
```

---

### Task 8: Episode replay view

The four zones. This is the largest single piece of UI; keep the data plumbing in `app.js` and the markup skeleton in `episode.html`.

**Files:**
- Create: `viz/search/episode.html`, `viz/search/app.js`
- Modify: `viz/search/style.css`

**Interfaces:**
- Consumes: `manifest.json`, `trace/<model>/<strategy>/<key>.json`, `gt/<key>.json`.
- Produces: nothing downstream.

- [ ] **Step 1: Build the state model in `app.js`**

One module-level object holds `{trace, gt, t}` where `t` is the sim index from 0 to `result.sims`. Two derived functions, both pure so they can be reasoned about independently:

- `queueAt(t)` — replay the pops from 0 to `t` to reconstruct which candidates remain unsimulated and each board's current `w`, then return the remaining candidates sorted by `bp * w` descending. The board weight after `k` failures is read off the pops rather than recomputed, because the trace records `w` at each pop and that is the ground truth of what the search actually did.
- `greenAt(boardId)` — return the opener and setup `(edge, depth)` sets for that board: `gt.root` for the depth-0 board, `gt.finish["<parent_edge>_<parent_depth>"]` for a child board, and empty sets when `gt` is null.

- [ ] **Step 2: Zone A, the scene**

Render an inline SVG sized from `scene.bounds`, drawing in this order: static rectangles, the goal marker, movable object rectangles, the robot, then one circle per entry of `scene.contacts`. Each contact circle is filled by the model's `q` for its best depth at the current board, stroked green when that `(edge, depth)` is green, and dimmed with a small pop-order number once simulated.

- [ ] **Step 3: Zone B, the priority queue**

Render `queueAt(t)` as rows: rank, board tag, `(edge, depth)`, a two-segment bar for `bp` and the `×w` shrinkage, and the green badge. Color rows by `board_id`. Above the list, show `best green currently at #k`, computed by finding the first green row — this is the number the whole page exists to make visible.

- [ ] **Step 4: Zone C, the timeline**

An `<input type="range">` from 0 to `result.sims`, plus one tick per pop marked pass or fail. Moving it sets `t` and re-renders A, B and D. Left and right arrow keys step it.

- [ ] **Step 5: Zone D, rank space**

Two 60×5 grids side by side for the current board: the left colored by the model's rank within that board's `grid` (rank 1 darkest), the right showing green cells only. Below them, the raw range of the grid, so a viewer can see the magnitudes without the page ever comparing them across models. When `board.grid` is null (uniform prior or tracing was off for that board), show an explicit "no grid recorded" placeholder rather than an empty box.

- [ ] **Step 6: Cross-highlighting**

Hovering any of: a contact circle in A, a row in B, or a cell in D highlights the corresponding element in the other two. Implement with a single `hover = {edge, depth}` state field and a re-render, not with per-element event wiring between zones.

- [ ] **Step 7: Verify against real data**

Serve as in Task 7 and open an episode with a high `rank_best_green` (the top of the default sort). Confirm: scrubbing re-sorts zone B, the `best green at #k` number changes as boards get demoted, hovering cross-highlights all three zones, and the console is clean. Then open a `has_gt: false` episode and confirm badges and zone D's right panel are suppressed rather than blank-but-clickable.

- [ ] **Step 8: Commit**

```bash
git add viz/search/episode.html viz/search/app.js viz/search/style.css
git commit -m "feat: four-zone episode replay for the search visualization"
```

---

### Task 9: Registry documentation

**Files:**
- Modify: `docs/experiments/horizon_q_model_registry.md`
- Modify: `docs/experiments/eval_set_registry.md`

- [ ] **Step 1: Add `train_h5:` to every model registry entry**

Walk every entry. Where the training H5 is already named in prose (for example `beast2c_d20_ceil.h5`, `d20_plus_setup_only.h5`, `round3/h5/d20_plus_setup_only_HARD.h5`), add an explicit `train_h5:` field with that path. Where no entry records one, write `train_h5: unrecorded`. Do not infer a value from a similar entry — an inferred path here would be worse than an honest gap.

- [ ] **Step 2: Add the GT schema block to the eval set registry**

Under the `testset_gt.h5` row, add a fenced block listing the datasets verified on 2026-07-26: `chain_depth (N,) int8`, `contact_px (N,60,2) f32`, `ctx (N,5,64,64) f16`, `edges_agree (N,) int8`, `f_grid (N,60,5) f32`, `is_solution_node (N,) int8`, `n_reach_edges/n_tried/n_win (N,) int32`, `node_kind (N,) str in {root, depth2, depth2_noop}`, `object_id (N,) str`, `parent_depth (N,) int16`, `parent_edge (N,) int16`, `r_mask/value_mask/value_target (N,60,5) f32`, `robot_goal (N,3) f32`, `setup_moved (N,) int8`, `xml (N,) str`. Record the row counts `root=1117, depth2=49622, depth2_noop=15717`, the two join keys — `(xml, object_id)` for a root and `(xml, object_id, parent_edge, parent_depth)` for a finish state — and the fact that this file carries no `ceiling_mask`, so its `0` cells are hard zeros from an exhaustive sweep rather than ceiling-optimistic values.

- [ ] **Step 3: Commit**

```bash
git add docs/experiments/horizon_q_model_registry.md docs/experiments/eval_set_registry.md
git commit -m "docs: train_h5 per model entry, testset_gt.h5 schema block"
```

---

### Task 10: Production trace run

Four arms over the full test set. This is a scaled job, so it goes through the pre-flight discipline rather than straight to a full launch.

**Files:** none (execution only)

- [ ] **Step 1: Invoke the scaled-run skill**

Before submitting anything, use the `scaled-run` skill. It owns smoke-testing, time calibration and SLURM sizing. Use `compute-resources` to pick the partition; heavy CPU work goes to a SLURM CPU partition, not an interactive node, and no `--time` limit is set unless the user asks for one.

- [ ] **Step 2: Calibrate on 20 episodes**

Run one arm over `--start 0 --end 20` and record wall time per episode. Multiply by 1018 and by 4 arms to size the array. The spec's estimate is about 8 CPU-hours per arm.

- [ ] **Step 3: Launch all four arms**

Arms are the cross product of two models and two strategies:

```
ceiling  = $NAMO_SCRATCH/round3/models/d20_plus_setup_only_splitloss/checkpoints/epoch011-val_loss1.6952.ckpt
hard     = $NAMO_SCRATCH/round3/models/setup_split_HARD_seed1/checkpoints/epoch011-val_loss0.8787.ckpt
strategy off  = --discount off
strategy conf = --discount conf --tau 0.15
```

Each writes to `$NAMO_SCRATCH/viz/search/trace/<model>/<strategy>/`, with `--trace-model` set to `ceiling` or `hard`.

- [ ] **Step 4: Build the manifest and verify counts**

```bash
PYTHONPATH="$PWD/build_python:$PWD/python" python scripts/viz/build_manifest.py \
  --data-root "$NAMO_SCRATCH/viz/search"
```

Expected: four arms printed, each with 1018 episodes. Then confirm the tier split in `manifest.json` is easy 238 / medium 409 / hard 371 and that exactly 37 episodes carry `has_gt: false`.

- [ ] **Step 5: Record the result**

Append the run to `docs/experiments/RESULTS.md` and update the relevant experiment card, following `docs/experiments/WORKFLOW.md`. Report by difficulty and horizon, never aggregate-only.

---

## Self-Review Notes

Spec coverage checked section by section: data model (Tasks 2, 4, 5), trace generation and the byte-identical gate (Task 5), index page (Task 7), episode view zones A-D (Task 8), green-badge semantics (Tasks 3, 4, 8), rank space (Task 8 Step 5), the 981/1018 coverage handling (Task 4 Step 5, Task 6, Task 8 Step 7), side deliverables (Task 9), and the compute run (Task 10).

Two places deliberately tell the implementer to check reality instead of trusting this document: the pixel-convention assertion in Task 1 Step 2 and the divisions-file shape in Task 6 Step 3. Both are cases where guessing would produce silently wrong numbers.
