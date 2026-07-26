"""On-disk contract for one episode's search trace. Pure data, no simulator imports.

Consumed by the static page in viz/search/. Bump schema_version if any field changes meaning."""
import hashlib
import os

SCHEMA_VERSION = 3   # v3: each board carries the geometry AND the region decomposition of ITS OWN state
# v2: meta carries the generator's full search-parameter set (meta["search"])


def episode_filename(xml_path, object_id):
    stem = os.path.splitext(os.path.basename(xml_path))[0]
    digest = hashlib.sha1(os.path.realpath(xml_path).encode()).hexdigest()[:8]
    return f"{stem}__{object_id}__{digest}.json"


def rle_encode(rows):
    """ROW-WISE run-length encode a 2-D integer grid. THE format for board["regions"]["rle"].

    `rows` = a sequence of equal-length integer sequences, indexed [ix][iy] (x-major, matching
    WavefrontSnapshot.region_map, whose cell (ix, iy) is the world square
    [x0 + ix*res, x0 + (ix+1)*res) x [y0 + iy*res, y0 + (iy+1)*res)).

    Returns ONE FLAT list of alternating value/count integers: [v0, n0, v1, n1, ...]. Runs NEVER
    cross a row boundary -- a run is always a contiguous span of one row ix, so a consumer can draw
    each run as a single rectangle (x = x0 + ix*res, width = res, y = y0 + iy_start*res,
    height = n*res) with no bookkeeping. Concatenating the runs in order reproduces exactly the
    row-major (C-order) flatten of the grid, so sum(counts) == nx*ny always."""
    out = []
    for row in rows:
        prev = None
        n = 0
        for v in row:
            v = int(v)
            if v == prev:
                n += 1
                continue
            if prev is not None:
                out.append(prev)
                out.append(n)
            prev = v
            n = 1
        if prev is not None:
            out.append(prev)
            out.append(n)
    return out


def rle_decode(flat, nx, ny):
    """Inverse of rle_encode: the (nx, ny) grid as a list of nx rows of ny ints."""
    rows = []
    row = []
    for i in range(0, len(flat), 2):
        v, n = flat[i], flat[i + 1]
        row.extend([v] * n)
        while len(row) >= ny:
            rows.append(row[:ny])
            row = row[ny:]
    return rows


def make_board(board_id, depth, parent_edge, parent_depth, pool, grid, w0, free_strikes,
               geom=None, regions=None):
    """geom / regions (v3, both None on a trace written without --trace-out geometry):
      geom    = {"movable": {name: [x, y, theta]}, "robot": [x, y, theta], "contacts": [[x, y] x60]}
                -- the poses AT THIS BOARD'S STATE (sizes stay in the episode-level `scene`, they
                never move). contacts = the 60 push points of the target object at this state.
      regions = {"nx", "ny", "res", "origin": [x0, y0], "labels": {"<id>": "robot"|"goal"|
                "robot_goal"|"region_N"}, "rle": [...]} -- the wavefront region decomposition AT
                THIS BOARD'S STATE. rle is rle_encode()'s flat value/count list over the (nx, ny)
                region-id grid; id 0 = no region (obstacle, or a border-touching component the
                exporter drops)."""
    return {"board_id": board_id, "depth": depth,
            "parent_edge": parent_edge, "parent_depth": parent_depth,
            "n_candidates": len(pool), "pool": pool, "grid": grid,
            "w0": w0, "free_strikes": free_strikes,
            "geom": geom, "regions": regions}


def make_pop(t, board_id, obj, edge, depth, q, bp, w, opened):
    """`w` is the board weight AS THE POP SAW IT -- i.e. BEFORE this pop's own failure demotes it. A board's
    post-failure weight is therefore NOT in pops[]; consumers must recompute it from meta["search"]."""
    return {"t": t, "board_id": board_id, "obj": obj, "edge": edge, "depth": depth,
            "q": q, "bp": bp, "w": w, "se": bp * w, "opened": bool(opened)}


def build_trace(meta, scene, boards, pops, result):
    return {"schema_version": SCHEMA_VERSION, "meta": meta, "scene": scene,
            "boards": boards, "pops": pops, "result": result}
