"""On-disk contract for one episode's search trace. Pure data, no simulator imports.

Consumed by the static page in viz/search/. Bump schema_version if any field changes meaning."""
import hashlib
import os

SCHEMA_VERSION = 2   # v2: meta carries the generator's full search-parameter set (meta["search"])


def episode_filename(xml_path, object_id):
    stem = os.path.splitext(os.path.basename(xml_path))[0]
    digest = hashlib.sha1(os.path.realpath(xml_path).encode()).hexdigest()[:8]
    return f"{stem}__{object_id}__{digest}.json"


def make_board(board_id, depth, parent_edge, parent_depth, pool, grid, w0, free_strikes):
    return {"board_id": board_id, "depth": depth,
            "parent_edge": parent_edge, "parent_depth": parent_depth,
            "n_candidates": len(pool), "pool": pool, "grid": grid,
            "w0": w0, "free_strikes": free_strikes}


def make_pop(t, board_id, obj, edge, depth, q, bp, w, opened):
    """`w` is the board weight AS THE POP SAW IT -- i.e. BEFORE this pop's own failure demotes it. A board's
    post-failure weight is therefore NOT in pops[]; consumers must recompute it from meta["search"]."""
    return {"t": t, "board_id": board_id, "obj": obj, "edge": edge, "depth": depth,
            "q": q, "bp": bp, "w": w, "se": bp * w, "opened": bool(opened)}


def build_trace(meta, scene, boards, pops, result):
    return {"schema_version": SCHEMA_VERSION, "meta": meta, "scene": scene,
            "boards": boards, "pops": pops, "result": result}
