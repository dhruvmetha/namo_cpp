"""Parallel H5 render for the CS-side generation driver.

build_train_h5.build_h5 renders every kept (state,action) row serially (~64 ms/row) and
re-renders the whole accumulating buffer each generation; at ~10^5 rows that is hours. This
renders across processes and merges the shard H5s (byte-identical rows, just sharded).

CRITICAL: the pool uses the **spawn** start method, NOT fork. The parent has already imported
torch + the sage visualizer (build_train_h5 / eval_gen pull them in), which spin up OpenMP/MKL
threadpools; forking then leaves the children blocked on inherited locked mutexes (a hard hang
we hit on the first gen-0 attempt — 48-80 workers, ~0% CPU, all in S state). Spawn re-imports
in fresh interpreters, so there is no inherited lock. This lives in a real module (not the
run_gen_cs __main__ script) so spawn can import the worker cleanly.
"""
import multiprocessing as mp
import os

from ._bootstrap import ensure_paths
ensure_paths()
from .build_train_h5 import _write_h5   # noqa: E402


def build_specs(buffer, cfg):
    """Row specs (meta, step, pi_weight, value_target, v_weight, is_solved) — NO rendering yet.
    Replicates build_train_h5.build_h5's weight/target logic exactly."""
    specs = []
    for k, entries in buffer.solves.items():
        meta = buffer.meta[k]
        if not entries:
            continue
        tmin = min(e["T"] for e in entries)
        raw = [2.0 ** (-(e["T"] - tmin)) for e in entries]
        z = sum(raw)
        for e, w0 in zip(entries, raw):
            w_traj = w0 / z
            T = e["T"]
            v_w = cfg.vhead_recency_decay ** max(0, buffer.generation - e["generation"])
            m = dict(meta); m["_gen"] = e["generation"]
            for i, st in enumerate(e["steps"]):
                specs.append((m, st, w_traj / T, cfg.gamma ** (T - 1 - i), v_w, 1))
    for e in buffer.fails:
        meta = buffer.meta.get(e["episode"])
        if meta is None:
            continue
        v_w = cfg.vhead_recency_decay ** max(0, buffer.generation - e["generation"])
        m = dict(meta); m["_gen"] = e["generation"]
        for st in e["steps"]:
            specs.append((m, st, 0.0, 0.0, v_w, 0))
    return specs


def _render_shard(args):
    render_config, specs, out_path = args
    from .build_train_h5 import _Renderer, _EnvCache, _render_row
    renderer = _Renderer(render_config)
    envc = _EnvCache()
    rows = []
    for (m, st, pi_w, val_t, v_w, is_solved) in specs:
        try:
            rows.append(_render_row(renderer, envc, m, st, pi_w, val_t, v_w, is_solved))
        except Exception as ex:
            print(f"  [render] skip {m.get('xml_key')}: {ex}", flush=True)
    _write_h5(out_path, rows)
    return (out_path, len(rows))


def _merge_h5(parts, out_h5):
    import h5py
    paths = [p for (p, n) in parts if n > 0]
    total = sum(n for (p, n) in parts)
    os.makedirs(os.path.dirname(out_h5), exist_ok=True)
    with h5py.File(out_h5, "w") as f:
        f.attrs["n_samples"] = total
        if total == 0:
            return
        with h5py.File(paths[0], "r") as first:
            keys = [k for k in first.keys()]
            layout = {k: (first[k].shape[1:], first[k].dtype, first[k].compression) for k in keys}
        dsets = {k: f.create_dataset(k, shape=(total,) + shp, dtype=dt, compression=comp)
                 for k, (shp, dt, comp) in layout.items()}
        off = 0
        for p in paths:
            with h5py.File(p, "r") as pf:
                n = pf["chosen_edge"].shape[0]
                for k in keys:
                    dsets[k][off:off + n] = pf[k][:]
                off += n


def build_h5_parallel(buffer, cfg, out_h5, render_config, n_workers):
    specs = build_specs(buffer, cfg)
    n_bc = sum(1 for s in specs if s[5] == 1)
    if not specs:
        _write_h5(out_h5, [])
        return {"n_rows": 0, "n_bc_rows": 0, "n_v_only": 0, "out_h5": out_h5}
    n_workers = max(1, min(n_workers, len(specs)))
    # CONTIGUOUS chunks (not round-robin): build_specs emits all of one episode's rows
    # consecutively, so contiguous shards keep a room's rows in ONE worker -> its _EnvCache
    # reuses make_env instead of every worker cold-loading every room.
    chunk = (len(specs) + n_workers - 1) // n_workers
    shards = [specs[i * chunk:(i + 1) * chunk] for i in range(n_workers)]
    tmpdir = out_h5 + ".shards"
    os.makedirs(tmpdir, exist_ok=True)
    tasks = [(render_config, shards[i], os.path.join(tmpdir, f"part{i}.h5"))
             for i in range(n_workers) if shards[i]]
    if len(tasks) == 1:
        parts = [_render_shard(tasks[0])]
    else:
        ctx = mp.get_context("spawn")            # NOT fork — see module docstring
        with ctx.Pool(len(tasks)) as pool:
            parts = pool.map(_render_shard, tasks)
    _merge_h5(parts, out_h5)
    for (p, _n) in parts:
        try:
            os.remove(p)
        except OSError:
            pass
    return {"n_rows": len(specs), "n_bc_rows": int(n_bc), "n_v_only": len(specs) - int(n_bc),
            "out_h5": out_h5}
