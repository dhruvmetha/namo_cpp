"""Buffer -> training H5 (one file, both heads).

For every kept (state, action) we render the model ctx ONCE (deferred from collection) and
attach the labels + weights both heads need. One row per (state, action):

  ctx           (5,64,64) f16   scene crop (rendered from the stored state, robot_goal-conditioned
                                 exactly like the deploy scorer / eval_reactive_argmax)
  contact_px    (60,2)   f32
  r_mask        (60,5)   f32     legal cells = reachable edges x 5 depths
  chosen_edge / chosen_depth     the taken action
  pi_weight     f32              BC weight: uniform within a trajectory; across an episode's
                                 trajectories mass ~ 2^-(T-T_min); episode totals 1. 0 on failed rows.
  value_target  f32              MC return gamma^(T-1-i) for solved step i; 0 for failed rows.
  v_weight      f32              recency: decay^(gen_now - gen_row).
  is_solved     i8               1 = BC-eligible (from a solved trajectory); 0 = V-only (failed).
  xml           str              room id (split grouping)
  generation    i32

Rendering reuses LiveScorer.render_ctx / contact_px_live via a MODEL-FREE subclass (render_ctx
touches only the visualizer, never the model), so no ckpt is needed to build gen-0 (arm A) data.
"""
from typing import Dict, List, Tuple
import os

import numpy as np

from ._bootstrap import ensure_paths
ensure_paths()
from live_scorer import LiveScorer, CROP_M, OUT                   # noqa: E402
from scorer_beam import make_env, FALLBACK_GOAL                   # noqa: E402
from namo.core.xml_goal_parser import extract_goal_with_fallback  # noqa: E402
from namo.paths import resolve                                    # noqa: E402
import namo_rl                                                    # noqa: E402

from .config import LoopConfig, NUM_DEPTHS
from .buffer import SolveBuffer


class _Renderer(LiveScorer):
    """LiveScorer without the model — render_ctx/contact_px_live only need the visualizer."""
    def __init__(self, render_config: str, crop_m: float = CROP_M, num_depths: int = NUM_DEPTHS):
        self.device = "cpu"
        self.model = None
        self.crop_m = crop_m
        self.num_depths = num_depths
        from sage_learning.visualizer import NAMODataVisualizer, NAMOXMLParser
        self.viz = NAMODataVisualizer(namo_config_path=render_config)
        self._XMLParser = NAMOXMLParser
        self.last_fell_back = False


def _rlstate(qpos, qvel) -> "namo_rl.RLState":
    s = namo_rl.RLState(); s.qpos = list(qpos); s.qvel = list(qvel); return s


class _EnvCache:
    def __init__(self):
        self._c: Dict[str, Tuple[object, tuple]] = {}

    def get(self, xml_key: str):
        if xml_key not in self._c:
            xml = str(resolve(xml_key))
            env = make_env(xml)
            goal = extract_goal_with_fallback(xml, FALLBACK_GOAL)
            self._c[xml_key] = (env, goal, xml)
        return self._c[xml_key]


def _render_row(renderer, envc, meta, st, pi_weight, value_target, v_weight, is_solved):
    env, goal, xml = envc.get(meta["xml_key"])
    env.set_full_state(_rlstate(st["qpos"], st["qvel"]))
    ctx, _ = renderer.render_ctx(env, meta["object_id"], goal, xml)
    cpx = renderer.contact_px_live(env, meta["object_id"])
    r_mask = np.zeros((60, NUM_DEPTHS), dtype=np.float32)
    for e in st["reachable_edges"]:
        if 0 <= e < 60:
            r_mask[e, :] = 1.0
    return {
        "ctx": ctx.astype(np.float16), "contact_px": cpx.astype(np.float32), "r_mask": r_mask,
        "chosen_edge": int(st["edge"]), "chosen_depth": int(st["depth"]),
        "pi_weight": float(pi_weight), "value_target": float(value_target),
        "v_weight": float(v_weight), "is_solved": int(is_solved),
        "xml": meta["xml_key"], "generation": int(meta.get("_gen", 0)),
    }


def build_h5(buffer: SolveBuffer, cfg: LoopConfig, out_h5: str, render_config: str) -> dict:
    renderer = _Renderer(render_config)
    envc = _EnvCache()
    rows: List[dict] = []

    # --- BC rows (solved trajectories) + their V targets ---
    for k, entries in buffer.solves.items():
        meta = buffer.meta[k]
        if not entries:
            continue
        tmin = min(e["T"] for e in entries)
        raw = [2.0 ** (-(e["T"] - tmin)) for e in entries]
        z = sum(raw)
        for e, w0 in zip(entries, raw):
            w_traj = w0 / z                       # episode's trajectory weights sum to 1
            T = e["T"]
            v_w = cfg.vhead_recency_decay ** max(0, buffer.generation - e["generation"])
            m = dict(meta); m["_gen"] = e["generation"]
            for i, st in enumerate(e["steps"]):
                rows.append(_render_row(renderer, envc, m, st,
                                        pi_weight=w_traj / T,
                                        value_target=cfg.gamma ** (T - 1 - i),
                                        v_weight=v_w, is_solved=1))

    # --- V-only rows (failed rollouts -> return 0) ---
    for e in buffer.fails:
        meta = buffer.meta.get(e["episode"])
        if meta is None:
            continue
        v_w = cfg.vhead_recency_decay ** max(0, buffer.generation - e["generation"])
        m = dict(meta); m["_gen"] = e["generation"]
        for st in e["steps"]:
            rows.append(_render_row(renderer, envc, m, st,
                                    pi_weight=0.0, value_target=0.0, v_weight=v_w, is_solved=0))

    _write_h5(out_h5, rows)
    n_bc = sum(r["is_solved"] for r in rows)
    return {"n_rows": len(rows), "n_bc_rows": int(n_bc), "n_v_only": len(rows) - int(n_bc),
            "out_h5": out_h5}


def _write_h5(path: str, rows: List[dict]) -> None:
    import h5py
    os.makedirs(os.path.dirname(path), exist_ok=True)
    n = len(rows)
    with h5py.File(path, "w") as f:
        f.attrs["n_samples"] = n
        if n == 0:
            return
        f.create_dataset("ctx", data=np.stack([r["ctx"] for r in rows]), compression="lzf")
        f.create_dataset("contact_px", data=np.stack([r["contact_px"] for r in rows]))
        f.create_dataset("r_mask", data=np.stack([r["r_mask"] for r in rows]))
        f.create_dataset("chosen_edge", data=np.array([r["chosen_edge"] for r in rows], np.int16))
        f.create_dataset("chosen_depth", data=np.array([r["chosen_depth"] for r in rows], np.int16))
        f.create_dataset("pi_weight", data=np.array([r["pi_weight"] for r in rows], np.float32))
        f.create_dataset("value_target", data=np.array([r["value_target"] for r in rows], np.float32))
        f.create_dataset("v_weight", data=np.array([r["v_weight"] for r in rows], np.float32))
        f.create_dataset("is_solved", data=np.array([r["is_solved"] for r in rows], np.int8))
        f.create_dataset("generation", data=np.array([r["generation"] for r in rows], np.int32))
        dt = h5py.string_dtype(encoding="utf-8")
        f.create_dataset("xml", data=np.array([r["xml"] for r in rows], dtype=object), dtype=dt)
