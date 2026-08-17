#!/usr/bin/env python3
"""LIVE 1-push SCORER bridge.

Given a RUNNING `namo_rl.RLEnvironment` at some state, render the 5-channel tight crop the trained
scorer expects and run the model on it — the SAME (60,5) P(this push opens a path to the robot goal)
the H5-fed scorer produces, but computed from the LIVE env instead of a precomputed H5.

Design (faithfulness by construction):
  - The 5-channel tight crop is rendered by the EXACT training renderer
    (`sage_learning.visualizer.NAMODataVisualizer.generate_all_masks_highres`, unified-wavefront path)
    fed an episode_data dict assembled from live env queries (get_observation / get_object_info /
    get_reachable_objects). Same draw code, same WavefrontSnapshotExporter region BFS, same
    highres=1024 -> tight crop 0.5 m @224 -> resize 64 (INTER_AREA) as the H5 build.
  - contact_px (60,2) is pure geometry (object theta + half-extents), identical formula to the H5.
  - The scorer is loaded once via eval_scorer.load_scorer (auto-detects arch from the state_dict).

NOTE on config: the v3 car data + the test manifest were collected/rendered with
`namo_config_complete_skill15_car_1x.yaml` (robot_size 0.035, primitives motion_primitives_1x_car.dat,
high_level_resolution 0.01). That is what built the H5 region masks and the reachable/valid labels, so
this script uses it for BOTH the env and the renderer by default (NOT namo_config_car.yaml, robot 0.052,
which would shift the wavefront/reachability off the training distribution). Override with --env-config /
--render-config.

`--validate` runs both gates and prints PASS/FAIL.
"""
import argparse
import json
import math
import os
import sys

import numpy as np

from pathlib import Path
REPO = Path(__file__).resolve().parents[2]; SAGE = os.environ.get("SAGE_REPO", "")
for _p in (f"{REPO}/build_python", f"{REPO}/python", SAGE, f"{REPO}/scripts", f"{REPO}/scripts/sandbox"):
    if _p and _p not in sys.path:
        sys.path.insert(0, _p)

import cv2  # noqa: E402
import torch  # noqa: E402

# Reuse the verified loader + geometry from the sandbox eval script (no re-implementation).
from eval_scorer import load_scorer, contact_px as contact_px_fn, match_episode  # noqa: E402
from namo.paths import SCRATCH, H5, MANIFESTS, resolve  # noqa: E402

CHANS = ["static", "movable", "target_object", "robot_region", "goal_sample_region"]
TIGHT = [f"local_tight_{c}" for c in CHANS]
OUT = 64
CROP_M = 0.5

DEFAULT_CKPT = str(SCRATCH / "sage_outputs/scorer/e4seed_s1/namo-classifier"
                   "/p2y7ihae/checkpoints/epoch029-val_loss0.2956.ckpt")
SKILL15_CFG = f"{REPO}/config/namo_config_complete_skill15_car_1x.yaml"


# --------------------------------------------------------------------------------------------------
# LiveScorer
# --------------------------------------------------------------------------------------------------
class LiveScorer:
    """Loads the scorer + renderer once; scores a live env state."""

    def __init__(self, ckpt=DEFAULT_CKPT, render_config=SKILL15_CFG, device=None,
                 num_depths=5, crop_m=CROP_M, network="edge_crossattn"):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = load_scorer(ckpt, num_depths, self.device, network)
        self.crop_m = crop_m
        self.num_depths = num_depths
        from sage_learning.visualizer import NAMODataVisualizer, NAMOXMLParser
        self.viz = NAMODataVisualizer(namo_config_path=render_config)
        self._XMLParser = NAMOXMLParser
        self.last_fell_back = False  # set True if the renderer used legacy BFS for the last call

    # -- target / goal helpers ---------------------------------------------------------------------
    @staticmethod
    def find_target_by_center(env, object_center, tol=0.02):
        """Return (object_name, dist) of the movable closest to object_center."""
        obs = env.get_observation()
        best, bd = None, 1e9
        for k, v in obs.items():
            if k == "robot_pose" or not k.endswith("_pose"):
                continue
            name = k[:-len("_pose")]
            d = (v[0] - object_center[0]) ** 2 + (v[1] - object_center[1]) ** 2
            if d < bd:
                bd, best = d, name
        return best, math.sqrt(bd)

    def xml_goal(self, xml_file):
        try:
            return tuple(self._XMLParser(xml_file).parse_environment().robot_goal)
        except Exception:
            return (0.0, 0.0, 0.0)

    # -- episode_data assembly from live env -------------------------------------------------------
    def _episode_data(self, env, target_object, robot_goal, xml_file, region_samples=None):
        """region_samples: optional list of (x,y[,theta]) for the TARGET region whose opening we
        score (the RO neighbour region). When given, the `goal_sample_region` channel is rendered
        from THESE points (matching how the scorer was trained — per-adjacency RO openings), and
        robot_goal is seeded to the first region sample (matches RO's ML-seed convention). When
        None, falls back to a single point at robot_goal (the legacy final-goal conditioning)."""
        obs = env.get_observation()
        oi = env.get_object_info()
        tp = obs[f"{target_object}_pose"]
        try:
            reach = [list(env.get_reachable_objects())]
        except Exception:
            reach = [[]]
        if region_samples:
            rs = [[float(p[0]), float(p[1])] for p in region_samples]
            first = region_samples[0]
            rg = [float(first[0]), float(first[1]),
                  float(first[2]) if len(first) > 2 else 0.0]
        else:
            rg = [float(robot_goal[0]), float(robot_goal[1]),
                  float(robot_goal[2]) if len(robot_goal) > 2 else 0.0]
            rs = [[rg[0], rg[1]]]
        return {
            # first state = current live observation (keys: 'robot_pose', '<obj>_pose')
            "state_observations": [dict(obs)],
            # walls carry pos_x/quat (static), movables carry size only — exactly what the
            # visualizer's _extract_env_info_from_episode expects.
            "static_object_info": {k: dict(v) for k, v in oi.items()},
            # target pose only affects non-scorer channels (target_goal / goal_mask) — placeholder.
            "action_sequence": [{"object_id": target_object,
                                 "target": [tp[0], tp[1], tp[2]]}],
            "robot_goal": rg,
            # goal_sample_region channel is rasterized from these points.
            "region_goals_sampled": rs,
            "reachable_objects_before_action": reach,
            "xml_file": xml_file,
        }

    def render_ctx(self, env, target_object, robot_goal, xml_file, region_samples=None):
        """Return (ctx (5,64,64) float32, meta dict)."""
        ep = self._episode_data(env, target_object, robot_goal, xml_file, region_samples)
        import io
        import contextlib
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            res = self.viz.generate_all_masks_highres(ep, tight_crop_size_meters=self.crop_m, fast_scorer=True)
        self.last_fell_back = "falling back to legacy BFS" in buf.getvalue()
        if res is None or res.get("local_tight") is None:
            raise RuntimeError("renderer returned None (missing target/region_goals_sampled)")
        lt = res["local_tight"]
        chans = [cv2.resize(lt[k].astype(np.float32), (OUT, OUT), interpolation=cv2.INTER_AREA)
                 for k in TIGHT]
        return np.stack(chans).astype(np.float32), res.get("local_tight_metadata", {})

    def contact_px_live(self, env, target_object):
        """(60,2) edge contact pixels in the 64-frame — pure geometry from live pose/size."""
        obs = env.get_observation()
        oi = env.get_object_info()
        th = obs[f"{target_object}_pose"][2]
        hw = oi[target_object]["size_x"]
        hd = oi[target_object]["size_y"]
        return np.array([contact_px_fn(e, hw, hd, th, self.crop_m, OUT) for e in range(60)],
                        dtype=np.float32)

    # -- scoring -----------------------------------------------------------------------------------
    def score_ctx(self, ctx, contact_px, h=1, raw=False):
        ct = torch.from_numpy(ctx[None]).float().to(self.device)
        cp = torch.from_numpy(contact_px[None]).float().to(self.device)
        # budget-Q ckpt: query at budget h (default 1; M3 uses h=2 for zero-sim foresight). HL-Gauss
        # head -> E[bin] in [0,1]; the sigmoid below is monotone so rankings/pools are unchanged.
        kw = {"H": torch.full((1,), int(h), dtype=torch.long, device=self.device)} \
            if getattr(self.model.network, "budget_cond", False) else {}
        if getattr(self.model.network, "action_motion_dim", 0) > 0:
            from namo.rl_loop.action_motion import action_motion_from_contact_px
            kw["action_motion"] = action_motion_from_contact_px(
                cp, encoding=self.model.network.action_motion_encoding,
                feature_dim=self.model.network.action_motion_dim)
        with torch.no_grad():
            t = self.model(ct, cp, **kw)[0]
            is_hl = t.dim() == 3
            if is_hl:
                from src.model.hl_gauss import HLGauss
                t = HLGauss(num_bins=t.shape[-1]).value(t)
            logits = t.cpu().numpy()
        # raw=True + HL-Gauss: return the E[bin] value DIRECTLY (already in [0,1]); the default sigmoid on
        # top squashes [0,1]->[0.5,0.73] (monotone, ranking-safe, but it MUSHES the magnitudes the search uses).
        if raw and is_hl:
            return logits
        return 1.0 / (1.0 + np.exp(-logits))

    def score_state(self, env, target_object, robot_goal, xml_file, region_samples=None, h=1, raw=False):
        """Return (60,5) P for `target_object` from the LIVE env at its current state.
        region_samples (RO neighbour region) conditions the goal_sample_region channel; see
        _episode_data. Pass it for region-opening; omit for the legacy final-goal conditioning."""
        ctx, _ = self.render_ctx(env, target_object, robot_goal, xml_file, region_samples)
        cpx = self.contact_px_live(env, target_object)
        return self.score_ctx(ctx, cpx, h=h, raw=raw)

    def topk(self, env, target_object, robot_goal, xml_file, k=5, candidate_edges=None):
        """Top-k (edge, depth, P) among reachable candidates (all depths per reachable edge)."""
        P = self.score_state(env, target_object, robot_goal, xml_file)
        if candidate_edges is None:
            env.get_reachable_objects()  # warm the wavefront before per-object edge query
            candidate_edges = env.get_reachable_edges(target_object)
        cands = [(e, d) for e in candidate_edges for d in range(P.shape[1])]
        cands.sort(key=lambda ed: -P[ed[0], ed[1]])
        return [(e, d, float(P[e, d])) for (e, d) in cands[:k]]


# --------------------------------------------------------------------------------------------------
# Convenience module-level function (task signature). Lazily builds a default scorer.
# --------------------------------------------------------------------------------------------------
_DEFAULT = {}


def score_state(env, target_object, robot_goal, xml_file, ckpt=DEFAULT_CKPT):
    if ckpt not in _DEFAULT:
        _DEFAULT[ckpt] = LiveScorer(ckpt=ckpt)
    return _DEFAULT[ckpt].score_state(env, target_object, robot_goal, xml_file)


# --------------------------------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------------------------------
def make_env(xml, cfg):
    import namo_rl
    env = namo_rl.RLEnvironment(str(resolve(xml)), cfg, False)  # resolve(): remap legacy data paths onto this box
    env.reset()
    return env


def topk_hit(P, valid, cand_idx, ks):
    flat = P.reshape(-1)
    order = sorted(cand_idx, key=lambda j: -flat[j])
    return {k: any((j // 5, j % 5) in valid for j in order[:k]) for k in ks}


def gate1_crop_match(scorer, env_cfg, n_samples, seed=0):
    """Recreate scenes from the TRAIN scorer H5 and compare live ctx to the stored model input."""
    import h5py
    from scipy import ndimage
    h5 = str(H5 / "v3_scorer_e4_data/data.h5")
    f = h5py.File(h5, "r")
    N = int(f.attrs["n_samples"])
    rng = np.random.default_rng(seed)
    idx = rng.choice(N, size=min(n_samples * 4, N), replace=False)  # oversample; some may fail match
    xmls = f["xml"]
    oc_all = f["object_center"]

    def boundary_dist(mask64, pts):
        """median px distance from each contact pt (x,y) to the nearest target-object pixel."""
        d = ndimage.distance_transform_edt(mask64 < 0.5)
        xs = np.clip(np.round(pts[:, 0]).astype(int), 0, 63)
        ys = np.clip(np.round(pts[:, 1]).astype(int), 0, 63)
        return float(np.median(d[ys, xs]))

    per_chan = {c: [] for c in CHANS}
    pix_match = {c: [] for c in CHANS}
    cpx_errs = []          # per-row max abs err of live vs STORED H5 contact_px
    cpx_align = []         # per-row median boundary dist of LIVE contact_px (geometric truth)
    fellback = 0
    done = []
    print("\n[Gate 1] crop-match vs train scorer H5 (v3_scorer_e4_data)", flush=True)
    for i in idx:
        if len(done) >= n_samples:
            break
        x = xmls[i]
        x = x.decode() if isinstance(x, bytes) else str(x)
        x = str(resolve(x))   # remap legacy data path onto this box before existence check
        if not os.path.exists(x):
            continue
        oc = oc_all[i]
        ctx_h5 = f["ctx"][i].astype(np.float32)          # (5,64,64)
        cpx_h5 = f["contact_px"][i].astype(np.float32)   # (60,2)
        try:
            env = make_env(x, env_cfg)
        except Exception as e:
            print(f"  skip (env build failed): {os.path.basename(x)}: {e}")
            continue
        env.get_reachable_objects()  # warm
        tgt, dist = scorer.find_target_by_center(env, oc)
        if tgt is None or dist > 0.005:   # only rows whose state == env-reset state (reproducible)
            continue
        rg = scorer.xml_goal(x)
        try:
            ctx_live, _ = scorer.render_ctx(env, tgt, rg, x)
        except Exception as e:
            print(f"  skip (render failed): {os.path.basename(x)}: {e}")
            continue
        if scorer.last_fell_back:
            fellback += 1
        cpx_live = scorer.contact_px_live(env, tgt)
        for ci, c in enumerate(CHANS):
            mae = float(np.mean(np.abs(ctx_live[ci] - ctx_h5[ci])))
            per_chan[c].append(mae)
            pix_match[c].append(float(np.mean(np.abs(ctx_live[ci] - ctx_h5[ci]) < 0.1)))
        cpx_errs.append(float(np.max(np.abs(cpx_live - cpx_h5))))
        cpx_align.append(boundary_dist(ctx_live[2], cpx_live))  # ch 2 = target_object
        done.append(x)
    f.close()

    print(f"  scenes compared: {len(done)}  (renderer legacy-BFS fallbacks: {fellback})")
    print(f"  {'channel':<22}{'mean MAE':>10}{'max MAE':>10}{'frac|diff|<0.1':>16}")
    geom_ok = True
    for c in CHANS:
        if not per_chan[c]:
            continue
        mae = np.mean(per_chan[c]); mx = np.max(per_chan[c]); pm = np.mean(pix_match[c])
        print(f"  {c:<22}{mae:>10.5f}{mx:>10.5f}{pm:>16.4f}")
        if c in ("static", "movable", "target_object") and mx > 0.02:
            geom_ok = False
    cpx_errs = np.array(cpx_errs) if cpx_errs else np.array([np.nan])
    frac_cpx = float(np.mean(cpx_errs < 0.5))
    print(f"  contact_px vs STORED H5: median={np.median(cpx_errs):.4f}px  p95={np.percentile(cpx_errs,95):.4f}px  "
          f"max={np.max(cpx_errs):.4f}px  frac rows <0.5px={frac_cpx*100:.1f}%")
    print(f"  contact_px boundary alignment (live cpx -> object edge): median={np.median(cpx_align):.3f}px "
          f"(geometric ground truth; ~1px = on the boundary)")
    # contact_px is correct if (a) it lands on the object boundary (geometric truth) AND
    # (b) it agrees with the stored value for the vast majority (rare H5-join collisions excepted).
    cpx_ok = (np.median(cpx_align) < 2.0) and (frac_cpx >= 0.95)
    return {"geom_ok": geom_ok, "cpx_ok": cpx_ok, "n": len(done),
            "per_chan": {c: (float(np.mean(per_chan[c])) if per_chan[c] else None) for c in CHANS},
            "cpx_median": float(np.median(cpx_errs)), "cpx_frac": frac_cpx,
            "cpx_align": float(np.median(cpx_align)), "fellback": fellback}


def gate2_functional(scorer, env_cfg, per_div, seed=0):
    """Live recall@k on test scenes with a known 1-push solution, cross-checked vs the H5-fed model.

    For each matched test episode:
      candidate set = manifest `tried` edges x all depths (deploy-time realistic, mirrors eval_scorer).
      valid set     = manifest `valid` (e,d) (opens the path).
      recall_live[k]       : rank live-ctx P over candidate, hit@k
      recall_h5[k]         : rank H5-ctx  P over the SAME candidate, hit@k  (isolates rendering)
      recall_live_reach[k] : candidate = LIVE get_reachable_edges (pure live pipeline)
    Also accumulates per-channel MAE(live, h5) on TEST scenes (extends Gate 1).
    """
    import h5py
    epf = json.load(open(str(MANIFESTS / "v3_test_episodes.json")))
    KS = [1, 3, 5]
    divs = ["hard", "med", "easy"]
    rng = np.random.default_rng(seed)

    agg = {d: {"live": {k: [] for k in KS}, "h5": {k: [] for k in KS},
               "live_reach": {k: [] for k in KS}, "n": 0} for d in divs}
    mae_test = {c: [] for c in CHANS}
    cpx_test = []
    reach_overlap = []  # |live_reach_edges ∩ manifest_tried_edges| / |manifest_tried_edges|
    print("\n[Gate 2] functional recall@k (live vs H5), test scenes with known 1-push solution",
          flush=True)

    for div in divs:
        h5p = f"{H5}/v3_test_{div}_lzf_tight_data/data.h5"
        if not os.path.exists(h5p):
            continue
        f = h5py.File(h5p, "r")
        N = int(f.attrs["n_samples"])
        xml = [x[0].decode() if isinstance(x[0], bytes) else str(x[0]) for x in f["xml_file"][:]]
        e_gt = f["edge_idx_a1"][:, 0].astype(int)
        d_gt = f["depth_idx_a1"][:, 0].astype(int)
        oc = f["local_tight_object_center"][:]
        oth = f["local_tight_object_theta"][:, 0]
        osz = f["target_object_size"][:]
        cmm = f["local_tight_crop_size_meters"][:, 0]
        order = rng.permutation(N)
        seen = set()
        kept = 0
        for i in order:
            if kept >= per_div:
                break
            gt = (int(e_gt[i]), int(d_gt[i]))
            rec, dm = match_episode(epf.get(xml[i]), oc[i], gt)
            if rec is None or dm > 0.01:
                continue
            key = (xml[i], round(float(oc[i, 0]), 4), round(float(oc[i, 1]), 4))
            if key in seen or not os.path.exists(str(resolve(xml[i]))):
                continue
            valid = {tuple(t) for t in rec["valid"]}
            tried = {tuple(t) for t in rec["tried"]}
            if not valid:
                continue
            # ---- H5-side ctx + contact_px (the model's training-distribution input) ----
            ctx_h5 = np.stack([cv2.resize(f[k][i].astype(np.float32), (OUT, OUT),
                                          interpolation=cv2.INTER_AREA) for k in TIGHT])
            cpx_h5 = np.array([contact_px_fn(ee, float(osz[i, 0]), float(osz[i, 1]),
                                             float(oth[i]), float(cmm[i]), OUT)
                               for ee in range(60)], dtype=np.float32)
            # ---- LIVE-side ----
            try:
                env = make_env(str(resolve(xml[i])), env_cfg)
            except Exception:
                continue
            env.get_reachable_objects()
            tgt, dist = scorer.find_target_by_center(env, oc[i])
            if tgt is None or dist > 0.02:
                continue
            rg = scorer.xml_goal(str(resolve(xml[i])))
            try:
                ctx_live, _ = scorer.render_ctx(env, tgt, rg, str(resolve(xml[i])))
            except Exception:
                continue
            cpx_live = scorer.contact_px_live(env, tgt)
            live_reach_edges = list(env.get_reachable_edges(tgt))

            seen.add(key)
            kept += 1
            agg[div]["n"] += 1

            for ci, c in enumerate(CHANS):
                mae_test[c].append(float(np.mean(np.abs(ctx_live[ci] - ctx_h5[ci]))))
            cpx_test.append(float(np.max(np.abs(cpx_live - cpx_h5))))
            tried_edges = sorted({ee for ee, _ in tried})
            if tried_edges:
                ov = len(set(live_reach_edges) & set(tried_edges)) / len(set(tried_edges))
                reach_overlap.append(ov)

            P_live = scorer.score_ctx(ctx_live, cpx_live)
            P_h5 = scorer.score_ctx(ctx_h5, cpx_h5)
            cand_manifest = [ee * 5 + dd for ee in tried_edges for dd in range(5)]
            cand_livereach = [ee * 5 + dd for ee in set(live_reach_edges) for dd in range(5)]
            hl = topk_hit(P_live, valid, cand_manifest, KS)
            hh = topk_hit(P_h5, valid, cand_manifest, KS)
            hr = topk_hit(P_live, valid, cand_livereach, KS) if cand_livereach else {k: False for k in KS}
            for k in KS:
                agg[div]["live"][k].append(hl[k])
                agg[div]["h5"][k].append(hh[k])
                agg[div]["live_reach"][k].append(hr[k])
        f.close()

    # ---- report ----
    print(f"  test-scene crop MAE (live vs H5), n={len(cpx_test)}:")
    print(f"    {'channel':<22}{'MAE':>10}")
    for c in CHANS:
        if mae_test[c]:
            print(f"    {c:<22}{np.mean(mae_test[c]):>10.5f}")
    if cpx_test:
        print(f"    contact_px max abs err (px): {np.max(cpx_test):.4f}")
    if reach_overlap:
        print(f"    live get_reachable_edges vs manifest tried edges: "
              f"mean overlap {np.mean(reach_overlap)*100:.1f}%")
    print()
    print(f"  {'div':<6}{'n':>4}   "
          f"{'live @1/3/5':>18}   {'H5 @1/3/5':>18}   {'live(reach) @1/3/5':>20}")
    tot = {"live": {k: [] for k in KS}, "h5": {k: [] for k in KS}, "live_reach": {k: [] for k in KS}}
    for div in divs:
        a = agg[div]
        if not a["n"]:
            continue

        def fmt(src):
            return "/".join(f"{np.mean(a[src][k])*100:4.0f}" for k in KS)
        for src in tot:
            for k in KS:
                tot[src][k].extend(a[src][k])
        print(f"  {div:<6}{a['n']:>4}   {fmt('live'):>18}   {fmt('h5'):>18}   {fmt('live_reach'):>20}")

    def fmt_tot(src):
        return "/".join(f"{np.mean(tot[src][k])*100:4.0f}" for k in KS)
    nall = len(tot["live"][1])
    print(f"  {'ALL':<6}{nall:>4}   {fmt_tot('live'):>18}   {fmt_tot('h5'):>18}   {fmt_tot('live_reach'):>20}")

    # PASS if live recall tracks H5 recall (the bridge reproduces the model input).
    ok = True
    diffs = []
    for k in KS:
        if not tot["live"][k]:
            ok = False
            break
        dl = abs(np.mean(tot["live"][k]) - np.mean(tot["h5"][k]))
        diffs.append(dl)
        if dl > 0.10:  # >10pp gap @k means the render diverges
            ok = False
    return {"ok": ok, "n": nall, "max_recall_gap": (max(diffs) if diffs else None),
            "mae_test": {c: (float(np.mean(mae_test[c])) if mae_test[c] else None) for c in CHANS},
            "reach_overlap": (float(np.mean(reach_overlap)) if reach_overlap else None)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=DEFAULT_CKPT)
    ap.add_argument("--env-config", default=SKILL15_CFG)
    ap.add_argument("--render-config", default=SKILL15_CFG)
    ap.add_argument("--validate", action="store_true")
    ap.add_argument("--gate1-n", type=int, default=30)
    ap.add_argument("--gate2-per-div", type=int, default=8)
    a = ap.parse_args()

    scorer = LiveScorer(ckpt=a.ckpt, render_config=a.render_config)
    print(f"device={scorer.device}  ckpt={os.path.basename(a.ckpt)}")
    print(f"env_config   = {a.env_config}")
    print(f"render_config= {a.render_config}")

    if not a.validate:
        print("loaded. pass --validate to run gates.")
        return

    g1 = gate1_crop_match(scorer, a.env_config, a.gate1_n)
    g2 = gate2_functional(scorer, a.env_config, a.gate2_per_div)

    print("\n==================== VERDICT ====================")
    g1geom = "PASS" if g1["geom_ok"] else "FAIL"
    g1cpx = "PASS" if g1["cpx_ok"] else "FAIL"
    print(f"  Gate 1 geometry channels match (static/movable/target):  {g1geom}")
    print(f"  Gate 1 contact_px match (<0.5 px):                        {g1cpx}")
    g2v = "PASS" if g2["ok"] else "FAIL"
    print(f"  Gate 2 live recall tracks H5 recall (gap<=10pp):          {g2v}"
          + (f"  (max gap {g2['max_recall_gap']*100:.0f}pp)" if g2["max_recall_gap"] is not None else ""))
    overall = "PASS" if (g1["geom_ok"] and g1["cpx_ok"] and g2["ok"]) else "CHECK"
    print(f"  OVERALL: {overall}")
    print("=================================================")


if __name__ == "__main__":
    main()
