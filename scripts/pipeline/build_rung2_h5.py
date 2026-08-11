#!/usr/bin/env python3
"""RUNG-2 tree-log -> training H5 (dense 60x5 value field: gamma^k / 0 / -1 / MASK).

Reads region_opening RUNG-2 EXHAUSTIVE Q1-guided depth-2 pkls (region_opening_rung2_car.yaml:
region_max_chain_depth=2, goal_strategy=scorer, region_selection_strategy=ml_first). Parses the search
TREE (primitive_trial_log + reachability_log, joined by chain_depth / parent_edge / parent_depth) and
emits ONE row per NODE-state:
  - the ROOT node (chain_depth=1, parent=None) -> start-state grid (like rung-1);
  - each EXPANDED depth-2 node (chain_depth=2, parent_edge=E, parent_depth=D) -> the POST-SHOVE state s'
    that its second pushes were searched from. s' is rendered by set_full_state on the depth-1 SETUP
    push's stored resulting_state (added to primitive_trial_log by region_opening.py, gated on
    exhaustive_mode). Faithful by construction (== the frontier ChainNode.state the search expanded).

Per node row (60x5 = (4 edges x 15 pts) x 5 car-d5 depths):
  ctx          (5,64,64) f16  crop from THAT node's state, robot_goal-conditioned exactly like
                              rung-1 / deploy (region_samples=None -> goal channel = robot_goal point).
  contact_px   (60,2)   f32   edge contact pixels at that node's object pose.
  r_mask       (60,5)   f32   1 on reachable edges (all depths) at that node's state.
  value_target (60,5)   f32   gamma^k on a discovered solution path (opener leaf = gamma^0 = 1;
                              setup = gamma^1 = 0.9) | searched-to-budget-nothing = 0 |
                              unreachable edge = -1 | reachable-but-untried = MASK (placeholder).
  value_mask   (60,5)   f32   1 = in loss (win / setup / searched-0 / unreachable);
                              0 = reachable-but-untried (no executed signal).
  f_grid       (60,5)   f32   (value_target==1) opener/win layer (drop-in vanilla-scorer readers).
  meta: xml, object_id, robot_goal(3), chain_depth i8, parent_edge/parent_depth i16 (-1 at root),
        node_kind {root,depth2}, is_solution_node i8, n_reach_edges/n_tried/n_win i32, edges_agree i8.

LABEL SEMANTICS (grounded, not invented) — card docs/experiments/log/EXP-2026-07-10-exit-search-loop.md:
  - A 2-shove win = a depth-2 trial with success=True. Its cell (finisher) at the depth-2 node -> 1
    (opener leaf, gamma^0). Its SETUP (the depth-1 push (E,D) that spawned that node) at the ROOT -> 0.9
    (gamma^1). This within-rung-2 setup->gamma overlay is what un-buries setups (a searched depth-1 push
    whose subtree found a win is a POSITIVE, not the soft-0 it would otherwise get).
  - tried-but-not-on-a-win -> 0 (deliberate soft negative: searched, nothing found in budget).
  - reachable-but-untried -> MASK (no executed signal; forcing 0 is the measured "C15 poison").
  - unreachable edge -> trainable -1 (feasibility band; the value head learns reachability).
  - FREE 1-PUSH FINISHES (the bridge-patch): every winning depth-2 finisher is a target-1 opener leaf at
    a NEW post-shove state s' Q1 never saw -> emitted as the depth-2 node's grid. These patch the
    seed->lookahead bridge on-distribution.

NOT done here [USER]: the CROSS-RUNG re-stamp (flip a rung-1 `0` that a rung-2 search solves *through* ->
gamma). That is a POOL-MERGE concern over the combined rung-1+rung-2 H5s, not a per-episode label. FLAGGED.
"""
import sys, os, glob, argparse, json, pickle
from collections import defaultdict
from pathlib import Path

import numpy as np

# --- path bootstrap (portable across CS boxes / worktree) ---
REPO = Path(__file__).resolve().parents[2]
SAGE = os.environ.get("SAGE_REPO", "/common/home/dm1487/robotics_research/ktamp/sage_learning")
for _p in (f"{REPO}/build_python", f"{REPO}/python", f"{REPO}/scripts", f"{REPO}/scripts/sandbox", SAGE):
    if _p and _p not in sys.path:
        sys.path.insert(0, _p)

import namo_rl                                                       # noqa: E402
from scorer_beam import make_env, FALLBACK_GOAL                     # noqa: E402
from namo.rl_loop.build_train_h5 import _Renderer, _rlstate         # noqa: E402  (model-free renderer + RLState)

NUM_DEPTHS = 5
GAMMA = 0.9
MOVE_TOL = 0.01   # m: a depth-1 setup that moves the target object less than this (and <~3deg) is a NO-OP
                  # (edge went unreachable / push slipped) -> not a real setup. Its "2-push win" is really a
                  # 1-push opener the root sampling missed (post-shove state == start). We KEEP those winning
                  # finisher rows (recovered opener labels, ctx==start) but do NOT gamma-stamp the no-op setup.
CAR_CFG = f"{REPO}/config/namo_config_complete_skill15_car_1x.yaml"
ACTION_NORMALIZATION = np.array([0.5, 0.5, np.pi], np.float32)


def _getf(e, k, default=None):
    return e.get(k, default) if isinstance(e, dict) else getattr(e, k, default)


def _node_key(cd, pe, pd):
    return (int(cd),
            None if pe is None else int(pe),
            None if pd is None else int(pd))


def _labels_for_node(node_trials, reach, chain_depth, winning_setups, *, colossus0=False,
                     censored_setups=frozenset()):
    """value_target, value_mask, ceiling_mask, n_tried, n_win for one node's 60x5 grid.

    winning_setups: set of (edge,depth) at the ROOT that spawned >=1 depth-2 win (setup->gamma overlay).
    """
    vt = np.zeros((60, NUM_DEPTHS), np.float32)
    vm = np.zeros((60, NUM_DEPTHS), np.float32)
    cm = np.zeros((60, NUM_DEPTHS), np.float32)
    tried = {}                                    # (e,d) -> success bool (OR across dup entries)
    for t in node_trials:
        e, d = int(t["edge_idx"]), int(t["depth"])
        if 0 <= e < 60 and 0 <= d < NUM_DEPTHS:
            tried[(e, d)] = tried.get((e, d), False) or bool(t.get("success"))
    n_win = 0
    for e in range(60):
        for d in range(NUM_DEPTHS):
            if (e, d) in tried:
                if tried[(e, d)]:
                    vt[e, d] = 1.0; vm[e, d] = 1.0; n_win += 1          # opener leaf gamma^0
                elif chain_depth == 1 and (e, d) in winning_setups:
                    vt[e, d] = GAMMA; vm[e, d] = 1.0                     # setup gamma^1 (subtree won)
                elif colossus0 and chain_depth == 1 and (e, d) in censored_setups:
                    vm[e, d] = 0.0                                       # capped child unresolved -> UNKNOWN
                elif colossus0 and chain_depth == 1:
                    vt[e, d] = GAMMA * GAMMA; vm[e, d] = 1.0; cm[e, d] = 1.0
                elif colossus0:
                    vt[e, d] = GAMMA; vm[e, d] = 1.0; cm[e, d] = 1.0     # verified non-opener, may setup depth 3
                else:
                    vt[e, d] = 0.0; vm[e, d] = 1.0                       # searched-to-budget-nothing
            elif e in reach:
                vm[e, d] = 0.0                                           # reachable-but-untried -> MASK
            else:
                vt[e, d] = -1.0; vm[e, d] = 1.0                          # unreachable edge -> feasibility -1
    return vt, vm, cm, len(tried), n_win


def _action_contract(node, object_id, *aligned_grids):
    """Fail the Colossus build unless the live 300-action field and provenance are complete."""
    motion = np.asarray(node.get("action_motion"), dtype=np.float32)
    if motion.shape != (60, NUM_DEPTHS, 3) or not np.isfinite(motion).all():
        raise AssertionError(f"bad action_motion for {object_id}: {motion.shape}")
    if int(node.get("action_generator_slot_count", -1)) != 60 * NUM_DEPTHS:
        raise AssertionError(f"live generator slot count is not 300 for {object_id}")
    for grid in aligned_grids:
        if np.asarray(grid).shape != motion.shape[:2]:
            raise AssertionError(f"action/label/mask slot mismatch for {object_id}")
    if node.get("action_motion_frame") != "world_xy_object_yaw":
        raise AssertionError(f"bad action motion frame for {object_id}")
    if node.get("action_motion_units") != "normalized":
        raise AssertionError(f"bad action motion units for {object_id}")
    normalization = np.asarray(node.get("action_motion_normalization"), dtype=np.float32)
    if normalization.shape != (3,) or not np.allclose(normalization, ACTION_NORMALIZATION):
        raise AssertionError(f"bad action normalization for {object_id}: {normalization}")
    target_state = np.asarray(node.get("target_object_state"), dtype=np.float32)
    if target_state.shape != (5,) or not np.isfinite(target_state).all():
        raise AssertionError(f"bad target object state for {object_id}: {target_state.shape}")
    recorded_pose = np.asarray((node.get("state_observation") or {}).get(f"{object_id}_pose"), dtype=np.float32)
    if recorded_pose.ndim != 1 or recorded_pose.shape[0] < 3 or not np.allclose(recorded_pose[:3], target_state[:3], atol=1e-6):
        raise AssertionError(f"target object state does not match board state for {object_id}")
    primitive_id = str(node.get("primitive_database_id") or "")
    primitive_sha = str(node.get("primitive_database_sha256") or "")
    shape_family = str(node.get("shape_family") or "")
    if not primitive_id or len(primitive_sha) != 64 or shape_family not in {"square", "wide", "tall"}:
        raise AssertionError(f"incomplete primitive provenance for {object_id}")
    return {
        "action_motion": motion,
        "target_object_state": target_state,
        "primitive_database_id": primitive_id,
        "primitive_database_sha256": primitive_sha,
        "shape_family": shape_family,
    }


def _episode_rows(e, renderer, env, xml, stats, hand_checks, *, colossus0=False, family_select=False, sel_seed=0):
    st = _getf(e, "algorithm_stats") or {}
    tl = st.get("primitive_trial_log") or []
    rl = st.get("reachability_log") or []
    goal = _getf(e, "robot_goal") or FALLBACK_GOAL
    if not tl or not rl:
        stats["skipped_no_tree"] += 1
        return []

    if colossus0 and bool(st.get("root_opener_rejected", False)):
        stats["skipped_root_opener_rejected"] += 1
        return []

    # index trials by node; collect depth-1 setup states
    trials_by_node = defaultdict(list)
    setup_state = {}                              # (edge,depth) chain_depth-1 -> {'qpos','qvel'}
    for t in tl:
        cd = int(t.get("chain_depth", 1))
        trials_by_node[_node_key(cd, t.get("parent_edge"), t.get("parent_depth"))].append(t)
        if cd == 1 and t.get("resulting_state") is not None:
            setup_state[(int(t["edge_idx"]), int(t["depth"]))] = t["resulting_state"]

    # reachability nodes by key + the root/start object pose, for NO-OP setup detection
    node_by_key = {_node_key(int(r.get("chain_depth", 1)), r.get("parent_edge"), r.get("parent_depth")): r
                   for r in rl}
    root_node = next((r for r in rl if int(r.get("chain_depth", 1)) == 1), None)
    root_obs = (root_node or {}).get("state_observation") or {}

    def _setup_moved(pe, pd, obj):
        """Did the depth-1 setup (pe,pd) actually move the target object? Compares the depth-2 node's
        recorded post-shove object pose to the root/start pose. A no-op setup -> its 'win' is a recovered
        1-push opener, NOT a real 2-push setup, so it must NOT get gamma."""
        nd = node_by_key.get((2, int(pe), int(pd)))
        r = root_obs.get(f"{obj}_pose") if obj else None
        o = ((nd or {}).get("state_observation") or {}).get(f"{obj}_pose") if obj else None
        if o is None or r is None:
            return True                                       # unknown -> conservatively treat as real setup
        dth = abs(((float(o[2]) - float(r[2]) + np.pi) % (2 * np.pi)) - np.pi)
        return (abs(float(o[0]) - float(r[0])) > MOVE_TOL or abs(float(o[1]) - float(r[1])) > MOVE_TOL
                or dth > 0.05)

    # winning setups = only MOVED setups -> gamma. no-op-setup wins are recovered 1-push openers.
    winning_setups = set()
    for t in tl:
        if (int(t.get("chain_depth", 1)) == 2 and t.get("success")
                and t.get("parent_edge") is not None and t.get("parent_depth") is not None):
            pe_i, pd_i = int(t["parent_edge"]), int(t["parent_depth"])
            obj_j = (node_by_key.get((2, pe_i, pd_i)) or {}).get("object_id")
            if _setup_moved(pe_i, pd_i, obj_j):
                winning_setups.add((pe_i, pd_i))
                stats["two_shove_solutions_genuine"] += 1
            else:
                stats["recovered_1push_openers"] += 1

    censored_setups = {
        (int(t["parent_edge"]), int(t["parent_depth"]))
        for t in tl
        if (int(t.get("chain_depth", 1)) == 2 and t.get("parent_edge") is not None
            and t.get("parent_depth") is not None and bool(t.get("finish_sweep_censored", False)))
    } - winning_setups

    # FAMILY-SELECT [EXP-2026-08-09 corpus]: cap rendered children per episode — all can't be
    # rendered (6.1M swept children ~ 244GB). Policy: N_LIVE random live + N_CHAMP top-scored dead
    # (the champions/impostors the duels need; score = the d20 ordering score RECORDED at collection,
    # so selection needs no pre-render pass) + N_RAND random dead (unbiased filler).
    selected_children = None
    if family_select:
        d1_score = {(int(t2.get("edge_idx", -1)), int(t2.get("depth", -1))): float(t2.get("score", 0.0))
                    for t2 in tl if int(t2.get("chain_depth", 1)) == 1}
        live_keys, swept_keys = set(), set()
        for t2 in tl:
            if int(t2.get("chain_depth", 1)) == 2 and t2.get("parent_edge") is not None:
                k = (int(t2["parent_edge"]), int(t2["parent_depth"]))
                swept_keys.add(k)
                if t2.get("success"):
                    live_keys.add(k)
        dead_keys = sorted(swept_keys - live_keys, key=lambda k: -d1_score.get(k, 0.0))
        import random as _r
        rng = _r.Random(sel_seed)
        live_pick = rng.sample(sorted(live_keys), min(4, len(live_keys)))
        champ_pick = dead_keys[:4]
        rest = dead_keys[4:]
        rand_pick = rng.sample(rest, min(2, len(rest)))
        selected_children = set(live_pick) | set(champ_pick) | set(rand_pick)

    rows = []
    root_row_idx = None
    for node in rl:
        cd = int(node.get("chain_depth", 1))
        pe, pd = node.get("parent_edge"), node.get("parent_depth")
        obj = node.get("object_id")
        reach = set(int(x) for x in (node.get("reachable_edges") or []))
        node_trials = trials_by_node.get(_node_key(cd, pe, pd), [])

        # --- restore this node's state ---
        if cd == 1:
            env.reset()
            node_kind = "root"; setup_moved_flag = -1
        else:
            if selected_children is not None and (pe is None or pd is None or
                    (int(pe), int(pd)) not in selected_children):
                stats["family_select_skipped"] = stats.get("family_select_skipped", 0) + 1
                continue
            key = (int(pe), int(pd)) if (pe is not None and pd is not None) else None
            rs = setup_state.get(key)
            if rs is None:
                stats["skipped_no_setup_state"] += 1                    # parent setup lacked resulting_state
                continue
            env.set_full_state(_rlstate(rs["qpos"], rs["qvel"]))
            moved = _setup_moved(pe, pd, obj)                           # no-op setup -> ctx==start (recovered opener)
            node_kind = "depth2" if moved else "depth2_noop"
            setup_moved_flag = 1 if moved else 0

        # QC: does the restored state reproduce the search's recorded reachable-edge set?
        try:
            env.get_reachable_objects()                                 # warm wavefront before per-object query
            live = set(int(x) for x in env.get_reachable_edges(obj))
        except Exception:
            live = None
        agree = live is not None and live == reach
        stats["edges_agree" if agree else "edges_disagree"] += 1

        vt, vm, cm, n_tried, n_win = _labels_for_node(
            node_trials, reach, cd, winning_setups, colossus0=colossus0,
            censored_setups=censored_setups,
        )
        r_mask = np.zeros((60, NUM_DEPTHS), np.float32)
        for ed in reach:
            if 0 <= ed < 60:
                r_mask[ed, :] = 1.0

        ctx, _ = renderer.render_ctx(env, obj, goal, xml)
        cpx = renderer.contact_px_live(env, obj)

        is_sol = 1 if ((cd == 2 and n_win > 0) or (cd == 1 and len(winning_setups) > 0)) else 0
        win_ranks = [int(t.get("rank", -1)) + 1 for t in node_trials if t.get("success")]
        node_censored = any(bool(t.get("finish_sweep_censored", False)) for t in node_trials)
        node_audit = any(bool(t.get("finish_miss_audit_selected", False)) for t in node_trials)
        row = dict(
            ctx=ctx.astype(np.float16), contact_px=cpx.astype(np.float32), r_mask=r_mask,
            value_target=vt, value_mask=vm, ceiling_mask=cm,
            f_grid=(vt == 1.0).astype(np.float32),
            xml=str(xml), object_id=str(obj), robot_goal=np.array(goal[:3], np.float32),
            chain_depth=np.int8(cd),
            parent_edge=np.int16(-1 if pe is None else int(pe)),
            parent_depth=np.int16(-1 if pd is None else int(pd)),
            node_kind=node_kind, is_solution_node=np.int8(is_sol), setup_moved=np.int8(setup_moved_flag),
            n_reach_edges=np.int32(len(reach)), n_tried=np.int32(n_tried), n_win=np.int32(n_win),
            edges_agree=np.int8(1 if agree else 0),
            finish_sweep_censored=np.int8(1 if node_censored else 0),
            finish_miss_audit_selected=np.int8(1 if node_audit else 0),
            winner_rank=np.int32(min(win_ranks) if win_ranks else 0),
        )
        if colossus0:
            row.update(_action_contract(node, obj, vt, vm, cm, r_mask))
        rows.append(row)
        if cd == 1:
            root_row_idx = len(rows) - 1

    stats["nodes_root"] += sum(1 for r in rows if r["node_kind"] == "root")
    stats["nodes_depth2"] += sum(1 for r in rows if r["node_kind"] == "depth2")
    stats["nodes_depth2_noop"] += sum(1 for r in rows if r["node_kind"] == "depth2_noop")
    stats["free_1push_finishes"] += sum(int(r["n_win"]) for r in rows if r["node_kind"] == "depth2")
    stats["recovered_finish_cells"] += sum(int(r["n_win"]) for r in rows if r["node_kind"] == "depth2_noop")

    # hand-check: for each depth-2 win, verify (root setup cell == gamma) and (depth-2 finisher cell == 1)
    if root_row_idx is not None and len(hand_checks) < 6:
        root_vt = rows[root_row_idx]["value_target"]
        for r in rows:
            if r["node_kind"] != "depth2" or int(r["n_win"]) == 0:
                continue
            pe_i, pd_i = int(r["parent_edge"]), int(r["parent_depth"])
            wins = list(zip(*np.where(r["value_target"] == 1.0)))
            if not wins:
                continue
            we, wd = int(wins[0][0]), int(wins[0][1])
            hand_checks.append(dict(
                xml=os.path.basename(str(xml)), obj=r["object_id"],
                setup=(pe_i, pd_i), root_setup_target=float(root_vt[pe_i, pd_i]),
                finisher=(we, wd), depth2_finisher_target=float(r["value_target"][we, wd]),
            ))
            if len(hand_checks) >= 6:
                break
    return rows


def build(pkl_glob, out_h5, render_config, limit=None, shard_idx=0, shard_count=1, *, colossus0=False, family_select=False):
    pkls = sorted(glob.glob(pkl_glob, recursive=True))
    if shard_count > 1:
        pkls = pkls[shard_idx::shard_count]
    print(f"[rung2] {len(pkls)} pkls (shard {shard_idx}/{shard_count}) from {pkl_glob}")
    renderer = _Renderer(render_config)
    env_cache = {}

    rows = []
    hand_checks = []
    stats = dict(episodes=0, skipped_no_tree=0, skipped_no_setup_state=0, skipped_duplicate_episode=0,
                 skipped_root_opener_rejected=0,
                 edges_agree=0, edges_disagree=0, nodes_root=0, nodes_depth2=0, nodes_depth2_noop=0,
                 free_1push_finishes=0, recovered_finish_cells=0, two_shove_solutions=0,
                 two_shove_solutions_genuine=0, recovered_1push_openers=0, ep_with_solution=0)
    seen_episodes = set()
    for p in pkls:
        try:
            d = pickle.load(open(p, "rb"))
        except Exception as ex:
            print(f"  [skip] {p}: {ex}")
            continue
        for e in (_getf(d, "episode_results") or []):
            st = _getf(e, "algorithm_stats") or {}
            tl = st.get("primitive_trial_log") or []
            if not tl:
                continue
            xml = _getf(e, "xml_file")
            if colossus0:
                episode_key = (os.path.realpath(str(xml)), str(st.get("chosen_object_id")),
                               str(st.get("neighbour_region_label")))
                if episode_key in seen_episodes:
                    stats["skipped_duplicate_episode"] += 1
                    continue
                seen_episodes.add(episode_key)
            if xml not in env_cache:
                env_cache.clear()                                       # single-slot: each xml unique -> avoid env leak
                env_cache[xml] = make_env(xml)
            env = env_cache[xml]
            n_sol = sum(1 for t in tl if int(t.get("chain_depth", 1)) == 2 and t.get("success"))
            stats["two_shove_solutions"] += n_sol
            if n_sol > 0:
                stats["ep_with_solution"] += 1
            ep_rows = _episode_rows(e, renderer, env, xml, stats, hand_checks, colossus0=colossus0, family_select=family_select, sel_seed=abs(hash((xml, e.get('episode_id', 0)))) & 0xffffffff)
            rows.extend(ep_rows)
            stats["episodes"] += 1
            if limit and len(rows) >= limit:
                break
        if limit and len(rows) >= limit:
            break

    _write(out_h5, rows)
    return rows, stats, hand_checks


def _write(path, rows):
    import h5py
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with h5py.File(path, "w") as f:
        f.attrs["n_samples"] = len(rows)
        f.attrs["num_depths"] = NUM_DEPTHS
        f.attrs["gamma"] = GAMMA
        f.attrs["label_scheme"] = ("value_target{-1,0,gamma^k,1}+value_mask over the search TREE: "
                                   "win/opener-leaf=1 | setup(subtree won)=gamma=0.9 | searched-nothing=0 | "
                                   "unreachable-edge=-1 | reachable-untried=MASK. row=one tree node "
                                   "(root start-state OR depth-2 post-shove state).")
        f.attrs["generation"] = 2
        has_action_motion = bool(rows) and all("action_motion" in row for row in rows)
        if has_action_motion:
            f.attrs["action_motion_frame"] = "world_xy_object_yaw"
            f.attrs["action_motion_units"] = "normalized"
            f.attrs["action_motion_normalization"] = json.dumps({"dx_m": 0.5, "dy_m": 0.5, "dtheta_rad": "pi"})
            f.attrs["action_motion_layout"] = "[row, edge=60, push_depth=5, (dx,dy,dtheta)=3]"
            f.attrs["primitive_provenance"] = "row-level"
            f.attrs["primitive_database_sha256s"] = json.dumps(sorted({r["primitive_database_sha256"] for r in rows}))
        if not rows:
            return
        dt = h5py.string_dtype(encoding="utf-8")
        stack = lambda k: np.stack([r[k] for r in rows])
        arr = lambda k, npy: np.array([r[k] for r in rows], npy)
        # lzf + PER-ROW chunks: the ctx crop is 40KB/row uncompressed and dominates the file (this is
        # what made the beast H5 107GB, which OOM'd the merge, was slow to move Amarel->CS, and blew
        # /scratch quota). Masks are very sparse -> lzf compresses 18x (MEASURED on real ctx: 107GB ->
        # ~6GB); (1,)+shape chunks keep random-access reads O(1 chunk/row). This is a SIZE win
        # (storage / transfer / quota / merge-OOM) — NOT a training-speed win: NFS is round-trip-COUNT
        # bound not byte-bound, and local-uncompressed reads beat local-compressed (decode CPU, no IO
        # saved). Training speed is fixed separately (node-local staging + torch.compile) —
        # see memory reference_training_speedup. gzip=26x but heavier decode. [data-format lesson 2026-07-17]
        def _cds(name):
            a = stack(name)
            f.create_dataset(name, data=a, compression="lzf", chunks=(1,) + a.shape[1:])
        for _k in ("ctx", "contact_px", "r_mask", "value_target", "value_mask", "ceiling_mask",
                   "f_grid", "robot_goal"):
            _cds(_k)
        if has_action_motion:
            _cds("action_motion")
            _cds("target_object_state")
        f.create_dataset("chain_depth", data=arr("chain_depth", np.int8))
        f.create_dataset("parent_edge", data=arr("parent_edge", np.int16))
        f.create_dataset("parent_depth", data=arr("parent_depth", np.int16))
        f.create_dataset("is_solution_node", data=arr("is_solution_node", np.int8))
        f.create_dataset("setup_moved", data=arr("setup_moved", np.int8))    # depth2:1 moved / 0 no-op; root:-1
        f.create_dataset("n_reach_edges", data=arr("n_reach_edges", np.int32))
        f.create_dataset("n_tried", data=arr("n_tried", np.int32))
        f.create_dataset("n_win", data=arr("n_win", np.int32))
        f.create_dataset("edges_agree", data=arr("edges_agree", np.int8))
        f.create_dataset("finish_sweep_censored", data=arr("finish_sweep_censored", np.int8))
        f.create_dataset("finish_miss_audit_selected", data=arr("finish_miss_audit_selected", np.int8))
        f.create_dataset("winner_rank", data=arr("winner_rank", np.int32))
        f.create_dataset("xml", data=np.array([r["xml"] for r in rows], dtype=object), dtype=dt)
        f.create_dataset("object_id", data=np.array([r["object_id"] for r in rows], dtype=object), dtype=dt)
        f.create_dataset("node_kind", data=np.array([r["node_kind"] for r in rows], dtype=object), dtype=dt)
        if has_action_motion:
            for key in ("primitive_database_id", "primitive_database_sha256", "shape_family"):
                f.create_dataset(key, data=np.array([r[key] for r in rows], dtype=object), dtype=dt)


def _report(rows, stats, hand_checks, out_h5):
    n = len(rows)
    print("\n================ RUNG-2 H5 SUMMARY ================")
    print(f"out_h5 = {out_h5}")
    print(f"episodes = {stats['episodes']}  | skipped_no_tree={stats['skipped_no_tree']} "
          f"skipped_no_setup_state={stats['skipped_no_setup_state']} "
          f"skipped_duplicate_episode={stats['skipped_duplicate_episode']} "
          f"skipped_root_opener_rejected={stats['skipped_root_opener_rejected']}")
    print(f"tree nodes(rows) = {n}  (root={stats['nodes_root']}  depth2_moved={stats['nodes_depth2']}  "
          f"depth2_noop={stats['nodes_depth2_noop']})")
    print(f"depth-2 winning finishers (raw) = {stats['two_shove_solutions']}  "
          f"| episodes with >=1 win = {stats['ep_with_solution']}/{stats['episodes']}")
    print(f"  GENUINE 2-push solutions (setup MOVED the object -> gamma setup)   = {stats['two_shove_solutions_genuine']}")
    print(f"  RECOVERED 1-push openers (no-op setup -> opener at start, no gamma) = {stats['recovered_1push_openers']}")
    print(f"free 1-push finishes (opener-leaf cells at TRUE post-shove states) = {stats['free_1push_finishes']}")
    print(f"recovered-opener cells (opener-leaf at start via no-op setup)      = {stats['recovered_finish_cells']}")
    if n == 0:
        print("==================================================\n")
        return
    vt = np.stack([r["value_target"] for r in rows]); vm = np.stack([r["value_mask"] for r in rows])
    tot = vt.size
    n_one = int(((vt == 1.0) & (vm == 1)).sum())
    n_gam = int((np.isclose(vt, GAMMA) & (vm == 1)).sum())
    n_zero = int(((vt == 0.0) & (vm == 1)).sum())
    n_neg = int(((vt == -1.0) & (vm == 1)).sum())
    n_mask = int((vm == 0).sum())
    print(f"cells total = {tot}  (rows x 300)")
    print(f"  target= 1   (opener leaf / win) : {n_one:>8}  ({100*n_one/tot:.3f}%)")
    print(f"  target= {GAMMA} (setup, subtree won): {n_gam:>8}  ({100*n_gam/tot:.3f}%)")
    print(f"  target= 0   (searched-nothing)  : {n_zero:>8}  ({100*n_zero/tot:.3f}%)")
    print(f"  target=-1   (unreachable edge)  : {n_neg:>8}  ({100*n_neg/tot:.3f}%)")
    print(f"  MASKED      (reach-untried)     : {n_mask:>8}  ({100*n_mask/tot:.3f}%)")
    print(f"  trained cells (mask=1)          : {n_one+n_gam+n_zero+n_neg:>8}  "
          f"({100*(n_one+n_gam+n_zero+n_neg)/tot:.3f}%)")
    ag, dg = stats["edges_agree"], stats["edges_disagree"]
    print(f"QC restored-state reachable-edge agreement: {ag}/{ag+dg} agree "
          f"({100*ag/max(ag+dg,1):.1f}%)  [root=reset, depth2=set_full_state]")
    print(f"ctx shape={rows[0]['ctx'].shape} contact_px={rows[0]['contact_px'].shape} "
          f"r_mask={rows[0]['r_mask'].shape} value_target={rows[0]['value_target'].shape}")
    print("\n--- HAND-CHECK (solution-path setup got gamma, its opener got 1) ---")
    if not hand_checks:
        print("  (no 2-shove solution in this shard to hand-check)")
    for h in hand_checks:
        ok_setup = np.isclose(h["root_setup_target"], GAMMA)
        ok_fin = np.isclose(h["depth2_finisher_target"], 1.0)
        print(f"  {h['xml']} obj={h['obj']}: setup{h['setup']} root_target={h['root_setup_target']:.2f} "
              f"[{'OK' if ok_setup else 'BAD'}=gamma]  ->  finisher{h['finisher']} "
              f"depth2_target={h['depth2_finisher_target']:.2f} [{'OK' if ok_fin else 'BAD'}=1]")
    print("==================================================\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkl-glob", required=True, help="glob for *_results.pkl (recursive ** ok)")
    ap.add_argument("--out", required=True, help="output H5 path")
    ap.add_argument("--render-config", default=CAR_CFG)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--shard-idx", type=int, default=0)
    ap.add_argument("--shard-count", type=int, default=1)
    ap.add_argument("--gamma", type=float, default=GAMMA,
                    help="1-step setup discount: setup value = gamma^1, opener/finish = gamma^0 = 1. "
                         "Vary to build beast-0-{gamma} variants from the SAME collection.")
    ap.add_argument("--family-select", action="store_true",
                    help="FAMILY corpus: render per episode only 4 live + top-4-scored dead + 2 random dead children")
    ap.add_argument("--colossus0", action="store_true",
                    help="apply capped Colossus label grammar, reject direct-root episodes, and dedup episodes")
    a = ap.parse_args()
    GAMMA = a.gamma
    rows, stats, hand_checks = build(a.pkl_glob, a.out, a.render_config, a.limit, a.shard_idx, a.shard_count, family_select=a.family_select,
                                    colossus0=a.colossus0)
    _report(rows, stats, hand_checks, a.out)
