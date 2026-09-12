#!/usr/bin/env python3
"""VALUE-GUIDED GREEDY BEST-FIRST search — objective = MIN TOTAL SIMULATED PUSHES to open the path.

The design we converged on (not MCTS, not admissible-A*, not budget-layered):
  * One priority queue of UNSIMULATED candidate pushes. Pop the most-promising, SIMULATE it (the only
    expensive op, ~1s), check goal (free), stop on first open. NO pruning -> complete within Hmax.
  * Objective is min SIMS, not min depth -> NO layering, NO g-term. Values are compared ACROSS depths:
    a deep-but-near-solution node outranks a fresh shallow first push, because it needs fewer more sims.
  * Q (per-action prior, PRE-sim) guides EXPANSION (which pushes to add). V=mean_top5(Q(s,.)) (per-state,
    POST-sim leaf value) guides SELECTION (which branch to chase). priority = combine(Q(s,a), V(s)).
  * sim_budget = the single reactive<->search dial (tiny -> reactive single best path; larger -> search).

FAILURE-DISCOUNT extension (--discount, default off => bit-identical to the static queue):
  Per-BOARD weight w(b). Effective priority of candidate a on board b = combine(q,V)*w(b).
  Root board (depth 0): w=1 ALWAYS. Child board (depth 1, post-setup finish state): w starts 1, and is
  updated ONLY by FAILED sims of that board's OWN candidates (a failed finish push demotes its siblings ->
  children of wrong roots stop flooding the queue). Lazy stale-reinsert on pop (w only decreases). w floored
  at EPS (never pruned). Modes: off | gamma (w*=GAMMA/fail) | fitted (w=g_table[min(k,kmax)]) | conf
  (w*=(1-q_failed)^TAU). LIFETIME LOGGING is always on when --lifetime-out is set (cheap): per-board records.

Baseline: --prior uniform = identical loop, RANDOM order, no value -> proves the guidance is worth it.

  python scripts/sandbox/eval_bestfirst.py --ckpt <ckpt> --manifest <pure2push.txt> --hmax 2 \
      --sim-budget 900 --prior model --agg mean5 --combine q --discount off --start 0 --end 985 --out <json>
"""
import sys, os, json, time, argparse, random, heapq
import yaml
from types import SimpleNamespace
from pathlib import Path
REPO = Path(__file__).resolve().parents[2]
SAGE = os.environ.get("SAGE_REPO", "")
for _p in (f"{REPO}/build_python", f"{REPO}/python", f"{REPO}/scripts", f"{REPO}/scripts/sandbox",
           f"{REPO}/scripts/pipeline", SAGE):
    if _p and _p not in sys.path:
        sys.path.insert(0, _p)
from scorer_beam import (  # noqa: E402
    BeamPlanner,
    DATA_DIR,
    PRIM_PREFIX,
    make_env,
    make_action,
    read_manifest,
    FALLBACK_GOAL,
    CFG,
)
from eval_m3 import rank_first_pushes_h2, sample_goal_points, goal_open_pts  # noqa: E402
from namo.core.xml_goal_parser import extract_goal_with_fallback  # noqa: E402
from namo.paths import MANIFESTS, DATASETS, SCRATCH  # noqa: E402
from namo import eval_sets  # noqa: E402
from namo.strategies import PrimitiveGoalStrategy  # noqa: E402
from namo.planners import get_region_snapshot  # noqa: E402
from namo.planners.opening.best_first_region_opening import BestFirstRegionOpeningPlanner  # noqa: E402
from namo.planners.search_measurements import (  # noqa: E402
    SearchMeasurements, content_digest, file_digest, measurement_options, run_identity,
    runtime_fingerprints, state_record,
    clock_finish,
    save_run_row, write_json_artifact, atomic_write, append_jsonl,
    xml_input_coverage,
)
from viz.trace_schema import build_trace, episode_filename, make_board, make_pop, rle_encode  # noqa: E402

PURE2PUSH = str(MANIFESTS / "test_pure2_fromkey.txt")
DEFAULT_MODEL_WARMUP_REPEATS = 3


def pooled_boundary_tasks(snapshot, records):
    """Resolve fixed one-boundary tasks, retaining every legacy donor reference.

    Reachability is deliberately not used to trim the allowed object pool: a
    setup push can make another boundary object's primitives reachable.
    """
    robot = snapshot["robot_label"]
    tasks = {}
    for row_index, record in enumerate(records):
        target = record.get("region") or snapshot["goal_label"]
        if target not in snapshot["adjacency"].get(robot, set()):
            raise ValueError(f"one-keyhole target {target!r} is not adjacent to {robot!r}")
        pool, error = BestFirstRegionOpeningPlanner._boundary_objects(snapshot["edge_objects"], robot, target)
        if error or not pool:
            raise ValueError(error or f"no boundary objects for {target!r}")
        points = record.get("target_points")
        if points is None:
            bundle = snapshot["region_goals"][target]
            points = (bundle["samples"] if isinstance(bundle, dict) else
                      [(goal.x, goal.y, goal.theta) for goal in bundle.goals])
        samples = [[float(p[0]), float(p[1]), float(p[2]) if len(p) > 2 else 0.0] for p in points]
        if not samples:
            raise ValueError(f"no fixed target samples for {target!r}")
        definition = dict(target_region=target, target_samples=samples,
                          boundary_objects=pool, object_scope="boundary_pool")
        key = content_digest(definition)
        task = tasks.setdefault(key, dict(definition, source_records=[], certification_status="pending"))
        task["source_records"].append(dict(row_index=row_index, record_sha256=content_digest(record),
                                           object_id=record.get("object_id"), region=record.get("region")))
    return list(tasks.values())


# The search itself now lives in the package. This script keeps its CLI, its
# scene/capture helpers and its evaluation reporting, and calls the same code
# production calls, so the two cannot drift apart.
from namo.planners.opening.best_first_search import (  # noqa: E402
    _FAILTYPE,
    _finalize_boards,
    _unmoved,
    _update_w_on_fail,
    candidates,
    priority,
    solve_scene,
)


def _make_planner(prior, ckpt, warmup_repeats):
    """Build the search shell and warm a model prior before any episode timer starts."""
    if prior == "model":
        planner = BeamPlanner(ckpt=ckpt)
        planner.scorer.warmup(repeats=warmup_repeats)
        return planner, planner.scorer.device, warmup_repeats
    planner = SimpleNamespace(
        prim=PrimitiveGoalStrategy(data_dir=DATA_DIR, primitive_prefix=PRIM_PREFIX),
        scorer=None,
    )
    return planner, "none", 0


def _scene_dict(env, goal):
    """The scene as the viz draws it, world frame in meters, AT THE CURRENT STATE (call at s0).
    static = walls (pose baked into object_info), movable = boxes (pose from the observation)."""
    info = env.get_object_info(); obs = env.get_observation()
    static = [{"name": k, "x": v["pos_x"], "y": v["pos_y"], "hw": v["size_x"], "hd": v["size_y"],
               "qw": v["quat_w"], "qz": v["quat_z"]}
              for k, v in info.items() if "pos_x" in v]
    movable = [{"name": k, "x": obs[f"{k}_pose"][0], "y": obs[f"{k}_pose"][1], "theta": obs[f"{k}_pose"][2],
                "hw": v["size_x"], "hd": v["size_y"]}
               for k, v in info.items() if k != "robot" and f"{k}_pose" in obs and "pos_x" not in v]
    return {"bounds": list(env.get_world_bounds()), "static": static, "movable": movable,
            "robot": list(obs["robot_pose"]), "goal": list(goal)}


def _make_capture(env, exporter, xml, obj, hw, hd, mov_names, offsets_world):
    """VIZ ONLY (--trace-out). Returns capture(state) -> (geom, regions) AT `state`.

    Both halves recompute from the LIVE env, so they are only correct if the env really is at that
    state -- hence the set_full_state on entry (the scorer's forward pass moves the sim as a side
    effect; same restore convention as scripts/sandbox/eval_m3.py:73). ~54 ms/call, all of it the
    region decomposition, which genuinely differs state to state: that is the point. Called once per
    POP (the state that push reached) plus once for the root board (nothing reached it), i.e. ~sims+1
    times per episode."""
    def capture(state):
        env.set_full_state(state)
        obs = env.get_observation()
        opose = obs[f"{obj}_pose"]
        off = offsets_world(hw, hd, float(opose[2]))
        geom = {"movable": {m: [round(float(c), 6) for c in obs[f"{m}_pose"]] for m in mov_names},
                "robot": [round(float(c), 6) for c in obs["robot_pose"]],
                "contacts": [[round(float(opose[0] + dx), 6), round(float(opose[1] + dy), 6)]
                             for dx, dy in off]}
        snap = exporter.build_snapshot(xml_path=xml, config_path=CFG, use_current_state=True)
        rm = snap.region_map
        regions = {"nx": int(rm.shape[0]), "ny": int(rm.shape[1]), "res": float(snap.resolution),
                   "origin": [float(snap.bounds[0]), float(snap.bounds[2])],
                   "labels": {str(int(k)): v for k, v in snap.region_labels.items()},
                   "rle": rle_encode(rm.tolist())}
        env.set_full_state(state)          # the snapshot pass must not leak state back into the search
        return geom, regions
    return capture


def _evaluate_pooled_task(a, planner, env, xml, goal, s0, snapshot, initial_observation,
                          rec, search_params, measurement, g_table, pops, capture):
    """Evaluate one fixed boundary task and retain technical failures as records."""
    import namo_rl

    measured = SearchMeasurements(**measurement)
    common = dict(xml=xml, xml_path=xml, region=rec["target_region"], **rec,
                  source_manifest=a.key, source_manifest_sha256=file_digest(a.key),
                  schema_version=measurement["schema_version"], measurement=measurement,
                  method=a.prior, shuffle_seed=a.seed_base, seed_base=a.seed_base, sampler_seed=42,
                  search=search_params, call_cap=a.sim_budget, complete=False, solved=False,
                  final_goal_reachable=None, goal_result_kind="fixed_target_predicate",
                  total_calls=0, calls_until_success=None, technical_error=False)
    boards = []
    try:
        runtime = runtime_fingerprints(str(CFG), str(DATA_DIR), PRIM_PREFIX, namo_rl.__file__,
                                       a.ckpt if a.prior == "model" else None)
        initial = state_record(s0)
        gp = rec["target_samples"]
        problem = dict(xml_sha256=file_digest(xml), initial_state=initial,
                       target_samples=gp if a.success == "region" else list(goal),
                       target_region=rec["target_region"], boundary_objects=rec["boundary_objects"],
                       object_scope="boundary_pool")
        semantic_runtime = {key: value for key, value in runtime.items() if key != "code_commit"}
        common.update(run_identity(problem, dict(search_params, runtime=semantic_runtime), a.prior,
                                   runtime["checkpoint_sha256"], {"sampler": 42, "shuffle": a.seed_base}),
                      runtime_fingerprints=runtime, xml_sha256=problem["xml_sha256"],
                      initialized_state_digest=content_digest(initial),
                      semantic_protocol=dict(search_params, runtime=semantic_runtime),
                      xml_input_coverage=xml_input_coverage(xml))
        measured.checkpoint(s0, trigger="initial", snapshot=snapshot, observation=initial_observation)
        measured.start_clock()
        measured.begin_attempt(task_kind="boundary", target_region=rec["target_region"],
                               boundary_objects=rec["boundary_objects"], target_samples=gp,
                               local_call_cap=a.sim_budget, hmax=a.hmax)
        solution = {}
        is_open = ((lambda e: goal_open_pts(e, [point[:2] for point in gp])) if a.success == "region" else
                   (lambda e: e.is_robot_goal_reachable()))
        solved, sims, plen, boards, end = solve_scene(
            planner, env, goal, xml, s0, a.hmax, a.sim_budget, a.prior, a.agg, a.combine,
            random.Random(a.seed_base), restrict_obj=tuple(rec["boundary_objects"]), is_open=is_open,
            raw=a.raw, dive_bonus=a.dive_bonus, discount=a.discount, gamma=a.gamma, tau=a.tau,
            g_table=g_table, eps=a.eps, w0_mode=a.w0_mode, free_strike_q=a.free_strike_q,
            child_patience=a.child_patience, dedupe_noop=a.dedupe_noop, prune_jam_depth=a.prune_jam_depth,
            trace_out=pops, capture=capture, timing=measured.local_timer, region_samples=gp,
            measurements=measured, solution_out=solution)
        chain = [make_action(obj, push_goal) for obj, push_goal in solution.get("plan", [])]
        measured.end_attempt(end=end, success=solved, calls=sims, actions=chain)
        if solved:
            measured.committed(solution["state"], chain, attempt_id=0, opened=True, task_kind="boundary",
                               observation=env.get_observation() if measured.statistics is not None else None)
        clock_finish(measured.timing, "t_wall", measured._clock_origin)
        # An exhausted search can leave a speculative push in the live environment.
        # Its committed prefix is still the saved start; do not relabel that rollout as a commit.
        terminal_state = solution["state"] if solved else s0
        terminal = state_record(terminal_state)
        measured.checkpoint(terminal_state, trigger="terminal")
        measured.event("run_end", solved=solved, calls=sims, local_end=end,
                       terminal_state_digest=content_digest(terminal))
        common.update(solved=solved, final_goal_reachable=solved, sims=sims, plan_len=plen,
                      total_calls=sims, calls_until_success=sims if solved else None, complete=True,
                      outcome=end, local_end=end, failure_kind=None if solved else end,
                      failure_subkind=None, censored=not solved, terminal_state_kind="committed",
                      terminal_state_digest=content_digest(terminal), execution_digest=measured.execution_digest,
                      attempt_count=measured.attempt_count, commit_count=measured.commit_count,
                      attempt_digests=measured.attempt_digests)
        if measured.timing is not None:
            common.update(measured.timing, local_timing=measured.local_timing,
                          time_until_success=measured.timing["t_wall"] if solved else None)
        if measured.statistics is not None:
            common.update(statistics=measured.statistics, initial_state=initial, terminal_state=terminal)
    except Exception as exc:
        if "run_id" not in common:
            common.update(run_identity({"uninitialized_task": xml, "target": rec}, search_params,
                                       a.prior, None, {"shuffle": a.seed_base}),
                          identity_status="initialization_incomplete")
        common.update(measured.failure_fields(exc))
    return common, boards


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="", help="required only when --prior=model")
    ap.add_argument("--manifest", default="",
                    help="optional scene-list override; default derives sorted scenes directly from --key")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=985)
    ap.add_argument("--hmax", type=int, default=2, help="max pushes in the search chain")
    ap.add_argument("--sim-budget", type=int, default=3000, help="shared simulator-call cap for the pooled task")
    ap.add_argument(
        "--prior",
        default="model",
        choices=["model", "uniform", "geometric", "geometric_transport", "geometric_region"],
        help=(
            "score source: geometric is the corrected target-region score; "
            "geometric_transport is the legacy single-XML-goal path proxy; "
            "geometric_region is a provenance alias for geometric"
        ),
    )
    ap.add_argument("--agg", default="mean5", choices=["mean5", "max"], help="state-value aggregate (selection)")
    ap.add_argument("--combine", default="q", choices=["q", "blend", "product"])
    ap.add_argument("--discount", default="off", choices=["off", "gamma", "fitted", "conf"],
                    help="per-board failure demotion. off=static queue (BIT-IDENTICAL baseline).")
    ap.add_argument("--gamma", type=float, default=0.65, help="--discount gamma: w *= gamma per failed sim")
    ap.add_argument("--child-patience", type=int, default=1,
                    help="--discount gamma: demote after each block of this many failed child-board probes")
    ap.add_argument("--tau", type=float, default=1.0, help="--discount conf: w *= (1-q_failed)^tau")
    ap.add_argument("--gtable", default="", help="--discount fitted: JSON {k: g_norm} (g_norm[0]=1)")
    ap.add_argument("--eps", type=float, default=1e-3, help="floor on w (never prune)")
    ap.add_argument("--w0-mode", default="one", choices=["one", "v"],
                    help="child board starting credibility: one=1.0 (default) | v=board value V (model prior)")
    ap.add_argument("--free-strike-q", type=float, default=2.0,
                    help="boards reached via a setup with q >= this get 1 free strike (2.0 = disabled)")
    ap.add_argument("--key", default=str(eval_sets.PURE2PUSH),
                    help="key: {xml: [{object_id, region, ...}]}; resolve the full boundary pool for each fixed target")
    ap.add_argument("--only-key", default="",
                    help="optional donor subset; pooled task identity and seeds do not depend on shard indices")
    ap.add_argument("--seed-base", type=int, default=7000,
                    help="RNG base for the uniform baseline; model is deterministic so only matters for --prior uniform.")
    # RAW is the default: the hl_gauss head already emits a value in [0,1], so the extra sigmoid is a
    # leftover from the old sigmoid_bce head. It squashes [0.09,0.98] into [0.52,0.73], which is inert
    # under combine=q + discount=off (monotonic -> same queue order) but destroys the magnitudes that
    # the bootstrap target min(cap, 0.9*V-hat) and any multiplicative search term (q*w) depend on.
    # --sigmoid restores the legacy scale; every result registered before 2026-08-07 was run that way.
    ap.add_argument("--raw", dest="raw", action="store_true", default=True,
                    help="use raw HL-Gauss E[bin] for the priority (DEFAULT)")
    ap.add_argument("--sigmoid", dest="raw", action="store_false",
                    help="legacy: squash the value through a sigmoid before ranking (pre-2026-08-07 default)")
    ap.add_argument("--dive-bonus", type=float, default=0.0, help="CASCADE dive bonus per push-already-done (default 0)")
    ap.add_argument("--success", default="region", choices=["region", "point"])
    ap.add_argument("--out", default=str(SCRATCH / "eval/bestfirst.json"))
    ap.add_argument("--leaf-out", default=str(SCRATCH / "eval/bestfirst.jsonl"))
    ap.add_argument(
        "--model-warmup-repeats",
        type=int,
        default=DEFAULT_MODEL_WARMUP_REPEATS,
        help="untimed synthetic forward passes immediately after loading a model prior",
    )
    ap.add_argument("--lifetime-out", default="", help="per-board lifetime JSONL (one row per board).")
    ap.add_argument("--no-dedupe-noop", dest="dedupe_noop", action="store_false",
                    help="ADOPTED 2026-07-27: a push that moves nothing reaches the state it started from, "
                         "so its child board is an exact duplicate of its parent and merely re-offers the "
                         "pushes just tried -- 27.4%% of all sims went there, and no solve ever came from "
                         "one. That child is now skipped; pass this flag to restore the old behaviour.")
    ap.set_defaults(dedupe_noop=True)
    ap.add_argument("--no-prune-jam-depth", dest="prune_jam_depth", action="store_false",
                    help="ADOPTED 2026-07-27: once a push jams at (state, edge, depth k), every deeper "
                         "depth on that edge is the same trajectory continued and jams identically, so it "
                         "is skipped without spending a simulation. Pass this to restore the old behaviour.")
    ap.set_defaults(prune_jam_depth=True)
    ap.add_argument("--trace-out", default="", help="per-episode search trace JSON dir (for viz/search)")
    ap.add_argument("--trace-lite", action="store_true",
                    help="record ordered pools/pops without per-pop geometry (same search order, smaller/faster trace)")
    ap.add_argument("--trace-model", default="", help="model label written into each trace's meta")
    ap.add_argument("--measurement-config", help="YAML containing independent measurement options")
    a = ap.parse_args()
    measurement = measurement_options(
        yaml.safe_load(Path(a.measurement_config).read_text())["measurement"] if a.measurement_config else None)
    if a.prior in {"geometric", "geometric_region"} and a.success != "region":
        ap.error(f"--prior {a.prior} requires --success region")
    if a.model_warmup_repeats < 0:
        ap.error("--model-warmup-repeats must be nonnegative")
    if (a.trace_out or a.lifetime_out) and not measurement["record_statistics"]:
        ap.error("expanded trace/lifetime output requires measurement.record_statistics=true")
    for target in (a.out, a.leaf_out, a.lifetime_out, a.trace_out):
        if target and Path(target).exists():
            raise FileExistsError(f"refusing existing output: {target}")

    import os as _os
    g_table = None
    if a.discount == "fitted":
        raw = json.load(open(a.gtable)); g_table = {int(k): float(v) for k, v in raw.items()}
    # Every knob that changes the ORDER the queue pops in. Written verbatim into each trace's meta so the
    # viz reconstructs bp = priority(q,V,combine) and the per-board w demotion with the same rule the search
    # ran, instead of assuming the defaults. Recording only -- the search reads `a`, never this dict.
    search_params = {"hmax": a.hmax, "sim_budget": a.sim_budget, "prior": a.prior, "agg": a.agg,
                     "combine": a.combine, "discount": a.discount, "gamma": a.gamma, "tau": a.tau,
                     "child_patience": a.child_patience,
                     "eps": a.eps, "w0_mode": a.w0_mode, "free_strike_q": a.free_strike_q,
                     "dive_bonus": a.dive_bonus, "raw": bool(a.raw),
                     "dedupe_noop": bool(a.dedupe_noop), "prune_jam_depth": bool(a.prune_jam_depth),
                     "gtable": ({str(k): v for k, v in g_table.items()} if g_table else None),
                     "object_scope": "boundary_pool", "seed_semantics": "explicit_per_problem_v1",
                     "success_predicate": a.success, "target_fraction": 0.2, "snapshot_seed": 42}
    key = json.load(open(a.key)); keyrp = {_os.path.realpath(k): v for k, v in key.items()}
    only = None
    if a.only_key:
        only_raw = json.load(open(a.only_key))
        only = {
            _os.path.realpath(xml): {(rec.get("object_id"), rec.get("region")) for rec in recs}
            for xml, recs in only_raw.items()
        }
    if a.prior == "model" and not a.ckpt:
        ap.error("--ckpt is required when --prior=model")
    planner, device, applied_warmup_repeats = _make_planner(
        a.prior, a.ckpt, a.model_warmup_repeats
    )
    search_params["model_warmup_repeats"] = applied_warmup_repeats
    print(f"device={device} hmax={a.hmax} sim_budget={a.sim_budget} prior={a.prior} "
          f"agg={a.agg} combine={a.combine} discount={a.discount} tau={a.tau} "
          f"model_warmup_repeats={applied_warmup_repeats} "
          f"dedupe_noop={a.dedupe_noop} prune_jam_depth={a.prune_jam_depth} "
          f"key={_os.path.basename(a.key)} success={a.success}", flush=True)
    xmls_all = read_manifest(a.manifest, None) if a.manifest else sorted(key)
    xmls = xmls_all[a.start:a.end]
    n = n_solved = n_already = n_norec = sims_tot = sims_solved = 0; t0 = time.time()
    atomic_write(a.leaf_out, b"")
    ltf = open(a.lifetime_out, "x") if a.lifetime_out else None
    errors = 0
    if a.trace_out and not a.trace_lite:
        from add_contact_px import contact_offsets_world
        # scipy (via the exporter's connected-components pass) + the exporter itself are imported ONLY
        # on the tracing path, so the flag-off run keeps its exact dependency set and startup cost.
        from namo.visualization.wavefront_snapshot import WavefrontSnapshotExporter
    if a.trace_out:
        os.makedirs(a.trace_out, exist_ok=True)
    ep_ctr = 0
    for xi, xml in enumerate(xmls):
        tasks = None
        task_index = 0
        try:
            xmlrp = _os.path.realpath(xml)
            if only is not None and xmlrp not in only:
                continue
            recs = key.get(xml) or keyrp.get(_os.path.realpath(xml))
            if not recs:
                n_norec += 1
                raise ValueError("source manifest contains no task record for this XML")
            env = make_env(xml)
            goal = extract_goal_with_fallback(xml, FALLBACK_GOAL)
            env.set_robot_goal(*goal); env.get_reachable_objects()
            snapshot = get_region_snapshot(env, goals_per_region=100, use_xml_goal=True, seed=42,
                                           **({"include_region_cells": True} if measurement["record_statistics"] else {}))
            tasks = pooled_boundary_tasks(snapshot, recs)
            s0 = env.get_full_state()
            initial_observation = env.get_observation() if measurement["record_statistics"] else None
            scene = _scene_dict(env, goal) if a.trace_out and not a.trace_lite else {}
            if a.trace_out and not a.trace_lite:
                exporter = WavefrontSnapshotExporter(env)      # one per env: static geometry never moves
                mov_names = [m["name"] for m in scene["movable"]]
            for task_index, rec in enumerate(tasks):
                if only is not None and not any(
                    (source["object_id"], source["region"]) in only[xmlrp] for source in rec["source_records"]
                ):
                    continue
                obj = rec["boundary_objects"][0]  # visualization anchor only, never a search restriction
                if a.success == "point" and rec["target_region"] != snapshot["goal_label"]:
                    raise ValueError("point success cannot evaluate a non-goal target region")
                pops = [] if a.trace_out else None
                capture = None
                ep_scene = scene
                if a.trace_out and not a.trace_lite:
                    env.set_full_state(s0)                     # the previous record left the env post-search
                    oi = env.get_object_info()[obj]
                    opose = env.get_observation()[f"{obj}_pose"]
                    off = contact_offsets_world(oi["size_x"], oi["size_y"], opose[2])
                    ep_scene = dict(scene, contacts=[[float(opose[0] + dx), float(opose[1] + dy)] for dx, dy in off])
                    capture = _make_capture(env, exporter, xml, obj, oi["size_x"], oi["size_y"],
                                            mov_names, contact_offsets_world)
                row, boards = _evaluate_pooled_task(a, planner, env, xml, goal, s0, snapshot,
                                                   initial_observation, rec, search_params,
                                                   measurement, g_table, pops, capture)
                save_run_row(a.leaf_out, row, statistics_dir=f"{Path(a.leaf_out).name}.statistics")
                if row.get("technical_error"):
                    errors += 1
                    continue
                solved, sims, plen, end = (row[name] for name in ("solved", "sims", "plan_len", "local_end"))
                n += 1; sims_tot += sims; n_solved += int(solved); sims_solved += sims if solved else 0
                if ltf is not None:
                    ep = {"ep": ep_ctr, "xml": xml, "object_id": obj, "region": rec.get("region"),
                          "solved": solved, "sims": sims, "end": end}
                    for row in _finalize_boards(boards, ep):
                        ltf.write(json.dumps(row) + "\n")
                if a.trace_out:
                    doc = build_trace(
                        meta={"xml": xml, "object_id": obj, "region": rec.get("region"),
                              "model": a.trace_model or os.path.basename(a.ckpt),
                              "strategy": (a.discount if a.discount == "off" else f"{a.discount}_tau{a.tau}"),
                              "search": search_params},
                        scene=ep_scene,
                        boards=[make_board(b["board_id"], b["depth"], b["parent_edge"], b["parent_depth"],
                                           b["pool"], b["grid"], b["w0"], b["free_strikes"],
                                           geom=b["geom"], regions=b["regions"]) for b in boards],
                        pops=pops, result={"solved": solved, "sims": sims, "plan_len": plen, "end": end})
                    json.dump(doc, open(os.path.join(a.trace_out, episode_filename(xml, obj)), "w"))
                ep_ctr += 1
            if xi % 20 == 0:
                print(f"  [{xi}/{len(xmls)}] episodes={n} solved={n_solved} avg_sims={sims_tot/max(n,1):.1f} "
                      f"({time.time()-t0:.0f}s)", file=sys.stderr, flush=True)
        except Exception as ex:
            if isinstance(ex, OSError):
                raise  # A failed publication cannot be repaired by writing more to the same disk.
            pending = tasks[task_index:] if tasks is not None else [{"source_records": keyrp.get(xmlrp, [])}]
            for rec in pending:
                failed = dict(xml=xml, xml_path=xml, **rec, method=a.prior, shuffle_seed=a.seed_base,
                              schema_version=measurement["schema_version"], measurement=measurement,
                              call_cap=a.sim_budget, identity_status="initialization_incomplete",
                              **run_identity({"uninitialized_task": xml, "target": rec}, search_params,
                                             a.prior, None, {"shuffle": a.seed_base}),
                              **SearchMeasurements().failure_fields(ex))
                save_run_row(a.leaf_out, failed)
                errors += 1
            print(f"  scene {xi} technical error: {ex}", file=sys.stderr, flush=True)
    if ltf is not None:
        ltf.close()
    res = {"ckpt": a.ckpt, **search_params, "seed_base": a.seed_base,
           "key": _os.path.basename(a.key), "n_episodes": n,
           "n_already_open": n_already, "n_no_record": n_norec, "technical_error_count": errors,
           "solve_rate": round(100.0 * n_solved / max(n, 1), 1),
           "avg_sims_all": round(sims_tot / max(n, 1), 2),
           "avg_sims_to_solve": round(sims_solved / max(n_solved, 1), 2)}
    write_json_artifact(a.out, res)
    if errors == 0:
        write_json_artifact(f"{a.leaf_out}.complete.json", {"expected_run_count": n, "accounted_run_count": n})
    print(json.dumps(res, indent=1), flush=True)


if __name__ == "__main__":
    main()
