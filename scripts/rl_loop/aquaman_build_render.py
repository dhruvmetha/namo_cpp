#!/usr/bin/env python3
"""Aquaman round-1 build, stage R (fat-sharded): trace JSONL -> per-shard compressed NPZ.

Per kept episode in the shard:
  ROW BOARDS (become training rows): root board + every depth-1 board the search expanded.
    Rendered ctx (scorer's own render_ctx — train/deploy consistency), contact_px, r_mask,
    plus per-cell outcome labels from the pops:
      opened cell -> 1.0 exact ; solution-path setup -> 0.9 exact ;
      simmed-failed cell -> recorded in guess_cells (target filled at assemble stage);
      untried -> masked.
  GRANDCHILD CTX (transient, for finish-cell guesses): for every failed depth-1 pop, render
    the ctx of the state it reached + its r_mask; assemble stage scores these with theta to
    produce V-hat for the failed finish cell, then discards them.
Output NPZ (compressed): rows_* arrays, guess_cells (row,e,d,child_ref,cap), gc_* arrays.
  child_ref: >=0 -> index into THIS shard's rows (a depth-1 board);  -(k+2) -> gc index k.
"""
import argparse
import json
import sys
import os
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
SAGE = os.environ.get("SAGE_REPO", "")
for _p in (f"{REPO}/build_python", f"{REPO}/python", f"{REPO}/scripts", f"{REPO}/scripts/sandbox", SAGE):
    if _p and _p not in sys.path:
        sys.path.insert(0, _p)

from scorer_beam import BeamPlanner, make_env, FALLBACK_GOAL, CFG  # noqa: E402
from namo.core.xml_goal_parser import extract_goal_with_fallback  # noqa: E402

CAP_ROOT, CAP_FINISH = 0.81, 0.9


def r_mask_of(env, obj):
    m = np.zeros((60, 5), np.float32)
    for e in env.get_reachable_edges(obj):
        m[e, :] = 1.0
    return m


def set_state_from_qpos(env, qpos, template_state):
    st = template_state
    arr = np.asarray(getattr(st, "qpos"))
    arr[:] = np.asarray(qpos, dtype=arr.dtype)
    env.set_full_state(st)
    return st


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard-file", required=True)
    ap.add_argument("--ckpt", required=True)          # renderer needs the scorer's ctx pipeline only
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    planner = BeamPlanner(a.ckpt, CFG)
    rows = {k: [] for k in ("ctx", "cpx", "rmask", "vt", "vm", "cm", "isroot", "xml", "obj")}
    guess = []            # (row_idx, e, d, child_ref, cap)
    gc = {"ctx": [], "cpx": [], "rmask": []}
    cur_xml, env, goal, template = None, None, None, None

    for line in open(a.shard_file):
        try:
            ep = json.loads(line)
        except json.JSONDecodeError:
            continue
        xml, obj = ep["xml"], ep["object_id"]
        if xml != cur_xml:
            env = make_env(xml)
            goal = extract_goal_with_fallback(xml, FALLBACK_GOAL)
            env.set_robot_goal(*goal)
            env.get_reachable_objects()
            template = env.get_full_state()
            cur_xml = xml
        # board_id -> its state qpos: root from root_qpos; depth-1 boards from the pop that spawned them.
        # A depth-1 board with id B was created by the (chronologically) matching successful-move pop on
        # the root; pops carry geom(=qpos) of the state they REACHED, and solve_scene creates the child
        # board immediately after that pop -> child board ids appear in pop order of moved root pops.
        boards = {b["board_id"]: b for b in ep["boards"]}
        pops = ep["pops"]
        board_state = {0: ep["root_qpos"]}
        moved_root_pops = [p for p in pops if p["board_id"] == 0 and not p["fail"] and not p["opened"]]
        child_ids = sorted(b for b in boards if b != 0)
        for bid, p in zip(child_ids, moved_root_pops):
            board_state[bid] = p["geom"]
        # per-board cell outcomes
        tried = {bid: {} for bid in boards}          # (e,d) -> pop
        for p in pops:
            tried[p["board_id"]][(p["edge"], p["depth"])] = p
        winner_pops = [p for p in pops if p["opened"]]
        # solution path: winning pop's board; if depth-1 board won, its spawning root cell is the setup
        setup_cell = {}                               # board_id==0 cells that are verified setups
        if winner_pops and winner_pops[0]["board_id"] != 0:
            wb = winner_pops[0]["board_id"]
            for bid, p in zip(child_ids, moved_root_pops):
                if bid == wb:
                    setup_cell[(p["edge"], p["depth"])] = True
        ep_row0 = len(rows["ctx"])
        present = [b for b in [0] + child_ids if b in board_state]
        rowidx = {b: ep_row0 + i for i, b in enumerate(present)}
        for bid in present:
            st_q = board_state[bid]
            set_state_from_qpos(env, st_q, template)
            ctx, _ = planner.scorer.render_ctx(env, obj, goal, xml, None)
            cpx = planner.scorer.contact_px_live(env, obj)
            rm = r_mask_of(env, obj)
            vt = np.zeros((60, 5), np.float32)
            vm = np.zeros((60, 5), np.float32)
            cm = np.zeros((60, 5), np.float32)
            ri = len(rows["ctx"])
            for (e, d), p in tried[bid].items():
                if p["opened"]:
                    vt[e, d] = 1.0; vm[e, d] = 1.0
                elif bid == 0 and (e, d) in setup_cell:
                    vt[e, d] = 0.9; vm[e, d] = 1.0
                else:
                    # simmed, failed: guess cell. child_ref: root cell whose push MOVED -> its child
                    # board (row in this shard); failed-finish or jammed cell -> grandchild ctx (or cap-only).
                    cap = CAP_ROOT if bid == 0 else CAP_FINISH
                    ref = None
                    if bid == 0 and not p["fail"]:
                        for cbid, mp in zip(child_ids, moved_root_pops):
                            if mp is p and cbid in rowidx:
                                ref = ("row", rowidx[cbid])
                                break
                    if ref is None and p.get("geom") and not p["fail"]:
                        gi = len(gc["ctx"])
                        set_state_from_qpos(env, p["geom"], template)
                        gctx, _ = planner.scorer.render_ctx(env, obj, goal, xml, None)
                        gc["ctx"].append(gctx.astype(np.float16))
                        gc["cpx"].append(planner.scorer.contact_px_live(env, obj))
                        gc["rmask"].append(r_mask_of(env, obj))
                        ref = ("gc", gi)
                    vt[e, d] = cap; vm[e, d] = 1.0; cm[e, d] = 1.0   # default: mute cap (assemble may two-side it)
                    if ref is not None:
                        guess.append((ri, e, d,
                                      ref[1] if ref[0] == "row" else -(ref[1] + 2),
                                      cap))
            rows["ctx"].append(ctx.astype(np.float16))
            rows["cpx"].append(cpx)
            rows["rmask"].append(rm)
            rows["vt"].append(vt)
            rows["vm"].append(vm)
            rows["cm"].append(cm)
            rows["isroot"].append(1 if bid == 0 else 0)
            rows["xml"].append(xml)
            rows["obj"].append(obj)
    np.savez_compressed(
        a.out,
        ctx=np.array(rows["ctx"]), cpx=np.array(rows["cpx"], dtype=np.float32),
        rmask=np.array(rows["rmask"]), vt=np.array(rows["vt"]), vm=np.array(rows["vm"]),
        cm=np.array(rows["cm"]), isroot=np.array(rows["isroot"], dtype=np.int8),
        xml=np.array(rows["xml"], dtype=object), obj=np.array(rows["obj"], dtype=object),
        guess=np.array(guess, dtype=np.float64).reshape(-1, 5),
        gc_ctx=np.array(gc["ctx"]) if gc["ctx"] else np.zeros((0, 5, 64, 64), np.float16),
        gc_cpx=np.array(gc["cpx"], dtype=np.float32) if gc["cpx"] else np.zeros((0, 60, 2), np.float32),
        gc_rmask=np.array(gc["rmask"]) if gc["rmask"] else np.zeros((0, 60, 5), np.float32))
    print(f"rows={len(rows['ctx'])} guesses={len(guess)} gc={len(gc['ctx'])}")


if __name__ == "__main__":
    main()
