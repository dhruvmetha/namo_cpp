"""SolveBuffer — persistent, off-policy store of solved trajectories across generations.

Retention per episode (GPT-5.5 fix — setup diversity, NOT best-only):
  - keep only solves with T <= T_min + slack (T_min = shortest solve for that episode),
  - dedup by the first-two action ids (keep shortest, tie -> most recent),
  - <= buf_max_per_first_action solves per first-action bucket,
  - <= buf_max_solves_per_episode total.
Every solve records its generation stamp; all generations are kept (off-policy).

Failed rollouts (state -> return 0) are stored SEPARATELY for the V-head (subsampled by
vhead_fail_keep_frac). Revalidation re-executes a sample of stored solves from s0 and drops
any that no longer open (guards the ~0.3 mm set_full_state sim jitter near threshold).
"""
from typing import Dict, List, Optional, Set, Tuple
import os
import pickle
import random

from ._bootstrap import ensure_paths
ensure_paths()
from scorer_beam import make_env, make_action, FALLBACK_GOAL       # noqa: E402
from eval_m3 import sample_goal_points, goal_open_pts              # noqa: E402
from namo.core.xml_goal_parser import extract_goal_with_fallback  # noqa: E402
from namo.strategies import Goal                                   # noqa: E402
import namo_rl                                                     # noqa: E402

from .config import LoopConfig


def _keystr(k: Tuple) -> str:
    return "|".join(str(x) for x in k)


def _aid(ed) -> int:
    return int(ed[0]) * 5 + int(ed[1])


class SolveBuffer:
    def __init__(self):
        self.solves: Dict[str, List[dict]] = {}   # episode_keystr -> [solve entry]
        self.meta: Dict[str, dict] = {}           # episode_keystr -> {difficulty, horizon, xml_key, object_id, object_center, region}
        self.fails: List[dict] = []               # failed-rollout entries (for V-head)
        self.generation: int = -1

    # ---------- persistence ----------
    @classmethod
    def load(cls, path: str) -> "SolveBuffer":
        if not os.path.exists(path):
            return cls()
        with open(path, "rb") as f:
            d = pickle.load(f)
        b = cls()
        b.solves, b.meta, b.fails, b.generation = d["solves"], d["meta"], d["fails"], d["generation"]
        return b

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"solves": self.solves, "meta": self.meta,
                         "fails": self.fails, "generation": self.generation}, f)

    # ---------- ingest ----------
    def ingest(self, rollout_dicts: List[dict], cfg: LoopConfig,
               rng: Optional[random.Random] = None) -> None:
        """Fold one generation's rollout records into the buffer + apply retention."""
        rng = rng or random.Random(0)
        self.generation = max(self.generation, cfg.generation)
        new_solves: Dict[str, List[dict]] = {}
        for r in rollout_dicts:
            k = _keystr((r["xml_key"], r["object_id"],
                         round(r["object_center"][0], 4), round(r["object_center"][1], 4)))
            self.meta.setdefault(k, {
                "difficulty": r["difficulty"], "horizon": r["horizon"],
                "xml_key": r["xml_key"], "object_id": r["object_id"],
                "object_center": r["object_center"], "region": r["region"]})
            if r["solved"]:
                entry = {"actions": [[s["edge"], s["depth"]] for s in r["steps"]],
                         "T": r["T"], "generation": r["generation"], "steps": r["steps"]}
                new_solves.setdefault(k, []).append(entry)
            elif rng.random() < cfg.vhead_fail_keep_frac:
                self.fails.append({"episode": k, "generation": r["generation"], "steps": r["steps"]})
        for k, entries in new_solves.items():
            merged = self.solves.get(k, []) + entries
            self.solves[k] = _retain(merged, cfg)

    # ---------- queries ----------
    def first_actions_by_episode(self) -> Dict[Tuple, Set[int]]:
        """{episode_key_tuple: set(first_action_ids)} — feeds the collector's forced trigger."""
        out: Dict[Tuple, Set[int]] = {}
        for k, entries in self.solves.items():
            m = self.meta[k]
            key = (m["xml_key"], m["object_id"],
                   round(m["object_center"][0], 4), round(m["object_center"][1], 4))
            out[key] = {_aid(e["actions"][0]) for e in entries if e["actions"]}
        return out

    def stats(self, hard_episode_keys: Optional[Set[str]] = None) -> dict:
        """Buffer composition: unique solves per tier + hard-episode positive coverage."""
        by_tier: Dict[str, int] = {"easy": 0, "med": 0, "hard": 0}
        eps_with_solve_by_tier: Dict[str, int] = {"easy": 0, "med": 0, "hard": 0}
        for k, entries in self.solves.items():
            tier = self.meta[k]["difficulty"]
            by_tier[tier] = by_tier.get(tier, 0) + len(entries)
            if entries:
                eps_with_solve_by_tier[tier] = eps_with_solve_by_tier.get(tier, 0) + 1
        cov = None
        if hard_episode_keys:
            covered = sum(1 for k in hard_episode_keys if self.solves.get(k))
            cov = covered / max(1, len(hard_episode_keys))
        return {"unique_solves_by_tier": by_tier,
                "episodes_with_solve_by_tier": eps_with_solve_by_tier,
                "n_fail_records": len(self.fails),
                "hard_positive_coverage": cov}

    # ---------- revalidation ----------
    def revalidate(self, cfg: LoopConfig, frac: float, rng: random.Random) -> int:
        """Re-execute a random sample of stored solves from s0; drop any that no longer open.
        Returns #dropped."""
        dropped = 0
        for k, entries in list(self.solves.items()):
            keep = []
            for e in entries:
                if rng.random() >= frac:
                    keep.append(e); continue
                if _replay_opens(self.meta[k], e, cfg):
                    keep.append(e)
                else:
                    dropped += 1
            self.solves[k] = keep
        return dropped


def _retain(entries: List[dict], cfg: LoopConfig) -> List[dict]:
    if not entries:
        return []
    tmin = min(e["T"] for e in entries)
    entries = [e for e in entries if e["T"] <= tmin + cfg.buf_len_slack]
    # dedup by (first, second) action ids -> keep shortest T, tie -> most recent generation
    by_pair: Dict[Tuple[int, int], dict] = {}
    for e in entries:
        a = e["actions"]
        pk = (_aid(a[0]), _aid(a[1]) if len(a) > 1 else -1)
        cur = by_pair.get(pk)
        if cur is None or (e["T"], -e["generation"]) < (cur["T"], -cur["generation"]):
            by_pair[pk] = e
    # <= buf_max_per_first_action per first-action bucket
    by_first: Dict[int, List[dict]] = {}
    for e in by_pair.values():
        by_first.setdefault(_aid(e["actions"][0]), []).append(e)
    kept: List[dict] = []
    for es in by_first.values():
        es.sort(key=lambda x: (x["T"], -x["generation"]))
        kept.extend(es[:cfg.buf_max_per_first_action])
    kept.sort(key=lambda x: (x["T"], -x["generation"]))
    return kept[:cfg.buf_max_solves_per_episode]


def _replay_opens(meta: dict, entry: dict, cfg: LoopConfig) -> bool:
    from namo.paths import resolve
    xml = str(resolve(meta["xml_key"]))
    env = make_env(xml)
    goal = extract_goal_with_fallback(xml, FALLBACK_GOAL)
    env.set_robot_goal(*goal); env.get_reachable_objects()
    s0 = env.get_full_state()
    gp = sample_goal_points(env)
    if not gp:
        return False
    env.set_full_state(s0)
    for st in entry["steps"]:
        g = Goal(x=st["x"], y=st["y"], theta=st["theta"], edge_idx=st["edge"], depth=st["depth"])
        env.step(make_action(meta["object_id"], g))
        if goal_open_pts(env, gp, cfg.open_frac):
            return True
    return False
