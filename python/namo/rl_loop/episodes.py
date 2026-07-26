"""EpisodeSpec + pool loading.

An episode is (scene xml, target object, goal region), matched by object_center — the
per-episode invariant (never `xml` alone). We normalise the two on-disk key formats to
one EpisodeSpec and tag each with difficulty (per-episode solve_rate bin, NEVER a file
label) and horizon (1push/2push).

Formats handled:
  - validset (build_episode_validsets.py / onepush_episodes.json): per record
    {object_id, object_center, object_theta, region, solve_rate, valid, tried}  -> 1push
  - pure2push (build_2push_validset.py / pure2push[_divisions].json): per record
    {object_id, object_center, ..., solve_rate_first_push, is_2push_solvable, division} -> 2push
"""
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import json

from ._bootstrap import ensure_paths
ensure_paths()
from eval_common import bin_of              # noqa: E402  (hard<0.05, med<0.30, else easy)
from namo.paths import resolve              # noqa: E402


@dataclass(frozen=True)
class EpisodeSpec:
    xml_key: str                 # original key string (room id — split/hold-out by THIS)
    xml: str                     # resolved on-box path
    object_id: str
    object_center: Tuple[float, float]
    object_theta: Optional[float]
    region: Optional[str]
    difficulty: str              # "easy" | "med" | "hard"
    horizon: str                 # "1push" | "2push"
    solve_rate: Optional[float]

    @property
    def key(self) -> Tuple[str, str, float, float]:
        """Hashable episode identity used everywhere downstream."""
        return (self.xml_key, self.object_id,
                round(self.object_center[0], 4), round(self.object_center[1], 4))


def _spec_from_validset(xml_key: str, r: dict) -> EpisodeSpec:
    sr = r.get("solve_rate")
    return EpisodeSpec(
        xml_key=xml_key, xml=str(resolve(xml_key)),
        object_id=r["object_id"], object_center=tuple(r["object_center"]),
        object_theta=r.get("object_theta"), region=r.get("region"),
        difficulty=bin_of(sr) if sr is not None else "med",
        horizon="1push", solve_rate=sr,
    )


def _spec_from_pure2push(xml_key: str, r: dict) -> EpisodeSpec:
    sr = r.get("solve_rate_first_push")
    div = r.get("division") or (bin_of(sr) if sr is not None else "med")
    return EpisodeSpec(
        xml_key=xml_key, xml=str(resolve(xml_key)),
        object_id=r["object_id"], object_center=tuple(r["object_center"]),
        object_theta=r.get("object_theta"), region=r.get("region"),
        difficulty=div, horizon="2push", solve_rate=sr,
    )


def load_pool(key_path: str) -> List[EpisodeSpec]:
    """Load a per-episode key json into a flat list of EpisodeSpec (auto-detects format)."""
    key = json.load(open(key_path))
    specs: List[EpisodeSpec] = []
    for xml_key, recs in key.items():
        for r in recs:
            if "solve_rate_first_push" in r or "valid_first_push" in r:
                specs.append(_spec_from_pure2push(xml_key, r))
            else:
                specs.append(_spec_from_validset(xml_key, r))
    return specs


def rooms_of(specs: List[EpisodeSpec]) -> List[str]:
    """Distinct room ids (xml keys), stable order."""
    seen, out = set(), []
    for s in specs:
        if s.xml_key not in seen:
            seen.add(s.xml_key); out.append(s.xml_key)
    return out
