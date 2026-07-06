"""Frozen 80/10/10 room-held-out split.

INVARIANT: hold out by ROOM (xml), never by row — one room has many episodes, so a
row-split leaks. We split the DISTINCT rooms 80/10/10 with a fixed seed, freeze the room
lists to json, and assert the three room sets are pairwise disjoint (0% leakage).

Build once (before gen-0); every generation loads the same frozen file.
"""
from dataclasses import dataclass
from typing import Dict, List
import json
import random
from pathlib import Path

from .episodes import EpisodeSpec, load_pool, rooms_of


@dataclass
class Split:
    train: List[str]      # room ids (xml keys)
    dev: List[str]
    test: List[str]

    def rooms_for(self, name: str) -> set:
        return set({"train": self.train, "dev": self.dev, "test": self.test}[name])


def build_split(pool_key: str, out_path: str, seed: int = 42,
                fracs=(0.8, 0.1, 0.1)) -> Split:
    """Create + freeze a room-held-out 80/10/10 split from a pool key json."""
    specs = load_pool(pool_key)
    rooms = rooms_of(specs)
    rng = random.Random(seed)
    order = rooms[:]
    rng.shuffle(order)
    n = len(order)
    n_tr = int(round(fracs[0] * n))
    n_dev = int(round(fracs[1] * n))
    train = sorted(order[:n_tr])
    dev = sorted(order[n_tr:n_tr + n_dev])
    test = sorted(order[n_tr + n_dev:])
    sp = Split(train=train, dev=dev, test=test)
    _assert_disjoint(sp)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"seed": seed, "pool_key": pool_key, "fracs": list(fracs),
                   "n_rooms": n, "train": train, "dev": dev, "test": test}, f)
    return sp


def load_split(path: str) -> Split:
    d = json.load(open(path))
    sp = Split(train=d["train"], dev=d["dev"], test=d["test"])
    _assert_disjoint(sp)
    return sp


def _assert_disjoint(sp: Split) -> None:
    tr, dv, te = set(sp.train), set(sp.dev), set(sp.test)
    assert not (tr & dv), f"room leakage train/dev: {len(tr & dv)}"
    assert not (tr & te), f"room leakage train/test: {len(tr & te)}"
    assert not (dv & te), f"room leakage dev/test: {len(dv & te)}"


def episodes_in(specs: List[EpisodeSpec], sp: Split, name: str) -> List[EpisodeSpec]:
    rooms = sp.rooms_for(name)
    return [s for s in specs if s.xml_key in rooms]
