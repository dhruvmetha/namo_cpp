"""LoopConfig — all knobs for one RL self-imitation generation, in one place.

Every stage (collect, buffer, train, eval) reads from this so a generation is fully
described by a single serialisable dict. Defaults follow the card
(EXP-2026-07-06-rl-only-self-imitation.md, ## Plan).
"""
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional
import json

from ._bootstrap import REPO

CAR_CONFIG = str(REPO / "config/namo_config_complete_skill15_car_1x.yaml")
PRIM_PREFIX = "1x_car_d5_"        # 60 edges x 5 depths, matches the scorer (60,5) head
NUM_DEPTHS = 5


@dataclass
class LoopConfig:
    # --- identity ---
    arm: str = "A"                       # "A" = uniform pi0 (no ckpt); "B" = ckpt-conditioned
    generation: int = 0
    run_root: str = ""                   # where this run's artifacts live (buffer, h5, ckpts, eval)
    ckpt: Optional[str] = None           # policy ckpt for arm B / gen>0 (None => uniform pi0)

    # --- MDP ---
    max_depth: int = 10                  # rollout horizon (pushes), early-stop on region open
    open_frac: float = 0.2               # region-open criterion: >= frac of s0 goal pts reachable
    gamma: float = 0.9                   # V-target discount on Monte-Carlo returns
    restrict_to_labeled_object: bool = True   # per-episode invariant: push only the labeled object

    # --- exploration (collection) ---
    temperature: float = 1.0             # softmax temperature over policy scores
    epsilon: float = 0.10                # uniform-over-reachable exploration floor
    rollouts_per_episode: int = 8        # ordinary rollouts attempted per episode per generation
    # forced first-push sweep (hard episodes lacking setup diversity)
    forced_enable: bool = True
    forced_min_distinct_first: int = 8   # trigger: < this many distinct successful first actions in buffer
    forced_top_initial: int = 8          # choose forced action from the policy's top-N initial candidates
    forced_max_attempts_per_action: int = 4   # per action, per generation

    # --- scoring budget passed to the (budget-cond) scorer; NoHz ckpts ignore it ---
    score_h: int = 1

    # --- buffer retention ---
    buf_max_solves_per_episode: int = 8
    buf_len_slack: int = 2               # keep only solves with T <= T_min + slack
    buf_max_per_first_action: int = 2    # <=2 solves per first-action bucket (setup diversity)
    revalidate_fraction: float = 0.0     # fraction of near-threshold solves to re-execute (0 = skip)

    # --- V-head data ---
    vhead_recency_decay: float = 0.5     # per-generation recency weight rho^(gen_now - gen_row)
    vhead_fail_keep_frac: float = 1.0    # fraction of failed-rollout states kept for V (subsample knob)

    # --- training ---
    max_epochs: int = 40
    batch_size: int = 128
    num_workers: int = 8
    base_lr: float = 3e-4
    train_pi: bool = True
    train_v: bool = True

    # --- eval (per-generation dev report) ---
    eval_max_pushes: int = 10            # greedy open@1..K on dev rooms
    eval_open_ks: tuple = (1, 2, 5, 10)

    # --- data / splits ---
    split_file: str = ""                 # frozen 80/10/10 room split json (episodes.py builds it)
    pool_key: str = ""                   # episode-key json defining the rollout pool (per-episode records)
    car_config: str = CAR_CONFIG

    def to_json(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def from_json(cls, path: str) -> "LoopConfig":
        with open(path) as f:
            d = json.load(f)
        d["eval_open_ks"] = tuple(d.get("eval_open_ks", (1, 2, 5, 10)))
        return cls(**d)
