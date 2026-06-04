#!/usr/bin/env python3
"""Corpus resolver — the single source of truth for corpus paths + params.

A *corpus* is one data-generation batch declared in config/corpora/<id>.yaml.
Artifact paths are DERIVED from the id and the scratch roots (env-overridable),
never hardcoded in scripts or docs. Usable from both Python and bash.

    # bash (paths become shell vars):
    eval "$(python scripts/corpus.py v3_aug9 paths)"
    echo "$ENVS  $PHASE1_OUT  $MASKS_FILTERED_DIR  $H5_DIR"

    # python:
    from corpus import load, paths_for
    cfg = load("v3_aug9");  p = paths_for("v3_aug9")

Roots default to /scratch/dm1487/* and are overridable via env
(NAMO_SCRATCH / NAMO_DATASETS / NAMO_OUTPUTS / NAMO_MANIFESTS / NAMO_H5) —
set them once in scripts/amarel/activate.sh and nothing else hardcodes paths.
"""
import json
import os
import sys
from pathlib import Path

try:
    import yaml
except ImportError:
    yaml = None

REPO = Path(__file__).resolve().parent.parent
CORPORA_DIR = REPO / "config" / "corpora"

SCRATCH = Path(os.environ.get("NAMO_SCRATCH", "/scratch/dm1487"))
DATASETS = Path(os.environ.get("NAMO_DATASETS", SCRATCH / "datasets"))
OUTPUTS = Path(os.environ.get("NAMO_OUTPUTS", SCRATCH / "outputs"))
MANIFESTS = Path(os.environ.get("NAMO_MANIFESTS", SCRATCH / "manifests"))
H5 = Path(os.environ.get("NAMO_H5", SCRATCH / "h5"))


def load(corpus_id):
    f = CORPORA_DIR / f"{corpus_id}.yaml"
    if not f.exists():
        sys.exit(f"no corpus config: {f}")
    if yaml is None:
        sys.exit("PyYAML not installed in this interpreter")
    cfg = yaml.safe_load(f.read_text())
    cfg.setdefault("id", corpus_id)
    return cfg


def paths_for(corpus_id):
    """Every artifact path for a corpus, derived from its id. One place, no hardcoding.

    layout (config field):
      'nested' (default, go-forward) → $OUTPUTS/<id>/<phase>; retiring a corpus
        is one prefix delete.
      'flat' (legacy) → $OUTPUTS/<id>_<phase>; matches existing v1/v2/v3 dirs, so
        the config describes data already on disk without moving anything.
    """
    cfg = load(corpus_id)
    cid = cfg["id"]
    layout = cfg.get("layout", "nested")
    base = OUTPUTS / cid
    p = {"ENVS": DATASETS / cfg["env_family"], "H5_DIR": H5 / cid, "LAYOUT": layout}
    if layout == "flat":
        def out_of(ph):
            return OUTPUTS / f"{cid}_{ph}"
    else:
        def out_of(ph):
            return base / ph
        p.update({
            "CORPUS_ROOT": base,
            "MASKS_DIR": base / "masks",
            "MASKS_FILTERED_DIR": base / "masks_filtered",
            "META": base / "meta.json",
        })
    for ph in cfg["phases"]:
        name = ph["name"]
        p[f"{name.upper()}_OUT"] = out_of(name)
        p[f"{name.upper()}_MANIFEST"] = MANIFESTS / f"{cid}_{name}.txt"
    return p


def _cmd_paths(cid):
    for k, v in paths_for(cid).items():
        print(f'{k}="{v}"')  # eval-able in bash


def _cmd_show(cid):
    out = {
        "config": load(cid),
        "paths": {k: str(v) for k, v in paths_for(cid).items()},
    }
    print(json.dumps(out, indent=2, default=str))


def _cmd_phasevars(cid):
    """Per-phase params as eval-able bash vars: P<N>_DEPTH/_K/_MINE[/_SEEDS/_REASONS]."""
    for ph in load(cid)["phases"]:
        n = "".join(c for c in ph["name"] if c.isdigit()) or ph["name"]
        pre = f"P{n}"
        print(f'{pre}_DEPTH="{ph["depth"]}"')
        print(f'{pre}_K="{ph["k"]}"')
        mf = ph.get("mine_from")
        mf = " ".join(mf) if isinstance(mf, list) else (mf or "")
        print(f'{pre}_MINE="{mf}"')
        if ph.get("seed_sweep"):
            print(f'{pre}_SEEDS="{" ".join(str(s) for s in ph["seed_sweep"])}"')
        if ph.get("mine_reasons"):
            print(f'{pre}_REASONS="{" ".join(ph["mine_reasons"])}"')


def main():
    cmds = {"paths": _cmd_paths, "show": _cmd_show, "phasevars": _cmd_phasevars}
    if len(sys.argv) < 3 or sys.argv[2] not in cmds:
        sys.exit("usage: corpus.py <id> {paths|show|phasevars}")
    cmds[sys.argv[2]](sys.argv[1])


if __name__ == "__main__":
    main()
