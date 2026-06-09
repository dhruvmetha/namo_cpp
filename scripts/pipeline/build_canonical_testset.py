#!/usr/bin/env python3
"""Build the CANONICAL NAMO test set — ONE clean, geometry-verified source of truth.

Reconciles the historically-confusing test artifacts (1-push validset keyed by `outputs/test_*_phase1`
SYMLINKS, vs the `pure2push` manifest keyed by `car_envs/v3/test` REAL paths) into a single set keyed by
canonical `realpath`, then GEOMETRY-GATES every scene against the training corpus (name-based checks are
meaningless — train/test use incompatible naming schemes; see verify_geom_disjoint.py).

Outputs (under --out-dir):
  manifests/canonical_scenes.txt   one realpath per line (geometry-clean scenes only)
  labels/onepush.json              realpath -> [episode...]  (exhaustive 1-push answer key, re-keyed from the validset)
  stats/canonical_stats.json       full statistics (counts, difficulty mix, solve-rate histogram, leak audit)

The 1-push answer key is the EXISTING champion validset (exhaustive, built by build_episode_validsets.py).
The 2-push answer key is added separately by build_2push_validset.py once the depth-2 collection lands.

Reuses geom_sig from verify_geom_disjoint.py and bin_of from eval_common.py (single sources of truth).
"""
import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))          # scripts/pipeline
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "scripts"))  # scripts (eval_common)
from verify_geom_disjoint import geom_sig, load_xmls

try:
    from eval_common import bin_of
except Exception:                                                    # keep builder self-contained
    def bin_of(sr):
        return "hard" if sr < 0.05 else ("med" if sr < 0.30 else "easy")

RP = os.path.realpath


def corpus_of(real):
    if "/v3/test/aug9_car/" in real:
        return "aug9"
    if "/v3/test/feb_car/" in real:
        return "feb"
    return "other"


def train_full_sigs(train_h5, workers=32):
    """Set of full-geometry signatures present in the training corpus (the leak reference)."""
    xmls = list(dict.fromkeys(load_xmls(train_h5)))
    with ThreadPoolExecutor(workers) as ex:
        sigs = {f for f, _ in ex.map(geom_sig, xmls, chunksize=64) if f}
    return sigs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--validset", required=True, help="1-push answer key (v3_test_validsets.json)")
    ap.add_argument("--pure2push", required=True, help="manifest of extra genuine depth-2 scenes (txt)")
    ap.add_argument("--train-xmls", required=True, help="training corpus h5/txt/json (leak reference)")
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args()

    os.makedirs(os.path.join(a.out_dir, "manifests"), exist_ok=True)
    os.makedirs(os.path.join(a.out_dir, "labels"), exist_ok=True)
    os.makedirs(os.path.join(a.out_dir, "stats"), exist_ok=True)

    # --- 1-push validset (nested by bucket OR flat by xml) -> realpath -> [episodes] ---
    vs = json.load(open(a.validset))
    buckets = vs if all(k in ("hard", "med", "easy") for k in vs) else {"_": vs}
    onepush = defaultdict(list)                  # realpath -> [episode dicts]
    for _, ents in buckets.items():
        for xml, eps in ents.items():
            real = RP(xml)
            eps_list = eps if isinstance(eps, list) else [eps]
            for ep in eps_list:
                onepush[real].append(ep)
    onepush_reals = set(onepush)

    # --- pure2push extra scenes -> realpath ---
    p2p_reals = {RP(l.strip()) for l in open(a.pure2push) if l.strip() and not l.startswith("#")}

    canonical = onepush_reals | p2p_reals

    # --- geometry gate every canonical scene against train ---
    print(f"hashing train corpus for leak reference ...", file=sys.stderr)
    tr_sigs = train_full_sigs(a.train_xmls)
    print(f"  {len(tr_sigs)} unique train scene signatures", file=sys.stderr)

    def sig_of(real):
        return real, geom_sig(real)
    with ThreadPoolExecutor(32) as ex:
        sigmap = dict(ex.map(sig_of, sorted(canonical), chunksize=32))

    clean, leaked, unparseable = [], [], []
    seen_geom = {}                                # full_sig -> first realpath (intra-test dedup audit)
    for real in sorted(canonical):
        full, _ = sigmap[real]
        if full is None:
            unparseable.append(real)
            continue
        if full in tr_sigs:
            leaked.append(real)
            continue
        clean.append(real)
        seen_geom.setdefault(full, real)

    # --- per-episode 1-push difficulty / solve-rate over CLEAN scenes ---
    diff = Counter()
    sr_hist = Counter()                           # solve_rate bucketed to 0.05
    n_eps = 0
    n_unsolvable_1push = 0                         # episodes with solve_rate == 0 -> genuine >=2-push candidates
    for real in clean:
        for ep in onepush.get(real, []):
            n_eps += 1
            sr = float(ep.get("solve_rate", 0.0))
            diff[bin_of(sr)] += 1
            sr_hist[round(sr // 0.05 * 0.05, 2)] += 1
            if not ep.get("valid"):
                n_unsolvable_1push += 1

    clean_set = set(clean)
    stats = {
        "n_canonical_scenes": len(canonical),
        "n_geom_clean": len(clean),
        "n_geom_leaked_into_train": len(leaked),
        "n_unparseable": len(unparseable),
        "leak_examples": leaked[:10],
        "unparseable_examples": unparseable[:10],
        "n_unique_geometries_in_clean": len(seen_geom),
        "n_geom_duplicate_scenes": len(clean) - len(seen_geom),
        "sources": {
            "onepush_validset_scenes": len(onepush_reals),
            "pure2push_scenes": len(p2p_reals),
            "overlap_both": len(onepush_reals & p2p_reals),
            "onepush_only": len(onepush_reals - p2p_reals),
            "pure2push_only": len(p2p_reals - onepush_reals),
        },
        "corpus_clean": dict(Counter(corpus_of(r) for r in clean)),
        "onepush_coverage": {
            "scenes_with_1push_key": len(onepush_reals & clean_set),
            "scenes_without_1push_key (need collection)": len(clean_set - onepush_reals),
            "n_episodes": n_eps,
            "difficulty_1push": dict(diff),
            "n_episodes_unsolvable_in_1push (>=2-push candidates)": n_unsolvable_1push,
            "solve_rate_histogram_0.05": dict(sorted(sr_hist.items())),
        },
    }

    # --- write outputs (realpath-keyed) ---
    man = os.path.join(a.out_dir, "manifests", "canonical_scenes.txt")
    with open(man, "w") as f:
        f.write("\n".join(clean) + "\n")
    op = os.path.join(a.out_dir, "labels", "onepush.json")
    json.dump({r: onepush[r] for r in clean if r in onepush}, open(op, "w"))
    sp = os.path.join(a.out_dir, "stats", "canonical_stats.json")
    json.dump(stats, open(sp, "w"), indent=2)

    print(json.dumps(stats, indent=2))
    print(f"\nwrote:\n  {man}\n  {op}\n  {sp}")
    if leaked:
        print(f"\n⚠ {len(leaked)} scenes LEAK into train (excluded from canonical set)")
    else:
        print(f"\n✓ 0 geometry leaks — all {len(clean)} clean scenes are disjoint from train")


if __name__ == "__main__":
    main()
