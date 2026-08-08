#!/usr/bin/env python3
"""Warn about trained checkpoints that no doc mentions.

Why: a checkpoint nobody indexed is a checkpoint nobody can reuse, and six months later it is
indistinguishable from scratch. On 2026-08-08 a single session produced 36 model dirs; the ones
that mattered got written up only because someone remembered them. This makes forgetting loud.

It is a WARNING, never a hard fail. A blocking check on something that lags real work by hours
just teaches everyone to pass --no-verify, which is how the portability guard got ignored.

  python scripts/experiments/check_registry_coverage.py            # warn on unregistered
  python scripts/experiments/check_registry_coverage.py --days 14  # only recent ones
  python scripts/experiments/check_registry_coverage.py --quiet    # hook mode: silent when clean
"""
import argparse
import os
import re
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DOCS = REPO / "docs"
# Names that are scaffolding, not results. Matched case-insensitively ANYWHERE in the dir name --
# these words land as suffixes at least as often as prefixes ("..._blackwell_smoke").
SKIP = ("smoke", "test", "tmp", "debug", "scratch", "probe")


def mentioned(token, docs):
    """Is `token` named in the docs?

    Two things a naive \\b search gets wrong, both of which fired on real docs 2026-08-08:
      * case -- the dir is AL10, the card writes aL10;
      * underscores -- \\b does not break on '_', so \\bBL10\\b misses "BL10_s1", and cards
        routinely compress a seed sweep to "AJ2_s{1,2,3}", where the literal "AJ2_s1" never
        appears at all. The boundary has to be alphanumeric-only for either to match.
    """
    return re.search(rf"(?<![A-Za-z0-9]){re.escape(token)}(?![A-Za-z0-9])", docs, re.I) is not None


def scratch_root():
    """$NAMO_SCRATCH via namo.paths, so this file carries no box path of its own."""
    sys.path.insert(0, str(REPO / "python"))
    try:
        from namo.paths import SCRATCH
        return Path(SCRATCH)
    except Exception:
        env = os.environ.get("NAMO_SCRATCH")
        return Path(env) if env else None


def find_model_dirs(root, max_depth=4):
    """Dirs holding checkpoints/*.ckpt. Bounded walk -- the scratch tree is large."""
    out = []
    if not root or not root.exists():
        return out
    for base in sorted(root.glob("*/")):
        for depth in range(1, max_depth + 1):
            pat = "/".join(["*"] * depth) + "/checkpoints"
            for ck in base.glob(pat):
                if ck.is_dir() and any(ck.glob("*.ckpt")):
                    out.append(ck.parent)
    return out


def documented_names():
    """Every token that appears in any doc -- registry, cards, RESULTS."""
    blob = []
    for p in DOCS.rglob("*.md"):
        try:
            blob.append(p.read_text(errors="ignore"))
        except OSError:
            pass
    return "\n".join(blob)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=float, default=None, help="only flag dirs modified within N days")
    ap.add_argument("--quiet", action="store_true", help="print nothing when everything is covered")
    a = ap.parse_args()

    root = scratch_root()
    if root is None:
        if not a.quiet:
            print("registry-coverage: no $NAMO_SCRATCH -- skipped", file=sys.stderr)
        return 0

    docs = documented_names()
    cutoff = time.time() - a.days * 86400 if a.days else None
    missing = []
    for d in find_model_dirs(root):
        name = d.name
        if any(w in name.lower() for w in SKIP):
            continue
        if cutoff and d.stat().st_mtime < cutoff:
            continue
        # covered if its own name, its seed-stripped stem, or the brace-sweep form is in the docs
        stem = re.sub(r"_s\d+$", "", name)
        if mentioned(name, docs) or mentioned(stem, docs) or mentioned(stem + "_s", docs):
            continue
        missing.append((d, d.stat().st_mtime))

    if not missing:
        if not a.quiet:
            print("registry-coverage: all trained checkpoints are mentioned in docs/")
        return 0

    by_stem = {}
    for d, mt in missing:
        by_stem.setdefault(re.sub(r"_s\d+$", "", d.name), []).append((d, mt))
    print(f"\n⚠  {len(missing)} trained checkpoint dir(s) in {len(by_stem)} group(s) appear in NO doc:",
          file=sys.stderr)
    for stem, items in sorted(by_stem.items(), key=lambda kv: -max(m for _, m in kv[1])):
        newest = time.strftime("%Y-%m-%d", time.localtime(max(m for _, m in items)))
        print(f"   {stem:<22} x{len(items):<3} newest {newest}   {items[0][0].parent}", file=sys.stderr)
    print("   → add a row to docs/experiments/horizon_q_model_registry.md "
          "(ckpt path, train_h5, eval protocol, aggregate, status) before the context is lost.",
          file=sys.stderr)
    print("   (warning only — does not block the commit)\n", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
