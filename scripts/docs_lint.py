#!/usr/bin/env python
"""Doc-hygiene linter for the NAMO markdown corpus.

Finds the two failure modes that accumulate as docs get moved and renamed:
  1. Broken cross-references  — a [text](target.md) or [[wikilink]] whose target
     file does not exist (resolved relative to the linking file).
  2. Orphan docs              — a doc under docs/ that nothing else links to
     (no incoming links), so it is only discoverable by `ls`.

Also reports broken links to code (relative paths ending in a source extension),
since those rot the same way when files move — line-number anchors (#L123) are
NOT verified, only the file's existence.

Usage:
    python scripts/docs_lint.py            # human report, exit 1 if broken links
    python scripts/docs_lint.py --orphans  # also list orphan docs
    python scripts/docs_lint.py --json      # machine-readable

No third-party deps. Run from the repo root.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

# Dirs we never want to scan or count (vendored / generated / disposable).
SKIP_DIRS = {".git", "node_modules", ".pytest_cache", "build_python", "outputs"}

# A markdown inline link: [text](target)  — capture the target.
INLINE_LINK = re.compile(r"\[[^\]]*\]\(([^)]+)\)")
# An Obsidian-style wikilink: [[target]] or [[target|alias]].
WIKILINK = re.compile(r"\[\[([^\]|#]+)(?:[#|][^\]]*)?\]\]")
# Wikilinks with these prefixes are auto-memory notes living in ~/.claude, not repo docs.
MEMORY_PREFIXES = ("project_", "feedback_", "reference_", "user_")
# A valid wikilink slug is kebab/snake words only — filters out code like [["obj_1"]].
SLUG_RE = re.compile(r"^[a-z0-9][a-z0-9_-]*$")

CODE_EXTS = {".py", ".cpp", ".hpp", ".h", ".c", ".cc", ".sh", ".yaml", ".yml", ".json", ".tex"}

# Auto-memory dir for THIS project (machine-specific; absent on other boxes → skipped).
MEMORY_DIR = (
    Path.home()
    / ".claude/projects/-common-home-dm1487-robotics-research-ktamp-namo/memory"
)


def iter_md_files():
    for p in REPO.rglob("*.md"):
        if any(part in SKIP_DIRS for part in p.relative_to(REPO).parts):
            continue
        yield p


def strip_anchor(target: str) -> str:
    """Drop a trailing #anchor / #L123-L456 fragment."""
    return target.split("#", 1)[0]


def is_external(target: str) -> bool:
    return target.startswith(("http://", "https://", "mailto:", "ftp://"))


def classify(target_file: Path) -> str:
    ext = target_file.suffix.lower()
    if ext == ".md":
        return "doc"
    if ext in CODE_EXTS:
        return "code"
    return "other"


def resolve(link_from: Path, target: str) -> Path | None:
    """Resolve a link target to an absolute path, or None if not a checkable file ref."""
    target = target.strip()
    if not target or is_external(target) or target.startswith("#"):
        return None
    bare = strip_anchor(target).strip()
    if not bare:
        return None  # pure in-page anchor
    # Absolute-in-repo (rare) vs relative-to-linking-file.
    base = REPO if bare.startswith("/") else link_from.parent
    return (base / bare.lstrip("/")).resolve()


def wikilink_resolve(name: str) -> Path | None:
    """Wikilinks are bare names; treat as resolved iff a matching *.md exists anywhere.

    Returns the match if found, else None. Names may or may not carry a .md suffix.
    """
    stem = name.strip()
    if stem.endswith(".md"):
        stem = stem[:-3]
    for p in iter_md_files():
        if p.stem == stem:
            return p
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--orphans", action="store_true", help="also report orphan docs")
    ap.add_argument("--json", action="store_true", help="machine-readable output")
    args = ap.parse_args()

    md_files = sorted(iter_md_files())
    md_set = {p.resolve() for p in md_files}

    broken_doc: list[tuple[str, str, str]] = []   # (from, target, note)
    broken_code: list[tuple[str, str, str]] = []
    broken_wiki: list[tuple[str, str]] = []
    memory_links = 0  # valid [[..]] into ~/.claude memory notes (informational)
    memory_backlog: dict[str, list[str]] = {}  # slug -> [docs asking for it]
    incoming: dict[Path, int] = {p.resolve(): 0 for p in md_files}

    def memory_note_exists(slug: str) -> bool:
        if not MEMORY_DIR.is_dir():
            return True  # can't check on this box → assume valid, don't cry wolf
        return (MEMORY_DIR / f"{slug}.md").exists()

    for md in md_files:
        text = md.read_text(errors="replace")
        rel_from = md.relative_to(REPO)

        for m in INLINE_LINK.finditer(text):
            target = m.group(1)
            resolved = resolve(md, target)
            if resolved is None:
                continue
            kind = classify(resolved)
            if resolved.exists():
                if kind == "doc" and resolved in incoming:
                    incoming[resolved] += 1
                continue
            # broken
            if kind == "doc":
                # Try to suggest where it moved (same basename elsewhere in tree).
                base = strip_anchor(target).split("/")[-1]
                cands = [str(p.relative_to(REPO)) for p in md_files if p.name == base]
                note = f"moved? -> {cands}" if cands else "no file with that name exists"
                broken_doc.append((str(rel_from), target, note))
            elif kind == "code":
                broken_code.append((str(rel_from), target, ""))

        for m in WIKILINK.finditer(text):
            name = m.group(1).strip()
            slug = name[:-3] if name.endswith(".md") else name
            if not SLUG_RE.match(slug):
                continue  # not a real wikilink (e.g. a code literal like [["obj_1"]])
            if slug.startswith(MEMORY_PREFIXES):
                if memory_note_exists(slug):
                    memory_links += 1
                else:
                    # Per the memory system: a [[name]] with no note yet is a
                    # "write this later" marker, not an error — collect as backlog.
                    memory_backlog.setdefault(slug, [])
                    if str(rel_from) not in memory_backlog[slug]:
                        memory_backlog[slug].append(str(rel_from))
                continue
            hit = wikilink_resolve(name)
            if hit is not None and hit.resolve() in incoming:
                incoming[hit.resolve()] += 1
            elif hit is None:
                broken_wiki.append((str(rel_from), name))

    # Orphans: docs under docs/ with zero incoming links (exclude obvious entrypoints).
    ENTRYPOINTS = {"INDEX.md", "README.md"}
    orphans = [
        str(p.relative_to(REPO))
        for p in md_files
        if p.resolve() in incoming
        and incoming[p.resolve()] == 0
        and "docs/" in str(p.relative_to(REPO))
        and p.name not in ENTRYPOINTS
    ]

    if args.json:
        print(json.dumps({
            "broken_doc_links": broken_doc,
            "broken_code_links": broken_code,
            "broken_wikilinks": broken_wiki,
            "unwritten_memory_notes": {k: sorted(v) for k, v in memory_backlog.items()},
            "orphan_docs": sorted(orphans),
            "n_md": len(md_files),
        }, indent=2))
        return 1 if (broken_doc or broken_wiki) else 0

    print(f"Scanned {len(md_files)} markdown files under {REPO}\n")

    print(f"== BROKEN DOC->DOC LINKS ({len(broken_doc)}) ==")
    for frm, tgt, note in broken_doc:
        print(f"  {frm}\n      -> {tgt}   [{note}]")
    if not broken_doc:
        print("  (none)")

    print(f"\n== BROKEN DOC->CODE LINKS ({len(broken_code)}) ==")
    for frm, tgt, _ in broken_code:
        print(f"  {frm}\n      -> {tgt}")
    if not broken_code:
        print("  (none)")

    print(f"\n== BROKEN WIKILINKS [[..]] ({len(broken_wiki)}) ==")
    for frm, name in broken_wiki:
        print(f"  {frm}  ->  [[{name}]]")
    if not broken_wiki:
        print("  (none)")
    print(f"  ({memory_links} valid [[..]] links into ~/.claude memory notes — not broken)")

    print(f"\n== UNWRITTEN MEMORY NOTES (backlog, {len(memory_backlog)}) ==")
    print("  (docs link [[slug]] but no memory note exists yet — write-later markers)")
    for slug in sorted(memory_backlog):
        print(f"  [[{slug}]]  <- asked for by: {', '.join(memory_backlog[slug])}")
    if not memory_backlog:
        print("  (none)")

    if args.orphans:
        print(f"\n== ORPHAN DOCS (no incoming md link, {len(orphans)}) ==")
        for o in sorted(orphans):
            print(f"  {o}")
        if not orphans:
            print("  (none)")

    return 1 if (broken_doc or broken_wiki) else 0


if __name__ == "__main__":
    sys.exit(main())
