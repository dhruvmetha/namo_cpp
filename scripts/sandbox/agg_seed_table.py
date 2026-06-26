#!/usr/bin/env python3
"""Aggregate reactive@2 + best-first@2(combine=q) across ALL v2/v3/v4 seeds into the error-bar table.
Reads reactarg_<label>/ (open2/n) + bfq_<label>/ (jsonl s@2, s@900). Prints markdown; --slack posts to U07N1DR8S94
via a one-line marker the cron picks up. Pure stdlib. labels: {Hz,NoHz}_v{2,3,4}_s{1,2,3}."""
import json, glob, os, statistics as st, argparse
E = "/scratch/dm1487/eval"

def _dir(prefix, label):
    """case-insensitive dir match: eval-chains wrote capital (bfq_Hz_v4_s2), v2 dirs are lowercase."""
    target = f"{prefix}_{label}".lower()
    for d in glob.glob(f"{E}/{prefix}_*"):
        if os.path.basename(d).lower() == target:
            return d
    return None

def react(label):
    n = op = 0
    dd = _dir("reactarg", label)
    for f in (glob.glob(f"{dd}/shard_*.json") if dd else []):
        try:
            j = json.load(open(f)); n += j.get("n", 0); op += j.get("open2", 0)
        except Exception:
            pass
    return 100 * op / n if n else None

def bf(label):
    n = s2 = s900 = 0
    dd = _dir("bfq", label)
    for f in (glob.glob(f"{dd}/shard_*.jsonl") if dd else []):
        for line in open(f):
            try:
                r = json.loads(line)
            except Exception:
                continue
            n += 1
            if r.get("solved"):
                if r.get("sims", 1e9) <= 2: s2 += 1
                if r.get("sims", 1e9) <= 900: s900 += 1
    return (100 * s2 / n if n else None, 100 * s900 / n if n else None)

# v2 uses the historical label scheme (reactarg_Hz_v2 / _s2 / _s3 ; bfq_hz_v2_s1..)
V2 = {
 "Hz-v2":  (["Hz_v2","Hz_v2_s2","Hz_v2_s3"],   ["hz_v2_s1","hz_v2_s2","hz_v2_s3"]),
 "NoHz-v2":(["NoHz_v2","NoHz_v2_s2","NoHz_v2_s3"], ["nohz_v2_s1","nohz_v2_s2","nohz_v2_s3"]),
}
def cells_for(ver):
    out = {}
    for arch in ["Hz", "NoHz"]:
        rl = [f"{arch}_v{ver}_s{s}" for s in (1, 2, 3)]
        bl = [f"{arch.lower()}_v{ver}_s{s}" for s in (1, 2, 3)]
        # v3 s1 historically stored without _s1 suffix on the reactive side
        if ver == 3:
            rl = [f"{arch}_v3", f"{arch}_v3_s2", f"{arch}_v3_s3"]
        out[f"{arch}-v{ver}"] = (rl, bl)
    return out

def ms(xs):
    xs = [x for x in xs if x is not None]
    if not xs: return None
    return (st.mean(xs), st.pstdev(xs) if len(xs) > 1 else 0.0, len(xs))

def fmt(t):
    return f"{t[0]:.1f}±{t[1]:.1f}({t[2]}s)" if t and t[2] > 1 else (f"{t[0]:.1f}(1s)" if t else "—")

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--slack", action="store_true"); a = ap.parse_args()
    rows = []
    allcells = {**V2, **cells_for(3), **cells_for(4)}
    for name, (rl, bl) in allcells.items():
        rvals = [react(x) for x in rl]
        b2 = [bf(x)[0] for x in bl]; b9 = [bf(x)[1] for x in bl]
        r = ms(rvals); b = ms(b2); b900 = ms(b9)
        rows.append((name, r, b, b900))
    lines = ["| cell | reactive@2 | best-first@2 (q) | dive tax | s@900 |", "|---|---|---|---|---|"]
    for name, r, b, b900 in rows:
        dt = f"{r[0]-b[0]:+.1f}" if (r and b) else "—"
        lines.append(f"| {name} | {fmt(r)} | {fmt(b)} | {dt} | {fmt(b900)} |")
    table = "\n".join(lines)
    print(table)
    if a.slack:
        print("\n@@SLACK_TABLE_START@@\n" + table + "\n@@SLACK_TABLE_END@@")

if __name__ == "__main__":
    main()
