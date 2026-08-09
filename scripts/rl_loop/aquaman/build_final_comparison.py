#!/usr/bin/env python3
"""Build the end-of-day comparison across every arm evaluated on the canonical gate.

Reads the gate JSONs produced by aquaman_agg.py and emits one markdown table plus the
difficulty x horizon splits for the v4 arms. Written to a file so the Slack post is exactly
what the numbers say, with nothing retyped by hand.
"""
import json
from pathlib import Path

R0 = Path("/common/users/dm1487/scratch_namo/aquaman/round0")

# arms whose numbers are already registered (3-seed pooled, canonical gate)
REGISTERED = [
    ("random",            "-",      "-",   1.7,  70.1,  3.3),
    ("theta0 (deployed)", "0.9",    "on",  22.6, 92.0, 39.7),
    ("arm A",             "0.9",    "on",  27.7, 91.5, 38.0),
    ("Bfix",              "0.9",    "on",  28.9, 87.1, 41.8),
    ("BfixNR",            "0.9",    "OFF", 11.2, 81.5, 15.2),
    ("BNG",               "0.9",    "on",  32.1, 88.6, 38.4),
    ("ARJ (v1 floor)",    "0.9",    "on",  27.7, 91.0, 42.5),
]
LIVE = [("AJ2", "gate_aj2.json", "0.5", "on"), ("AJ2NR", "gate_aj2.json", "0.5", "OFF"),
        ("AJ3", "gate_aj3.json", "0.9", "on"), ("AJ3NR", "gate_aj3.json", "0.9", "OFF"),
        ("AJ4", "gate_aj4.json", "0.9", "on"), ("AJ4NR", "gate_aj4.json", "0.9", "OFF")]


def get(gate, arm):
    g = json.load(open(R0 / gate))
    if arm not in g:
        return None
    return (g[arm]["2push"]["hard"]["solve@5"], g[arm]["2push"]["hard"]["solve@900"],
            g[arm]["1push"]["hard"]["solve@1"])


rows = [(n, s, a, x, y, z) for n, s, a, x, y, z in REGISTERED]
for arm, gate, setup, aux in LIVE:
    if not (R0 / gate).exists():
        continue
    v = get(gate, arm)
    if v:
        rows.append((arm, setup, aux, v[0], v[1], v[2]))

out = ["| arm | setup | aux | 2p-hard@5 | 2p-hard@900 | 1p-hard@1 |",
       "|---|---|---|--:|--:|--:|"]
best = max(r[3] for r in rows)
for n, s, a, x, y, z in rows:
    mark = " **<-- best**" if x == best else ""
    out.append(f"| {n} | {s} | {a} | {x}{mark} | {y} | {z} |")

splits = []
gp = R0 / "gate_aj4.json"
if gp.exists():
    g = json.load(open(gp))
    for arm in ("AJ4", "AJ4NR"):
        if arm not in g:
            continue
        for hz in ("1push", "2push"):
            for t in ("easy", "medium", "hard"):
                d = g[arm].get(hz, {}).get(t)
                if d:
                    splits.append(f"{arm:<6} {hz:<6} {t:<7} n={d['n']:<5} @1={d['solve@1']:<6}"
                                  f"@5={d['solve@5']:<6}@30={d['solve@30']:<6}@900={d['solve@900']:<6}"
                                  f"s2s={d['avg_sims_to_solve']}")

txt = "\n".join(out) + "\n\n```\n" + "\n".join(splits) + "\n```\n"
(R0 / "final_comparison.md").write_text(txt)
print(txt)
