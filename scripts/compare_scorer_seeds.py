#!/usr/bin/env python3
"""Compare paired multi-seed outputs from scripts/eval_scorer.py."""

import argparse
import json
from pathlib import Path
from statistics import mean


TIERS = ("easy", "med", "hard")


def load_result(path):
    with open(path) as f:
        return json.load(f)


def identity_sequence(path):
    rows = []
    with open(path) as f:
        for line in f:
            row = json.loads(line)
            rows.append((
                row["i"], row["xml"], row["object_id"], row.get("region"),
                tuple(row["object_center"]), row["sr"],
            ))
    return rows


def jsonl_path(path):
    path = Path(path)
    return path.with_suffix(".jsonl")


def metrics(result, tier):
    row = result["divisions"][tier]
    decomp = row["failure_decomp_at1_pct"]
    return {
        "n": row["n"],
        "hit1": row["scorer_realistic"]["@1"],
        "hit5": row["scorer_realistic"]["@5"],
        "wrong_contact": decomp["wrong_edge"],
        "wrong_depth": decomp["right_edge_wrong_depth"],
        "missing": row["valid_missing_from_pool"],
    }


def compare(baseline_paths, treatment_paths):
    if len(baseline_paths) != len(treatment_paths):
        raise ValueError("baseline and treatment must have the same number of seeds")
    if len(baseline_paths) != 3:
        raise ValueError("the pre-registered verdict requires exactly three paired seeds")

    baselines = [load_result(p) for p in baseline_paths]
    treatments = [load_result(p) for p in treatment_paths]
    reference_ids = None
    for bp, tp, baseline, treatment in zip(
            baseline_paths, treatment_paths, baselines, treatments):
        if baseline["mode"] != "live_canonical" or treatment["mode"] != "live_canonical":
            raise ValueError("all inputs must be live_canonical evaluations")
        baseline_ids = identity_sequence(jsonl_path(bp))
        treatment_ids = identity_sequence(jsonl_path(tp))
        if baseline_ids != treatment_ids:
            raise ValueError(f"paired identity mismatch: {bp} vs {tp}")
        if reference_ids is None:
            reference_ids = baseline_ids
        elif baseline_ids != reference_ids:
            raise ValueError(f"cross-seed identity mismatch: {bp}")

    rows = []
    for tier in TIERS:
        for seed_idx, (baseline, treatment) in enumerate(zip(baselines, treatments), start=1):
            b = metrics(baseline, tier)
            t = metrics(treatment, tier)
            if b["n"] != t["n"]:
                raise ValueError(f"seed {seed_idx} {tier}: episode-count mismatch")
            if b["missing"] or t["missing"]:
                raise ValueError(f"seed {seed_idx} {tier}: valid actions missing from candidate pool")
            rows.append({
                "tier": tier, "seed": seed_idx, "n": b["n"],
                "baseline": b, "treatment": t,
            })

    hard = [row for row in rows if row["tier"] == "hard"]
    hard_hit_deltas = [row["treatment"]["hit1"] - row["baseline"]["hit1"] for row in hard]
    hard_wrong_deltas = [row["treatment"]["wrong_contact"] - row["baseline"]["wrong_contact"]
                         for row in hard]
    guardrail = all(
        abs(row["treatment"]["hit1"] - row["baseline"]["hit1"]) <= 2.0
        for row in rows if row["tier"] in ("easy", "med")
    )
    positive_hard = sum(delta > 0 for delta in hard_hit_deltas)
    mean_hard_positive = mean(hard_hit_deltas) > 0
    mean_hard_wrong_falls = mean(hard_wrong_deltas) < 0
    if positive_hard == len(hard) and mean_hard_wrong_falls and guardrail:
        verdict = "STRONG CONFIRMATION"
    elif positive_hard >= 2 and mean_hard_positive:
        verdict = "PARTIAL CONFIRMATION"
    else:
        verdict = "FAILURE TO REPLICATE"
    return rows, {
        "verdict": verdict,
        "positive_hard_seeds": positive_hard,
        "num_seeds": len(hard),
        "mean_hard_hit1_delta": mean(hard_hit_deltas),
        "mean_hard_wrong_contact_delta": mean(hard_wrong_deltas),
        "easy_med_guardrail": guardrail,
        "identities": len(reference_ids),
    }


def markdown(rows, summary):
    lines = [
        "| tier | seed | n | baseline @1 | treatment @1 | delta | baseline @5 | treatment @5 | delta |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for tier in TIERS:
        tier_rows = [row for row in rows if row["tier"] == tier]
        for row in tier_rows:
            b, t = row["baseline"], row["treatment"]
            lines.append(
                f"| {tier} | {row['seed']} | {row['n']} | {b['hit1']:.1f}% | {t['hit1']:.1f}% | "
                f"{t['hit1'] - b['hit1']:+.1f} pp | {b['hit5']:.1f}% | {t['hit5']:.1f}% | "
                f"{t['hit5'] - b['hit5']:+.1f} pp |"
            )
        lines.append(
            f"| {tier} | mean | {tier_rows[0]['n']} | "
            f"{mean(row['baseline']['hit1'] for row in tier_rows):.1f}% | "
            f"{mean(row['treatment']['hit1'] for row in tier_rows):.1f}% | "
            f"{mean(row['treatment']['hit1'] - row['baseline']['hit1'] for row in tier_rows):+.1f} pp | "
            f"{mean(row['baseline']['hit5'] for row in tier_rows):.1f}% | "
            f"{mean(row['treatment']['hit5'] for row in tier_rows):.1f}% | "
            f"{mean(row['treatment']['hit5'] - row['baseline']['hit5'] for row in tier_rows):+.1f} pp |"
        )
    lines.extend(["", "| tier | seed | baseline wrong contact | treatment wrong contact | delta | baseline right-contact/wrong-depth | treatment right-contact/wrong-depth | delta |",
                  "|---|---:|---:|---:|---:|---:|---:|---:|"])
    for tier in TIERS:
        tier_rows = [row for row in rows if row["tier"] == tier]
        for row in tier_rows:
            b, t = row["baseline"], row["treatment"]
            lines.append(
                f"| {tier} | {row['seed']} | {b['wrong_contact']:.1f}% | {t['wrong_contact']:.1f}% | "
                f"{t['wrong_contact'] - b['wrong_contact']:+.1f} pp | {b['wrong_depth']:.1f}% | "
                f"{t['wrong_depth']:.1f}% | {t['wrong_depth'] - b['wrong_depth']:+.1f} pp |"
            )
        lines.append(
            f"| {tier} | mean | {mean(row['baseline']['wrong_contact'] for row in tier_rows):.1f}% | "
            f"{mean(row['treatment']['wrong_contact'] for row in tier_rows):.1f}% | "
            f"{mean(row['treatment']['wrong_contact'] - row['baseline']['wrong_contact'] for row in tier_rows):+.1f} pp | "
            f"{mean(row['baseline']['wrong_depth'] for row in tier_rows):.1f}% | "
            f"{mean(row['treatment']['wrong_depth'] for row in tier_rows):.1f}% | "
            f"{mean(row['treatment']['wrong_depth'] - row['baseline']['wrong_depth'] for row in tier_rows):+.1f} pp |"
        )
    lines.extend([
        "",
        f"**Verdict: {summary['verdict']}.** Hard exact @1 improved in "
        f"{summary['positive_hard_seeds']}/{summary['num_seeds']} seeds; mean hard exact @1 delta "
        f"{summary['mean_hard_hit1_delta']:+.1f} pp; mean hard wrong-contact delta "
        f"{summary['mean_hard_wrong_contact_delta']:+.1f} pp; easy/medium two-point guardrail "
        f"{'passed' if summary['easy_med_guardrail'] else 'failed'}; "
        f"{summary['identities']} canonical episode identities matched across all inputs.",
    ])
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", nargs="+", required=True)
    parser.add_argument("--treatment", nargs="+", required=True)
    parser.add_argument("--out")
    args = parser.parse_args()
    rows, summary = compare(args.baseline, args.treatment)
    report = markdown(rows, summary)
    if args.out:
        Path(args.out).write_text(report)
    print(report, end="")


if __name__ == "__main__":
    main()
