#!/usr/bin/env python3
"""Build the fixed-generator command list for the one-million-XML Colossus-0 run."""
import argparse
import math
import shlex
from pathlib import Path


def templates(root: Path, family: str):
    if family == "aug9":
        return sorted((root / "templates/aug9_car_v3").glob("*/*.xml"))
    return sorted((root / "templates/feb_car").glob("*/*.xml"))


def emit_family(args, family: str, n_base_envs: int, seed_base: int):
    generator_root = Path(args.generator_root)
    family_templates = templates(generator_root, family)
    if not family_templates:
        raise RuntimeError(f"no {family} templates found")
    family_out = Path(args.output_root) / "gen" / family
    counts = [n_base_envs // len(family_templates)] * len(family_templates)
    for index in range(n_base_envs % len(family_templates)):
        counts[index] += 1
    num_objects = generator_root / (
        "templates/aug9_car_v3/num_objects.json" if family == "aug9" else "templates/feb_car/num_objects.json"
    )

    commands = []
    for template_index, (template, count) in enumerate(zip(family_templates, counts)):
        offset = 0
        for chunk_index in range(math.ceil(count / args.chunk_envs)):
            chunk = min(args.chunk_envs, count - offset)
            seed = seed_base + template_index * 10_000_000 + offset
            parts = [
                args.python,
                str(generator_root / "generate_envs.py"),
                str(template),
                "--namo-config", args.namo_config,
                "--num-envs", str(chunk),
                "--output-dir", str(family_out),
                "--num-workers", "1",
                "--start-seed", str(seed),
                "--run-id-offset", str(offset),
                "--num-objects-json", str(num_objects),
                "--object-size-range", "0.06", "0.16",
                "--object-half-height", "0.05",
                "--goal-size", "0.02",
                "--clearance-radius", "0.0",
                "--min-goal-distance", "0.0",
                "--runtime-validate",
            ]
            commands.append(" ".join(shlex.quote(part) for part in parts))
            offset += chunk
    return commands


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--generator-root", required=True)
    parser.add_argument("--namo-config", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--python", required=True)
    parser.add_argument("--aug9-base-envs", type=int, default=360_000)
    parser.add_argument("--feb-base-envs", type=int, default=800_000)
    parser.add_argument("--chunk-envs", type=int, default=20)
    parser.add_argument("--aug9-seed-base", type=int, default=4_200_000_000)
    parser.add_argument("--feb-seed-base", type=int, default=5_200_000_000)
    args = parser.parse_args()

    commands = emit_family(args, "aug9", args.aug9_base_envs, args.aug9_seed_base)
    commands += emit_family(args, "feb", args.feb_base_envs, args.feb_seed_base)
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as handle:
        handle.writelines(f"{command}\n" for command in commands)
    print(f"wrote {len(commands)} commands to {output}")


if __name__ == "__main__":
    main()
