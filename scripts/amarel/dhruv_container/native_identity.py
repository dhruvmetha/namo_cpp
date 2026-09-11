"""Record actual loaded libraries; compare Focal toolchain and runtime identities."""

import argparse
import hashlib
import json
from pathlib import Path
import platform
import subprocess


PACKAGES = ("gcc-9", "g++-9", "libc6:amd64", "libc6-dev:amd64", "libstdc++6:amd64",
            "libgcc-s1:amd64", "binutils", "cmake", "libyaml-cpp0.6:amd64", "libglfw3:amd64")


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def compare(reference, actual):
    mismatches = [key for key in ("compiler", "glibc", "packages", "python")
                  if reference.get(key) != actual.get(key)]
    for name in sorted(reference["libraries"].keys() | actual["libraries"].keys()):
        # These two artifacts are intentionally rebuilt; record but don't
        # require byte equality before the numerical replay comparison.
        # CPU-only runs intentionally do not bind the workstation GPU driver.
        # CUDA user-space libraries from the frozen Python env still compare.
        if name.startswith(("namo_rl.", "libmujoco.", "libcuda.so.")):
            continue
        expected = reference["libraries"].get(name, {}).get("sha256")
        if expected != actual["libraries"].get(name, {}).get("sha256"):
            mismatches.append("library:" + name)
    return mismatches


def library_records(paths):
    records = {}
    for path in sorted(paths):
        if path.name in records:
            raise ValueError("duplicate loaded library basename: " + path.name)
        records[path.name] = dict(path=str(path), sha256=digest(path))
    return records


def capture():
    # Same import order as validate_full_namo_alignment.py.
    import cv2  # noqa: F401
    import namo_rl  # noqa: F401
    import torch  # noqa: F401

    paths = {Path(line.split()[-1]).resolve()
             for line in Path("/proc/self/maps").read_text().splitlines()
             if len(line.split()) >= 6 and line.split()[-1].startswith("/")
             and ".so" in line.split()[-1]}
    libraries = library_records(paths)
    packages = subprocess.check_output(["dpkg-query", "-W", "-f=${Package}=${Version}\n", *PACKAGES], text=True)
    return dict(compiler=subprocess.check_output(["/usr/bin/c++", "-dumpfullversion"], text=True).strip(),
                glibc=platform.libc_ver()[1], packages=dict(line.split("=", 1) for line in packages.splitlines()),
                libraries=libraries, python=platform.python_version(), host=platform.node())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference", type=Path)
    args = parser.parse_args()
    identity = capture()
    if args.reference:
        identity["mismatches"] = compare(json.loads(args.reference.read_text()), identity)
    with args.output.open("x") as handle:
        json.dump(identity, handle, indent=2, sort_keys=True)
        handle.write("\n")
    if identity.get("mismatches"):
        raise SystemExit("native identity mismatch: " + ", ".join(identity["mismatches"]))
    print("Native identity recorded; reference checks passed." if args.reference else "Native identity recorded.")


if __name__ == "__main__":
    main()
