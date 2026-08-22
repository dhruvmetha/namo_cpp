#!/usr/bin/env python3
"""Keep sharded scene labelling running until every scene has a pkl, refilling boxes that die.

Written because a box died silently overnight: rlab6 stopped after 10 of 31 scenes with no error in
its log and no process left, while four sibling shards ran fine. A one-shot fan-out loses that
shard's work and nobody notices until morning. This re-checks every `--interval` seconds and
relaunches whatever is missing, so a dead box costs one interval instead of a night.

Resumption is by MISSING INDEX, not by count. The collector writes `<host>_env_%06d_results.pkl`
using each scene's index in its shard manifest, and workers finish out of order, so the completed
set is not a prefix and `--start-idx <n_done>` would skip scenes that never ran. This parses the
indices actually present and writes a fresh manifest of only the gaps.

BOXES must list only glibc >= 2.38 hosts. `build_python/namo_rl*.so` is built on ilab3 (2.39) and
westeros and arrakis sit on 2.35, where the import dies with `GLIBC_2.38 not found`. That is how
355 scenes silently did nothing on westeros.

  python scripts/pipeline/sweep_label_shards.py --root <wave-dir> --manifest <all.txt> \\
      --boxes rlab7:128 rlab5:64 rlab4:48 ilab3:48 rlab6:32 --interval 300
"""
import argparse
import os
import re
import subprocess
import time

# this file lives at <repo>/scripts/pipeline/, so THREE dirnames reach the root.
# Two lands in scripts/, which made every remote `cd $REPO && source env.ilab.sh`
# fail at the && and produce a silent no-op launch with an empty collect.log.
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CONFIG = "python/namo/data_collection/region_opening_exhaustive_2push_multihop_car.yaml"
PKL_IDX = re.compile(r"_env_(\d+)_results\.pkl$")


def sh(host, cmd, timeout=60):
    try:
        r = subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15",
                            f"{host}.cs.rutgers.edu", cmd],
                           capture_output=True, text=True, timeout=timeout)
        return r.stdout.strip()
    except Exception:
        return ""


def busy(host, shard):
    """Live processes alone do NOT mean a box is working. Reclaim it if its shard is finished.

    The collection hangs after writing its last pkl: the parent stays in sleep and all 48 workers
    sit as ZN zombies it never reaps. So `pgrep` keeps returning ~49 on a box with nothing left to
    do, the box reads as busy forever, and the supervisor never refills it. Two boxes sat idle-but-
    busy this way while 1644 scenes waited. Plain SIGTERM does not shift the parent either, hence
    the -9.
    """
    mf = os.path.join(shard, "manifest.txt")
    if os.path.exists(mf):
        want = sum(1 for l in open(mf) if l.strip())
        if want and len(done_indices(shard)) >= want:
            sh(host, "pkill -9 -u dm1487 -f modular_parallel_collection")
            return False
    out = sh(host, "pgrep -u dm1487 -fc modular_parallel_collection")
    return out.isdigit() and int(out) > 0


def done_indices(shard):
    pkls = os.path.join(shard, "pkls")
    got = set()
    for dirpath, _d, files in os.walk(pkls):
        for f in files:
            m = PKL_IDX.search(f)
            if m:
                got.add(int(m.group(1)))
    return got


def launch(host, workers, shard, n, timeout_s):
    """Start one shard on `host`, detached, and confirm it actually started.

    The backgrounding has to sit INSIDE a subshell, after the setup chain has run in the
    foreground. Writing it as `cd X && source Y && nohup python ... &` looks equivalent and is not:
    `&` binds looser than `&&`, so the WHOLE chain becomes one background job and ssh tears the
    session down before it reaches nohup. That failed silently for hours, every round reporting a
    launch, `$!` coming back as 0, and not one line appended to collect.log. `setsid` then keeps the
    child off the session's process group so the teardown cannot reach it either.
    """
    inner = ("nohup python -m namo.data_collection.modular_parallel_collection "
             f"--config-yaml {CONFIG} --manifest {shard}/manifest.txt "
             f"--output-dir {shard}/pkls --start-idx 0 --end-idx {n} --workers {workers} "
             f"--search-timeout {timeout_s} >> {shard}/collect.log 2>&1 < /dev/null &")
    cmd = (f"cd {REPO} && source env.ilab.sh >/dev/null 2>&1 && "
           "export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 "
           "NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 && "
           f"PYTHONPATH=\"{REPO}/build_python:{REPO}/python\" setsid bash -c '{inner}' ; "
           "sleep 3 ; pgrep -u dm1487 -fc modular_parallel_collection")
    out = sh(host, cmd, timeout=90)
    live = int(out) if out.strip().isdigit() else 0
    if live == 0:
        print(f"  !! {host}: launch produced no processes, see {shard}/collect.log", flush=True)
    return live


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="wave dir holding one subdir per box")
    ap.add_argument("--manifest", required=True, help="every scene that must end up labelled")
    ap.add_argument("--boxes", nargs="+", required=True, help="host:workers, glibc>=2.38 only")
    ap.add_argument("--interval", type=int, default=300)
    ap.add_argument("--search-timeout", type=int, default=600)
    ap.add_argument("--max-rounds", type=int, default=200)
    args = ap.parse_args()

    all_scenes = [l.strip() for l in open(args.manifest) if l.strip()]
    boxes = [(b.split(":")[0], int(b.split(":")[1])) for b in args.boxes]
    total_w = sum(w for _h, w in boxes)

    # Completion is tracked BY SCENE PATH in one accumulating file, not by counting pkls per shard.
    # Each round rewrites a shard's manifest, so an index means something different afterwards and
    # archived pkls no longer sit where done_indices() would find them. Without this file every
    # scene finished in an earlier round looks unlabelled again and gets re-simulated forever.
    done_file = os.path.join(args.root, "completed.txt")
    completed = set()
    if os.path.exists(done_file):
        completed = {l.strip() for l in open(done_file) if l.strip()}

    for rnd in range(args.max_rounds):
        for host, _w in boxes:
            shard = os.path.join(args.root, host)
            mf = os.path.join(shard, "manifest.txt")
            if not os.path.exists(mf):
                continue
            scenes = [l.strip() for l in open(mf) if l.strip()]
            completed.update(scenes[i] for i in done_indices(shard) if i < len(scenes))
        with open(done_file, "w") as f:
            f.write("\n".join(sorted(completed)) + "\n")

        # A busy box is already working its manifest. Those scenes are neither completed nor free,
        # and handing them to an idle box duplicates the work rather than finishing sooner.
        idle_hosts, in_flight = [], set()
        for host, w in boxes:
            if busy(host, os.path.join(args.root, host)):
                mf = os.path.join(args.root, host, "manifest.txt")
                if os.path.exists(mf):
                    in_flight.update(l.strip() for l in open(mf) if l.strip())
            else:
                idle_hosts.append((host, w))

        pending = [s for s in all_scenes if s not in completed and s not in in_flight]

        if not pending and not in_flight:
            print(f"[round {rnd}] all {len(all_scenes)} scenes labelled "
                  f"({len(completed)} completed)", flush=True)
            return

        idle = idle_hosts
        print(f"[round {rnd}] pending={len(pending)} in_flight={len(in_flight)} "
              f"idle={[h for h, _ in idle]}", flush=True)

        if idle and pending:
            idle_w = sum(w for _h, w in idle)
            off = 0
            for host, w in idle:
                take = max(1, len(pending) * w // idle_w)
                slice_ = pending[off:off + take]
                if not slice_:
                    break
                shard = os.path.join(args.root, host)
                os.makedirs(os.path.join(shard, "pkls"), exist_ok=True)
                # ARCHIVE, never delete. Finished pkls ARE the results. They must leave the shard's
                # working dir because their `_env_%06d_` index refers to the OLD manifest and the
                # rewritten one restarts at 0, but they stay under --root so the answer key still
                # sees them. build_2push_validset.py keys on realpath(xml) read from inside each
                # pkl, not on the filename, so same-named files in sibling dirs are fine.
                archive = os.path.join(args.root, "done", f"{host}_r{rnd}")
                os.makedirs(archive, exist_ok=True)
                subprocess.run(
                    f"find {shard}/pkls -name '*.pkl' -exec mv -t {archive} {{}} + 2>/dev/null",
                    shell=True)
                # a fresh manifest of only the gaps, so index 0..n-1 maps to real work again
                with open(os.path.join(shard, "manifest.txt"), "w") as f:
                    f.write("\n".join(slice_) + "\n")
                live = launch(host, w, shard, len(slice_), args.search_timeout)
                print(f"  {host}: {len(slice_)} scenes, {w} workers, {live} procs up", flush=True)
                off += take
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
