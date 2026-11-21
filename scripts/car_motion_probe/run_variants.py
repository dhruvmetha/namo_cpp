#!/usr/bin/env python3
"""Measure car variants with the same probes: each v2 change alone, then combined.

Each variant is built from the pool scene by scripts/car_v2/convert_v1_to_v2.py with
ctrl_units="omega", so the unchanged C++ push controller (which sends wheel speeds in
rad/s) can drive the PWM motor; then probe_drive.py and probe_push.py run on it exactly
as on the v1 baseline. The "v1" variant is v1 rebuilt by the converter and must reproduce
the baseline numbers.

  python run_variants.py --xml <pool scene> --config <car config> --out <dir> \\
      [--build-python <checkout>/build_python] [--only v1,M1,...]
"""
import argparse
import json
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "car_v2"))
import car_v2_block as V        # noqa: E402
import convert_v1_to_v2 as C    # noqa: E402

VARIANTS = {
    "v1":         dict(V.V1),
    "M1_caster":  V.V1 | dict(caster="one_frictionless"),
    "M2_wheel":   V.V1 | dict(wheel="v2"),
    "M3_motor":   V.V1 | dict(actuator="pwm"),
    "body_mvp":   V.V1 | dict(body="mvp"),
    "M1+M2":      V.V1 | dict(caster="one_frictionless", wheel="v2"),
    "mvp+M1+M2":  V.V1 | dict(body="mvp", caster="one_frictionless", wheel="v2"),
    "v2":         dict(V.V2),
}


def row(name, opts, out):
    d = json.loads((out / "drive_metrics.json").read_text())
    p = json.loads((out / "push_metrics.json").read_text())
    a, st, sf, ss, arc = d["a_straight"], d["d_stop"], d["b_spin_fast"], d["b_spin_slow"], d["c_arc"]
    caster_drag = sum(v["fn"] * v["ft_over_fn"] for k, v in a["steady_loads"].items() if k.startswith("c"))
    pc, pp = p["push_cpp"], p["push"]
    return dict(
        variant=name, opts=opts, weight_N=d["weight_N"], geometry=d["geometry"],
        steady_cms=a["steady_speed_cms"], start_peak_cms=a["steady_speed_cms"] * (1 + a["overshoot_pct"] / 100),
        start_airborne_ms=a["airborne_ms"], start_settle_5pct_ms=a["settle_to_5pct_ms"], start_jerk_max=a["jerk_max"],
        steady_pitch_deg=a["pitch_steady_deg"][0], caster_drag_N=caster_drag,
        stop_reverse_mms=st["max_reverse_speed_mms"], stop_settle_ms=st["settle_to_1mms_ms"],
        stop_airborne_ms=st["airborne_ms"], stop_dist_mm=st["final_dist_mm"],
        spin_fast_ratio=sf["ratio"], spin_fast_drift_mm_per_turn=sf["drift_per_turn_mm"],
        spin_slow_ratio=ss["ratio"], spin_slow_drift_mm_per_turn=ss["drift_per_turn_mm"],
        arc_radius_cm=arc["fit_radius_cm"], arc_cmd_radius_cm=arc["cmd_radius_cm"],
        arc_yaw_ratio=arc["actual_yaw_rate_dps"] / arc["cmd_yaw_rate_dps"],
        push_done=p["done"], push_stopped_early=p["info"].get("stopped_early"),
        push_failure=p["info"].get("failure_reason"), push_ticks=p["push_ticks"],
        push_v_mean_cms=pc["v_mean_cms"], push_stall_frac=pc["stall_frac_v_below_half_cruise"],
        push_car_mm=pc["car_travel_mm"], push_block_mm=pc["block_travel_mm"],
        push_contact_frac=pp["block_contact_frac"], push_contact_onsets=pp["block_contact_onsets"],
        push_big_ctrl_steps=pc["big_ctrl_steps"], push_wheel_slip_mean=pp["wheel_slip_mean"],
        push_yaw_rate_max_dps=pc["yaw_rate_max_dps"], push_jerk_rms=pp["jerk_rms_after_100ms"],
        replay_drift_mm=p["replay_vs_cpp"]["car_xy_err_max_mm"],
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xml", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--build-python", default=None)
    ap.add_argument("--only", default=None)
    args = ap.parse_args()
    root = ET.parse(args.xml).getroot()
    ok, why = C.is_v1_car(root)
    assert ok, why
    names = args.only.split(",") if args.only else list(VARIANTS)
    rows = []
    for name in names:
        opts = VARIANTS[name] | dict(ctrl_units="omega")
        out = Path(args.out) / "variants" / name
        (out / "traces").mkdir(parents=True, exist_ok=True)
        scene = out / "scene.xml"
        scene.write_text("<?xml version='1.0' encoding='utf-8'?>\n" +
                         ET.tostring(C.convert_root(root, **opts), encoding="unicode") + "\n")
        py = sys.executable
        subprocess.run([py, str(HERE / "probe_drive.py"), "--xml", str(scene), "--out", str(out)],
                       check=True, stdout=open(out / "drive_stdout.txt", "w"), stderr=subprocess.STDOUT)
        push = [py, str(HERE / "probe_push.py"), "--xml", str(scene), "--config", args.config, "--out", str(out),
                "--object", "obstacle_1_movable", "--edge", "30", "--depth", "4"]   # the baseline push
        if args.build_python:
            push += ["--build-python", args.build_python]
        subprocess.run(push, check=True, stdout=open(out / "push_stdout.txt", "w"), stderr=subprocess.STDOUT)
        rows.append(row(name, opts, out))
        print(json.dumps(rows[-1]), flush=True)
    (Path(args.out) / "variants" / "summary.json").write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
