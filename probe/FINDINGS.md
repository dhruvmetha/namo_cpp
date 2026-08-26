# Can this MuJoCo setup represent accumulating block rotation?

Capability probe, 2026-08-26. Nothing here is adopted, and nothing here is a fit. Scene copies live in scratch; `config/`, `datasets/` and the pool XMLs were never touched.

## Answer

**Yes, but not through any parameter.** Rotation is reachable only by changing the pusher's contact *geometry*. Every constant in the defensible range is provably inert, and I can say why rather than just reporting that a sweep came back empty.

Torsional friction, the leading suspect, is falsified twice over. Setting the block's torsional coefficient to `0.0001` gives results **bit-identical** to the current `0.005`, and raising it to `0.5` or `5.0` *reduces* rotation instead of increasing it. In the measured yaw-torque budget it carries 0.005 mNm out of about 4 mNm, which is 0.14%. It never mobilises, because the block barely spins against the floor in the first place.

## What the sim is actually doing

The car's 7 cm front face and the block's 7 cm end face form a **flat-on-flat box mate**. Measured through the whole push: a contact manifold 4.29 to 4.32 cm wide, 3 to 4 points, **both ends loaded simultaneously**, penetrating 0.005 to 0.14 mm.

That mate is a kinematic constraint on relative yaw. The block's heading tracks the car's heading to within 0.3 degrees for the entire 5.5 s push. The 1.8 degrees that appears in step 1 and unwinds over steps 2 to 5 is not the block rotating, it is the car's own yaw wobble being copied onto a block welded to its face.

The controller is **not** the cause. I rebuilt the push with an idealised pusher on a single slide joint, velocity-servoed, mass 10 kg: it cannot yaw, cannot slip, cannot lose traction. It produces the same null at every offset from 0.0 to 3.5 cm. Pure-pursuit and CTE-PD are exonerated.

## Why no constant can work

The push force needed to keep the block moving equals its floor drag, `mu*m*g`. The floor's resistance to yaw *also* scales with `mu*m*g`. The ratio is therefore invariant, and the rotation rate follows the sliding limit surface as

    coupling = d / c^2

with `d` the moment arm of the contact patch and `c` the block's yaw friction radius. Neither `mu` nor `m` appears. Sweeping them is guaranteed to do nothing, which is exactly what the sweep found.

I verified the invariance on the point-pusher case, where rotation is not suppressed: friction over a 20x range gives -3.06 to -2.84 deg/cm, mass over a 10x range gives -3.21 to -2.84 deg/cm. Flat across both.

Two measured quantities set the outcome:

| quantity | value | how |
|---|---|---|
| MuJoCo's floor support radius `c` | **8.28 cm**, invariant | all four block-floor contacts sit at the extreme corners, always |
| uniform-pressure radius of a real 7x15 block | 4.39 cm | integrated over the footprint |
| `c` implied by the hardware's own coupling | **~9.0 cm** | from 2.126 at 3.0 cm and 2.416 at 3.5 cm |
| largest moment arm available on a 7 cm face | 3.50 cm | a point contact at the very corner |

The ground model is fine. MuJoCo's 8.28 cm and the hardware's implied 9.0 cm agree well. **The entire discrepancy is at the pusher interface**, where the flat mate collapses the effective moment arm from the commanded 3.0 cm to a measured 1.3 cm and then adds an angular constraint that removes what is left.

## The joint sweep

One-at-a-time nulls are weak, so I ran the full product: block friction x torsional x mass x pusher friction x solref x cone x condim x impratio, at two offsets. **10,368 runs.**

Median |coupling| 0.006 deg/cm. Hardware needs 2.126 to 2.416.

31 combinations do clear 2.126 with translation intact, and they are **not** a mechanism. Running the best of them across the full offset profile:

| offset cm | 0.5 | 1.0 | 2.0 | 2.5 | 3.0 | 3.5 |
|---|---|---|---|---|---|---|
| best sweep candidate | -0.000 | -0.000 | -0.000 | +0.000 | +0.001 | **-2.865** |
| hardware | 0.199 | flat after 3 deg | - | - | 2.126 | 2.416 |

Exactly zero at every offset, then a cliff at one. Hardware's coupling grows in proportion to offset, with `k/offset` constant at 0.709 and 0.690 across two independent trials. **Zero settings out of 10,368 clear 1.5 deg/cm at both 3.0 and 3.5 cm.** These are numerical accidents at one degenerate geometry.

## What does work, and why it is not a fix

Replacing the car's flat front face with a cylinder or a 1 cm face, **on the real scene with the real car and the real controller**, changes nothing else:

| pusher | off 3.5 | off 3.0 | off 1.0 | off 0.5 |
|---|---|---|---|---|
| current flat 7 cm | +2.5 deg | +0.1 | -0.1 | +0.1 |
| cylinder | **+27.3 deg** / 1.190 | +22.4 / 1.123 | +12.8 / 0.517 | -18.0 / -0.541 |
| 1 cm face | +29.6 / 1.184 | +24.6 / 1.141 | +8.0 / 0.281 | -14.2 / -0.443 |
| hardware | +47.4 / 2.416 | +27.3 / 2.126 | +1.2 / flat | ~+1 / 0.199 |

So the behaviour is representable. It was suppressed by interface geometry, not by any constant.

But it fails your two-sided test, and it fails it on the control case. At 0.5 cm offset the cylinder gives **-18.0 degrees where hardware self-squares to about +1**. Coupling at the corners is also about half the hardware value. Both pushes also trip the stuck detector and abort at 1601 of 2950 ticks.

The real interface behaves like a **point contact at corner offsets and like a flat mate near centre**. No single geometry I tested switches between the two regimes, and that switch is the thing that would actually have to be modelled.

## What this says about the sim-real gap

This is a structural limit, not a calibration error, and it is worth writing down as a characterisation.

MuJoCo represents two nominally flat faces as a *perfect planar mate* over the full overlap. Real faces are flat to maybe 0.1 mm and touch at a few high spots, so a real interface has far lower angular stiffness and cannot hold a block square. The sim's push therefore carries an angular constraint the hardware does not have, and it is strongest exactly where the label matters most, at corner contacts.

Practical read: solve rates that depend on **which** push opens a region stay trustworthy, since translation is unaffected. Anything that depends on the block's **final heading** after a corner push is systematically wrong, and wrong in one direction, since the sim always under-rotates.

## Reproduction

    source env.ilab.sh
    export PYTHONPATH="<main-checkout>/build_python:$PWD/python"
    python probe/one.py <scene.xml> <config.yaml> <edge> <tag>   # one push, per-tick
    python probe/joint_sweep.py                                  # 10,368-run sweep, ~14 s on 26 cores

`probe/rig.py` is the idealised pusher. `probe/torque_probe.py` is the yaw-torque budget and contact-manifold reader. `probe/variants.py` writes scene copies to scratch and never edits a source XML.

Bindings came from the main checkout's prebuilt `.so`, read-only. No build was run in the main tree and no C++ was changed.
