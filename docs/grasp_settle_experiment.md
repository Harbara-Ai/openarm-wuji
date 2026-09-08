# Preload and bounded-reference Lift experiment

## Checkpoint and branch

The pre-experiment SE(3) evaluator and causal-recorder baseline is preserved by local
commit `36cbb5b` on `main`. This experiment lives on
`experiment/grasp-settle-gate` and has not been pushed.
The previous paced-Lift implementation is preserved at `3b1189a`.

Run the complete seed-7 experiment with:

```powershell
./scripts/run_grasp_settle_experiment.ps1
```

The command rebuilds the model, regenerates the annotated GIF and reports, records the
then-current schema-v2 episode, and requires deterministic replay. `settled_after_slip` is the
expected seed-7 outcome for this branch, so height success does not masquerade as a
stable grasp.

## State machine

```text
grasp_close
  -> first frame with synergy >= 0.7 and enough finger contacts
freeze_synergy
  -> record the first qualifying synergy (seed 7: 0.72)
preload
  -> constant arm targets, synergy increases at most 0.01/frame to 0.76
preload_settle
  -> constant arm and hand targets, evaluate only new settle frames
lift_s_curve or regrasp_required
```

The settle window reuses the existing eight-frame contact-hold length. Every frame in
that window must retain the configured multi-finger contact condition, while object pose
relative to the grasp center must remain below the explicitly external CD-WM 8 mm and
6 degree gates. Closing and preload-ramp frames never enter this window. Least-squares
slopes of the hand-contact resultant force norm and moment norm must both be <= 0
over the window. This is a provisional non-growth gate, not a force-closure proof,
absolute force ceiling, or hardware-calibrated noise test. Large constant loads can pass.

Closing retains its 55-frame timeout. Preload and preload-settle each have separate
30-frame budgets. Exhaustion returns `preload_timeout` or `preload_unstable`, sets
`regrasp_required=true`, and rejects Lift. The controller reports a regrasp request;
automatic repositioning is not implemented. Arm *commands* remain fixed, while the
physical joints can move under load. Synergy is a position command, not a Newton-valued
force controller. If a seed freezes above the preload target, preload never opens it.

## Cartesian reference limits

The reference uses two quintic minimum-jerk segments, `10u^3 - 15u^4 + 6u^5`:
50 mm in 0.5 s, followed by 70 mm in 0.7 s. Position, velocity, and acceleration
are continuous at 0, 0.5, and 1.2 s; jerk is bounded but can jump at these joins.
The analytic segment peaks are `1.875 D/T`, `10/sqrt(3) D/T^2`, and `60 D/T^3`.
The resulting global reference peaks are 0.1875 m/s, 1.1547 m/s², and 24 m/s³.
Explicit config limits are 0.19 m/s, 1.16 m/s², and 24.1 m/s³, rounded above those
analytic requirements. Infeasible settings are rejected before robot motion. These
values are engineering reference limits derived from the requested motion, not CD-WM
published acceleration/jerk limits or verified hardware limits.

At this historical checkpoint, Lift retained the final preload synergy with no
first-frame closure jump, while IK remained position-only and joint targets passed
through the existing 0.015 rad/frame cap.
The actual simulated palm therefore need not satisfy the reference limits. The report
includes 30 Hz finite-difference measurements: seed 7 reaches 0.2425 m/s, 2.7656 m/s²,
and 66.06 m/s³, with a maximum position tracking error of 14.34 mm. These are sampled
diagnostics, not continuous-time upper bounds. Physical acceleration/jerk enforcement
and palm orientation control remain open controller work.

## Historical seed-7 result at 3b1189a

The experiment freezes at synergy `0.72` on closing frame 24. Eight subsequent settle
frames pass at `1.20 mm` translation drift and `0.58°` rotation drift. Compared with the
checkpoint, pre-lift cube displacement falls from about `17.5 mm` to `5.9 mm`.

The first version retained the old aggressive joint-space Lift: the gripper moved about
`87 mm` in its first 0.5 seconds, the cube peaked at `73.7 mm`, and then dropped. The
controlled follow-up replaces that jump with cubic Cartesian waypoints whose published
endpoint is the CD-WM external protocol's `50 mm / 0.5 s`. A measured Lift-only gravity
compensation calibration of `29.77 mm` gives `50.12 mm` actual gripper travel for seed 7.
It is controller-specific calibration, not a new success threshold.

With the paced Lift, the cube peaks at `96.3 mm` and finishes at `93.5 mm`, so it no
longer drops. Relative object-in-palm motion still reaches `50.8 mm` and `36.6°` before
the final window settles to `0.60 mm` and `0.82°`. The exclusive result is therefore:

```text
task_success = true
grasp_stable = false
outcome = settled_after_slip
```

Wuji's mapping identifies `finger1` as the thumb. At the last pre-Lift settle frame,
its mean cube-local contact lies near the `x=-35 mm, y=-35 mm` edge rather than centered
on a face opposing the other four fingers. The other fingers span the `y=-35 mm`,
`x=+35 mm`, and `y=+35 mm` faces. This non-opposed, edge-heavy contact layout is a
plausible source of the measured lateral force and torque when table support disappears.
It is evidence of a grasp-geometry problem, not proof that the thumb alone causes all
slip. The next controlled experiment should keep the paced Lift and change only the
pregrasp pose/hand synergy to place the thumb and finger contacts on opposing faces.

## Current seed-7 result and preload selection

The current run freezes at 0.72, preloads for four frames to 0.76, and settles for eight
frames. The settle window measures 4.53 mm / 3.43° relative drift; resultant slopes are
-0.05855 N/frame and -0.002447 Nm/frame. All gates pass. The episode has 122 actions:
12 reach, 31 approach, 24 grasp-close, four preload, eight preload-settle, and 43 Lift.

| Metric | Previous 3b1189a | Current preload + S curve |
| --- | ---: | ---: |
| Actual palm lift in first 0.5 s | 50.12 mm | 50.32 mm |
| Task success / stable grasp | true / false | true / false |
| Outcome | settled_after_slip | settled_after_slip |
| Object peak / final height | 96.31 / 93.48 mm | 97.35 / 95.97 mm |
| Maximum relative translation / rotation | 50.83 mm / 36.56° | 43.96 mm / 16.92° |
| Peak hand-contact resultant force / moment | 37.07 N / 1.540 Nm | 9.23 N / 0.108 Nm |
| Establishment translation / rotation | 23.44 mm / 2.02° | 28.62 mm / 5.72° |

Establishment spans the start of closing through the end of preload-settle. The final
Lift window settles to 0.65 mm / 0.41°, but the earlier slip remains a stability failure.
Reference tracking uses a 27.3 mm Lift gravity compensation calibrated on seed 7.
The pregrasp geometry is unchanged; this result cannot isolate the effects of preload,
trajectory shape, and compensation from one another.

The old 0.93 target was tested first: it passed the static gate but dropped during
Lift, with a 697.88 N peak contact resultant and 30.10 Nm moment. This is consistent
with a severe contact transient; it does not alone prove an elastic-energy mechanism.
A seed-7 sweep at fixed 29.77 mm Lift compensation gave:

| Preload target | Outcome | Peak force | Relative rotation |
| --- | --- | ---: | ---: |
| 0.74 | settled_after_slip | 9.2 N | 20.8° |
| 0.76 | settled_after_slip | 9.2 N | 16.8° |
| 0.78 | settled_after_slip | 11.7 N | 20.0° |
| 0.80 | settled_after_slip | 9.3 N | 18.7° |
| 0.82 | drop | 22.3 N | 40.9° |
| 0.85 | drop | 10.7 N | 37.7° |
| 0.88 | never_lift (preload gate rejected) | 8.4 N | N/A |
| 0.90 | settled_after_slip | 50.7 N | 88.1° |
| 0.93 | drop | 697.9 N | 43.2° |

0.76 is a provisional tuning choice. After that choice, compensation was recalibrated;
both stages used seed 7. These rows are not independent episodes for a success-rate
estimate, and the tuning is not evidence of multi-seed robustness. No stable successes
were found. The complete settings and negative results are saved in
`outputs/reach_grasp_lift/preload_sweep_seed7.json`. Reproduce the sweep with:

```powershell
$env:PYTHONPATH = "$PWD/src"
./.venvs/openarm-sim/Scripts/python.exe scripts/preload_sweep.py --output outputs/reach_grasp_lift/preload_sweep_seed7.json
```

The next evaluation should use held-out seeds and report both height success and stable
grasp success. Adjusting thumb opposition and improving physical trajectory tracking
remain necessary candidates for addressing the residual slip.
