# Grasp geometry and stabilization diagnostics

## Scope and controlled variables

This iteration diagnoses grasp geometry before any ACT training or palm-pose scan. It
keeps preload at `0.76`, the two-segment minimum-jerk Lift reference, contact/friction
settings, and cube randomization unchanged. The safe starting point is commit
`a8a14dc` on `experiment/grasp-settle-gate`.

Run the fixed-seed diagnostic with:

```powershell
$env:PYTHONPATH = "$PWD/src"
./.venvs/openarm-sim/Scripts/python.exe scripts/grasp_geometry_diagnostics.py `
  --model outputs/reach_grasp_lift/reach_grasp_lift.mjb `
  --config configs/reach_grasp_lift.json `
  --synergies configs/wuji_hand_left_synergies.json `
  --output outputs/grasp_geometry_diagnostics `
  --seeds 0 7 11 19 29
```

Each seed gets full JSON and CSV time series. Every row contains both world poses,
`T_palm^-1 * T_object`, drift from Lift start, object/palm linear and angular velocity,
commanded and measured palm orientation, contact-face/edge geometry, and net wrench.
Rows are labeled `close`, `preload`, `lift_transient`, or `lift_hold` for analysis while
retaining the original controller phase.

## Post-settle stability

`grasp_stable` remains the strict full-Lift CD-WM comparison and is never rescued by a
later settling event. `post_settle_stable` separately evaluates the final 0.5 seconds.
It requires final relative translation/rotation drift below the same external 8 mm / 6°
limits. RMS relative translation/angular speed and their drift slopes must also be below
`8 mm / 0.5 s = 16 mm/s` and `6° / 0.5 s = 12°/s`. Maximum and RMS speeds and continuous
drift slopes are all reported. These are explicitly external-derived diagnostic gates,
not calibrated Wuji thresholds.

The controller now holds the final Lift waypoint for a full 0.5 seconds before this
classification. The Lift reference itself is unchanged.

## 6D palm pose IK

The DLS task vector now stacks translation error with a quaternion rotation-vector
error and uses both translational and rotational site Jacobians. The same configured
palm orientation is used for Reach, Approach/Preload, and Lift IK targets.

The kinematic solver reaches roughly micrometre position residual and `0.001°`
orientation residual. Position actuators under gravity still create a systematic
orientation bias. A separate, explicit simulation-only rotation-vector calibration is
therefore applied to the internal IK command while telemetry preserves all three values:

- desired/commanded palm quaternion;
- compensated IK command quaternion;
- actual palm quaternion and geodesic error.

Seed 7 ends Reach at about `0.44°` error. During Lift, however, the five-seed maximum
orientation error is `6.2–6.9°` and palm orientation drift is `4.8–5.0°`. The pose IK is
working, but physical tracking is not yet reliable enough to claim completion or begin a
palm-pose scan. The 6D path also under-tracks the 50 mm reference during the first 0.5 s
(`25.0–39.9 mm` in the four Lifted runs). This is recorded as a controller limitation;
the external protocol result is not forced to pass.

## Contact opposition and wrench diagnostics

Every MuJoCo hand/cube contact is transformed into the cube frame and classified as
`+X`, `-X`, `+Y`, `-Y`, `+Z`, or `-Z`. Reports include edge/corner distances and flags,
thumb and other-finger face distributions, force-weighted dominant faces, contact
centroids and COM offsets, thumb/other resultants and force angle, force-line distance
to COM, world net force, horizontal force, and world net moment.

The first fixed five seeds produced:

| Seed | Task | Strict | Post-settle | Outcome | Full drift mm/° | Post drift mm/° | Thumb / other dominant face | Opposition frames | Peak Fxy / torque |
| ---: | :---: | :---: | :---: | --- | ---: | ---: | --- | ---: | ---: |
| 0 | yes | no | no | persistent_slip | 57.0 / 17.1 | 2.67 / 3.93 | -Y / +X | 0.0% | 8.35 N / 0.138 Nm |
| 7 | yes | no | yes | settled_after_slip | 52.5 / 17.7 | 1.39 / 1.60 | -Y / +Y | 77.6% | 8.28 N / 0.198 Nm |
| 11 | yes | no | yes | settled_after_slip | 53.7 / 16.5 | 0.96 / 1.37 | -Y / +X | 0.0% | 7.93 N / 0.136 Nm |
| 19 | yes | no | yes | settled_after_slip | 53.9 / 17.9 | 1.06 / 1.55 | -Y / +X | 0.0% | 8.02 N / 0.128 Nm |
| 29 | no | no | no | never_lift (Reach timeout) | 0 / 0 | 0 / 0 | N/A | N/A | N/A |

Under the new fixed orientation the thumb is consistently on `-Y` and its aggregated
edge/corner rate is zero in the four completed lifts. This removes the previous
systematic thumb-at-`(-35,-35)` observation for these seeds. It does not yet create
reliable opposition: only seed 7 has `+Y` as the other-finger dominant face; the other
three successful grasps are dominated by `+X`.

Four of five seeds reach the height target, zero are strict stable grasps, three are
post-settle stable, one continues rotating too quickly in the final window, and one never
passes Reach. The large `52–57 mm` transient drift remains unacceptable for expert
demonstrations even when the final pose settles.

## Decision before pose scan

Do not start the yaw/XY scan yet. First fix 6D palm tracking so the measured orientation
stays within the configured tolerance during Lift and restore physical tracking of the
unchanged 50 mm / 0.5 s reference without increasing drop rate. Once that controller
check passes on fixed seeds, use these opposition/wrench metrics to rank a small yaw and
XY candidate set. No trajectory with strict drift failure is eligible as an ACT expert
demonstration.
