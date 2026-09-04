# Reach–Grasp–Lift task

## Current scope

The first task milestone implements the scene, deterministic reset, and `Reach` phase.
It intentionally stops at a pre-grasp point. `Grasp`, contact classification, `Lift`,
and LeRobotDataset recording are not claimed yet.

Run the complete build and smoke check from PowerShell:

```powershell
./scripts/run_reach_only_smoke.ps1
```

The command rebuilds `outputs/reach_grasp_lift/reach_grasp_lift.mjb` from the pinned
OpenArm pedestal and Wuji Hand sources, then writes a machine-readable report and GIF.
The MJB is generated locally because it embeds mesh assets from two upstream roots.

## Scene and reset contract

- The official OpenArm pedestal fixes the arm roots at `z=0.698 m`.
- The table top is at `z=0.40 m`.
- A `70 x 70 x 70 mm`, `0.12 kg` free-joint cube starts near
  `[0.48, 0.15, 0.435] m` and receives seeded XY noise of at most `25 mm` per axis.
- The left arm begins at `[0, 0, 0, pi/2, 0, 0, 0] rad`; the Wuji Hand begins open.
- `task_front` is a fixed scene camera. `camera_wrist_left` remains the wrist camera.
- `wuji_grasp_center` is a site rigidly attached to the palm and marks the point the
  task controller moves, rather than treating an arbitrary body origin as the grasp point.

`ReachGraspLiftTask.reset(seed)` resets all MuJoCo state, writes both arm home targets,
opens the hand, samples the cube position, clears velocities, settles the physics, and
resynchronizes the backend frame clock and controller history. Reusing a seed reproduces
the cube pose.

## Reach expert

The target is `120 mm` above the settled cube center. Position-only damped least-squares
IK uses the translational site Jacobian:

```text
delta_q = J^T (J J^T + lambda^2 I)^-1 delta_x
```

IK is solved on a separate kinematic `MjData`, so planning does not mutate the live task
state. The resulting seven-joint target is approached at no more than `0.06 rad` per
control frame; the three hand synergies remain zero. A documented `34 mm` upward
Cartesian calibration offsets gravity sag measured with the current position actuators
and attached hand. This number is simulation-specific and must be recalibrated or
removed for another controller or real hardware.

Reach succeeds only after the grasp center remains within `12 mm` of the un-biased
pre-grasp target for five consecutive frames. A failed kinematic solve reports
`ik_unreachable`; a control-loop timeout reports `reach_timeout`.
The seed-7 reference result is 24 frames, 8.9 mm final error, and 3.3 mm minimum error.

## Why this is not ACT data yet

The current GIF and post-action backend records are diagnostic evidence. They must not
be used to train ACT. The training recorder will preserve the causal transition:

```text
observation_t -> expert action_t -> observation_t+1
```

LeRobot will generate episode/frame/timestamp metadata. Backend wall-clock timestamp,
simulation time, velocity diagnostics, cube pose, seed, and failure labels remain task
telemetry rather than ad-hoc policy observation keys.

## Next acceptance test

Implement one grasp style first (power grasp is the current default candidate): descend
from pre-grasp, close the synergy gradually, and require contact between the cube and
at least two distinct Wuji finger groups. The result must distinguish `grasped`,
`grasp_empty`, collision, and timeout before any Lift or ACT training is added.
