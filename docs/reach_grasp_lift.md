# Reach–Grasp–Lift task

## Current scope

The task implements the scene, deterministic reset, collision-free `Reach`, a
force-filtered `Grasp`, height/contact-verified `Lift`, and causally aligned recording of
the complete state machine. LeRobotDataset conversion and broad randomization robustness
are not claimed yet.

Run the complete build and smoke check from PowerShell:

```powershell
./scripts/run_reach_only_smoke.ps1
./scripts/run_grasp_smoke.ps1
./scripts/run_lift_smoke.ps1
./scripts/run_episode_recording_smoke.ps1
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
- The task-only `[pi, 0, pi]` mount rotation keeps the fingers pointing toward the
  workspace while the palm faces down. This remains a simulation assumption until the
  real tool transform is measured.
- `task_front` is a fixed scene camera. `camera_wrist_left` remains the wrist camera.
- `wuji_grasp_center` is a site rigidly attached to the palm and marks the point the
  task controller moves, rather than treating an arbitrary body origin as the grasp point.
  It is placed in hidden render group 4, so it cannot leak into policy camera images.

`ReachGraspLiftTask.reset(seed)` resets all MuJoCo state, writes both arm home targets,
opens the hand, samples the cube position, clears velocities, settles the physics, and
resynchronizes the backend frame clock and controller history. Reusing a seed reproduces
the cube pose.

## Reach expert

The safe pre-grasp target is `100 mm` behind and `50 mm` above the settled cube center.
This avoids the earlier joint-space trajectory that swept through and pushed the cube.
Position-only damped least-squares IK uses the translational site Jacobian:

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
The seed-7 reference result is 12 frames, 9.2 mm final error, and 5.7 mm minimum error.

## Grasp expert and contact labels

MuJoCo has no high-level `grasp()` API. The expert moves the grasp center to `35 mm`
above the cube center, keeps the 7-D arm command at that target, and increases the
`open_close` synergy by `0.03` per 30 Hz frame. MuJoCo itself computes collision,
friction, contact forces, and resulting cube motion.

`FingerContactMonitor` inspects `data.contact`, maps each unnamed hand collision geom
through its owning body to `finger1`–`finger5`, and uses `mj_contactForce` to reject
normal forces below `0.1 N`. Grasp succeeds when:

- the close synergy is at least `0.7`;
- at least two distinct finger groups contact the cube; and
- that condition remains true for eight consecutive frames.

The seed-7 reference reaches synergy `0.93`, ends with all five finger groups in contact,
and moves the cube only `0.18 mm` during the open-hand approach. A five-seed diagnostic
Grasp-only run passed 5/5, but this is not yet the final benchmark. Failure labels include
`grasp_ik_unreachable`, `approach_collision`, `approach_timeout`, and `grasp_empty`.

## Lift expert and separated outcomes

Lift keeps the successful hand command active and moves the grasp center upward by
`120 mm`, limiting arm commands to `0.015 rad` per 30 Hz frame. No weld, equality,
adhesion, or other fake attachment is enabled: the cube rises only through MuJoCo
contact and friction.

The evaluator no longer treats height plus multi-finger contact as complete success:

- `task_success` means the cube stayed at least `80 mm` above reset height for 15 frames.
- `grasp_stable` measures full object-in-palm relative SE(3) drift.
- `contact_diagnostics` stores fingers, contact directions, individual world forces,
  resultant force, and resultant moment, but cannot make a grasp successful.

The relative-pose comparison uses the public CD-WM parallel-jaw criterion as an external
baseline: 50 mm gripper motion within 0.5 s, at least 25 mm object lift, translation
drift below 8 mm, rotation drift below 6 degrees, and a two-frame Lift anchor margin.
Closing-phase establishment translation and rotation are recorded separately. These
thresholds are not claimed to be calibrated for the Wuji Hand.

For seed 7, Lift reaches `98.3 mm` and finishes at `92.5 mm`, so
`task_success=true`. However, maximum object-in-palm drift is `53.4 mm` and `106.0°`.
The final 15 frames settle to `1.55 mm` and `1.09°`, producing
`grasp_stable=false` and `outcome=settled_after_slip`. It is no longer reported as a
complete stable-grasp success. The previous five-seed 3/5 count used the old height and
contact criterion and must be replaced by a new benchmark under the SE(3) taxonomy.

## Episode recorder and why this is not ACT data yet

The complete state machine now records the causal transition:

```text
observation_t -> expert action_t -> observation_t+1
```

All state-machine actions pass through one hook, so the stored action is the bounded
10-D vector actually returned by `send_action`, not an intended pre-limit command. The
seed-7 smoke contains 102 transitions: 12 Reach, 31 Approach, 31 Grasp-close, and 28
Lift. A fresh seeded replay reproduces actions, measured state, both world poses,
relative SE(3), 838 individual contacts, resultants, and simulation time exactly; camera
rasterization stays within 2/255 intensity levels.

Backend wall-clock timestamp, simulation time, velocity diagnostics, cube pose, seed,
contact force, and failure labels remain task telemetry rather than ad-hoc policy
observation keys. The saved NPZ is an intermediate validation format. LeRobot must still
generate its own dataset episode/frame/timestamp metadata during conversion, so this is
not yet an ACT training dataset. See `episode_recording.md` for the schema.

## Next acceptance test

Run at least 30 fixed seeds with this recorder, preserve successful and failed outcomes,
and report failure categories. Then convert the successful demonstrations to
LeRobotDataset and pass a 100-step DataLoader smoke test.
