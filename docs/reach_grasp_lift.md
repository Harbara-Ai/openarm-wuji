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
Six-dimensional damped least-squares IK stacks position with a quaternion
rotation-vector error and uses both translational and rotational site Jacobians:

```text
delta_q = J_6D^T (J_6D J_6D^T + lambda^2 I)^-1 [delta_x, w delta_theta]
```

IK is solved on a separate kinematic `MjData`, so planning does not mutate the live task
state. The resulting seven-joint target is approached at no more than `0.06 rad` per
control frame; the three hand synergies remain zero. A documented `34 mm` upward
Cartesian calibration offsets gravity sag measured with the current position actuators
and attached hand. This number is simulation-specific and must be recalibrated or
removed for another controller or real hardware.

The desired palm orientation is held across Reach, Approach/Preload, and Lift. A
simulation-only rotation-vector compensation separates the desired quaternion from the
internal IK command so actuator load bias is visible in telemetry rather than hidden by
a relaxed threshold. Reach succeeds only after the grasp center remains within `12 mm`
and `2°` of the un-biased target for five consecutive frames. A failed kinematic solve reports
`ik_unreachable`; a control-loop timeout reports `reach_timeout`.
The current seed-7 Reach result is 12 frames, 9.6 mm final position error, and about
0.44° final orientation error.

## Grasp expert and contact labels

MuJoCo has no high-level `grasp()` API. The expert moves the grasp center to `35 mm`
above the cube center, keeps the 7-D arm command at that target, and increases the
`open_close` synergy by `0.03` per 30 Hz frame. MuJoCo itself computes collision,
friction, contact forces, and resulting cube motion.

`FingerContactMonitor` inspects `data.contact`, maps each unnamed hand collision geom
through its owning body to `finger1`–`finger5`, and uses `mj_contactForce` to reject
normal forces below `0.1 N`. Closing freezes at the first frame where:

- the close synergy is at least `0.7`;
- at least two distinct finger groups contact the cube.

Seed 7 freezes at `0.72`. Preload holds arm targets and increases synergy by at most
`0.01/frame` to the provisional target `0.76`. Preload-settle then holds both commands
for a fresh eight-frame window: contact must persist, relative SE(3) must pass the
external 8 mm / 6° gate, and resultant force/moment norm slopes must both be nonpositive.
This gate measures non-growth; it cannot rule out excessive constant loads. The current
seed-7 settle window is 4.53 mm / 3.43°. Closing has a 55-frame budget, while preload
and preload-settle have separate 30-frame budgets; timeout rejects Lift and requests a
regrasp. The arm is held by its normal actuators, not by a rigid constraint. Synergy
commands hand joint positions, so this is not direct force regulation.

The approach still moves the cube only `0.18 mm`. The historical 5/5 Grasp-only result
predates preload and must not be used as its success rate. Failure labels now also
include `preload_timeout` and `preload_unstable`. See `grasp_settle_experiment.md` for
the full parameter sweep and negative results.

## Lift expert and separated outcomes

Lift keeps the successful hand command active and moves the grasp center upward by
`120 mm`. The first 0.5 seconds follow quintic minimum-jerk waypoints ending at the
external protocol's `50 mm` commanded gripper displacement; later waypoints continue
toward 120 mm. Joint commands remain limited to `0.015 rad` per 30 Hz frame. With the
new 6D constraint, the measured seed-7 palm displacement is only `24.97 mm` in that
window even though the reference endpoint remains 50 mm. This under-tracking is reported
as an external-protocol failure and must be fixed at the controller layer rather than
hidden in evaluation.
Lift keeps the final preload synergy without a closure jump. The reference limits are
0.19 m/s, 1.16 m/s², and 24.1 m/s³, validated before motion. The two segments join with
continuous position, velocity, and acceleration. Actual sampled palm motion exceeds
these reference bounds (0.2425 m/s, 2.7656 m/s², 66.06 m/s³); tracking diagnostics are
stored separately. These are not guaranteed physical-motion bounds.
No weld, equality, adhesion, or other fake attachment is enabled: the cube rises only
through MuJoCo contact and friction.

The evaluator no longer treats height plus multi-finger contact as complete success:

- `task_success` means the cube stayed at least `80 mm` above reset height for 15 frames.
- `grasp_stable` measures full object-in-palm relative SE(3) drift.
- `post_settle_stable` independently checks whether the final fixed 0.5-second hold has
  converged after any initial slip; it never changes `grasp_stable` to true.
- `contact_diagnostics` stores fingers, contact directions, individual world forces,
  resultant force, and resultant moment, but cannot make a grasp successful.

The relative-pose comparison uses the public CD-WM parallel-jaw criterion as an external
baseline: 50 mm gripper motion within 0.5 s, at least 25 mm object lift, translation
drift below 8 mm, rotation drift below 6 degrees, and a two-frame Lift anchor margin.
Establishment from the start of closing through preload-settle is recorded separately. These
thresholds are not claimed to be calibrated for the Wuji Hand.

For seed 7, the preloaded Lift peaks at `109.92 mm` and finishes at `92.40 mm`, so
`task_success=true`. However, maximum object-in-palm drift is `52.54 mm` and `17.71°`.
The final 0.5-second hold settles to `1.39 mm` and `1.60°`, producing
`grasp_stable=false` and `outcome=settled_after_slip`. It is no longer reported as a
complete stable-grasp success. The previous five-seed 3/5 count used the old height and
contact criterion and must be replaced by a new benchmark under the SE(3) taxonomy.

The new cube-frame contact diagnostics show that fixed palm orientation removes the old
systematic thumb-corner contact in the first five diagnostic seeds, but other fingers
are still dominated by a non-opposing `+X` face in three of four completed lifts. Strict
stability remains 0/5. See `grasp_geometry_diagnostics.md` for the full controlled run
and the decision to defer pose scan until physical 6D tracking improves.

## Episode recorder and why this is not ACT data yet

The complete state machine now records the causal transition:

```text
observation_t -> expert action_t -> observation_t+1
```

All state-machine actions pass through one hook, so the stored action is the bounded
10-D vector actually returned by `send_action`, not an intended pre-limit command. The
schema-v3 seed-7 smoke contains 140 transitions: 12 Reach, 41 Approach, 24 Grasp-close,
four Preload, eight Preload-settle, and 51 Lift (including 15 frames at the final
reference). A fresh seeded replay reproduces actions, measured state, both world poses,
relative SE(3), 1,343 individual contacts, resultants, and simulation time exactly;
camera rasterization stays within 1/255 intensity level in the current run.

Backend wall-clock timestamp, simulation time, velocity diagnostics, cube pose, seed,
contact force, and failure labels remain task telemetry rather than ad-hoc policy
observation keys. The saved NPZ is an intermediate validation format. LeRobot must still
generate its own dataset episode/frame/timestamp metadata during conversion, so this is
not yet an ACT training dataset. See `episode_recording.md` for the schema.

## Next acceptance test

First restore reliable measured 6D palm tracking during Lift and the physical 50 mm /
0.5 s motion while keeping the current reference and preload fixed. Repeat the same
small fixed-seed diagnostic. Only then run a limited yaw/XY grasp-pose scan using
opposition, edge avoidance, wrench, and SE(3) quality metrics. Do not record ACT
demonstrations until strict stable successes repeat on unseen seeds.
