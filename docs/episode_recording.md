# Causal episode recording

## Purpose

`CausalEpisodeRecorder` records the complete scripted Reach–Grasp–Lift state machine
without changing its control behavior. Every row has one unambiguous meaning:

```text
observation_t -> bounded action_t actually returned by send_action -> observation_t+1
```

The state machine sends every phase through the same recording hook: `reach`,
`approach`, `grasp_close`, `preload`, `preload_settle`, and `lift_s_curve`.
Older episodes can contain `grasp_settle` and `lift`; phases are metadata, not policy
features. Reset creates frame 0;
the first action creates frame 1. Adjacent rows must share the same boundary frame and
27-D measured state.

Run the reference recording and deterministic replay from PowerShell:

```powershell
./scripts/run_episode_recording_smoke.ps1
```

The reference seed produces `outputs/reach_grasp_lift/reach_grasp_lift_seed0007.npz`
locally and a tracked summary in `outputs/reach_grasp_lift/episode_report.json`. The NPZ
is ignored by Git because future multi-episode image data will be large.

## Schema version 3

Policy-aligned arrays have `T` rows:

| Field | Shape | Meaning |
| --- | --- | --- |
| `observation_state` | `[T, 27]` | 7 measured arm positions followed by 20 measured hand positions |
| `observation_front_rgb` | `[T, H, W, 3]` | Front RGB at time `t`, `uint8`, HWC |
| `observation_wrist_rgb` | `[T, H, W, 3]` | Wrist RGB at time `t`, `uint8`, HWC |
| `action` | `[T, 10]` | 7 bounded arm targets plus 3 bounded hand synergies actually sent |
| `next_observation_state` | `[T, 27]` | Measured state resulting from the same row's action |
| `phase` | `[T]` | State-machine phase that emitted the action |

The final front/wrist images close the shifted image chain without duplicating every
`observation_t+1` image. `frame_index`, `sim_time`, and their `next_` counterparts make
the causal relationship checkable.

Task metadata includes schema version, task name and text, episode index, seed, control
frequency, serialized outcome, `task_success`, `grasp_stable`, `post_settle_stable`, and
the exclusive outcome label.

The serialized result additionally includes frozen/preload synergy, preload and settle
durations, wrench slopes, and `trajectory_diagnostics`. Reference Cartesian limits and
actual sampled derivative peaks are reported separately. These result fields are task
telemetry and do not change the 27-D state or 10-D action contracts.

Schema v3 also records, at every state boundary:

- cube world position and scalar-first `(w, x, y, z)` quaternion;
- grasp-center world position and quaternion;
- desired and compensated-IK palm quaternions, actual orientation error, and object/palm
  linear and angular velocities;
- the complete cube pose expressed in the grasp-center frame;
- every hand–cube contact point, its finger, world normal, and the world force acting on
  the cube;
- the same contact position, normal, and force in the cube frame, cube face, distance to
  edge/corner, and edge/corner flags;
- the contact torque and moment about the cube center, plus their resultant wrench.

Contacts are variable-length and are stored without pickle using
`contact_sample_offsets` plus flat contact arrays. Cube/grasp poses, relative SE(3),
contacts, simulation time, host time, phase, seed, and outcomes are task telemetry. They
are deliberately not extra keys in the ACT policy observation.

## Validation and replay

Saving rejects non-finite or incorrectly shaped state/action data, non-consecutive
frames, non-advancing simulation time, mismatched image shape/type, and discontinuous
state or image chains. Validation recomputes relative SE(3) from the two world poses and
recomputes resultants from individual contacts. Replay resets a new MuJoCo instance with
the stored seed, sends the stored bounded actions, and compares every policy observation,
world/relative pose, contact point/wrench, simulation time, and camera frame.

For seed 7, the current schema-v3 episode has 140 transitions over 4.666 simulated
seconds: 12 Reach, 41 Approach, 24 Grasp-close, four Preload, eight Preload-settle, and
51 Lift, with 1,343 individual contact records. Action, state, pose, contacts, and
simulation time replay exactly. GPU rasterization differs by at most one intensity level
out of 255 in the current run and is checked against the explicit 2/255 tolerance.

## Outcome taxonomy

Labels are evaluated in the following order, which makes them mutually exclusive:

1. `approach_push`: open-hand approach displacement exceeded the existing project safety
   limit `grasp.max_approach_cube_displacement_m` (currently 25 mm).
2. `never_lift`: no approach push occurred and peak object lift was below the external
   baseline's 25 mm minimum.
3. `drop`: the object reached at least 25 mm but finished below 25 mm.
4. `rigid_success`: the object stayed lifted and the complete Lift stability window met
   both external relative-pose limits.
5. `settled_after_slip`: the complete window violated a limit, but the final fixed
   0.5-second hold passes `post_settle_stable` pose, RMS relative-speed, and trend gates.
6. `persistent_slip`: the object stayed lifted but the post-settle window did not
   converge. The result also exposes `continuous_slip=true` for reporting.

`task_success` is independently true after object height stays at or above the project
goal of 80 mm for 15 frames. `grasp_stable` is independently true only when both
relative-pose drift tests pass. Contacts never make either boolean true; they are retained
as `contact_diagnostics` to explain the result.

## CD-WM external baseline

The public [CD-WM Grasp Dataset card](https://huggingface.co/datasets/BWangCN/cdwm-grasp-dataset)
defines its parallel-jaw rigid set using a 50 mm gripper lift over 0.5 s, at least 25 mm
actual object lift, less than 8 mm relative translation drift, and less than 6 degrees
relative rotation drift after a two-frame anchor margin. It permits one establishment
reorientation while the jaws seat the object.

This project records establishment translation and rotation from the start of
`grasp_close` to the start of `lift`, then anchors Lift stability two logged frames after
Lift starts. It outputs continuous maximum drift against that anchor, not only an endpoint
delta. The 8 mm and 6 degree limits are explicitly marked
`external_baseline_not_wuji_calibrated`; they are comparison values, not claimed final
thresholds for the Wuji Hand. Multi-seed simulation and real-hardware measurements are
required before selecting project-specific limits.

Seed 7 reaches the project height goal but moves relative to the palm by 52.54 mm and
17.71 degrees at maximum. Its final 0.5-second window settles to 1.39 mm and 1.60 degrees,
so it is `task_success=true`, `grasp_stable=false`, `post_settle_stable=true`, and
`outcome=settled_after_slip`, never a complete stable-grasp success.

## Relationship to LeRobot and ACT

This NPZ is an auditable intermediate format, not yet a `LeRobotDataset`. Its policy
fields already match the plugin contract, but the next milestone must write successful
episodes using LeRobot 0.6.2's dataset schema, encode videos/features, and pass a
DataLoader smoke test. ACT training should begin only after that conversion and after
collecting enough varied successful demonstrations; one scripted seed is only a pipeline
test, not useful training coverage.
