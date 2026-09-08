# Contact-target grasp optimization: minimum-loop result

Date: 2026-09-05  
Seed: 7  
Status: **Stage-1 kinematic candidate accepted; static grasp rejected**

Reproduce from the repository root (with the simulation environment active):

```powershell
$env:PYTHONPATH='src'
.\.venvs\openarm-sim\Scripts\python.exe scripts/contact_grasp_experiment.py `
  --model outputs/reach_grasp_lift/reach_grasp_lift.mjb `
  --task-config configs/reach_grasp_lift.json `
  --optimization-config configs/contact_grasp_optimization.json `
  --synergies configs/wuji_hand_left_synergies.json `
  --seed 7
```

This experiment keeps the existing Reach–Grasp–Lift implementation as the baseline and
adds a separate whole-hand kinematic optimizer. The decision variables are seven arm
joints, which determine the reachable palm SE(3), and all twenty Wuji finger joints.
The policy-facing three-synergy action contract is unchanged.

## Automatic model discovery

The compiled MuJoCo model has no explicit fingertip sites and its fingertip geoms have
no names. The implementation therefore walks each finger body chain, selects its leaf
body, and then selects the outermost collision-enabled geom on that body.

| Finger | Distal body | Evaluation geom | Independent joints |
|---|---|---:|---:|
| Thumb (`finger1`) | `wuji_left_finger1_link4` | 47 | 4 |
| Index (`finger2`) | `wuji_left_finger2_link4` | 57 | 4 |
| Middle (`finger3`) | `wuji_left_finger3_link4` | 67 | 4 |
| Ring (`finger4`) | `wuji_left_finger4_link4` | 77 | 4 |
| Little (`finger5`) | `wuji_left_finger5_link4` | 87 | 4 |

The position objective uses a smooth representative point at each distal collision
geom, with a configurable standoff derived from its MuJoCo bounding radius. Surface
distance is retained for penetration. The direction from the actual fingertip surface
toward the cube surface is used for normal alignment; this avoids both an undocumented
mesh-frame normal and discontinuous nearest-face position gradients.

## Stage-1 kinematic solve

The target region is the central `36 x 36 mm` square on the requested cube face. This
leaves 17 mm between the region boundary and the 70 mm cube boundary. Position,
direction, non-tip/cube clearance, fingertip penetration, self-collision, joint limits,
and displacement from the nominal grasp all contribute to the numerical least-squares
objective. Optimization runs in contact-acquisition, alignment/clearance, and
collision-polish stages. Thumb, index, middle, and ring are required; little is optional.
Three named, geometry-motivated initializations replace an undirected pose grid.
Thresholds and weights are in `configs/contact_grasp_optimization.json`.

### Requested `thumb -X / fingers +X`

The current-grasp initialization reduced objective cost from `17.1490` to `0.2309` and
passed the kinematic gate. The other two explicit initializations were retained in the
report but did not pass, so the solver did not select them merely for having a different
palm seed.

| Finger | Position error | Direction error | Tip penetration |
|---|---:|---:|---:|
| Thumb | 2.08 mm | 0 deg | 0 mm |
| Index | 3.30 mm | 0 deg | 0 mm |
| Middle | 0.88 mm | 0 deg | 0 mm |
| Ring | 0.80 mm | 0 deg | 0 mm |

`wuji_left_finger1_joint2` reached the configured joint-limit margin. The nearest
non-tip geom retained 0.986 mm rather than the soft 1.000 mm clearance target, a 14 μm
shortfall, but there was no actual non-tip/cube penetration. Summed inter-finger
penetration was 1.89 μm, below the configured 50 μm numerical hard tolerance.

### Fallback `thumb -Y / fingers +Y`

The fallback was not run after the requested `-X/+X` topology passed Stage 1. It remains
configured and is only attempted when every explicit `-X/+X` initialization fails.

## Stage-2 squeeze and wrist-lock test

The accepted kinematic `-X/+X` candidate was executed once. The twenty finger actuator
targets were independent.
The squeeze increments used separate gains for thumb, index/middle, and ring/little.
After settling, the seven arm joints were numerically locked every 2 ms and table
collision was disabled for a one-second gravity hold.

| Metric | Existing baseline | Contact-target diagnostic candidate |
|---|---:|---:|
| Complete contact loss | 0.164 s | 0.066 s |
| Static success | false | false |
| Maximum relative translation drift | 413.70 mm | 405.13 mm |
| Maximum relative rotation drift | 102.19 deg | 110.39 deg |
| Palm translation drift | 0 | 0 |
| Horizontal residual contact force before removal | 7.329 N | 0.662 N |
| Net contact moment before removal | 0.0310 Nm | 0.1342 Nm |

The horizontal residual force improved, but torque became more than four times worse.
The physical execution did not preserve the intended topology: the thumb had no contact,
index contacted `-Y`, middle/ring contacted `+Z`, and little contacted `+Y`. All
index/middle/ring contacts were classified as edge contacts. The candidate therefore
does not satisfy the one-second static acceptance gate and must not be used as an expert
demonstration.

## Conclusion and next decision

The upgraded Stage-1 solver demonstrates that the Wuji kinematics can represent the
requested opposition topology. The remaining failure is the transition from a kinematic
candidate to dynamic contact: moving and squeezing the free cube destroys the planned
topology before table removal. The next useful change is therefore a light-pregrasp,
contact-event-driven execution that stops each independent finger at its assigned face,
followed by force-balanced squeeze. A force-closure QP becomes useful after that executor
can reproduce the Stage-1 contact set. All future candidates must pass the unchanged
penetration, edge, topology, and one-second static gravity gates.
