# Seed 7 finger2 path-aware contact planning

This experiment is intentionally limited to finger2 path feasibility. It does
not modify or execute the preload gate, squeeze, Lift, RL, or ACT paths.

## Reproduce

```powershell
$env:PYTHONPATH = "src;scripts"
$env:OPENBLAS_NUM_THREADS = "1"
$env:OMP_NUM_THREADS = "1"
.\.venvs\openarm-sim\Scripts\python.exe scripts\finger2_path_planning_experiment.py `
  --model outputs\reach_grasp_lift\reach_grasp_lift.mjb `
  --task-config configs\reach_grasp_lift.json `
  --optimization-config configs\contact_grasp_optimization.json `
  --synergies configs\wuji_hand_left_synergies.json `
  --seed 7 `
  --output outputs\reach_grasp_lift\finger2_path_planning_seed7.json
```

The planner holds the palm pose, cube pose, and all non-tested digits fixed.
The other four digits use their open configuration. For each semantic region,
it solves a fingertip endpoint, a Cartesian-normal pregrasp, and one
non-contact waypoint. Both path segments use 21 samples (41 unique samples in
total).

## Contact-region search

Distances are MuJoCo signed geom distances. Positive values are separated;
the configured 1 mm clearance is a soft objective rather than a claimed
hardware-safe threshold.

| region seed (cube Y,Z) | endpoint error | endpoint | q_mid | geom53 min | geom55 min | global non-tip min | minimum location | predicted first geom | path |
| --- | ---: | --- | --- | ---: | ---: | ---: | --- | ---: | --- |
| center (0, 0) | 0.094 mm | pass | found | 0.964 mm | 0.987 mm | 0.964 mm | pre_to_mid, alpha=0 | 57 | pass |
| upper (0, +10 mm) | 0.041 mm | pass | found | 0.980 mm | 0.994 mm | 0.980 mm | pre_to_mid, alpha=0 | 57 | pass |
| lower (0, -10 mm) | 0.159 mm | fail | not run | — | — | — | — | — | fail |
| +Y-side (+10 mm, 0) | 0.225 mm | fail | not run | — | — | — | — | — | fail |
| -Y-side (-10 mm, 0) | 0.102 mm | pass | found | 0.962 mm | 0.985 mm | 0.962 mm | pre_to_mid, alpha=0 | 57 | pass |

The kinematic best is `upper`. Its optimized contact point in the cube frame is
`[0.035000, -0.004398, 0.020861] m`. The full decision-vector order is seven
arm joints followed by four joints each for finger1 through finger5:

```text
q_pre  = [-0.679939, -0.166056, 0.041506, 0.824200, -0.027765, -0.050299, -0.322114,
           0.047500, 0, 0, 0, 0.204303, 0.370000, 0.514028, 0.528769,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
q_mid  = [-0.679939, -0.166056, 0.041506, 0.824200, -0.027765, -0.050299, -0.322114,
           0.047500, 0, 0, 0, 0.212426, 0.355000, 0.460683, 0.760235,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
q_star = [-0.679939, -0.166056, 0.041506, 0.824200, -0.027765, -0.050299, -0.322114,
           0.047500, 0, 0, 0, 0.220548, 0.357389, 0.459837, 0.991701,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
```

The JSON report contains the complete 41-sample clearance curve. For the
kinematic best, per-geom minima are geom51 = 15.020 mm, geom53 = 0.980 mm, and
geom55 = 0.994 mm. The endpoint prediction is the designated tip geom57.

## Single-finger MuJoCo dynamics

Every kinematically feasible candidate was restored to the same initial
physics state and executed over `q_pre -> q_mid -> q_star`, with three seconds
per moving segment. Palm/wrist and all other digits were numerically fixed;
the cube remained free on the table. Contacts were checked at the 2 ms physics
step.

| candidate | first valid contact | time | face / location | cube translation | cube rotation | prediction match | pass |
| --- | --- | ---: | --- | ---: | ---: | --- | --- |
| upper | geom55, distal non-tip | 38 ms | +X edge | 4.605 mm | 5.738 deg | no | no |
| center | geom55, distal non-tip | 38 ms | +X edge | 4.605 mm | 5.739 deg | no | no |
| -Y-side | geom55, distal non-tip | 36 ms | +X edge | 4.373 mm | 5.458 deg | no | no |

For `upper`, geom55 was already at -0.331 mm signed distance when its contact
force crossed 0.1 N. The actual finger2 configuration was
`[0.198457, 0.369467, 0.507789, 0.528646]`, while the commanded configuration
was `[0.204484, 0.369667, 0.512843, 0.533912]`. This shows why a collision-free
synchronous joint interpolation is insufficient: the four position-controlled
joints do not trace that interpolation synchronously in dynamics. Geom57 later
contacts, but only after geom55 and at an edge, so it cannot count as tip-first.

## Decision

The finite semantic search found kinematically collision-free waypoint paths,
but none is executable as a valid tip-first single-finger path in the current
MuJoCo controller and fixed palm pose. The result is therefore
`finger2_required_contact_rejected`, not `single_finger_path_pass`.

The next grasp topology should use thumb (`-X`), middle (`+X`), and ring (`+X`)
as required contacts. Finger2 and the little finger should be optional/support
contacts. This report stops before implementing that topology.
