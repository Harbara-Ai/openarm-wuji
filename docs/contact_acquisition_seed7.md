# Seed 7 contact-acquisition experiment

This experiment executes only the Stage-1 `thumb -X / index-middle-ring +X`
candidate.  The table remains enabled; no lift, table removal, gravity-only
test, preload-gate tuning, QP, RL, or ACT is involved.

Run it with:

```powershell
$env:PYTHONPATH = "$PWD/src;$PWD/scripts"
.\.venvs\openarm-sim\Scripts\python.exe scripts/contact_acquisition_experiment.py `
  --model (Resolve-Path outputs/reach_grasp_lift/reach_grasp_lift.mjb) `
  --task-config (Resolve-Path configs/reach_grasp_lift.json) `
  --optimization-config (Resolve-Path configs/contact_grasp_optimization.json) `
  --synergies (Resolve-Path configs/wuji_hand_left_synergies.json) `
  --seed 7
```

The executor maintains independent per-finger states
`PREGRASP -> APPROACH -> CONTACT_CANDIDATE -> CONTACT_ACQUIRED -> CONTACT_HOLD -> SQUEEZE`.
Stage-1 now constructs pre-contact in Cartesian space: each required designated
tip is targeted along the assigned face normal with a configured 5 mm clearance,
then solved with a point-Jacobian IK while the palm is fixed.  Every candidate is
checked for non-tip cube penetration, self-collision, and swept-path clearance
before it is executed.  Optional finger5 is left open.  The controller then
approaches each finger in one of three
sequencing modes: `synchronized`, `thumb_first`, and `fingers_first`.
Acquisition requires a designated fingertip geom (not merely a distal pad), the
configured face, edge margin, normal alignment, force, and five consecutive
confirmation frames.  Non-tip contacts are retained as telemetry and classified
as `non_tip_contact`/`non_tip_unilateral_push`; they cannot acquire topology.
Cube pose/velocity and contact wrenches are recorded at every control step as
diagnostic telemetry only.

## Current seed-7 result

| strategy | outcome | required topology | max cube translation | max cube rotation |
|---|---|---:|---:|---:|
| synchronized | `path_infeasible` | 0/4 | 0 mm | 0 deg |
| thumb_first | `path_infeasible` | 0/4 | 0 mm | 0 deg |
| fingers_first | `path_infeasible` | 0/4 | 0 mm | 0 deg |

The first observed contact is `finger2` on `+X` at about 34 ms, with a normal
force of about 0.42 N.  Raw MuJoCo telemetry identifies it as geom `55` on
`wuji_left_finger2_link4`; the optimizer-designated fingertip geom is `57`, so
this first contact is the distal-link/pad geometry, not the fingertip geom.
Before the required four-finger topology is confirmed,
the cube reaches 2.88 mm translation and 3.62 degrees rotation, so all three
strategies are stopped and labelled `unilateral_push`.

The isolated finger2 run still fails the pre-contact feasibility gate: its
Cartesian-normal target is reachable for the designated tip, but geom `53`
(finger2 link3) remains penetrating before the target can be approached.  This
is now reported as `non_tip_contact`/`path_infeasible`, rather than being counted
as a valid fingertip acquisition.  The swept-path diagnostic retains geom
metadata, nearest face/point, normal velocity, TTC, and prediction-vs-actual
fields for the next geometry iteration.

The Stage-1 endpoint candidate remains kinematically valid (4/4 required
fingers, no tip penetration), but the whole closure path is not yet feasible:
the distal non-tip geometry and neighboring-finger clearance fail before
contact.  This is the intended stricter gate; it prevents reporting a static
endpoint as an executable grasp.

The JSON report contains the full per-finger and cube time-series.  Its
`time_series` fields are task telemetry and are deliberately not appended to
the LeRobot policy observation.
