# Seed 7 thumb-middle-ring topology

This experiment follows the finger2 rejection by introducing a separate
three-contact topology:

- required: finger1/thumb on `-X`, finger3/middle on `+X`, finger4/ring on `+X`
- optional and held open during this acquisition test: finger2 and finger5

The experiment does not run squeeze, preload, Lift, RL, or ACT. The table stays
enabled. Optional and strategy-inactive digits are numerically fixed, while
active and already acquired required fingers retain MuJoCo actuator dynamics.

## Clean initialization

Contact planning must start after `run_reach(seed=7)`, not after the legacy
power-grasp state. `run_reach` puts the palm in place without first disturbing
the cube. Reusing `run_grasp` produced an apparently valid 3/3 endpoint around
a cube/palm state that the old grasp had already displaced; releasing that
state caused immediate table settling and was not a valid acquisition trial.

Each 30 Hz target can also be interpolated over the 2 ms MuJoCo substeps. This
option changes only the diagnostic executor trajectory, not any task gate.

## Finite semantic initialization comparison

All errors below come from clean `run_reach(seed=7)` state. Endpoint tolerance
remains 6 mm and is not relaxed.

| initialization | endpoint success | satisfied required fingers | thumb error | middle error | ring error | non-tip penetration | self penetration |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| current_grasp | no | finger1 | 5.31 mm | 20.97 mm | 20.06 mm | 0 | 0.015 mm |
| three_finger_opposition_same_palm | no | finger1 | 4.14 mm | 6.05 mm | 8.45 mm | 0 | 0.047 mm |
| three_finger_power_small_yaw | no | finger1 | 4.16 mm | 7.49 mm | 7.72 mm | 0 | 0.017 mm |

Because none of the three finite semantic starts yields a valid endpoint,
single-finger or synchronized acquisition is not executed for the clean state.
This is `kinematic_candidate_invalid`, not an acquisition-gate failure.

## Pregrasp fixes verified during diagnosis

Two executor errors were corrected before drawing the clean-state conclusion:

1. isolated finger tests now build a pregrasp only for the required finger in
   that test, rather than allowing the other topology fingers to contaminate it;
2. adaptive Cartesian retreat now really executes all five configured attempts
   (0, 3, 6, 9, and 12 mm) instead of reporting an unexecuted final 12 mm step.

The executor also clears all generalized velocities at the pregrasp teleport,
so residual cube velocity from an earlier phase cannot be labeled a new push.

## Decision

The thumb-middle-ring topology is now represented correctly in configuration,
but the present three palm/hand initializations do not provide a clean seed-7
endpoint for middle and ring. We should not tune the contact gate or proceed to
squeeze/Lift.

The next bounded problem is grasp geometry initialization: derive one or more
palm poses from the clean table state that place the middle and ring workspaces
over the `+X` interior while keeping the thumb over `-X`. That should be done
with visual/kinematic workspace diagnostics before adding any new acquisition
controller logic.

## Reproduction outputs

- `outputs/reach_grasp_lift/three_finger_acquisition_seed7_reach_current.json`
- `outputs/reach_grasp_lift/three_finger_acquisition_seed7_reach_opposition.json`
- `outputs/reach_grasp_lift/three_finger_acquisition_seed7_reach_power_yaw.json`

