# Frozen-synergy grasp-settle experiment

## Checkpoint and branch

The pre-experiment SE(3) evaluator and causal-recorder baseline is preserved by local
commit `36cbb5b` on `main`. This experiment lives on
`experiment/grasp-settle-gate` and has not been pushed.

Run the complete seed-7 experiment with:

```powershell
./scripts/run_grasp_settle_experiment.ps1
```

The command rebuilds the model, regenerates the annotated GIF and reports, records the
schema-v2 episode, and requires deterministic replay. `drop` is the expected seed-7
outcome for this branch, so an honest task failure does not masquerade as an
infrastructure-test failure.

## State machine

```text
grasp_close
  -> first frame with synergy >= 0.7 and enough finger contacts
freeze_synergy
  -> hold the exact first qualifying synergy
grasp_settle
  -> evaluate only new settle frames
lift or regrasp_required
```

The settle window reuses the existing eight-frame contact-hold length. Every frame in
that window must retain the configured multi-finger contact condition, while object pose
relative to the grasp center must remain below the explicitly external CD-WM 8 mm and
6 degree gates. Closing frames never enter this window. Closing plus settling may not
exceed the existing `grasp.max_close_steps=55` budget. Exhausting the budget returns
`grasp_unstable`, sets `regrasp_required=true`, and rejects Lift; a reposition/regrasp
motion is deliberately not invented until its strategy is specified.

## Seed-7 result

The experiment freezes at synergy `0.72` on closing frame 24. Eight subsequent settle
frames pass at `1.20 mm` translation drift and `0.58°` rotation drift. Compared with the
checkpoint, pre-lift cube displacement falls from about `17.5 mm` to `5.9 mm`.

The lower frozen grasp does not survive the unchanged aggressive Lift. The cube peaks at
`73.7 mm`, then finishes at `15.0 mm`; Lift drift reaches `82.2 mm` and `73.3°`.
The exclusive result is therefore:

```text
task_success = false
grasp_stable = false
outcome = drop
```

This is a useful negative result: freeze-and-settle fixes the pre-lift establishment
condition but is not sufficient under the current joint-space Lift dynamics. The next
controlled experiment should keep this gate and replace only Lift with orientation-held
Cartesian waypoints at the external 50 mm / 0.5 s reference speed.
