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
schema-v2 episode, and requires deterministic replay. `settled_after_slip` is the
expected seed-7 outcome for this branch, so height success does not masquerade as a
stable grasp.

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

The first version retained the old aggressive joint-space Lift: the gripper moved about
`87 mm` in its first 0.5 seconds, the cube peaked at `73.7 mm`, and then dropped. The
controlled follow-up replaces that jump with cubic Cartesian waypoints whose published
endpoint is the CD-WM external protocol's `50 mm / 0.5 s`. A measured Lift-only gravity
compensation calibration of `29.77 mm` gives `50.12 mm` actual gripper travel for seed 7.
It is controller-specific calibration, not a new success threshold.

With the paced Lift, the cube peaks at `96.3 mm` and finishes at `93.5 mm`, so it no
longer drops. Relative object-in-palm motion still reaches `50.8 mm` and `36.6°` before
the final window settles to `0.60 mm` and `0.82°`. The exclusive result is therefore:

```text
task_success = true
grasp_stable = false
outcome = settled_after_slip
```

Wuji's mapping identifies `finger1` as the thumb. At the last pre-Lift settle frame,
its mean cube-local contact lies near the `x=-35 mm, y=-35 mm` edge rather than centered
on a face opposing the other four fingers. The other fingers span the `y=-35 mm`,
`x=+35 mm`, and `y=+35 mm` faces. This non-opposed, edge-heavy contact layout is a
plausible source of the measured lateral force and torque when table support disappears.
It is evidence of a grasp-geometry problem, not proof that the thumb alone causes all
slip. The next controlled experiment should keep the paced Lift and change only the
pregrasp pose/hand synergy to place the thumb and finger contacts on opposing faces.
