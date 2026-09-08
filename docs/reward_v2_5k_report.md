# Reward V2 controlled 5K report

Date: 2026-09-07. SAC, MLP, 228D observation, 20D independent delta-position
action, fixed palm, reset, success gate, penetration termination, Lift, telemetry,
and ACT integration were unchanged. V2 was trained from a fresh SAC initialization
with seed 7 for exactly 5,000 environment steps.

## Reward change

Reward V1 used persistent absolute proximity and per-step contact reward:

```text
0.5 absolute_mean_proximity
+ 1.0 [min(N/3, 1) + 0.2 thumb_contact]
+ stability/hold - slip/penetration/cube_motion/action costs + success
```

Reward V2 is:

```text
2.0 approach_progress
+ 1.0 contact_transition
+ 0.1 contact_maintain
+ 0.5 beta stability
+ 0.5 beta translation_rotation_hold
- 0.2 beta bounded_slip
- 2.0 penetration
- 1.0 cube_motion
- 0.01 action - 0.02 delta_action
+ 10.0 success
```

`approach_progress` is the step difference of
`0.3 mean(phi_i) + 0.7 phi_third`, with `phi_i=exp(-max(d_i,0)/0.025)`.
Contact transitions give 0.25 per newly acquired finger plus one-step 0.5 bonuses
when crossing two and three contacts. The thumb bonus was removed. Stability, SE(3)
hold, and bounded slip begin after a 0.10 s grace and a 0.20 s ramp. All values are in
`configs/rl/grasp_stage1.json`.

## Learning behavior

| Metric | Reward V1 5K | Reward V2 5K |
|---|---:|---:|
| Mean episode return | 15.813 | -1.859 |
| Episodes with any contact | 6/41 (14.6%) | 10/41 (24.4%) |
| Episodes with at least two contacts | 0/41 | 0/41 |
| Episodes with at least three contacts | 0/41 | 0/41 |
| Maximum simultaneous contacts | 1 | 1 |
| Held two contacts for 0.1 s / 0.3 s | not recorded | 0 / 0 |
| Stable success | 0 | 0 |

Raw return values are not directly comparable because V2 intentionally removed a
positive reward paid every timestep. Within V2, first/last ten-episode return changed
from -1.786 to -1.874, so there is no upward learning trend.

The V2 third-closest distance averaged 35.112 mm over the first 250 steps and
35.437 mm over the last 250 steps; it became 0.325 mm worse. Its best 50-step-bin
average was about 34.01 mm, not a sustained coverage improvement.

V2 first/last 250-step mean finger distances were:

| Finger | First 250 | Last 250 |
|---|---:|---:|
| thumb / finger1 | 22.34 mm | 22.94 mm |
| finger2 | 34.67 mm | 35.03 mm |
| finger3 | 39.64 mm | 39.85 mm |
| finger4 | 35.96 mm | 36.52 mm |
| finger5 | 30.84 mm | 32.17 mm |

## Policy and exploration diagnostics

The stochastic training actions remained active across all fingers: mean absolute
actions were 0.514, 0.517, 0.514, 0.515, and 0.517 for fingers 1–5. Final latent
Gaussian policy standard deviation averaged 0.871 (range 0.851–0.895). Entropy
coefficient alpha fell from 1.0 to 0.230, almost identical to V1's final 0.230 and
latent standard deviation 0.873. Exploration therefore did not collapse to the thumb.

The deterministic policy nevertheless reproduced essentially the same geometry as V1.
Across evaluation seeds 7–11, final mean distances were 6.56, 35.40, 40.05, 36.75,
and 32.73 mm for fingers 1–5. V1 values were 5.75, 35.41, 40.05, 36.75, and 32.73 mm.
The third-closest distance was 35.401 mm for V2 versus 35.415 mm for V1, a negligible
0.014 mm difference. All five deterministic finger action groups were non-zero, but
their learned joint directions did not move fingers 2–5 toward the cube.

## Deterministic evaluation, seeds 7–11

Every seed ran 120 steps with zero contact, zero multi-contact milestone, zero hold
milestone, zero stable success, and no failure termination. Final third-closest distance
was approximately 35.4 mm for every seed. V2 did not break the single-thumb proximity
behavior in the actual task geometry.

## Decision

1. Reward V2 did not break the single-thumb local optimum under the required 5K test.
2. SAC did not begin a sustained multi-finger approach; contact frequency improved, but
   the acceptance metric stayed at one simultaneous contact.
3. The gate for a 10K extension was not met, so no 10K run is performed.
4. Because V1 and V2 have nearly identical exploration statistics and deterministic
   geometry, the next experiment should stop reward tuning and test an early
   action-space curriculum or demonstration/replay seeding, as scoped in the request.

Evidence: `outputs/rl_grasp_stage1/sac_v2_5000_steps.json`, V2 checkpoint
`outputs/rl_grasp_stage1/sac_grasp_stage1_v2_5000_steps.zip`, and the preserved V1
reports/checkpoint in the same directory.
