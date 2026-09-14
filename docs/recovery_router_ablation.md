# Recovery router ablation

## Outcome

**Case A.** A simple persistence/hysteresis router makes Recovery net-positive, raises matched Approach-stage success, reduces cube failures, and does not increase timeout. Freeze this router and proceed to Grasp/Preload as the next separate stage.

Best router: `selective_hysteresis`.
Frozen deployable config: `D:/yl/embodied ai/openarm-wuji-learning/configs/recovery_router.json`.

No Reach, Approach, or Recovery ACT was retrained. Grasp/Preload/Lift was not
executed. The only experimental variable is when the frozen step-1500
Recovery policy is allowed to take over.

## Why the old router fails

Historical matched seeds 1400-1599 reproduced the original problem:

- Reach: `154/200`.
- Baseline Approach: `119/154`
  (`77.3%`).
- Old first-trigger Recovery: `96/154`
  (`62.3%`).
- Rescued `20`, regressed
  `43`; net
  `-23`.

The old artifact did not store the requested 3/5/10-frame causal histories.
An instrumented matched replicate was therefore run with the same frozen
policies and seeds. Feature and outcome labels in the diagnosis below always
come from the same exact MuJoCo snapshot fork. The replicate again produced a
net `-23`, although individual seeds are not bitwise repeatable across fresh
processes.

## Matched outcome taxonomy at first trigger

| type | programmatic definition | count |
|---|---|---:|
| A | Approach fails; Recovery succeeds (useful switch) | 22 |
| B | Approach succeeds; Recovery fails (harmful switch) | 45 |
| C | both succeed (neutral) | 58 |
| D | both fail (neutral) | 18 |

## Legacy trigger precision

Precision is `Approach FAIL + Recovery SUCCESS` divided by all states where
that trigger is primary. A high Recovery success rate alone is insufficient
because it includes states Approach would also solve.

| primary trigger | states | Approach succeeds | Recovery succeeds | useful precision | net |
|---|---:|---:|---:|---:|---:|
| moving_away | 71 | 57 | 41 | 12.7% | -16 |
| terminal_plateau | 0 | 0 | 0 | 0 | 0 |
| gate_stall | 39 | 28 | 22 | 12.8% | -6 |
| near_timeout_terminal | 0 | 0 | 0 | 0 | 0 |
| cube_displacement | 33 | 18 | 17 | 24.2% | -1 |

`moving_away` and `gate_stall` are over-sensitive at first detection. Their
Approach self-recovery rates are high, so a single legacy hit is not evidence
that takeover is beneficial.

## Type A versus Type B feature distributions

Values are median [p25, p75] at the first legacy trigger. All features use
only current/past frames; none uses future outcome information.

| feature | Type A: useful | Type B: harmful |
|---|---:|---:|
| approach_elapsed_frames | 20.00 [20.00, 21.75] frames | 22.00 [21.00, 23.00] frames |
| terminal_error_m | 9.02 [7.46, 16.24] mm | 14.00 [7.38, 21.58] mm |
| error_slope_5_m_s | -63.11 [-130.65, -2.45] mm/s | -17.50 [-111.68, 50.31] mm/s |
| moving_away_consecutive_frames | 1.00 [0.00, 2.00] frames | 2.00 [0.00, 2.00] frames |
| non_improving_consecutive_frames | 1.00 [0.00, 2.00] frames | 2.00 [0.00, 2.00] frames |
| cube_displacement_m | 1.89 [0.50, 3.60] mm | 0.24 [0.08, 1.48] mm |
| cube_radial_velocity_m_s | 21.86 [-0.19, 80.72] mm/s | -0.15 [-4.16, 8.15] mm/s |
| arm_joint_velocity_l2_rad_s | 0.45 [0.32, 0.69] rad/s | 0.48 [0.28, 0.64] rad/s |
| grasp_center_velocity_m_s | 182.46 [137.72, 222.44] mm/s | 224.59 [120.09, 312.67] mm/s |
| remaining_timeout_frames | 70.00 [68.25, 70.00] frames | 68.00 [67.00, 69.00] frames |
| basin_exit_distance_m | 1.85 [0.94, 6.21] mm | 8.93 [0.40, 16.85] mm |
| entered_12mm_basin | 90.9% | 100.0% |

The complete JSON also contains 3/5/10-frame slopes and recent minima, cube
motion change, joint/EEF velocity, gate durations, elapsed/remaining time,
and basin-exit motion.

## Offline candidate evaluation on first-trigger futures

This table is a screening test only. Persistence rules that intentionally
wait cannot be faithfully scored from one first-trigger future, so they were
kept only when a dynamic closed-loop test was scientifically necessary.

| candidate | switches | rescued | regressed | net | predicted success | timeout | cube failure |
|---|---:|---:|---:|---:|---:|---:|---:|
| patience_25 | 6 | 2 | 1 | +1 | 116 | 21 | 18 |
| gate_stall_persistent_10 | 0 | 0 | 0 | +0 | 115 | 21 | 19 |
| gate_stall_persistent_5 | 0 | 0 | 0 | +0 | 115 | 21 | 19 |
| gate_stall_persistent_8 | 0 | 0 | 0 | +0 | 115 | 21 | 19 |
| late_terminal_only | 0 | 0 | 0 | +0 | 115 | 21 | 19 |
| moving_away_persistent_3 | 0 | 0 | 0 | +0 | 115 | 21 | 19 |
| moving_away_persistent_3_error15 | 0 | 0 | 0 | +0 | 115 | 21 | 19 |
| moving_away_persistent_4 | 0 | 0 | 0 | +0 | 115 | 21 | 19 |
| moving_away_persistent_4_error15 | 0 | 0 | 0 | +0 | 115 | 21 | 19 |
| moving_away_persistent_5 | 0 | 0 | 0 | +0 | 115 | 21 | 19 |
| moving_away_persistent_5_error15 | 0 | 0 | 0 | +0 | 115 | 21 | 19 |
| patience_30 | 0 | 0 | 0 | +0 | 115 | 21 | 19 |
| patience_40 | 0 | 0 | 0 | +0 | 115 | 21 | 19 |
| plateau_persistent_10 | 0 | 0 | 0 | +0 | 115 | 21 | 19 |
| plateau_persistent_5 | 0 | 0 | 0 | +0 | 115 | 21 | 19 |
| plateau_persistent_8 | 0 | 0 | 0 | +0 | 115 | 21 | 19 |
| cube_outward_5mm_s | 31 | 8 | 8 | +0 | 115 | 26 | 14 |
| cube_outward | 32 | 8 | 8 | +0 | 115 | 27 | 13 |
| selective_hysteresis | 32 | 8 | 8 | +0 | 115 | 27 | 13 |
| cube_displacement_only | 33 | 8 | 9 | -1 | 114 | 28 | 13 |
| patience_20 | 117 | 18 | 38 | -20 | 95 | 52 | 8 |
| old_first_trigger | 143 | 22 | 45 | -23 | 92 | 59 | 4 |
| patience_10 | 143 | 22 | 45 | -23 | 92 | 59 | 4 |
| patience_15 | 143 | 22 | 45 | -23 | 92 | 59 | 4 |

## Closed-loop matched router ablation

Each row uses seeds 1400-1599 and contains its own matched baseline branch
from the same prefix. `H_exec=1`; success/safety gates and all policies are
unchanged; Recovery is locked after at most one switch.

| router | switches | Recovery success | baseline -> routed success | rescued | regressed | net | timeout | cube failure |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| selective_hysteresis | 74 (47.7%) | 48 (64.9%) | 112 -> 129 (83.2%) | 22 | 5 | +17 | 20 -> 20 | 23 -> 6 |
| cube_outward_5mm_s | 51 (33.1%) | 33 (64.7%) | 111 -> 120 (77.9%) | 13 | 4 | +9 | 21 -> 25 | 22 -> 9 |
| patience_25 | 130 (84.4%) | 77 (59.2%) | 114 -> 93 (60.4%) | 11 | 32 | -21 | 20 -> 43 | 20 -> 18 |

## Best router details

`selective_hysteresis` uses a minimum 12-frame patience window and switches only on:

- cube displacement over the legacy 3 mm diagnostic threshold while radial
  cube velocity remains outward;
- moving-away for at least five consecutive frames with error above 15 mm;
- gate-stall/plateau with at least eight non-improving frames; or
- the existing near-timeout terminal signal.

Matched result:

- Approach-stage success: `112` ->
  `129` out of `155` Reach
  successes (`83.2%`).
- Recovery attempts/successes: `74` /
  `48`
  (`64.9%`).
- Rescued `22`, regressed `5`, net
  `+17`; exact paired p=`0.001514`.
- Timeout: `20` -> `20`.
- Cube safety failure: `23` ->
  `6`.
- Cube displacement median/p90:
  `1.87` /
  `30.25` mm ->
  `2.09` /
  `11.98` mm.

Contribution by primary trigger at the actual delayed switch:

| primary trigger | attempts | baseline success | Recovery success | net |
|---|---:|---:|---:|---:|
| cube_displacement | 56 | 29 | 40 | +11 |
| moving_away | 16 | 2 | 8 | +6 |
| terminal_plateau | 2 | 0 | 0 | +0 |

## Decision

A simple persistence/hysteresis router makes Recovery net-positive, raises matched Approach-stage success, reduces cube failures, and does not increase timeout. Freeze this router and proceed to Grasp/Preload as the next separate stage.

A learned router is not justified by this experiment: the simple
persistence/hysteresis rule already produces positive matched net benefit,
83%+ Approach-stage success, lower cube failure, and no timeout increase.
