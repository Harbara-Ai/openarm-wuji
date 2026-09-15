# Frozen staged ACT with scripted Lift

## Decision

**Case B: GraspSecure passes, but many grasps fail under Lift. The current
static GraspSecure/preload gate is not sufficient to identify a load-bearing
grasp.** Keep Lift scripted; improve and validate a load-bearing GraspSecure
criterion before considering Lift learning.

This is not an interface failure. Across all 64 actual Lift attempts the exact
20-D hand controller target present at the GraspSecure terminal state was held
unchanged (maximum change `0.0`
rad). No reset, snapshot restore, expert state, or expert action occurred at the
GraspSecure-to-Lift handoff.

## Frozen protocol

- Reach, Approach, Recovery, the `selective_hysteresis` router, and GraspSecure
  step 1500 were loaded frozen. No policy was trained or updated.
- Lift reused the configured two-segment quintic minimum-jerk Cartesian
  trajectory, IK tolerances, `0.015 rad` arm target rate limit, controller
  gains, and grasp parameters. The reference lasts 36 control frames at 30 Hz.
- After the reference, the final scripted arm target and the unchanged 20-D
  GraspSecure hand target were held for 30 frames (1.0 s).
- The pre-Lift 25 mm cube-motion safety reference was retained. A rejected
  GraspSecure pass did not enter Lift.
- Lift success requires a completed trajectory, cube height continuously at or
  above 80 mm for the full 1.0 s terminal hold, no environment support during
  that hold, and no drop. Finger count, topology, force, antipodality, slip,
  and SE(3) drift are diagnostics only.

## Conditional GraspSecure -> Lift

The runner used seeds 4000--4076 and needed 77 frozen full-pipeline rollouts to
obtain 30 safe GraspSecure terminal states. There were 32 formal GraspSecure
passes; 2 exceeded the 25 mm safety reference and correctly did not enter
Lift.

| Metric | Result |
| --- | ---: |
| Formal GraspSecure passes | 32/50 Approach-stage successes |
| Safety rejects after GraspSecure | 2 |
| Actual scripted Lift attempts | 30 |
| Lift successes / safe attempts | 14/30 (46.7%) |
| Lift successes / all GraspSecure passes | 14/32 (43.8%) |
| Drops | 12 |
| Hold/height failures | 2 |
| Environment-support failures | 2 |

The 16 actual Lift failures comprise 12 drops, 2 terminal height-not-held
cases, and 2 cases with environment support during the required hold.

## Fresh 100-run end-to-end result

Seeds 5000--5099 were run independently after the conditional collection.

| Stage probability | Count | Rate |
| --- | ---: | ---: |
| P(Reach) | 74/100 | 74.0% |
| P(Approach-stage success \| Reach) | 61/74 | 82.4% |
| P(GraspSecure \| Approach) | 39/61 | 63.9% |
| P(Lift \| GraspSecure), including safety rejects | 17/39 | 43.6% |
| P(Lift \| safe GraspSecure actually lifted) | 17/34 | 50.0% |
| **P(full success)** | **17/100** | **17.0%** |

The product of the four formal stage probabilities is
`0.170`, exactly matching the observed `17/100`. The
smallest full-run conditional probability is Lift after GraspSecure (43.6%).

### Mutually exclusive first failure stage

| Stage | Count |
| --- | ---: |
| Reach | 26 |
| Approach | 0 |
| Recovery | 13 |
| Grasp formation | 3 |
| Preload formation | 15 |
| Terminal hold / pre-Lift safety | 9 |
| Lift | 8 |
| Lift hold | 9 |

The 17 actual full-run Lift failures comprise 8 drops, 7 environment-support
failures, and 2 height-not-held failures. Five additional formal GraspSecure
passes were stopped by the pre-Lift 25 mm safety guard.

## Does preload predict load-bearing success?

| Population | Lift success preload L2 | Lift failure preload L2 | Median difference | Cliff's delta |
| --- | ---: | ---: | ---: | ---: |
| Conditional | 1.365 rad | 1.284 rad | +0.081 rad | +0.473 |
| Fresh full | 1.360 rad | 1.294 rad | +0.065 rad | +0.260 |
| Pooled descriptive | 1.360 rad | 1.291 rad | +0.069 rad | +0.357 |

Larger preload is associated with better Lift outcome, especially in the
conditional set, but it is not sufficient. The pooled ranges overlap:
successful grasps include `1.217` rad,
while failures extend to `1.453` rad.
The full-run failure mean (`1.303`
rad) is also far above the historical GraspSecure-failure mean of 0.726 rad.
The existing scalar preload gate filters very weak closures but cannot by
itself identify load-bearing geometry.

## Load and contact diagnostics

| Diagnostic (median) | Conditional success | Conditional failure | Full success | Full failure |
| --- | ---: | ---: | ---: | ---: |
| Max cube/palm translation drift | 39.2 mm | 57.1 mm | 42.8 mm | 54.2 mm |
| Max cube/palm rotation drift | 13.6 deg | 91.9 deg | 15.8 deg | 79.6 deg |
| Longest zero-hand-contact run | 0 frames | 1 frames | 0 frames | 2 frames |

All 14 conditional Lift successes ended with all five fingers above the 0.1 N
diagnostic threshold. In the fresh full set, 15/17 successes ended with all
five and 2/17 with thumb/index/middle/ring. By contrast, 10/17 fresh-full Lift
failures ended with no finger above threshold. This topology separation is
explanatory evidence, not a newly imposed success gate.

Peak-force distributions are heavy-tailed, so medians are more representative
than means. Across the 34 fresh full Lift attempts the per-finger peak normal
force medians were: thumb `1.23` N, index
`1.40` N, middle
`1.46` N, ring
`2.13` N, and little
`1.39` N. Per-frame contact points, world-frame
force vectors, resultants, moments, topology, and preload are retained in each
Lift NPZ.

The external 8 mm / 6 deg values were not reintroduced as hard gates. Even
physical Lift successes have substantial establishment/settling motion (fresh
full success medians 42.8 mm and 15.8 deg), so those CD-WM values remain
diagnostic baselines rather than calibrated Wuji acceptance thresholds.

## Interpretation and next step

The existing scripted Lift can carry real GraspSecure ACT outputs, so the
pipeline and handoff are functional. It does so for only 46.7% of safe
conditional handoffs and 50.0% of safe handoffs in the independent full run.
Together with the moderate-but-overlapping preload effect and frequent contact
collapse in failures, this supports **Case B**, not reliable full-pipeline
freeze and not immediate Lift learning.

The next experiment should keep Lift scripted and use these labeled terminal
states to improve a load-bearing GraspSecure criterion. It should test
geometry/contact persistence and a small physical load probe as predictors,
with preload retained as a continuous feature. This report does not change a
gate, train a policy, or start that experiment.

## Artifacts

- Machine summary: `D:/yl/embodied ai/openarm-wuji-learning/outputs/staged_act_with_scripted_lift/summary.json`
- Conditional raw summary: `D:/yl/embodied ai/openarm-wuji-learning/outputs/staged_act_with_scripted_lift/conditional/summary.json`
- Full raw summary: `D:/yl/embodied ai/openarm-wuji-learning/outputs/staged_act_with_scripted_lift/full/summary.json`
- Exact GraspSecure terminal snapshots: `D:/yl/embodied ai/openarm-wuji-learning/outputs/staged_act_with_scripted_lift/conditional/graspsecure_terminal_states` and `D:/yl/embodied ai/openarm-wuji-learning/outputs/staged_act_with_scripted_lift/full/graspsecure_terminal_states`
- Per-frame Lift trajectories: `D:/yl/embodied ai/openarm-wuji-learning/outputs/staged_act_with_scripted_lift/conditional/lift` and `D:/yl/embodied ai/openarm-wuji-learning/outputs/staged_act_with_scripted_lift/full/lift`
