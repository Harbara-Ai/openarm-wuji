# ACT best configuration D: step 500 → 1000 → 2000

## Conclusion

**Forced decision: Case D. Both binary and continuous Reach metrics have failed to improve by step 2000. Stop adding training steps and increase/modify demonstrations.**

The exact-resume run is valid, but additional optimization on the same 20 demonstrations does not make closed-loop Reach converge. Step 1000 gives a small continuous improvement over step 500, while step 2000 clearly regresses:

- Valid Reach remains `0/20` at all three checkpoints.
- Seeds entering the position-only 12 mm basin change `6 → 7 → 0`.
- Mean minimum pregrasp error changes `15.45 → 13.06 → 19.64 mm`.
- Reach arm H=1 MAE changes `0.02369 → 0.02434 → 0.02914 rad`.
- Training loss nevertheless falls `2.215 → 1.728 → 1.057`.

This is not evidence for continuing the same run. The policy at step 2000 is smoother but biased away from the pregrasp target, rather than reaching the basin and merely oscillating out of it. Among these checkpoints, step 1000 is the best closed-loop checkpoint, but it is not a valid Reach policy.

## Fixed experimental contract

No requested experimental variable changed:

```text
20 successful demonstrations / 2804 frames
observation.state: 27D actual qpos
action: 27D absolute controller target
front + wrist RGB: 240 x 320
ResNet18 ImageNet initialization
phase-balanced sampling: 25/25/20/10/10/10
chunk_size = 20
n_action_steps = 20
batch size = 8
AdamW, learning rate = 1e-5
training seed = 1000
H_exec = 1
evaluation seeds = 0..19
same MuJoCo model, reset, controller, grasp and task gates
```

The added fields are rollout diagnostics only. They are not passed to ACT and do not change the 27D policy observation or the controller command.

## Resume integrity

The run restored complete policy, optimizer, RNG and data-order state:

| Segment | Restored step | Target step | Stream start sample | New samples |
|---|---:|---:|---:|---:|
| first resume | 500 | 1000 | 4000 | 4000 |
| second resume | 1000 | 2000 | 8000 | 8000 |

The actual phase fractions remained matched to the requested sampler:

| Segment | Reach | Approach | Grasp | Preload | Lift | Hold |
|---|---:|---:|---:|---:|---:|---:|
| 500→1000 | 24.50% | 25.05% | 20.28% | 10.20% | 9.85% | 10.12% |
| 1000→2000 | 24.98% | 24.94% | 19.91% | 10.09% | 10.08% | 10.01% |
| cumulative through 2000 | 25.04% | 24.93% | 19.96% | 10.06% | 9.99% | 10.02% |

Checkpoint `002000` contains the policy, pre/postprocessors, optimizer state, RNG state and `training_step.json` with `step: 2000`.

## Reach definitions

These metrics must not be conflated:

- **Valid Reach:** position error ≤ 12 mm **and** palm orientation error ≤ 2° for 5 consecutive frames.
- **12 mm dwell diagnostic:** position error ≤ 12 mm only. It can diagnose basin entry but is not a successful Reach.
- `first_frame_enter_12mm` and `last_frame_inside_12mm` do not imply continuous dwell; the continuous quantity is `longest_consecutive_frames_inside_12mm`.

All per-seed closest-approach windows, first/last frames and the error trajectory around closest approach are stored in the machine-readable summary.

## Closed-loop comparison

| Metric | step 500 | step 1000 | step 2000 |
|---|---:|---:|---:|
| endpoint training loss | 2.215 | 1.728 | 1.057 |
| Valid Reach | 0/20 | 0/20 | 0/20 |
| Valid Approach | 0/20 | 0/20 | 0/20 |
| Task success | 0/20 | 0/20 | 0/20 |
| minimum pregrasp error, mean | 15.45 mm | **13.06 mm** | 19.64 mm |
| minimum pregrasp error, median | 16.67 mm | **13.33 mm** | 19.77 mm |
| minimum pregrasp error, best | 2.49 mm | **1.82 mm** | 12.26 mm |
| seeds entering position-only 12 mm | 6/20 | **7/20** | 0/20 |
| mean frames inside 12 mm | 0.75 | **0.90** | 0.00 |
| max consecutive frames inside 12 mm | 3 | 3 | 0 |
| seeds with ≥5 consecutive position-only frames | 0/20 | 0/20 | 0/20 |
| seeds entering position + orientation gate | 5/20 | **6/20** | 0/20 |
| max consecutive position + orientation frames | 3 | 3 | 0 |

Step 1000 therefore makes a modest improvement in basin entry, but not in the required dwell length. The improvement is not sustained: step 2000 does not enter the position basin for any seed, and even its best seed remains 0.259 mm outside the 12 mm threshold.

## Per-seed position-basin diagnosis

Each cell is `minimum pregrasp error / total frames inside 12 mm, longest consecutive run`.

| seed | step 500 | step 1000 | step 2000 |
|---:|---:|---:|---:|
| 0 | 13.00 mm / 0, 0 | 14.00 mm / 0, 0 | 16.56 mm / 0, 0 |
| 1 | 23.19 mm / 0, 0 | 13.19 mm / 0, 0 | 17.68 mm / 0, 0 |
| 2 | 7.88 mm / 2, 2 | 11.04 mm / 1, 1 | 23.51 mm / 0, 0 |
| 3 | 18.81 mm / 0, 0 | 12.73 mm / 0, 0 | 24.92 mm / 0, 0 |
| 4 | 20.57 mm / 0, 0 | 4.06 mm / 5, 3 | 21.87 mm / 0, 0 |
| 5 | 12.39 mm / 0, 0 | 25.98 mm / 0, 0 | 22.86 mm / 0, 0 |
| 6 | 11.90 mm / 1, 1 | 10.15 mm / 2, 2 | 21.12 mm / 0, 0 |
| 7 | 18.37 mm / 0, 0 | 18.23 mm / 0, 0 | 19.80 mm / 0, 0 |
| 8 | 20.52 mm / 0, 0 | 16.72 mm / 0, 0 | 16.70 mm / 0, 0 |
| 9 | 10.76 mm / 1, 1 | 9.07 mm / 1, 1 | 24.01 mm / 0, 0 |
| 10 | 24.46 mm / 0, 0 | 6.64 mm / 4, 1 | 14.93 mm / 0, 0 |
| 11 | 6.80 mm / 4, 3 | 1.82 mm / 4, 2 | 27.21 mm / 0, 0 |
| 12 | 21.57 mm / 0, 0 | 14.15 mm / 0, 0 | 15.21 mm / 0, 0 |
| 13 | 15.05 mm / 0, 0 | 25.63 mm / 0, 0 | 24.43 mm / 0, 0 |
| 14 | 3.86 mm / 5, 3 | 13.74 mm / 0, 0 | 21.96 mm / 0, 0 |
| 15 | 21.04 mm / 0, 0 | 6.20 mm / 1, 1 | 19.74 mm / 0, 0 |
| 16 | 2.49 mm / 2, 2 | 13.48 mm / 0, 0 | 13.97 mm / 0, 0 |
| 17 | 18.28 mm / 0, 0 | 12.97 mm / 0, 0 | 17.68 mm / 0, 0 |
| 18 | 14.73 mm / 0, 0 | 17.37 mm / 0, 0 | 12.26 mm / 0, 0 |
| 19 | 23.34 mm / 0, 0 | 14.01 mm / 0, 0 | 16.44 mm / 0, 0 |

The non-contiguous total dwell is important. At step 1000, for example, seed 4 has five total frames inside 12 mm but a longest run of only three; seed 10 has four total frames but each is isolated. Neither is a valid Reach.

## Target stability and oscillation

Target motion is computed from the ACT-predicted absolute OpenArm target, dimensions `0:7`, before controller clipping. Episode boundaries are excluded from delta and reversal calculations.

| Metric | step 500 | step 1000 | step 2000 |
|---|---:|---:|---:|
| overall mean frame-to-frame target Δ L2 | 0.04579 rad | 0.04471 rad | **0.03106 rad** |
| closest ±10 frames mean target Δ L2 | 0.06373 rad | 0.06333 rad | **0.03450 rad** |
| closest ±10 frames mean second-difference L2 | 0.09075 rad | 0.08181 rad | **0.04511 rad** |
| closest-window direction reversal fraction | 45.62% | **43.13%** | 45.66% |

The magnitude of target jitter decreases substantially at step 2000, while direction reversals remain around 46%. Because step 2000 never enters the basin and has worse pregrasp error, the primary failure is not “the policy reaches the correct pose but target jitter prevents five-frame dwell.” It produces a smoother but biased trajectory.

The within-basin sample is too small for a strong oscillation estimate: there are only six qualifying transitions at step 500, five at step 1000, and none at step 2000. The robust closest-window statistics above are therefore the main comparison.

## Offline reconstruction and frame-0 conditioning

| Metric | step 500 | step 1000 | step 2000 |
|---|---:|---:|---:|
| Reach arm H=1 MAE | **0.02369** | 0.02434 | 0.02914 rad |
| Reach hand H=1 MAE | 0.01404 | **0.00575** | 0.02160 rad |
| Approach arm H=1 MAE | 0.02232 | 0.01531 | **0.01124 rad** |
| frame-0 paired arm MAE | **0.02965** | 0.03272 | 0.03954 rad |
| frame-0 image-conditioned arm spread / expert spread | 1.244 | 1.150 | 1.298 |
| front-only arm spread / expert spread | 1.071 | 0.964 | 1.166 |
| wrist-only arm spread / expert spread | 0.474 | 0.409 | 0.257 |

The image-conditioned spread remains non-zero and comparable to or larger than expert spread, so there is no new visual-conditioning collapse or observation/action shape failure. Front-camera conditioning dominates the reset action variation, while wrist-only variation weakens with training. Reach arm and frame-0 errors worsen despite lower total training loss; this is consistent with fitting the small phase-balanced dataset without improving the specific reset-to-pregrasp behavior.

Padding, episode boundaries and state/action round-trip checks pass at every checkpoint.

## Cube motion, contact and clipping

| Metric | step 500 | step 1000 | step 2000 |
|---|---:|---:|---:|
| mean maximum cube displacement | 3.81 mm | 29.82 mm | 3.97 mm |
| maximum cube displacement | 74.19 mm | 360.56 mm | 74.36 mm |
| episodes over 25 mm displacement | 1/20 | 4/20 | 1/20 |
| early contact before valid Approach | 1/20 | 5/20 | 1/20 |
| any contact | 1/20 | 5/20 | 1/20 |
| clipped action values | 2920 | 2842 | 3200 |
| sum clipping magnitude | 90.80 rad | 65.53 rad | 83.72 rad |
| max clipping magnitude | 0.0722 rad | 0.0652 rad | 0.0745 rad |

All clipping at all checkpoints is on action index 7, `wuji_left_finger1_joint1`; no OpenArm action dimension clips. At step 2000 this hand joint clips on every one of the `20 × 160 = 3200` rollout frames. The clipped-count trend therefore worsens at step 2000, although the mean magnitude per clipped value remains lower than at step 500. This persistent thumb-output bias is real but does not explain the arm Reach miss directly.

## Answers to the requested questions

1. **Does Reach begin to succeed?** No. It remains `0/20` at steps 500, 1000 and 2000.
2. **Does 12 mm dwell keep increasing despite 0/20?** No. It improves only slightly from step 500 to 1000, then collapses to zero at step 2000.
3. **Does pregrasp error keep decreasing?** No. Mean error improves `15.45 → 13.06 mm`, then regresses to `19.64 mm`.
4. **Does Reach arm MAE keep decreasing?** No. It worsens monotonically `0.02369 → 0.02434 → 0.02914 rad`.
5. **Can the policy enter the Reach basin but not hold it?** Steps 500 and 1000 can enter briefly, never for five frames. Step 2000 no longer enters at all. The final policy is therefore not merely a dwell failure.
6. **Is there target jitter near pregrasp?** Direction reversals remain frequent, but target-delta magnitude is much smaller at step 2000. Since Reach simultaneously regresses, temporal jitter is not the primary final blocker; smooth target bias is.
7. **Does clipping worsen?** By count, yes: step 2000 clips `wuji_left_finger1_joint1` on every frame. It remains isolated to the hand, not the Reach arm joints.
8. **Is there evidence for more training on the same data?** No. Lower loss coexists with worse Reach MAE, worse frame-0 arm MAE, zero basin entry and persistent clipping.

## Decision and next step

```text
Case D:
Both binary and continuous Reach metrics plateau/regress by step 2000.
Stop adding training steps and increase/modify demonstrations.
```

Retain step 1000 as the best closed-loop checkpoint from this curve for comparison, but do not present it as successful. The next matched experiment should first improve demonstration coverage of reset → pregrasp and sustained pregrasp dwell, rather than extending this 20-demo run. Useful additions would be more reset/cube variation, several stable pregrasp hold frames, and recovery trajectories around the 12 mm boundary. No new data or model changes were made in this experiment.

## Artifacts

- `outputs/act_phase_balanced_chunk20/act_train/checkpoints/001000/pretrained_model`
- `outputs/act_phase_balanced_chunk20/act_train/checkpoints/002000/pretrained_model`
- `outputs/act_phase_balanced_chunk20/rollouts/step_001000_seeds_0_19_h001`
- `outputs/act_phase_balanced_chunk20/rollouts/step_002000_seeds_0_19_h001`
- `outputs/act_phase_balanced_chunk20/diagnosis/offline_chunk_step_001000`
- `outputs/act_phase_balanced_chunk20/diagnosis/offline_chunk_step_002000`
- `outputs/act_phase_balanced_chunk20/diagnosis/reset_conditioning_step_001000`
- `outputs/act_phase_balanced_chunk20/diagnosis/reset_conditioning_step_002000`
- `outputs/act_phase_balanced_chunk20/training_curve_step_000500_001000_002000.json`
