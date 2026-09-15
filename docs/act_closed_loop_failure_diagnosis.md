# ACT closed-loop failure diagnosis

## Executive conclusion

**Final selection: Case B — ACT remains substantially underfit at the visually conditioned start of Reach.** Exact continuation from step 500 to the declared hard stop at step 2000 reduced balanced Reach arm first-action MAE from **0.1160 to 0.0414 rad**, but `H_exec=1` still produced **0/3 Reach**, **0/3 Approach**, and zero contacts on training seeds 0, 7, and 11. Mean minimum pregrasp error was **22.68 mm** against a 12 mm gate; the best seed reached 12.90 mm but never entered the tolerance even for one frame.

The decisive extra result is the controlled reset input ablation. At step 500, ACT's across-seed first-arm-action spread was only **2.97%** of the expert spread. At step 2000 it rose to **32.18%**, and swapping images while fixing state reproduced **33.45%**, whereas swapping state while fixing images produced only **2.01%**. Training is therefore learning useful visual conditioning, but it still reproduces only about one third of the expert's seed-dependent Reach variation; frame-0 arm MAE remains **0.0645 rad**. This is an on-demonstration-state failure, so it supports underfitting rather than an interface bug or pure off-manifold failure.

Closed-loop compounding remains a secondary amplifier, but it weakened substantially as the model fit improved. In the approximate periodic expert-pose reset diagnostic, step-2000 action-error growth is **2.02x by age 4** and **3.13x by age 19**, versus **2.75x and 4.99x** at step 500. This reset is not a bit-exact MuJoCo state restoration (velocity is finite-differenced and solver state is not recorded), so it is supporting sensitivity evidence rather than a clean causal measurement of covariate shift.

The original 100-frame open-loop execution is also not dominant. Replanning every frame failed at step 500 and again at step 2000. Input preprocessing, temporal alignment, padding/masking, action indexing, and normalization were re-audited without finding a semantic pipeline bug. No scripted grasp, dataset, observation/action interface, SAC component, or MuJoCo controller was changed in this diagnosis.

## Scope and fixed conditions

- Dataset: 20 successful demonstrations, 2,804 frames, 30 FPS.
- Observation: 27D actual joint position plus 240x320 RGB front and wrist cameras.
- Action: 27D absolute position-controller target, `[7D OpenArm, 20D Wuji]`.
- Policy: ACT, 100-action prediction chunk and `n_action_steps=100`; step-500 baseline continued exactly to the hard stop at step 2000.
- Main closed-loop seeds: 0, 7, 11; all are training seeds.
- Reach gate: 12 mm.

The detailed machine-readable outputs are in:

- `outputs/act_e2e_smoke/diagnosis/offline_chunk/offline_chunk_metrics.json`
- `outputs/act_e2e_smoke/diagnosis/pipeline_audit/pipeline_audit.json`
- `outputs/act_e2e_smoke/diagnosis/teacher_forced/summary.json`
- `outputs/act_e2e_smoke/diagnosis/teacher_forced_audit/teacher_forced_audit.json`
- `outputs/act_e2e_smoke/diagnosis/offline_chunk_step_002000/offline_chunk_metrics.json`
- `outputs/act_e2e_smoke/diagnosis/reset_conditioning_step_002000/reset_conditioning_summary.json`
- `outputs/act_e2e_smoke/diagnosis/teacher_forced_audit_step_002000/teacher_forced_audit.json`
- `outputs/act_e2e_smoke/rollouts/execution_horizon_ablation/h_*/summary.json`
- `outputs/act_e2e_smoke/rollouts/training_progress/step_002000_h001/summary.json`

## 1. Offline full-chunk reconstruction

The step-500 checkpoint was evaluated in inference mode at 288 real expert observations: 48 deterministic, evenly spaced samples from each of Reach, Approach, Grasp, Preload, Lift, and Hold. The expert action was **not** passed to ACT's VAE encoder; inference used ACT's deterministic zero latent. Every target was `[a_t, ..., a_t+99]` from the same episode, and padded target positions were excluded from every MAE.

Boundary checks passed:

- Dataset and manually reconstructed episode masks agree exactly.
- Maximum raw-to-native state and padded-action discrepancy is 0 rad.
- No target from the next episode enters an MAE calculation.

The balanced diagnostic gives a first-action MAE of **0.0402 rad overall**, **0.0481 rad arm**, and **0.0375 rad hand**. The older `0.084 rad` number compared only the three frame-0 predictions from seeds 0, 7, and 11 against their expert actions; it was not a dataset-wide offline average. The closest like-for-like result here is the balanced Reach value, `0.0786 rad` over 48 observations. The new phase-balanced measurement is the appropriate result for locating the failure.

## 2. MAE versus action horizon

| Predicted horizon | Valid observations | Overall MAE (rad) | Arm 7D MAE (rad) | Hand 20D MAE (rad) |
|---:|---:|---:|---:|---:|
| 1 | 288 | 0.040214 | 0.048085 | 0.037459 |
| 5 | 276 | 0.036063 | 0.044359 | 0.033160 |
| 10 | 259 | 0.039105 | 0.041037 | 0.038429 |
| 20 | 235 | 0.040837 | 0.035841 | 0.042586 |
| 50 | 195 | 0.043657 | 0.030331 | 0.048321 |
| 100 | 82 | 0.028132 | 0.052757 | 0.019513 |

The error does **not** grow monotonically with horizon. This is expected because later horizons have fewer valid samples and a different phase mixture. Horizon 100 has only 82 valid observations, biased toward early anchor states and repeated/easy future postures, so its low overall value must not be read as evidence that a 100-step open-loop rollout is safe.

For Reach observations specifically:

| Predicted horizon | Overall MAE (rad) | Arm 7D MAE (rad) | Hand 20D MAE (rad) |
|---:|---:|---:|---:|
| 1 | 0.078563 | 0.115988 | 0.065465 |
| 5 | 0.076608 | 0.108446 | 0.065465 |
| 10 | 0.069927 | 0.086520 | 0.064119 |
| 20 | 0.052066 | 0.041256 | 0.055850 |
| 50 | 0.075824 | 0.034729 | 0.090208 |
| 100 | 0.028036 | 0.049786 | 0.020423 |

Plots:

![Offline chunk MAE versus horizon](../outputs/act_e2e_smoke/diagnosis/offline_chunk/offline_chunk_mae_vs_horizon.png)

![Arm versus hand MAE](../outputs/act_e2e_smoke/diagnosis/offline_chunk/arm_vs_hand_mae.png)

## 3. Reach/Approach/Grasp/Lift phase-wise error

| Observation phase | Samples | First-action overall MAE (rad) | Arm MAE (rad) | Hand MAE (rad) |
|---|---:|---:|---:|---:|
| Reach | 48 | 0.078563 | **0.115988** | 0.065465 |
| Approach | 48 | 0.044980 | 0.041286 | 0.046272 |
| Grasp | 48 | 0.038299 | 0.030742 | 0.040944 |
| Preload | 48 | 0.036227 | 0.027171 | 0.039396 |
| Lift | 48 | 0.024461 | 0.033140 | 0.021423 |
| Hold | 48 | 0.018752 | 0.040182 | 0.011251 |

Reach is the worst phase, and the arm is the dominant Reach error: arm MAE is 77% larger than hand MAE (`0.1160 / 0.0655`). This matches the observed rollout: the policy misses the 12 mm Reach gate before the grasp controller could matter. Later low-error phases therefore cannot rescue the episode.

At the step-2000 hard stop, evaluated on the exact same 288 anchors:

| Observation phase | First-action overall MAE (rad) | Arm MAE (rad) | Hand MAE (rad) |
|---|---:|---:|---:|
| Reach | 0.020093 | **0.041365** | 0.012647 |
| Approach | 0.022319 | 0.032308 | 0.018822 |
| Grasp | 0.065240 | 0.012997 | **0.083526** |
| Preload | 0.036422 | 0.011197 | **0.045250** |
| Lift | 0.012821 | 0.017279 | 0.011261 |
| Hold | 0.011894 | 0.029162 | 0.005850 |

Reach arm improved by 64.3% from step 500, but the optimization did not improve
all phases: Grasp and Preload hand MAE rose to 0.0835 and 0.0453 rad. The
balanced first-action MAE therefore worsened from `0.02656` at step 1500 to
`0.02813` at step 2000 even while the logged train objective continued falling.

![Step-2000 phase-wise first-action MAE](../outputs/act_e2e_smoke/diagnosis/offline_chunk_step_002000/phasewise_action_mae.png)

An additional reset-conditioning check exposes what the aggregate MAE hides.
Across all 20 demonstrations, cube X/Y span `42.32/41.32 mm`, while the largest
frame-0 arm-state standard deviation is only `0.00087 rad`. Expert first arm
actions vary strongly with cube XY. At step 500, however, ACT's across-seed arm
RMS action spread is only **2.97%** of the expert spread, with frame-0 arm MAE
`0.1179 rad`. The checkpoint has largely collapsed to a seed-insensitive mean
Reach command.

| Arm joint | Expert std | Step-500 ACT std | Step-2000 ACT std | Step-2000 / expert |
|---:|---:|---:|---:|---:|
| 1 | 0.014746 | 0.000693 | 0.012144 | 82.35% |
| 2 | 0.034091 | 0.000183 | 0.002108 | 6.18% |
| 3 | 0.033885 | 0.000046 | 0.000579 | 1.71% |
| 4 | 0.030294 | 0.002023 | 0.017912 | 59.13% |
| 5 | 0.034295 | 0.000136 | 0.001552 | 4.52% |
| 6 | 0.005353 | 0.000622 | 0.010519 | 196.52% |
| 7 | 0.032139 | 0.000071 | 0.001596 | 4.97% |

The step-2000 controlled swap makes the attribution clearer:

| Frame-0 input condition | Arm RMS spread / expert | Arm MAE vs expert |
|---|---:|---:|
| Paired state and images | 32.18% | 0.064515 rad |
| Vary both images, fix seed-0 state | 33.45% | 0.064509 rad |
| Vary state, fix seed-0 images | 2.01% | 0.065313 rad |
| Vary front only | 19.46% | 0.065166 rad |
| Vary wrist only | 25.94% | 0.064650 rad |

Thus the learned cross-seed conditioning is genuinely visual, with both cameras
contributing. It improved substantially from step 500, but remains attenuated
to about one third of the expert arm spread. In particular, joints 2/3/5/7,
which encode most of the expert cube-Y response, retain only 1.7-6.2% of expert
variation. This is severe visual/Reach underfitting, not proof of a dead visual
branch.

![Phase-wise first-action MAE](../outputs/act_e2e_smoke/diagnosis/offline_chunk/phasewise_action_mae.png)

## 4. Execution-horizon ablation

Only the number of queued actions executed before re-observation/replanning was changed. Checkpoint, dataset, model weights, MuJoCo, controller, and grasp remained fixed. Each row aggregates 160-frame rollouts on seeds 0, 7, and 11.

| `H_exec` | Replans / episode | Reach | Approach | Grasp | Task | Mean min pregrasp error (mm) | Mean min approach error (mm) | Mean final cube displacement (mm) | Mean peak cube lift (mm) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 160 | 0/3 | 0/3 | 0/3 | 0/3 | 32.05 | 74.36 | 165.44 | 20.63 |
| 5 | 32 | 0/3 | 0/3 | 0/3 | 0/3 | 32.05 | 78.77 | 163.97 | 19.80 |
| 10 | 16 | 0/3 | 0/3 | 0/3 | 0/3 | 32.05 | 78.77 | 163.93 | 19.80 |
| 20 | 8 | 0/3 | 0/3 | 0/3 | 0/3 | 32.05 | 78.77 | 163.97 | 19.80 |
| 50 | 4 | 0/3 | 0/3 | 0/3 | 0/3 | 32.05 | 78.77 | 163.97 | 19.80 |
| 100 | 2 | 0/3 | 0/3 | 0/3 | 0/3 | 32.05 | 78.77 | 163.97 | 19.80 |

At `H_exec=1`, the queue was cleared and ACT was re-observed/re-run on all 160 frames in each episode. It improved mean minimum approach error by about 4.4 mm relative to `H_exec=100`, but did not change pregrasp error and remained far outside the 12 mm Reach gate. Seed 11 briefly reached two simultaneous contacts for one frame for every horizon; this is not a grasp.

The final cube-displacement column was recomputed from the saved trajectories
against each raw reset pose. The original rollout summaries used the first
post-action cube sample as their origin and therefore omitted the first control
interval; the evaluator now emits explicit final and maximum displacement from
the true reset pose.

Therefore, the original 100-step queued execution is an amplifier, not the dominant failure. Case A is rejected at step 500.

Code review then found that the original rollout summary used single-frame
Reach/Approach/Grasp thresholds and could count a height-only hold outside the
ordered grasp-to-lift interval. This could have created future false positives,
although it cannot change the step-500 result because no run crossed even the
single-frame Reach threshold. Before evaluating later checkpoints, the metric
was tightened to require 5 sustained Reach frames, 5 sustained Approach frames
with the cube-displacement gate, 8 sustained multi-finger Grasp frames, and 15
post-grasp frames above 80 mm while retaining at least two finger contacts.
In this report, the table's `Task` column means that ordered sustained rollout
milestone only; it is not the project's separate height-only `task_success`,
nor does it claim `grasp_stable` without the SE(3) stability evaluation.

## 5. Training-versus-rollout preprocessing parity

The audit compared the same seed and frame through the raw recorder, native LeRobot dataset, runtime observation preparation, and checkpoint processor.

| Item | Training path | Rollout path | Result |
|---|---|---|---|
| State | float32, 27D | float32, 27D | Exact before and after processor |
| Front | RGB CHW float32 `[0,1]` | RGB HWC uint8 -> official prepare -> CHW float32 `[0,1]` | Exact at reset |
| Wrist | RGB CHW float32 `[0,1]` | RGB HWC uint8 -> official prepare -> CHW float32 `[0,1]` | 12 channel values differ by 1 uint8 LSB at reset |
| Resolution | 240x320 | 240x320 | Same; no resize |
| Camera keys/order | front, wrist | front, wrist | Same |
| Joint order | 7 arm then 20 Wuji | 7 arm then 20 Wuji | Exact |
| Checkpoint processor | stored mean/std | same stored mean/std | Same |

`parity_exact=false` only because MuJoCo produced sparse 1-LSB wrist rasterization variation. Raw-to-native PNG decoding itself is exact, state is exact, and the full seeded control replay reproduces all actual states and controller targets bit-for-bit. This is renderer repeatability noise, not RGB/BGR, range, dtype, layout, camera-key, or normalization mismatch. The audit therefore sets `parity_passed=true` and `input_preprocessing_mismatch_found=false`.

## 6. Temporal alignment result

The recorder/exporter contract is verified as:

```text
observation_t
-> absolute controller target action_t
-> physics step
-> observation_t+1
```

Evidence across all 20 episodes and 2,804 frames:

- All observation state chains match exactly.
- All frame indices match and every timestamp is strictly increasing.
- A seeded 140-frame replay reproduces the recorded controller targets and next states with maximum absolute error 0 rad.
- Across nontrivial joint values, one controller step reduces mean absolute target error from 0.08797 to 0.08553 rad; 59.85% move closer in one step. A position-controlled physical system need not make every joint closer on every single substep.

No fixed one-frame shift or state/action indexing error was found. Case C is not supported by temporal alignment evidence.

Code review found that the audit JSON's top-level `one_frame_temporal_shift_found`
boolean originally omitted `exact_controller_target_replay`, although the
underlying temporal `passed` check already included it. The boolean was fixed,
the audit was rerun, and state, controller target, and temporal checks all pass.

## 7. Padding/masking audit

Episode lengths range from 136 to 146 frames (mean 140.2), while ACT predicts 100 actions. Of 280,400 possible chunk slots, 99,000 are episode-tail padding (**35.31%**). LeRobot repeats the last valid action in those slots and marks them with `action_is_pad`; ACT constructs `valid_mask = ~action_is_pad`, excluding them from both the L1 numerator and denominator. Direct checks at episode starts, interiors, penultimate frames, and final frames all match the expected mask.

Padding is therefore not reducing the reported loss artificially. There is, however, a real objective imbalance:

| Phase | Raw frames | Raw frame fraction | Valid ACT target slots | Valid target fraction |
|---|---:|---:|---:|---:|
| Reach | 267 | 9.52% | 2,002 | **1.10%** |
| Approach | 797 | 28.42% | 26,941 | 14.85% |
| Grasp close | 480 | 17.12% | 31,536 | 17.38% |
| Preload | 80 | 2.85% | 6,376 | 3.51% |
| Preload settle | 160 | 5.71% | 13,712 | 7.56% |
| Lift | 720 | 25.68% | 70,833 | **39.05%** |
| Hold | 300 | 10.70% | 30,000 | 16.54% |

This provides a concrete mechanism by which total loss can decrease while the episode-start controller stays poor: the target-slot objective assigns nearly 35 times more valid L1 elements to Lift than to Reach. This table counts potential L1 elements, not each phase's realized loss contribution. In addition, ACT's reported total objective is `L1 + 10 * KLD`, so it must not be interpreted as a direct action-MAE curve. The imbalance is a training-distribution property, not a masking software bug.

## 8. Normalization audit

The following table is computed from all 2,804 raw training actions. Actions are absolute position-actuator targets in radians.

| Index | Joint | Min | Max | Mean | Std |
|---:|---|---:|---:|---:|---:|
| 0 | openarm_left_joint1 | -0.761456 | 0.013085 | -0.495874 | 0.193148 |
| 1 | openarm_left_joint2 | -0.104013 | 0.103026 | 0.007886 | 0.045719 |
| 2 | openarm_left_joint3 | -0.052690 | 0.044356 | 0.000234 | 0.015116 |
| 3 | openarm_left_joint4 | 0.700670 | 1.718549 | 1.182995 | 0.267133 |
| 4 | openarm_left_joint5 | -0.117557 | 0.112691 | 0.004646 | 0.043167 |
| 5 | openarm_left_joint6 | -0.468500 | 0.255891 | 0.015783 | 0.207612 |
| 6 | openarm_left_joint7 | -0.047921 | 0.044064 | 0.003851 | 0.025805 |
| 7 | wuji_left_finger1_joint1 | 0.047500 | 0.923400 | 0.514583 | 0.409281 |
| 8 | wuji_left_finger1_joint2 | 0 | 0.380000 | 0.202639 | 0.177562 |
| 9 | wuji_left_finger1_joint3 | 0 | 0.760000 | 0.405278 | 0.355124 |
| 10 | wuji_left_finger1_joint4 | 0 | 0.760000 | 0.405278 | 0.355124 |
| 11 | wuji_left_finger2_joint1 | 0 | 0.836000 | 0.445806 | 0.390637 |
| 12 | wuji_left_finger2_joint2 | 0 | 0 | 0 | 0 |
| 13 | wuji_left_finger2_joint3 | 0 | 0.836000 | 0.445806 | 0.390637 |
| 14 | wuji_left_finger2_joint4 | 0 | 0.836000 | 0.445806 | 0.390637 |
| 15 | wuji_left_finger3_joint1 | 0 | 0.836000 | 0.445806 | 0.390637 |
| 16 | wuji_left_finger3_joint2 | 0 | 0 | 0 | 0 |
| 17 | wuji_left_finger3_joint3 | 0 | 0.836000 | 0.445806 | 0.390637 |
| 18 | wuji_left_finger3_joint4 | 0 | 0.836000 | 0.445806 | 0.390637 |
| 19 | wuji_left_finger4_joint1 | 0 | 0.836000 | 0.445806 | 0.390637 |
| 20 | wuji_left_finger4_joint2 | 0 | 0 | 0 | 0 |
| 21 | wuji_left_finger4_joint3 | 0 | 0.836000 | 0.445806 | 0.390637 |
| 22 | wuji_left_finger4_joint4 | 0 | 0.836000 | 0.445806 | 0.390637 |
| 23 | wuji_left_finger5_joint1 | 0 | 0.836000 | 0.445806 | 0.390637 |
| 24 | wuji_left_finger5_joint2 | 0 | 0 | 0 | 0 |
| 25 | wuji_left_finger5_joint3 | 0 | 0.836000 | 0.445806 | 0.390637 |
| 26 | wuji_left_finger5_joint4 | 0 | 0.836000 | 0.445806 | 0.390637 |

Checkpoint min/max match raw data exactly; checkpoint mean/std differ by at most `3.58e-7`, and the policy preprocessor/postprocessor copies the checkpoint statistics exactly. Constant joint targets remain consistent across training and inference.

For `finger1_joint1` (index 7):

- Controller range: `[0.0475, 1.6030]` rad.
- Training target: min 0.0475, max 0.9234, mean 0.51458, std 0.40928 rad.
- Normalized training range: `[-1.14123, 0.99887]`.
- Step-500 rollout prediction before controller clipping: min -0.04189, max 0.97916 rad.
- The same prediction in normalized units: min -1.35963, max 1.13511.
- Across the existing 3,200 rollout frames, 822 values (25.69%) are clipped on this joint, with maximum correction 0.08939 rad. No other joint clips.

The network occasionally extrapolates below a training minimum that is also the actuator lower limit. Unnormalization is correct, and the controller correctly clips to its physical range. This is model error, not a range-semantics or normalization bug.

By step 2000, the three H=1 rollouts clip only 9 of 12,960 action values
(`0.069%`), still exclusively joint index 7, with maximum correction
`0.01383 rad`. Clipping is therefore not the cause of the remaining Reach
failure.

## 9. Teacher-forced / expert-state-reset diagnosis

On training seed 0, ACT replans every frame (`H_exec=1`) while MuJoCo is reset to the recorded expert arm/hand/cube pose every `N` frames. The action queue is not allowed to hide observation feedback. Arm/hand/cube pose and simulation time are restored; velocity is reconstructed by a forward finite difference from recorded `q_t` to `q_{t+1}`. This uses one-step future information, and solver warm-start/contact internals are not restored. Re-rendered images also differ slightly from recorded images. Consequently, reset-row action MAE is useful as a policy-on-near-expert-observation check, while post-state error and error-versus-age are only supporting sensitivity evidence, not a pure causal estimate of behavioral-cloning covariate shift.

| Reset interval `N` | Resets | Mean pre-state MAE (rad) | Action MAE (rad) | Arm action MAE | Hand action MAE | One-step post-state MAE |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 140 | 0.000000 | 0.032913 | 0.034092 | 0.032500 | 0.012937 |
| 5 | 28 | 0.030111 | 0.059157 | 0.044771 | 0.064192 | 0.044201 |
| 10 | 14 | 0.065083 | 0.109617 | 0.054311 | 0.128974 | 0.077907 |
| 20 | 7 | 0.099016 | 0.140164 | 0.059315 | 0.168462 | 0.106456 |

The within-window error makes the mechanism explicit:

| Frames since kinematic expert-pose reset | Pre-state MAE (N=5) | Action MAE (N=5) | Pre-state MAE (N=10) | Action MAE (N=10) |
|---:|---:|---:|---:|---:|
| 0 | 0.000000 | 0.033853 | 0.000000 | 0.035698 |
| 1 | 0.013024 | 0.036720 | 0.013494 | 0.040804 |
| 2 | 0.032216 | 0.057758 | 0.034298 | 0.063285 |
| 3 | 0.046691 | 0.074199 | 0.050473 | 0.079785 |
| 4 | **0.058624** | **0.093254** | 0.061965 | 0.095076 |
| 9 | - | - | **0.120551** | **0.173962** |

The every-frame kinematic-reset (`N=1`) phase breakdown independently confirms that the near-expert-state predictions are poorest at Reach:

| Phase | Action MAE | Arm action MAE | Hand action MAE | One-step post-state MAE |
|---|---:|---:|---:|---:|
| Reach | 0.073088 | **0.104760** | 0.062003 | 0.025911 |
| Approach | 0.040864 | 0.035524 | 0.042733 | 0.016581 |
| Grasp | 0.033571 | 0.023515 | 0.037091 | 0.012086 |
| Preload | 0.025298 | 0.010709 | 0.030405 | 0.010530 |
| Preload settle | 0.028286 | 0.014653 | 0.033057 | 0.010697 |
| Lift | 0.020539 | 0.025909 | 0.018659 | 0.008378 |
| Hold | 0.012184 | 0.026813 | 0.007064 | 0.006734 |

Interpretation: the action interface is meaningful on near-expert states, and this approximate-reset experiment is consistent with rapid compounding once the state departs from the demonstrations. It does not, by itself, isolate covariate shift from the imperfect state restoration or establish Case D as the primary case. The cleaner result remains the offline reconstruction: expert-state Reach predictions are still materially inaccurate.

The same diagnostic at the hard-stop checkpoint shows that continued fitting
reduced both the on-pose floor and subsequent growth:

| Checkpoint | N=1 action MAE | N=5 mean | N=10 mean | N=20 mean | Age-4 growth | Age-9 growth | Age-19 growth |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 500 | 0.032913 | 0.059157 | 0.109617 | 0.140164 | 2.75x | 4.87x | 4.99x |
| 2000 | 0.026401 | 0.037412 | 0.046428 | 0.065129 | 2.02x | 2.14x | 3.13x |

At step 2000, every-frame-reset Reach arm action MAE is still `0.04338 rad`,
but the multi-frame excess is much smaller than at step 500. That co-movement
supports the ranking used below: compounding is real, yet a substantial portion
of it is downstream of correctable model underfit rather than proof that the
20 demonstrations intrinsically lack all required recovery coverage.

## 10. Root-cause ranking and case decision

### Root cause #1: visually conditioned Reach is still underfit

**Evidence:** at step 2000, balanced expert-state Reach arm MAE remains `0.0414 rad`, and the harder frame-0 arm MAE is `0.0645 rad`. ACT reproduces only 32.18% of expert cross-seed arm-action spread; joints 2/3/5/7 retain only 1.7-6.2% of their expert variation. `H_exec=1` remains 0/3 Reach with 22.68 mm mean minimum pregrasp error. Continued training substantially improved all of these on-manifold measures from step 500, directly showing that underfit was causal rather than merely correlated.

**Confidence:** Very high (0.95).

### Root cause #2: the 100-step objective heavily underweights episode-start Reach

**Evidence:** Reach is 9.52% of raw frames but only 1.10% of valid action-chunk L1 target slots, versus 39.05% for Lift. From step 1500 to 2000, Reach arm MAE improved by 36%, while Grasp and Preload hand MAE regressed by 26% and 22%; the balanced H1 MAE worsened despite a lower scalar training objective. The objective is trading error among phases rather than consistently fitting the complete sequence.

**Confidence:** High that this is a material optimization mechanism (0.90). It is a data-weighting property, not a padding-mask bug.

### Root cause #3: off-trajectory error compounds; long execution makes it worse

**Evidence:** the approximate reset diagnostic still shows 2.02x action-error growth by age 4 and 3.13x by age 19 at step 2000. However, these factors fell from 2.75x and 4.99x at step 500 as the model fit improved. Separately, `H_exec=1` improved step-500 approach error by only 4.4 mm and did not change Reach success relative to H=100.

**Confidence:** Medium-high that compounding contributes (0.80); low that 100-step execution is dominant (0.10); medium-low that missing recovery coverage is already proven to be the irreducible cause (0.40).

### Explicit exclusions

- Preprocessing/camera/state-order mismatch: not found; confidence high.
- One-frame temporal shift or wrong action indexing: not found; confidence high.
- Padding included in loss: not found; confidence high.
- Normalization/controller range semantics bug: not found; confidence high.
- Grasp quality as the cause of the original step-500 0/20: rejected for this stage, because the policy fails before Reach/Approach.

### Forced final case selection

**Case B — ACT is still substantially underfit.**

Case A is rejected because H=1 and H=5 do not materially recover behavior. Case
C is rejected by exact state/action replay plus preprocessing, padding, temporal,
and normalization audits. Case D remains a plausible later-stage problem, but
it is not primary yet: the policy still fails at frame 0 on real training
observations, and its cross-seed visual response is strongly attenuated. Case E
is not selected because the apparent compounding weakened markedly when the
same model was trained further, while on-manifold Reach fit and behavior moved
together. The dominant actionable issue is therefore model/objective fit before
collecting recovery demonstrations.

## 11. Completed Case-B experiment and recommended next step

The requested exact-resume experiment is complete. Dataset, architecture,
normalization, observation, action, controller, scripted grasp, and primary
hyperparameters remained unchanged; only optimizer steps advanced from 500 to
the hard stop at 2000. Each checkpoint was evaluated with the same 288 balanced
expert anchors and `H_exec=1` on seeds 0, 7, and 11.

Checkpoint continuity was verified at every resume; for the final segment all
153 Adam step tensors advanced uniformly from 1500 to 2000, and optimizer and
RNG state were restored rather than restarting training.

<!-- TRAINING_PROGRESS_START: update this table after each exact-resume checkpoint. -->

| Checkpoint | Reported train objective | Balanced H1 overall MAE | Reach H1 arm MAE | Mean min pregrasp | H=1 Reach / 3 | H=1 Approach / 3 | H=1 task / 3 | Decision |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 500 | 2.111 | 0.040214 | 0.115988 | 32.05 mm | 0 | 0 | 0 | Continue: Case-B threshold met |
| 1000 | 1.574 | 0.026482 | 0.092893 | 24.13 mm | 0 | 0 | 0 | Continue: offline and behavior improved, Reach arm remains underfit |
| 1500 | 1.135 | 0.026560 | 0.064621 | 20.57 mm | 0 | 0 | 0 | Continue once to hard stop: Reach still improves and remains underfit |
| 2000 | 0.861 | 0.028131 | 0.041365 | 22.68 mm | 0 | 0 | 0 | Hard stop; final Case B, change training design |

<!-- TRAINING_PROGRESS_END -->

The reported train objective is the final 10-update logger window, not a held-out
validation loss. ACT uses `total = L1 + 10 * KLD`; for reference, the step-1000
window was `1.574 = 0.157 + 10 * 0.142` up to logging precision, and step 1500
was `1.135 = 0.128 + 10 * 0.101`; step 2000 was
`0.861 = 0.117 + 10 * 0.074`. Checkpoint selection in this diagnosis is
therefore based on offline action reconstruction and rollout behavior rather
than that scalar alone.

At step 1000, balanced first-action MAE improved by 34.1% and mean minimum
pregrasp error improved by 24.7%, so continued optimization has a measurable
closed-loop effect. The improvement is highly asymmetric during Reach: hand
MAE fell from 0.0655 to 0.0162 rad, while arm MAE only fell from 0.1160 to
0.0929 rad. None of the three rollouts entered the 12 mm positional tolerance
even for one frame (best 22.98 mm), let alone held it for the required five, so
the evidence supports one more exact-resume checkpoint at step 1500 rather
than declaring recovery or changing the interface.

At step 1500, Reach arm MAE improved again to 0.0646 rad and mean minimum
pregrasp error fell to 20.57 mm (best 19.14 mm), but no seed entered the 12 mm
Reach tolerance even for one frame. Recomputed from the true reset pose, final
cube displacement fell to a 24.05 mm mean and two seeds left
the cube essentially unmoved. Balanced H1 MAE was flat at 0.0266 rad because the
improvement was phase-specific: Grasp hand MAE regressed to 0.0660 rad and
Preload hand MAE to 0.0371 rad. Because the primary early-phase metric and
closed-loop geometry are still improving together, one final unchanged
continuation to the pre-declared hard stop at step 2000 is evidence-based.

At step 2000, Reach arm reconstruction improved again to `0.0414 rad`, but the
behavioral gain was not consistent: mean minimum pregrasp error regressed from
20.57 to 22.68 mm. Seed 11 improved to 12.90 mm, while seeds 0 and 7 regressed
to 28.68 and 26.45 mm. None entered the 12 mm tolerance, there were no contacts,
and Grasp/Preload hand reconstruction worsened. Training was therefore stopped
at exactly 2000 rather than extending the same run.

The next clean single-variable experiment should initialize the existing
ResNet-18 visual backbone from ImageNet weights instead of the current
`pretrained_backbone_weights=null`, while keeping this dataset, 27D interface,
100-step ACT configuration, grasp, controller, and evaluation seeds fixed. The
reason is specific: the controlled swap proves that images carry the required
cross-seed signal and that ACT has started to use it, but its response is still
only one third of the expert spread after approximately 5.71 dataset-equivalent
sample passes. Evaluate the
same reset-conditioning spread, phase-aware MAE, and H=1 seeds at 500/1000/2000.

Do not simultaneously change sampling. If the pretrained-backbone ablation
still leaves Reach weak, the next independent ablation should correct the 1.10%
Reach target weighting with a phase-balanced anchor sampler or a shorter
training chunk. Only after on-demonstration Reach becomes accurate while H=1
still fails should Case D trigger perturbed/recovery demonstrations or a later
DAgger-style loop. No such data collection was performed in this round.

For diagnostics, retain `H_exec=1` so queued open-loop actions cannot confound
the comparison. This is not yet a claim that H=1 is the final deployment
setting.
