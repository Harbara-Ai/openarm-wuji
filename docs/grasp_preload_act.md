# Grasp + Preload ACT

## Outcome

**Case A: Grasp+Preload ACT works both standalone and after real staged handoff. Freeze it and attach scripted Lift.**

No Lift was executed or trained in this experiment.
The selected step-1500 policy is frozen in `configs/grasp_secure_stage.json`. Attaching scripted Lift means a guarded next evaluation, not a claim of hardware-ready reliability.

## Dataset and interface

- Successful episodes: 50
- Frames: 2107
- Sources: 20 nominal scripted starts + 30 real frozen staged handoffs
- Frozen staged collection: 92 rollouts, 53 upstream-success states
- Failed diagnostics excluded from BC: grasp_formation=23, upstream=39
- Observation: front RGB + wrist RGB at 240x320, plus 27D actual qpos
- Action: 27D absolute position-controller target
- Telemetry excluded from policy observation: contact topology/forces, cube pose, palm-relative pose, slip, and target-actual preload
- Native LeRobotDataset: v3.0; state/action exact round-trip passed

The evaluation preload threshold is the successful-expert terminal mean preload L2 10th percentile: 1.1775 rad.

## ACT configuration

Fresh ACT with ImageNet-initialized ResNet-18, chunk_size=20, n_action_steps=20, H_exec=1, batch size 8, AdamW at 1e-5, and seed 1000. Upstream policies and router remained frozen.

## Standalone closed-loop evaluation

| Step | Grasp+Preload | Grasp | Preload | Timeouts | Cube mean / p90 / max | >25 mm | Clipping | Failure stages |
|---:|---:|---:|---:|---:|---:|---:|---|---|
| 500 | 9/20 (45.0%) | 19/20 (95.0%) | 9/20 (45.0%) | 11 | 93.10 / 59.06 / 1356.05 mm | 5 | 0 (none) | grasp_formation=1, preload_formation=10 |
| 1000 | 0/20 (0.0%) | 14/20 (70.0%) | 1/20 (5.0%) | 20 | 278.54 / 982.99 / 3220.93 mm | 6 | 273 (wuji_left_finger1_joint1=273) | contact_acquisition=4, grasp_formation=2, preload_formation=13, terminal_hold=1 |
| 1500 | 14/20 (70.0%) | 20/20 (100.0%) | 15/20 (75.0%) | 6 | 9.85 / 19.18 / 34.26 mm | 1 | 12 (wuji_left_finger1_joint1=12) | preload_formation=5, terminal_hold=1 |
| 2000 | 6/20 (30.0%) | 17/20 (85.0%) | 7/20 (35.0%) | 14 | 8.42 / 17.14 / 34.16 mm | 1 | 129 (wuji_left_finger1_joint1=129) | contact_acquisition=1, grasp_formation=2, preload_formation=10, terminal_hold=1 |

Best checkpoint by closed-loop outcome: step 1500.

## Preload diagnosis at the selected checkpoint

| Population | Mean max preload L2 | Mean terminal preload L2 |
|---|---:|---:|
| Success | 1.4560 rad | 1.3530 rad |
| Failure | 1.1383 rad | 0.7264 rad |

At step 1500, success has a substantially larger terminal preload than failure (1.3530 vs 0.7264 rad), so target-actual preload is strongly associated with the measured outcome in this matched set. This is descriptive, not a causal estimate.

### Contact and force diagnostics

Finger order is thumb, index, middle, ring, little.

| Population | Mean peak force per finger (N) | Mean final force per finger (N) | Mean max total force (N) | Max-contact distribution |
|---|---|---|---:|---|
| Success (n=14) | 0.70, 2.38, 5.24, 4.01, 1.35 | 0.45, 0.77, 1.26, 1.58, 0.57 | 12.31 | {"5": 14} |
| Failure (n=6) | 1.06, 1.21, 1.82, 2.55, 3.23 | 0.31, 0.39, 0.53, 0.83, 1.79 | 6.87 | {"4": 2, "5": 4} |

Selected-checkpoint cube displacement mean / p90 / max was 9.85 / 19.18 / 34.26 mm; 1/20 exceeded the existing 25 mm cube-motion safety reference. With no Lift in this stage, physical drop is not applicable; this >25 mm count is reported as the escape/motion proxy.

## Real staged handoff

- Upstream successes / all rollouts: 12/20 (60.0%)
- Grasp+Preload successes / upstream successes: 9/12 (75.0%)
- Joint staged successes / all rollouts: 9/20 (45.0%)
- Failure stages after successful handoff: {"preload_formation": 2, "terminal_hold": 1}
- Mean maximum cube displacement: 15.31 mm
- Cube displacement p90 / max: 31.98 / 86.49 mm; 2/12 attempts exceeded the existing 25 mm motion reference
- Action clipping values: 0

The staged evaluation ran the frozen Reach, Approach, selective hysteresis router, and optional Recovery live. It did not inject an expert state or action before GraspSecure.

## Reproducibility caveat

A same-snapshot A/B check restored identical 27D qpos and identical front pixels, but 5 wrist pixels differed by one gray level. The first action differed by only 1.43e-5 rad; contact dynamics later amplified that perturbation. Therefore the 20-run rates are matched single-trial estimates, not confidence-bounded reliability numbers. This does not change the ranking observed here (step 1500 is well ahead), but a repeated robustness evaluation is warranted before hardware transfer.

## Decision

Case A: Grasp+Preload ACT works both standalone and after real staged handoff. Freeze it and attach scripted Lift.

Because the live staged set contains a cube-motion tail (maximum 86.49 mm), scripted Lift should be attached with the existing cube-motion safety reference active. Do not train Lift yet.

The machine-readable aggregate is stored at `outputs/grasp_preload_act/summary.json`.
