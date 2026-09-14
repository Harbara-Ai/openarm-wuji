# Staged ACT pipeline: Reach → Approach

## Scope and outcome

This run completed Stage 1–3 only: the best Reach-only ACT was frozen, a new
Approach-only ACT was trained from scratch, and both policies were composed with
explicit gates. Grasp/Preload ACT was **not started** and Lift remains scripted.

The main result is **19/20 Reach**, **16/19 conditional Approach**, and **16/20
joint Reach→Approach success** on the 20 training seeds. The real closed-loop
Reach endpoint is therefore usable as the Approach input; it is not the present
bottleneck. Approach is not yet robust enough to start Grasp/Preload training,
because three chained seeds still fail at the terminal approach/contact region.

## Shared policy interface

- Observation: front + wrist RGB at 240×320 and 27-D actual qpos
- Action: 27-D absolute controller target
- ACT: ImageNet ResNet-18, chunk size 20, `n_action_steps=20`
- Execution: `H_exec=1`; a new prediction is made from the current observation
- Training/evaluation seeds: `[0, 1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 17, 18, 19, 20, 21, 22]`
- No expert action is read during any reported ACT rollout

The staged API is `ReachPolicy` / `ApproachPolicy`, each exposing `reset()`,
`select_action()`, `is_success()`, and `is_timeout()`. The outer evaluator owns
the phase switch; phase is not hidden inside a monolithic policy.

## Policy 1: frozen Reach

- Checkpoint: `D:/yl/embodied ai/openarm-wuji-learning/outputs/act_reach_only/act_train/checkpoints/002000/pretrained_model`
- Dataset: 20 episodes, 427 frames
  (267 real Reach +
  160 terminal hold)
- Gate: pregrasp position error ≤12 mm for five consecutive frames
- Closed-loop result: 19/20; all 20 enter
  the 12 mm basin; no early cube contact and no action clipping
- Rollout: `outputs/act_reach_only/rollouts_step_002000/summary.json`

## Policy 2: Approach-only

The dataset starts at each demonstration's exact stable pregrasp state and ends
at the grasp-start state. It contains 20 episodes and 957
frames: 797 real Approach frames plus
160 terminal hold frames.
The Wuji 20-D target stays constant; no grasp-close action is present.

The scripted expert-target replay reaches the same gate on 20/20 seeds, with
mean minimum error 8.86 mm
and maximum cube displacement
0.283 mm. Thus the
dataset endpoint and gate are dynamically reachable.

| step | Approach success | mean min error (mm) | mean final error (mm) | max cube move (mm) | cube-gate / timeout failures | clipped values |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 500 | 0/20 | 32.73 | 71.41 | 31.55 | 7 / 13 | 469 |
| 1000 | 5/20 | 21.47 | 32.98 | 7.41 | 0 / 15 | 644 |
| 1500 | 12/20 | 7.59 | 14.50 | 59.13 | 4 / 4 | 645 |
| 2000 | 11/20 | 4.59 | 15.34 | 70.52 | 6 / 3 | 611 |

Step 1500 has the best isolated nominal-start count (12/20). Step 2000 is kept
as Policy 2 for the composed pipeline because it is substantially better after
the real Reach handoff (16/19 versus 11/19 at step 1500) and has much lower
nominal-start mean minimum error (4.59 mm). Its isolated result is still only
11/20: six seeds cross the 25 mm cube-displacement safety gate and three time
out. Therefore the answer to “safe Approach-only?” is **partially, not robustly**.

Selected checkpoint: `D:/yl/embodied ai/openarm-wuji-learning/outputs/act_staged/approach_only/act_train/checkpoints/002000/pretrained_model`

## Stage 3: real Reach → Approach handoff

No expert state or action is injected at the transition. The frame that satisfies
the Reach gate is passed directly to Approach ACT.

- Reach: 19/20
- Approach given Reach: 16/19
  (84.2%)
- Joint Reach→Approach: 16/20
- Remaining failures: seed 8 fails Reach; seed 2 crosses the Approach cube-motion
  gate; seeds 6 and 16 time out near the terminal/contact region
- Mean same-seed arm-state handoff distance: 0.00945 rad
- Mean nearest expert 27-D start distance: 0.03945 rad
- Mean cube-position handoff mismatch: 0.051 mm
- Successful handoffs' mean same-seed arm distance: 0.00984 rad
- Failed handoffs' mean same-seed arm distance: 0.00739 rad
- Rollout: `outputs/act_staged/reach_approach_step_002000/summary.json`

This is **not evidence of a harmful handoff distribution mismatch**. The chained
policy improves from 11/20 at exact expert starts to 16/19 conditional success,
and failed handoffs are not farther from the same-seed expert start than successful
handoffs (0.00739 versus 0.00984 rad arm L2). The remaining failure mode belongs
to Policy 2's terminal/contact stabilization: it can enter the grasp-pose basin
but sometimes fails to stop before pushing the cube or cannot hold the full gate.

## Decision

Do not begin Grasp/Preload ACT yet. Keep the staged architecture and collect a
small, targeted Approach correction/stop set around failed chained seeds 2, 6,
and 16, plus modest perturbed-pregrasp endpoints produced by Reach ACT. This is
not a request for more generic full-task demonstrations: the missing supervision
is specifically the last Approach frames and recovery/stop behavior.

Machine-readable summary: `outputs/act_staged/summary.json`

## Verification

- Native LeRobotDataset v3.0 export round-trip: state/action exact
- DataLoader smoke: state `[8,27]`, action `[8,20,27]`, both images
  `[8,3,240,320]`, padding mask `[8,20]`
- Scripted Approach endpoint replay: 20/20
- ACT checkpoints: 500, 1000, 1500, and 2000 steps
- Syntax check passed for the dataset builder, staged controller, replay, and
  evaluator; native exporter unit test passed
