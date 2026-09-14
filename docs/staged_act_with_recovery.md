# Staged ACT with independent Recovery

## Outcome

**Case D.** Recovery succeeds on a substantial fraction of both training and unseen trigger states, but first-trigger switching causes a statistically clear matched regression; keep Recovery and redesign or calibrate the trigger/switch logic before Grasp/Preload.

Selected Recovery checkpoint: `D:/yl/embodied ai/openarm-wuji-learning/outputs/staged_act_with_recovery/recovery_act_train/checkpoints/001500/pretrained_model`.
Proceed to Grasp+Preload ACT: `false`.

## Controlled setup

- Recovery training uses only the 30 successful correction episodes / 559
  frames. Original Approach demonstrations are not mixed in.
- Fresh ACT; normal sampling; ImageNet ResNet-18; chunk20;
  `n_action_steps=20`; `H_exec=1`; training seed 1000.
- Observation is front+wrist RGB plus 27D actual qpos. Action is the 27D
  absolute target actually sent to the MuJoCo position controller.
- Reach ACT and original Approach ACT are frozen. Grasp, Preload, Lift, RL and
  SmolVLA are not run.
- The router reuses the five mining detectors and permits exactly zero or one
  Recovery attempt. Recovery success completes the Approach stage; it never
  switches back to Approach.

## Recovery-only checkpoint curve: exact 30 training starts

| step | recovery success | cube failures | timeout | final error median / p90 | mean frames | clipped values |
|---:|---:|---:|---:|---:|---:|---:|
| 500 | 7/30 (23.3%) | 1 | 22 | 22.62 / 36.48 mm | 70.5 | 4 |
| 1000 | 10/30 (33.3%) | 1 | 19 | 19.94 / 36.31 mm | 62.7 | 1387 |
| 1500 | 19/30 (63.3%) | 0 | 11 | 10.96 / 28.44 mm | 40.1 | 927 |
| 2000 | 17/30 (56.7%) | 1 | 12 | 11.91 / 29.74 mm | 44.2 | 845 |

Checkpoint selection follows recovery success, cube safety, then timeout; it
does not use training loss or unseen results.

## Recovery by trigger source

| trigger | training correction starts | unseen first-trigger starts |
|---|---:|---:|
| moving_away | 8/17 (47.1%) | 45/76 (59.2%) |
| terminal_plateau | 2/2 (100.0%) | 0/0 (0.0%) |
| gate_stall | 3/4 (75.0%) | 24/42 (57.1%) |
| near_timeout_terminal | 3/3 (100.0%) | 0/0 (0.0%) |
| cube_displacement | 3/4 (75.0%) | 19/28 (67.9%) |

## Matched unseen seeds 1400-1599

Both branches share the exact same Reach and Approach prefix. At the first
near-failure trigger, the baseline continues original Approach while the new
branch restores that exact MuJoCo state and performs one Recovery attempt.

| metric | Reach -> Approach baseline | Reach -> Approach -> Recovery |
|---|---:|---:|
| Reach success | 154/200 | 154/200 |
| Approach-stage success given Reach | 119/154 (77.3%) | 96/154 (62.3%) |
| joint success | 119/200 | 96/200 |
| timeout | 18 | 54 |
| cube safety failure | 17 | 4 |
| cube displacement median / p90 | 1.73 / 26.34 mm | 2.55 / 8.55 mm |

- Direct Approach successes before any trigger:
  `8`.
- Recovery attempts: `146`; successes:
  `88`
  (`60.3%`).
- Mean Recovery length: `40.8` frames.
- Matched switch effect: rescued baseline failures
  `20`, regressed baseline successes
  `43`, both-success `68`,
  both-failure `15`.
- Two-sided exact McNemar/binomial p-value on discordant pairs: `0.005152`.

## Answers

1. **Training correction starts:** partially, but not reliably. Recovery
   succeeds on `19/30`
   (`63.3%`).
2. **Unseen near-failure states:** partially. It succeeds on
   `88/146`
   (`60.3%`), close to the training-start
   rate, so there is no large train-to-unseen collapse.
3. **Overall robustness improved:** no. Matched Approach-stage success falls
   from `119` to
   `96` out of
   `154` Reach successes.
4. **Safer-but-more-timeout tradeoff solved:** no. Cube failures fall from
   `17` to
   `4`, but timeouts rise from
   `18` to `54`.
5. **Enter Grasp+Preload ACT:** `false`. Redesign and retest
   the trigger/switch logic first.

## Decision

Recovery succeeds on a substantial fraction of both training and unseen trigger states, but first-trigger switching causes a statistically clear matched regression; keep Recovery and redesign or calibrate the trigger/switch logic before Grasp/Preload.
