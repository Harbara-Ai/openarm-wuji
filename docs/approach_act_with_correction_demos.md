# Approach ACT with targeted correction demonstrations

## Outcome

**Case B.** Correction demos improve cube safety but not overall recovery, while staged nominal Approach regresses; retry 20% corrections.

Selected checkpoint: `D:/yl/embodied ai/openarm-wuji-learning/outputs/approach_act_with_correction/act_train/checkpoints/000500/pretrained_model`.  Proceed to Grasp/Preload ACT:
`false`.

## Controlled setup

- Original Approach: 20 episodes / 957 frames.
- Successful corrections: 30 episodes / 559 frames.
- Actual training mix: original `70.0%`, correction
  `30.0%` (16000 samples total).
- Sampling is episode-first; correction slots are balanced by
  `(trigger_type, source_outcome)` before sampling an episode and frame.
- The ten correction strata received 466--492 samples each; no raw dataset
  file was modified.
- Fresh ACT; ImageNet ResNet-18; chunk20; `n_action_steps=20`; `H_exec=1`;
  27D actual state to 27D absolute controller target; front+wrist 240x320.
- Reach ACT, controllers, gates, MuJoCo, grasp, reward and action semantics were unchanged.

## Checkpoint results

| step | nominal Approach | train Reach | train Approach given Reach | unseen Reach | unseen Approach given Reach | unseen cube failures | unseen timeouts | recovery after trigger |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 500 | 17/20 | 19/20 | 15/19 (78.9%) | 156/200 | 111/156 (71.2%) | 8 | 37 | 71.2% |
| 1000 | 5/20 | 19/20 | 7/19 (36.8%) | 159/200 | 35/159 (22.0%) | 18 | 106 | 21.5% |
| 1500 | 0/20 | 19/20 | 0/19 (0.0%) | 157/200 | 0/157 (0.0%) | 0 | 157 | 0.0% |
| 2000 | 17/20 | 19/20 | 13/19 (68.4%) | 158/200 | 96/158 (60.8%) | 1 | 61 | 64.8% |

Checkpoint choice follows the requested priority: unseen conditional success,
then cube safety, timeout rate, and training-seed regression—not training loss.

## Matched unseen comparison: seeds 1200-1399

| policy | Reach | Approach given Reach | joint success | cube failures | timeouts | near-failure rate | recovery after near-failure | cube displacement median / p90 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Original Approach ACT | 157/200 | 116/157 (73.9%) | 116/200 | 20 | 21 | 94.9% | 72.5% | 1.58 / 26.07 mm |
| New step 500 | 156/200 | 111/156 (71.2%) | 111/200 | 8 | 37 | 100.0% | 71.2% | 6.48 / 11.39 mm |

The 1000-1199 mining result (115/160 conditional, 71.9%) is retained only as
historical context.  The table above is the valid matched comparison because
both policies see exactly the same unseen seeds 1200-1399.

## Nominal regression and safety

The original step-2000 baseline achieved
`11/20` from exact expert Approach starts
and `16/19`
after frozen Reach.  The selected correction model achieves
`17/20` and
`15/19`
respectively.  Thus exact expert-start nominal performance improves, but the
real frozen-Reach handoff regresses by one success.  Per-checkpoint
terminal-error, cube-motion, early-contact, and action-clipping distributions
are preserved in `summary.json`.

## Failure-mode comparison

| outcome | Original ACT | New step 500 |
|---|---:|---:|
| success | 116 | 111 |
| timeout | 21 | 37 |
| cube_displacement | 20 | 8 |
| reach_failure | 43 | 44 |

The correction model reduces formal cube-motion safety failures from
`20` to
`8` and cube-displacement p90
from `26.07` to
`11.39` mm.  This comes
with more timeouts (`21` to `37`)
and a higher median cube displacement
(`1.58` to
`6.48` mm).

## Recovery interpretation

`near_failure` uses the same five detectors used for mining: moving away,
terminal plateau, gate stall, near-timeout terminal, and cube displacement.
A recovery means a rollout triggered at least one detector and nevertheless
passed the unchanged Approach gate.  This is stricter and more informative
than looking only at aggregate success, while still treating overlapping
trigger types as one episode for the overall recovery rate.

| trigger | Original: episodes / recovery | New: episodes / recovery |
|---|---:|---:|
| moving_away | 148 / 72.3% | 144 / 75.7% |
| terminal_plateau | 146 / 72.6% | 141 / 73.8% |
| gate_stall | 116 / 82.8% | 104 / 84.6% |
| near_timeout_terminal | 14 / 0.0% | 34 / 0.0% |
| cube_displacement | 61 / 49.2% | 138 / 68.8% |

Overall trigger-after-recovery is `72.5%`
for the original policy and
`71.2%` for the selected model:
there is no aggregate recovery gain.  Cube-displacement triggers become much
more frequent (`61`
to `138`),
but fewer cross the unchanged 25 mm safety-failure gate.  The learned trade is
therefore "smaller pushes and more timeouts", not "more reliable recovery".

## Decision

Correction demos improve cube safety but not overall recovery, while staged nominal Approach regresses; retry 20% corrections.
