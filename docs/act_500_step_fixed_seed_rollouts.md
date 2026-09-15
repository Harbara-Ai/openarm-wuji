# ACT 500-step continuation with fixed-seed rollouts

Date: 2026-09-10

## Scope held fixed

This continuation changed only the total optimizer-step target, checkpoint
frequency, and console log frequency. It did not recollect or re-export data,
and it did not change the scripted grasp, MuJoCo model, task configuration,
camera/state observation, normalization processors, ACT architecture, action
meaning, or the 27-D controller interface.

- Dataset: the same 20 successful episodes / 2,804 frames
- Observation: 27-D actual qpos plus front and wrist RGB at 240 x 320
- Action: 27-D absolute MuJoCo position-controller target
- ACT: the same 52M-parameter configuration and 100-step action chunk
- Optimizer: the same AdamW state, batch size 8, seed 1000, and learning rate
- Evaluation: training seeds 0, 7, and 11; 160 control frames each

Training was stopped at every boundary and resumed from the explicitly named
checkpoint. LeRobot restored model weights, optimizer state, Python/NumPy/Torch
RNG, training step, and sampler offset. This avoided both weight-only loading
and reliance on the unsupported Windows `last` symlink.

## Training and rollout results

| Checkpoint | Train loss | Reach | Full task | Mean pregrasp error | Mean approach error | Mean cube displacement | Max contacts | Longest >=2 contacts |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 100 | 3.619 | 0/3 | 0/3 | 29.5 mm | 60.8 mm | 161.5 mm | 2 | 1 frame |
| 200 | 3.171 | 0/3 | 0/3 | 29.9 mm | 83.4 mm | 518.7 mm | 3 | 84 frames |
| 300 | 2.505 | 0/3 | 0/3 | 34.3 mm | 76.0 mm | 208.0 mm | 2 | 1 frame |
| 400 | 2.268 | 0/3 | 0/3 | 33.1 mm | 78.3 mm | 206.3 mm | 3 | 1 frame |
| 500 | 2.111 | 0/3 | 0/3 | 32.0 mm | 78.8 mm | 159.5 mm | 2 | 1 frame |

The configured reach position tolerance is 12 mm. All 15 rollouts remained
well outside it, so none was allowed to count a later collision as an ordered
grasp. Peak cube height change stayed around 15--24 mm and the required lift
hold was zero frames throughout.

At step 200, seed 11 produced the only long multi-contact segment: up to three
fingers and 84 frames with at least two contacts. It still occurred without a
valid reach/approach and never lifted the cube. Seeds 0 and 7 at the same
checkpoint pushed the cube off the table (approximately 0.69 m and 0.79 m
displacement), so this checkpoint is not a behavioral improvement.

## Offline error versus closed-loop behavior

The model is learning something on the demonstration distribution:

| Checkpoint | Mean first-action MAE | Mean first-action hand MAE |
| ---: | ---: | ---: |
| 100 | 0.445 rad | 0.532 rad |
| 200 | 0.170 rad | 0.158 rad |
| 300 | 0.096 rad | 0.084 rad |
| 400 | 0.087 rad | 0.075 rad |
| 500 | 0.084 rad | 0.072 rad |

However, this offline improvement did not produce a single valid reach. At
step 500 the mean first-action arm MAE is still about 0.119 rad, large enough
to matter during approach. The first ten predicted actions have approximately
0.084 rad mean error even before the simulated state has diverged far from the
expert path.

Action-range clipping begins at step 300. At step 500 it occurs 123 times over
the three rollouts, all on `finger1_joint1`: ACT predicts below its 0.0475 rad
lower actuator bound, with maximum correction about 0.089 rad. This is a real
model-output symptom, but it is not the sole cause because steps 100 and 200
also fail without any clipping.

## Step-500 expanded 20-seed evaluation

The unchanged step-500 checkpoint was additionally evaluated for 160 frames on
seeds 0--19. Seventeen of these seeds are represented by successful training
demonstrations; seeds 4, 13, and 15 were collection diagnostics excluded from
the BC dataset.

- Ordered reach, approach, grasp, and task success were all 0/20. The subset of
  17 successful-demonstration seeds was also 0/17.
- Mean minimum pregrasp error was 31.9 mm (best 25.5 mm on seed 4), still above
  the 12 mm reach tolerance. Mean minimum approach error was 73.9 mm.
- Mean cube displacement was 122.0 mm (median 128.0 mm; maximum 213.7 mm).
- Mean peak cube height change was 18.0 mm and the maximum was 23.0 mm, but no
  rollout entered a valid Lift phase and final lift was effectively zero.
- The episode-level maximum-contact distribution was: 0 contacts in 4 runs,
  1 in 4, 2 in 9, 3 in 1, 4 in 1, and 5 in 1. Twelve runs briefly reached at
  least two contacts and three reached at least three.
- The apparently rich contacts do not constitute grasp success. Seed 4 held at
  least two contacts for 37 frames and seed 13 for 39 frames, but both are
  excluded collection-diagnostic seeds and neither completed ordered reach or
  Lift. Across the 17 training-demonstration seeds, multi-contact lasted at
  most one frame.
- All 20 runs contained action clipping: 822 values total, with maximum absolute
  correction 0.0894 rad.

This larger sample confirms that the earlier 0/3 result was not an unlucky
choice of seeds. It also shows why raw contact count alone is misleading here:
the only sustained multi-contact cases occurred on geometry associated with
failed scripted collection episodes, outside the valid task sequence.

## Conclusion

Training reached 500 steps successfully and loss decreased from 3.619 at step
100 to 2.111 at step 500. Nevertheless, every fixed-seed rollout remains
0/3 for reach and 0/3 for the complete task. The behavioral result is therefore
**not improved enough to claim ACT expert reproduction**.

The failure occurs before Lift: the learned arm/hand trajectory does not enter
the correct reach/approach geometry and cannot establish an ordered stable
grasp. The existing 27-D action interface is functioning—each checkpoint loads,
drives MuJoCo, and records the exact predicted and bounded targets—but supervised
loss is currently decoupled from closed-loop success.

These results alone cannot cleanly separate four remaining explanations:

1. 500 updates (4,000 sampled frames, about 1.43 frame-level epochs) are still
   insufficient for this 52M-parameter ACT;
2. 20 highly correlated scripted demonstrations provide weak recovery coverage;
3. executing a 100-step chunk open-loop causes small early errors to compound
   before the next visual/state replan;
4. sequence padding/masking or action normalization may make total loss a poor
   proxy for the critical early reach segment.

Before changing grasp or collecting more data, the clean next diagnostic is an
offline, phase-aware reconstruction check of the full first 100-action chunk on
training episodes, followed by an inference-only action-execution-horizon
ablation. That would distinguish model/data error from open-loop compounding
without changing the learned 27-D action representation.

## Artifacts

```text
outputs/act_e2e_smoke/act_train/checkpoints/000100/
outputs/act_e2e_smoke/act_train/checkpoints/000200/
outputs/act_e2e_smoke/act_train/checkpoints/000300/
outputs/act_e2e_smoke/act_train/checkpoints/000400/
outputs/act_e2e_smoke/act_train/checkpoints/000500/
outputs/act_e2e_smoke/rollouts/step_000100/
outputs/act_e2e_smoke/rollouts/step_000200/
outputs/act_e2e_smoke/rollouts/step_000300/
outputs/act_e2e_smoke/rollouts/step_000400/
outputs/act_e2e_smoke/rollouts/step_000500/
outputs/act_e2e_smoke/rollouts/step_000500_seeds_0_19/
outputs/act_e2e_smoke/rollouts/checkpoint_comparison.json
```
