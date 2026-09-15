# ACT ResNet-18 ImageNet initialization ablation

## Question

Does initializing the existing ACT ResNet-18 visual backbone from ImageNet improve the current OpenArm + Wuji policy while every other experimental variable remains fixed?

## Controlled setup

The only intended change from the random-backbone baseline is:

```text
pretrained_backbone_weights:
  null
  -> ResNet18_Weights.IMAGENET1K_V1
```

The ImageNet run starts from a fresh ACT policy. Step 500 is not resumed from the random-initialization checkpoint. Step 1000 resumes the ImageNet step-500 checkpoint with optimizer, scheduler, random-number-generator, and dataloader position restored.

The following are unchanged:

- dataset: the same 20 successful demonstrations and 2,804 frames;
- observations: 27-D actual joint state plus `front` and `wrist` RGB at 240 x 320;
- actions: 27-D OpenArm + Wuji position-controller targets;
- cameras and preprocessing;
- ACT `chunk_size = 100` and `n_action_steps = 100`;
- batch size 8, training seed 1000, learning rate `1e-5`, and no image augmentation;
- MuJoCo model, reset, grasp controller, task gates, and rollout code;
- evaluation seeds 0--19, 160 control frames, and execution horizon 1.

The ImageNet checkpoint config records `ResNet18_Weights.IMAGENET1K_V1`. The official torchvision ResNet-18 weights were downloaded and cached locally.

## Training

| Checkpoint | Total loss | L1 loss | KL loss |
| --- | ---: | ---: | ---: |
| ImageNet step 500 | 2.236 | 0.173 | 0.206 |
| ImageNet step 1000 | 1.563 | 0.133 | 0.143 |

The loss decreases normally. This alone is not treated as closed-loop success.

## Offline action reconstruction

All values below are mean absolute error in radians. The random baseline is its step-500 checkpoint because that is where the existing phase-balanced and reset-conditioning diagnostics were recorded.

| Diagnostic | Random step 500 | ImageNet step 500 | ImageNet step 1000 |
| --- | ---: | ---: | ---: |
| All phases, H=1, 27-D | 0.04021 | 0.03278 | 0.01801 |
| All phases, H=1, arm 7-D | 0.04808 | 0.04366 | 0.02608 |
| Reach, H=1, 27-D | 0.07856 | 0.05773 | 0.02325 |
| Reach, H=1, arm 7-D | 0.11599 | 0.11419 | 0.06205 |
| Reach, H=1, hand 20-D | 0.06546 | 0.03798 | 0.00967 |

The padding and episode-boundary checks pass, and no cross-episode target is included in these errors.

## Frame-0 visual conditioning

`spread / expert` compares the RMS spread of the policy's first arm target across the 20 reset scenes against the RMS spread of the corresponding expert first targets.

| Diagnostic | Random step 500 | ImageNet step 500 | ImageNet step 1000 |
| --- | ---: | ---: | ---: |
| Paired frame-0 arm MAE (rad) | 0.11786 | 0.10569 | 0.06305 |
| Paired arm spread / expert | 2.97% | 67.49% | 93.50% |
| Vary images, fixed state | 2.80% | 67.49% | 93.57% |
| Vary state, fixed images | 1.54% | 0.17% | 0.23% |
| Vary front only | 0.15% | 58.06% | 81.63% |
| Vary wrist only | 2.82% | 11.43% | 16.08% |

This is the clearest positive result of the ablation. The random-backbone policy produces almost the same initial arm command for different images. The ImageNet policy instead responds strongly to scene variation, principally through the front camera. This diagnosis changes only policy inputs at inference; no expert action is passed to the policy.

## Matched step-1000 closed-loop rollouts

Both checkpoints are evaluated on exactly seeds 0--19 for 160 frames with H=1 replanning.

| Metric | Random init | ImageNet init |
| --- | ---: | ---: |
| Ordered Reach | 0/20 | 0/20 |
| Ordered Approach | 0/20 | 0/20 |
| Ordered multi-finger grasp | 0/20 | 0/20 |
| Task success | 0/20 | 0/20 |
| Episodes with any contact | 7/20 | 20/20 |
| Episodes reaching at least 2 simultaneous contacts | 1/20 | 20/20 |
| Episodes holding at least 2 contacts for 8 frames | 0/20 | 5/20 |
| Maximum simultaneous contacts | 2 | 5 |
| Mean minimum pregrasp error | 25.46 mm | 20.17 mm |
| Median minimum pregrasp error | 25.19 mm | 19.97 mm |
| Minimum pregrasp error over all seeds | 18.03 mm | 8.66 mm |
| Median maximum cube displacement | 35.13 mm | 17.16 mm |
| Episodes exceeding 25 mm cube displacement | 10/20 | 4/20 |
| Total clipped action values | 3,200 | 2,300 |
| Episodes with at least 150 clipped values | 20/20 | 12/20 |

The ImageNet mean maximum cube displacement is 57.16 mm versus 55.63 mm for random initialization, but this mean is dominated by ImageNet seed 11, where the cube leaves the workspace and reaches 804.04 mm displacement. The median and threshold count better describe the typical improvement, while the outlier is a serious safety failure and must not be hidden.

The new contacts do not count as successful grasps. They occur before the policy satisfies the ordered, sustained Reach gate. Thus the policy is often contacting or pushing the cube instead of completing `Reach -> Approach -> Grasp -> Lift`.

## Conclusion

ImageNet initialization is beneficial but not sufficient.

- It clearly breaks the prior near-image-invariant frame-0 behavior.
- It improves offline reconstruction, pregrasp proximity, contact richness, and typical cube displacement at the same training step.
- It does not produce a single valid Reach, Approach, grasp, or lift in 20 rollouts at step 1000.
- The seed-11 workspace-ejection outlier shows that the checkpoint is not safe for unrestricted rollout.

Therefore this ablation should be retained as the better visual initialization, but it must not be reported as an end-to-end solution. Blindly increasing training length is not yet justified by task success alone. A matched step-2000 continuation remains a clean optional follow-up; if Reach remains zero there, the next ablation should address phase imbalance or temporal/initial-action supervision while keeping ImageNet initialization.

## Artifacts

- ImageNet checkpoint: `outputs/act_imagenet_ablation/act_train/checkpoints/001000/pretrained_model`
- ImageNet rollout summary: `outputs/act_imagenet_ablation/rollouts/step_001000_seeds_0_19_h001/summary.json`
- Matched random rollout summary: `outputs/act_e2e_smoke/rollouts/step_001000_seeds_0_19_h001/summary.json`
- Offline metrics: `outputs/act_imagenet_ablation/diagnosis/offline_chunk_step_001000/offline_chunk_metrics.json`
- Reset-conditioning metrics: `outputs/act_imagenet_ablation/diagnosis/reset_conditioning_step_001000/reset_conditioning_summary.json`
- Training logs: `outputs/act_imagenet_ablation/train_000000_to_000500.log` and `outputs/act_imagenet_ablation/train_000500_to_001000.log`
