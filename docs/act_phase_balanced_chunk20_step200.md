# ACT phase-balanced sampling + chunk20：step 200 matched experiment

## 结论

这轮组合改动改善了 Reach 相关的连续指标，但 **没有解决 valid Reach failure**：

- 20 个评测 seed 中，`Reach success = 0/20`，因此 `Approach/task success = 0/20`；
- minimum pregrasp error 均值降至 17.08 mm，比现有 ImageNet chunk100 step 1000 的 20.17 mm 更低；
- Reach arm H=1 MAE 为 0.06992 rad，比现有 ImageNet chunk100 step 500 的 0.11419 rad 低 38.8%，但仍比训练到 step 1000 的 0.06205 rad 高 12.7%；
- 过早 closing/contact 更明显：18/20 达到至少两指接触、13/20 持续至少 8 帧，但这些 episode 全部没有先满足 Reach；
- 方块最大位移中位数达到 72.61 mm，15/20 超过 25 mm，说明策略经常通过接触推动方块，而不是先完成 pregrasp。

因此当前证据支持继续同一配置到 step 500 后再判断，但不支持现在增加 demonstrations，也不能把提前多指接触当成 grasp 改善。

## 实验约束

相对当前 ImageNet ACT，只改了两个变量：

1. 用 phase-balanced sampler 替换普通 episode-aware sampling；
2. `chunk_size` 和 `n_action_steps` 从 100 改为 20。

保持不变：

- 20 条 successful demonstrations、2,804 frames；
- 27D actual joint state；
- 27D absolute position-controller target；
- front + wrist RGB，240 x 320，30 FPS；
- ResNet-18 `ResNet18_Weights.IMAGENET1K_V1` 初始化；
- ACT 主体结构、batch size 8、AdamW、learning rate `1e-5`、seed 1000；
- MuJoCo、reset、grasp pipeline、controller 和 action semantics；
- rollout seeds 和 `H_exec = 1`。

本模型从 step 0 训练，checkpoint config 中 `pretrained_path = null`，没有从旧 ACT checkpoint resume。

## Phase-balanced sampling

原始 recorder phase 映射为六组：

```text
reach                         -> Reach
approach                      -> Approach
grasp_close                   -> Grasp
preload + preload_settle      -> Preload
lift                          -> Lift
hold                          -> Hold
```

采样器在每个 epoch 中按目标 quota 有放回采样，再用固定 seed 打乱。这样既不修改 dataset，也不修改 loss；只改变作为 observation anchor 的 frame 分布。

| Phase | 原始 frames | 原始比例 | 目标比例 | 200-step 实际样本 | 实际比例 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Reach | 267 | 9.52% | 25% | 410 | 25.63% |
| Approach | 797 | 28.42% | 25% | 408 | 25.50% |
| Grasp | 480 | 17.12% | 20% | 312 | 19.50% |
| Preload | 240 | 8.56% | 10% | 155 | 9.69% |
| Lift | 720 | 25.68% | 10% | 162 | 10.13% |
| Hold | 300 | 10.70% | 10% | 153 | 9.56% |

实际比例来自单进程、`num_workers=0` 时训练真实消费的前 `200 x 8 = 1,600` 个 sampler index，不是仅报告理论权重。

## Chunk20 与 padding

- dataset action target shape：`[B, 20, 27]`；
- policy output shape：`[B, 20, 27]`；
- inference 每帧重新 observe/predict，只执行第一个 action；
- 对全部 20 个 episode 的最后一帧做边界预检：第一项有效，后 19 项均为 padding，repeat-last action 完全一致；
- step-200 offline diagnostic 再次验证：padding mask 与 raw episode boundary 完全一致；
- dataset padded action 与 raw repeat-last action 最大误差为 0；
- state round-trip 最大误差为 0；
- 没有跨 episode target 进入 MAE。

## 训练

保存 checkpoint：step 50、100、150、200。step 150 是 `save_freq=50` 自动产生的额外恢复点。

| Step | Total loss | L1 loss | KL loss |
| ---: | ---: | ---: | ---: |
| 50 | 5.616 | 0.350 | 0.527 |
| 100 | 4.150 | 0.315 | 0.384 |
| 150 | 3.636 | 0.268 | 0.337 |
| 200 | 3.308 | 0.235 | 0.307 |

Loss 正常下降，但不作为 closed-loop 成功证据。

## 固定 seeds 0、7、11

| Seed | Reach | Approach | Task | min pregrasp | min approach | max cube displacement | max contacts | longest >=2 contacts | clipped values |
| ---: | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | false | false | false | 17.40 mm | 75.24 mm | 72.27 mm | 1 | 0 frames | 136 |
| 7 | false | false | false | 13.87 mm | 18.80 mm | 118.25 mm | 5 | 43 frames | 5 |
| 11 | false | false | false | 24.82 mm | 44.15 mm | 157.81 mm | 4 | 102 frames | 4 |

seed 7 出现过较大的 cube height 和五指接触，但它没有先通过 Reach，并伴随 118.25 mm 水平/空间位移，因此不能解释为成功 lift。

## Seeds 0--19 rollout

| Metric | phase-balanced chunk20 step 200 |
| --- | ---: |
| Valid Reach | 0/20 |
| Valid Approach | 0/20 |
| Ordered multi-finger grasp | 0/20 |
| Task success | 0/20 |
| Any contact | 20/20 |
| At least 2 simultaneous contacts | 18/20 |
| At least 2 contacts held >=8 frames | 13/20 |
| Maximum simultaneous contacts | 5 |
| Mean / median minimum pregrasp error | 17.08 / 16.59 mm |
| Best minimum pregrasp error | 5.31 mm |
| Mean / median minimum approach error | 28.75 / 26.69 mm |
| Mean / median maximum cube displacement | 72.84 / 72.61 mm |
| Episodes exceeding 25 mm cube displacement | 15/20 |
| Maximum cube displacement | 157.81 mm |
| Total clipped action values | 355 |

## Offline reconstruction at step 200

Ground-truth action 没有传给 ACT VAE encoder；以下是 deterministic inference path 的 absolute action MAE。

| Metric | MAE (rad) |
| --- | ---: |
| Overall H=1, 27D | 0.04905 |
| Overall H=1, arm 7D | 0.04165 |
| Overall H=1, hand 20D | 0.05164 |
| Reach H=1, arm 7D | 0.06992 |
| Reach H=1, hand 20D | 0.02448 |
| Approach H=1, arm 7D | 0.05193 |
| Approach H=1, hand 20D | 0.05310 |

## Frame-0 image conditioning

| Condition | arm action spread / expert spread |
| --- | ---: |
| Paired state + images | 69.82% |
| Vary images, fixed state | 69.84% |
| Vary state, fixed images | 0.26% |
| Vary front only | 58.26% |
| Vary wrist only | 29.88% |

ImageNet visual conditioning 没有退化回随机初始化时接近 image-invariant 的状态。当前 frame-0 paired arm MAE 为 0.03892 rad。

## 与现有 ImageNet chunk100 baseline 比较

现有 ImageNet run 没有保存 step-200 checkpoint，因此不存在严格相同 optimizer-step 的 chunk100 对照。下面同时列出可用的 step 500 和 step 1000，避免把训练步数差异隐藏起来。

| Metric | chunk100 normal step 500 | chunk100 normal step 1000 | chunk20 balanced step 200 |
| --- | ---: | ---: | ---: |
| Valid Reach | 0/20 | 0/20 | 0/20 |
| Mean minimum pregrasp error | 28.49 mm | 20.17 mm | 17.08 mm |
| Reach arm H=1 MAE | 0.11419 rad | 0.06205 rad | 0.06992 rad |
| Median max cube displacement | 139.02 mm | 17.16 mm | 72.61 mm |
| Episodes with displacement >25 mm | 17/20 | 4/20 | 15/20 |
| At least 2 simultaneous contacts | 10/20 | 20/20 | 18/20 |
| Sustained >=2 contacts | 3/20 | 5/20 | 13/20 |

相对旧 step 500，新配置用更少训练 step 就显著改善了 pregrasp error 和 Reach arm MAE。相对旧 step 1000，新配置的 pregrasp error 略好，但 Reach arm MAE 仍略差，并且 cube pushing 明显更严重。

## 最终回答

1. **是否改善 Reach？** 连续指标有改善，但 binary Reach success 没有改善。pregrasp 更近、Reach arm MAE 较早下降，但 Reach gate 仍为 0/20。
2. **step 200 是否出现至少一个 valid Reach？** 没有，0/20。
3. **Reach arm MAE 是否明显下降？** 相比现有 chunk100 step 500，从 0.11419 降至 0.06992 rad，下降 38.8%；但尚未优于旧 step 1000 的 0.06205 rad。
4. **是否仍存在过早 contact/closing？** 是，而且非常明显：20/20 有 contact、13/20 有持续多指接触，但没有任何 episode 先通过 Reach；15/20 推动方块超过 25 mm。
5. **下一步选择：A。** 保持所有设置不变，将同一配置继续到 step 500，然后再次执行相同 20-seed gate。理由是 step 200 已显示更高的样本效率，但训练仍明显不足，尤其是 Reach hand/Approach MAE。若 step 500 仍为 `Reach = 0/20` 或 cube pushing 不下降，再执行 B，将 phase balancing 和 chunk20 分开做 2 x 2 ablation。当前不选择 C；没有证据表明 demo 数量是此刻最先要改的变量。

## 产物

- sampler：`scripts/train_act_phase_balanced.py`
- checkpoint：`outputs/act_phase_balanced_chunk20/act_train/checkpoints/000200/pretrained_model`
- sampling report：`outputs/act_phase_balanced_chunk20/phase_sampling_report.json`
- rollout summary：`outputs/act_phase_balanced_chunk20/rollouts/step_000200_seeds_0_19_h001/summary.json`
- offline diagnostic：`outputs/act_phase_balanced_chunk20/diagnosis/offline_chunk_step_000200/offline_chunk_metrics.json`
- reset conditioning：`outputs/act_phase_balanced_chunk20/diagnosis/reset_conditioning_step_000200/reset_conditioning_summary.json`
- training log：`outputs/act_phase_balanced_chunk20/train_000000_to_000200.log`
