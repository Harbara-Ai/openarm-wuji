# ACT phase-balanced sampling + chunk20：续训至 step 500

## 结论

保持 step-200 实验的全部配置不变并精确恢复 optimizer、RNG 和 sampler stream 后，模型已续训到 step 500。结果是：

- `Reach success = 0/20`，`Approach success = 0/20`，`task success = 0/20`；
- Reach arm H=1 MAE 从 step 200 的 0.06992 rad 降到 0.02369 rad，下降 66.1%；
- overall H=1 MAE 从 0.04905 rad 降到 0.01959 rad，下降 60.1%；
- minimum pregrasp error 均值从 17.08 mm 小幅降到 15.45 mm，最佳值从 5.31 mm 降到 2.49 mm，但没有任何 seed 连续 5 帧通过 Reach gate；
- 过早接触和推块显著减少：超过 25 mm cube displacement 的 episode 从 15/20 降到 1/20，中位数从 72.61 mm 降到近似 0；
- 同时也不再形成有效抓取接触：仅 1/20 出现 contact，且 seed 14 的两指接触只持续 1 帧并推动方块 74.19 mm；
- action clipping 从 355 个值增加到 2,920 个值，说明闭环动作仍有明显的 joint-limit / distribution mismatch。

因此，续训明显改善了离线动作重建并抑制了大多数过早接触，但仍没有解决 closed-loop Reach。不能把 loss 或离线 MAE 的下降解释为 pipeline 已跑通。

## 配置与恢复语义

本轮没有改数据、grasp、controller、action semantics 或评测：

- 20 条 successful demonstrations，2,804 frames；
- observation state / action：27D；
- front + wrist RGB：240 x 320，30 FPS；
- ResNet-18 ImageNet 初始化；
- phase-balanced sampling；
- `chunk_size = n_action_steps = 20`；
- batch size 8、AdamW、learning rate `1e-5`、seed 1000；
- inference `H_exec = 1`；
- rollout seeds 0--19。

训练从 `000200` checkpoint 以 `resume=true` 恢复。step 500 checkpoint 同时包含 policy、optimizer、RNG 和 training step state。sampler 从第 `200 x 8 = 1,600` 个样本继续，而不是重新从 epoch 开头取样。

## 实际 phase sampling

step 200 到 500 的 2,400 个新增样本：

| Phase | 样本数 | 实际比例 |
| --- | ---: | ---: |
| Reach | 618 | 25.75% |
| Approach | 584 | 24.33% |
| Grasp | 478 | 19.92% |
| Preload | 239 | 9.96% |
| Lift | 237 | 9.88% |
| Hold | 244 | 10.17% |

从 step 0 累计到 step 500 的 4,000 个样本比例为 `25.70 / 24.80 / 19.75 / 9.85 / 9.98 / 9.93%`，与目标 `25 / 25 / 20 / 10 / 10 / 10%` 一致。

## 训练

| Step | Total loss | L1 loss | KL loss |
| ---: | ---: | ---: | ---: |
| 200 | 3.308 | 0.235 | 0.307 |
| 250 | 2.911 | 0.212 | 0.270 |
| 300 | 2.807 | 0.193 | 0.261 |
| 350 | 2.610 | 0.177 | 0.243 |
| 400 | 2.454 | 0.167 | 0.229 |
| 450 | 2.362 | 0.165 | 0.220 |
| 500 | 2.215 | 0.144 | 0.207 |

## 固定 seeds 0、7、11

| Seed | Reach | Approach | Task | min pregrasp | min approach | max cube displacement | max contacts | clipped values |
| ---: | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| 0 | false | false | false | 13.00 mm | 59.34 mm | ~0 mm | 0 | 159 |
| 7 | false | false | false | 18.37 mm | 49.03 mm | ~0 mm | 0 | 151 |
| 11 | false | false | false | 6.80 mm | 84.38 mm | 0.03 mm | 0 | 149 |

seed 11 曾接近 pregrasp threshold，但没有连续保持足够帧，随后轨迹也没有进入有效 Approach。

## Seeds 0--19 rollout

| Metric | step 200 | step 500 |
| --- | ---: | ---: |
| Valid Reach | 0/20 | 0/20 |
| Valid Approach | 0/20 | 0/20 |
| Ordered multi-finger grasp | 0/20 | 0/20 |
| Task success | 0/20 | 0/20 |
| Any contact | 20/20 | 1/20 |
| At least 2 simultaneous contacts | 18/20 | 1/20 |
| At least 2 contacts held >=8 frames | 13/20 | 0/20 |
| Maximum simultaneous contacts | 5 | 2 |
| Mean / median minimum pregrasp error | 17.08 / 16.59 mm | 15.45 / 16.67 mm |
| Best minimum pregrasp error | 5.31 mm | 2.49 mm |
| Mean / median minimum approach error | 28.75 / 26.69 mm | 51.05 / 48.92 mm |
| Mean / median maximum cube displacement | 72.84 / 72.61 mm | 3.81 / ~0 mm |
| Episodes exceeding 25 mm cube displacement | 15/20 | 1/20 |
| Maximum cube displacement | 157.81 mm | 74.19 mm |
| Total clipped action values | 355 | 2,920 |

step 500 唯一发生 cube contact 的 seed 14 达到 finger2 + finger4 两指接触，但只保持 1 帧，cube displacement 为 74.19 mm。这是碰撞推块，不是 grasp。

## Offline reconstruction

以下均使用 deterministic inference path，ground-truth action 没有输入 VAE encoder。

| Metric | step 200 | step 500 | 变化 |
| --- | ---: | ---: | ---: |
| Overall H=1, 27D | 0.04905 | 0.01959 rad | -60.1% |
| Overall H=1, arm 7D | 0.04165 | 0.02206 rad | -47.0% |
| Overall H=1, hand 20D | 0.05164 | 0.01872 rad | -63.7% |
| Reach H=1, arm 7D | 0.06992 | 0.02369 rad | -66.1% |
| Reach H=1, hand 20D | 0.02448 | 0.01404 rad | -42.6% |
| Approach H=1, arm 7D | 0.05193 | 0.02232 rad | -57.0% |
| Approach H=1, hand 20D | 0.05310 | 0.02563 rad | -51.7% |

Padding mask、raw repeat-last action、state round-trip 和 episode boundary 检查全部通过；没有跨 episode target 进入 MAE。

## Frame-0 image conditioning

| Condition | arm action spread / expert spread |
| --- | ---: |
| Paired state + images | 124.38% |
| Vary images, fixed state | 124.39% |
| Vary state, fixed images | 0.32% |
| Vary front only | 107.12% |
| Vary wrist only | 47.38% |

视觉条件化仍然存在，并未退化为 image-invariant policy；但输出 spread 已超过 expert spread，因此“看图有反应”不等于输出已经校准。frame-0 paired arm MAE 从 step 200 的 0.03892 rad 降到 0.02965 rad。

## 判断与下一步

1. 原配置已经按要求精确续训到 step 500，checkpoint 可加载。
2. 训练 loss、Reach arm MAE 和 cube pushing 均明显改善。
3. 仍没有任何 valid Reach，因此不能继续盲目训练并期待离线 MAE 自动转化为闭环成功。
4. 根据 step-200 报告预先约定的决策规则，下一步应选择 **B：将 phase balancing 与 chunk20 拆开做 2 x 2 ablation**。同时应把 action clipping 作为 ablation 的必查指标。
5. 本轮不修改 demonstrations、grasp、reward、action interface，也不继续超过 step 500。

## 产物

- step-500 checkpoint：`outputs/act_phase_balanced_chunk20/act_train/checkpoints/000500/pretrained_model`
- continuation sampling report：`outputs/act_phase_balanced_chunk20/phase_sampling_report_step_000200_to_000500.json`
- training log：`outputs/act_phase_balanced_chunk20/train_000200_to_000500.log`
- rollout summary：`outputs/act_phase_balanced_chunk20/rollouts/step_000500_seeds_0_19_h001/summary.json`
- offline diagnostic：`outputs/act_phase_balanced_chunk20/diagnosis/offline_chunk_step_000500/offline_chunk_metrics.json`
- reset conditioning：`outputs/act_phase_balanced_chunk20/diagnosis/reset_conditioning_step_000500/reset_conditioning_summary.json`
