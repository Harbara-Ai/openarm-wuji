# ACT phase balancing x chunk size 2 x 2 ablation

## 实验设计

本实验只改变两个因素：训练 observation-anchor sampling 和 ACT action chunk size。所有结果统一使用 step-500 checkpoint，并以 `H_exec = 1` 在 seeds 0--19 上闭环评测。

| ID | Sampling | chunk size | 来源 |
| --- | --- | ---: | --- |
| A | Normal | 100 | 现有 ImageNet baseline |
| B | Normal | 20 | 本轮从头训练 |
| C | Phase-balanced | 100 | 本轮从头训练 |
| D | Phase-balanced | 20 | 现有最新实验 |

保持不变：20 条 successful demonstrations、2,804 frames、27D actual qpos、27D absolute controller target、front+wrist 240 x 320 RGB、ResNet-18 ImageNet 初始化、batch size 8、AdamW、learning rate `1e-5`、training seed 1000、预处理、MuJoCo、grasp/controller/task gates 和 rollout seeds。

B、C 的 `pretrained_path = null`，均从 ImageNet-initialized backbone + 随机初始化 ACT head 从 step 0 开始，没有从 A 或 D resume。B、C 保存 step 100、200、300、400、500 checkpoint。

## Phase-balanced sampling

C 与 D 使用同一 deterministic exact-quota sampler：

| Phase | 目标比例 | 500 steps 实际比例 |
| --- | ---: | ---: |
| Reach | 25% | 25.70% |
| Approach | 25% | 24.80% |
| Grasp | 20% | 19.75% |
| Preload | 10% | 9.85% |
| Lift | 10% | 9.98% |
| Hold | 10% | 9.93% |

每个 run 消费 4,000 个 observation anchors。phase sampling 只改变 anchor 分布，不修改 action target、normalization 或 loss。

## Clipping 诊断语义

每帧保存的 policy predicted action 与 position controller 实际接收的 bounded target 做差：

```text
clip_correction = abs(predicted_action - sent_action)
```

统计包括全局 count/sum/mean/p95/max、27 个 joint 的独立统计以及两种 phase 视角：

- `achieved_gate_stage`：由实际 Reach/Approach/Grasp/Lift gate 决定；如果 Reach 未通过，全部帧都属于 Reach；
- `nominal_expert_time_phase`：20 条 expert demonstration 在同一 frame index 上的 modal phase，仅用于判断 clipping 是否集中在预期动作时段，不声称失败 policy 真正完成了该 phase。

expert-time modal timeline 为 Reach 12、Approach 41、Grasp 24、Preload 12、Lift 36、Hold 35 frames。clipping threshold 为 `1e-9 rad`。

<!-- RESULTS_START -->
## 核心 2 x 2 结果

| Sampling | chunk100 | chunk20 |
| --- | --- | --- |
| Normal | **A**: Reach 0/20; pregrasp 28.49 mm; Reach-arm MAE 0.11419; cube median 139.02 mm; clip 2,786 / 162.92 rad | **B**: Reach 0/20; pregrasp 16.28 mm; Reach-arm MAE 0.04003; cube median 18.26 mm; clip 2,442 / 80.01 rad |
| Phase-balanced | **C**: Reach 0/20; pregrasp 20.12 mm; Reach-arm MAE 0.05815; cube median 26.88 mm; clip 2,762 / 146.46 rad | **D**: Reach 0/20; pregrasp 15.45 mm; Reach-arm MAE 0.02369; cube median ~0 mm; clip 2,920 / 90.80 rad |

这里 `clip count / magnitude` 分别表示 clipped scalar values 数量和累计绝对修正量。四格都没有 binary Reach，因此连续指标只用于排序和归因，不能替代成功率。

## 完整 closed-loop 指标

| Metric | A Normal-100 | B Normal-20 | C Balanced-100 | D Balanced-20 |
| --- | ---: | ---: | ---: | ---: |
| Valid Reach | 0/20 | 0/20 | 0/20 | 0/20 |
| Valid Approach | 0/20 | 0/20 | 0/20 | 0/20 |
| Task success | 0/20 | 0/20 | 0/20 | 0/20 |
| Mean min pregrasp error | 28.49 mm | 16.28 mm | 20.12 mm | **15.45 mm** |
| Median min pregrasp error | 28.50 mm | **15.27 mm** | 20.74 mm | 16.67 mm |
| Best min pregrasp error | 23.22 mm | 9.28 mm | 10.20 mm | **2.49 mm** |
| Mean min approach error | 63.71 mm | 14.24 mm | **13.41 mm** | 51.05 mm |
| Median max cube displacement | 139.02 mm | 18.26 mm | 26.88 mm | **~0 mm** |
| Mean max cube displacement | 121.51 mm | 42.05 mm | 109.06 mm | **3.81 mm** |
| Maximum cube displacement | 205.08 mm | 267.89 mm | 1,226.56 mm | **74.19 mm** |
| Episodes >25 mm displacement | 17/20 | 6/20 | 11/20 | **1/20** |
| Any contact | 14/20 | 20/20 | 20/20 | 1/20 |
| >=2 simultaneous contacts | 10/20 | 20/20 | 20/20 | 1/20 |
| Persistent >=2 contacts, >=8 frames | 3/20 | 0/20 | 2/20 | 0/20 |
| Maximum simultaneous contacts | 4 | 4 | 5 | 2 |

B/C 的 contact 全部发生在 valid Reach 之前，因此属于 early contact。C 的 mean cube displacement 被一个约 1.23 m 的极端推块/掉出台面 seed 拉高；中位数和 `>25 mm` 数也仍比 D 差，所以结论不依赖该离群值。

D 的 mean/best pregrasp、cube safety 和离线 Reach error 最好，但 median pregrasp 略逊于 B。D 的 approach error 较高不能解释为有效 Approach failure，因为四格都没有先通过 Reach；在 ordered gate 下没有任何模型真正进入 Approach。

## Offline H=1 reconstruction

| Metric | A Normal-100 | B Normal-20 | C Balanced-100 | D Balanced-20 |
| --- | ---: | ---: | ---: | ---: |
| Reach arm MAE | 0.11419 | 0.04003 | 0.05815 | **0.02369 rad** |
| Reach hand MAE | 0.03798 | 0.02611 | 0.01887 | **0.01404 rad** |
| Approach arm MAE | 0.03847 | 0.02904 | 0.03796 | **0.02232 rad** |
| Frame-0 arm spread / expert spread | 67.49% | 96.78% | 89.08% | 124.38% |
| Frame-0 paired arm MAE | 0.10569 | **0.02522** | 0.05629 | 0.02965 rad |

四格的 padding mask、episode boundary 和 raw action round-trip 检查全部通过。四格 frame-0 output 都随图像变化；B/C/D 的 spread 接近或超过 expert spread，不存在视觉分支退化为 image-invariant 的证据。D 超过 100% 表示视觉响应偏大，不表示优于 expert。

## Factorial effects

以下 effect 用四格均值计算；对 error/displacement/clipping magnitude，负数表示改善：

| Metric | chunk20 - chunk100 | balanced - normal | interaction `D-C-B+A` |
| --- | ---: | ---: | ---: |
| Valid Reach count | 0 | 0 | 0 |
| Mean pregrasp error | **-8.44 mm** | -4.60 mm | +7.55 mm |
| Reach arm H=1 MAE | **-0.05431 rad** | -0.03619 rad | +0.03970 rad |
| Mean cube displacement | **-92.35 mm** | -25.34 mm | -25.80 mm |
| Clipping count | -93 | +227 | +502 |
| Clipping cumulative magnitude | **-69.28 rad** | -2.83 rad | +27.24 rad |

两个因素都独立改善 pregrasp 与 Reach-arm MAE，但 combination 的收益小于简单相加，因此在这些连续 Reach 指标上存在 antagonistic/diminishing-return interaction。这个 interaction 没有让 D 比 B 或 C 更差：D 仍拥有最低 mean pregrasp、最低 Reach-arm MAE 和最低 cube displacement。因此不能把整体结论定为“组合有害”。

## Action clipping 定位

| Metric | A Normal-100 | B Normal-20 | C Balanced-100 | D Balanced-20 |
| --- | ---: | ---: | ---: | ---: |
| Clipped values | 2,786 | **2,442** | 2,762 | 2,920 |
| Sum abs correction | 162.92 | **80.01** | 146.46 | 90.80 rad |
| Mean correction when clipped | 0.0585 | 0.0328 | 0.0530 | **0.0311 rad** |
| P95 correction | 0.0758 | 0.0627 | 0.0897 | **0.0515 rad** |
| Maximum correction | 0.0936 | 0.0868 | 0.1069 | **0.0722 rad** |

四格的全部 clipping 都来自同一个 joint：

```text
zero-based action index 7
wuji_left_finger1_joint1
controller ctrlrange = [0.0475, 1.603] rad
direction = policy target below lower bound
```

其余 26 个 joint 的 clipping count 全部为 0。按 nominal expert-time phase 的 `count / sum_abs_rad`：

| Phase | A Normal-100 | B Normal-20 | C Balanced-100 | D Balanced-20 |
| --- | ---: | ---: | ---: | ---: |
| Reach | 232 / 13.06 | 173 / 7.52 | 219 / 8.94 | 213 / 5.94 |
| Approach | 725 / 42.14 | 634 / 21.50 | 720 / 37.80 | 759 / 24.99 |
| Grasp | 418 / 24.60 | 374 / 12.11 | 407 / 21.07 | 447 / 13.79 |
| Preload | 204 / 11.96 | 187 / 5.86 | 208 / 12.24 | 214 / 6.55 |
| Lift | 612 / 36.26 | 532 / 16.40 | 620 / 33.35 | 659 / 20.40 |
| Hold | 595 / 34.90 | 542 / 16.62 | 588 / 33.06 | 628 / 19.13 |

因为四格 Reach 均为 0/20，按 `achieved_gate_stage` 定义全部 3,200 frames 都仍属于 Reach。nominal phase 表仅用于说明 lower-bound mismatch 贯穿整个 160-frame rollout，并非只在 Grasp/Lift 爆发。

结论需要区分 count 与 severity：

- D 的 2,920 次高 count 确实包含 `phase-balanced x chunk20` 的正 interaction（+502 counts）；
- 但 chunk20 把累计修正幅度平均降低 69.28 rad，是 clipping severity 的主导改善因素；
- phase balancing 对累计幅度的平均主效应只有 -2.83 rad；
- 所以“2,920 次”不是 chunk20 导致的。clipping 是四格共有的 thumb lower-bound calibration 问题，组合主要提高了小幅越界的频率。

## 必答问题

### Q1：chunk100 -> chunk20 单独是否改善 Reach？

**没有改善 binary Reach，仍为 0/20；但显著改善连续指标。** 在 normal sampling 下，mean pregrasp 从 28.49 降到 16.28 mm，Reach-arm MAE 从 0.11419 降到 0.04003 rad，`>25 mm` 推块从 17/20 降到 6/20，clipping magnitude 从 162.92 降到 80.01 rad。

### Q2：normal -> phase-balanced 单独是否改善 Reach？

**没有改善 binary Reach，仍为 0/20；但也独立改善连续指标。** 在 chunk100 下，mean pregrasp 从 28.49 降到 20.12 mm，Reach-arm MAE 从 0.11419 降到 0.05815 rad，cube displacement 中位数从 139.02 降到 26.88 mm。

### Q3：phase-balanced + chunk20 是否存在负 interaction？

**存在收益递减和 clipping-count interaction，但不存在让组合整体变差的 harmful interaction。** D 仍在最关键的 Reach-arm MAE、mean/best pregrasp 和 cube safety 上优于 B/C。

### Q4：哪一个配置 closed-loop 最好？

**D：phase-balanced + chunk20。** 它仍是 0/20 Reach，所以这里只是四个失败模型中的最佳候选，不是成功配置。

### Q5：哪个配置最适合继续到 2k / 5k？

**D。** 先只继续到 step 2k，在 step 1k/2k 做同样 20-seed gate；不要直接承诺 5k。500 steps 只相当于约 1.43 个 sampled dataset epochs，且 D 的 loss/MAE 到 step500 仍在下降，因此训练时长尚未被充分排除。

### Q6：action clipping 主要与哪个变量相关？

**根因不属于这两个 ablation 变量，而是 thumb joint1 的 absolute-target lower-bound mismatch。** 若只比较两个因素，chunk20 明显降低修正幅度；phase balancing 主要增加越界次数，尤其与 chunk20 组合时，但不是主要 severity 来源。

## 强制决策

```text
Case E:
None of the four configurations solves Reach;
training duration/data coverage is now the primary suspect.

best_config = phase-balanced sampling + chunk_size 20 + n_action_steps 20
next = continue training best_config to step 2k
```

选择 Case E 是因为唯一核心验收指标 `Valid Reach` 四格全部为 0/20。虽然两个因素都改善连续指标，但这不足以选择 Case C 作为最终结论。

下一步保持 D 的数据与接口不变续训到 step 2k，并在 step 1k/2k 复测。如果 step 2k 仍为 0/20，或 minimum pregrasp 不再改善，则停止加 steps，转为增加覆盖 cube pose / Reach correction 的 demonstrations。本轮不修改 clipping/controller；否则会破坏 2 x 2 的因果解释。

## 产物

- B checkpoint：`outputs/act_phase_chunk_2x2/normal_chunk20/act_train/checkpoints/000500/pretrained_model`
- C checkpoint：`outputs/act_phase_chunk_2x2/phase_chunk100/act_train/checkpoints/000500/pretrained_model`
- B rollout：`outputs/act_phase_chunk_2x2/normal_chunk20/rollouts/step_000500_seeds_0_19_h001/summary.json`
- C rollout：`outputs/act_phase_chunk_2x2/phase_chunk100/rollouts/step_000500_seeds_0_19_h001/summary.json`
- C sampling report：`outputs/act_phase_chunk_2x2/phase_chunk100/phase_sampling_report.json`
- unified machine-readable result：`outputs/act_phase_chunk_2x2/summary.json`
<!-- RESULTS_END -->
