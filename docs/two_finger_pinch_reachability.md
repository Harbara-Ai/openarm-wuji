# Fixed-palm thumb-index pinch reachability

## 结论

在当前 fixed-palm、free rigid cube、MuJoCo contact/friction、PD controller、joint-rate limit、penetration rule 和 seed-7 reset geometry 下，本轮没有找到稳定的 thumb + index pinch。

搜索共评估 1961 个不同的 8D thumb/index joint target。没有候选把双指接触连续保持到 0.2 s，更没有达到要求的 0.5 s。最佳具有几何 opposition 的近失解只连续保持 0.1333 s，并且平均切向滑移为 13.99 mm/s。因此不能把它报告为 stable pinch。

这支持当前诊断：在这套 fixed-palm geometry 和 rigid-contact setup 下，连最简单的 thumb-index stable pinch 都没有被 bounded search 找到；继续训练 two-finger SAC 不值得。下一步应先检查 palm-object 相对位姿、可达接触面和接触模型，而不是让 RL 搜索一个尚未证明存在的稳定解。

## 实验约束

本轮没有加载 policy、没有更新 reward，也没有训练 RL。搜索脚本直接调用当前 `WujiStaticGraspEnv.step_absolute_target()`：

- palm 保持 fixed；
- cube 保持当前自由刚体和当前桌面支撑；
- friction/contact、PD、joint limits、每步 joint-rate limit、reset 和失败规则均来自 `configs/rl/grasp_stage1_expert_pca5_reward_v3.json`；
- 只有 `finger1/thumb` 与 `finger2/index` 的 8 个 target joint 可变；
- middle、ring、little 始终保持配置中的 open pose `[0, ..., 0]`，且整个搜索中没有被动第三指接触；
- 每个候选都从相同 seed-7 reset 开始，给定同一个固定 target；先模拟 1.0 s 让 rate-limited PD 接近目标，再至少检查末端 0.5 s；前 30 个候选用 1.0 s 末端窗口复核；
- 最终最佳 target 原样用于 seeds 7–11，没有针对各 seed 重新优化。

搜索不是 SAC reward optimization。`diagnostic_objective` 只用于候选排序，优先级依次包含 strong success、合理 contact opposition、连续双接触、接触占比、低 slip/drift 和无 penetration exploit。

## 搜索空间与方法

搜索变量顺序为：

```text
[thumb_q1, thumb_q2, thumb_q3, thumb_q4,
 index_q1, index_q2, index_q3, index_q4]
```

所有范围均与 MuJoCo joint limits 取交集，实际搜索范围（rad）为：

| finger | q1 | q2 | q3 | q4 |
|---|---:|---:|---:|---:|
| thumb lower | 0.600 | -0.100 | 0.300 | 0.300 |
| thumb upper | 1.603 | 0.900 | 1.550 | 1.550 |
| index lower | 0.400 | -0.370 | 0.300 | 0.300 |
| index upper | 1.550 | 0.370 | 1.540 | 1.550 |

候选组成：

- 1 个已有 manual reference；
- 1600 个 8D Latin-hypercube 全局样本；
- 30 个几何 opposition 最好的 seed，各做 3 个尺度（0.15、0.075、0.035）的局部随机细化，每尺度 4 个候选，共 360 个。

总数为 `1 + 1600 + 360 = 1961`。

## Strong pinch 判定

本轮只把同时满足以下条件的候选称为 strong two-finger pinch：

1. thumb 和 index 有效接触连续 `>= 0.5 s`；
2. 末端窗口平均 tangential slip `< 0.010 m/s`；
3. cube-palm relative translation drift `< 8 mm`；
4. relative rotation drift `< 6 deg`；
5. penetration 不超过当前环境的 8 mm failure limit；
6. middle/ring/little 没有接触；
7. 两指作用在 cube 上的平均接触法向夹角 `> 90 deg`。

其中 8 mm / 6 deg 沿用项目中明确标记的 CD-WM external diagnostic，只是诊断 gate，不宣称是 Wuji 最终标准。10 mm/s 是当前 Reward V3 中已有的 contact-slip 尺度。连续指标全部保留在输出中。

## 全部候选的结果

| metric | result |
|---|---:|
| candidates | 1961 |
| 出现过至少 1 帧双接触 | 1658 |
| 连续双接触 `>= 0.1 s` | 56 |
| 连续双接触 `>= 0.2 s` | 0 |
| 连续双接触 `>= 0.5 s` | 0 |
| 平均 contact-normal angle `> 90 deg` | 591 |
| opposition 且连续双接触 `>= 0.1 s` | 16 |
| formal stable success | 0 |
| strong thumb-index pinch | 0 |

这些数值区分了“能同时碰到方块”和“能稳定对夹”：84.5% 的候选曾出现双接触，但没有一个维持到 0.2 s。

已有 manual reference：

```text
thumb = [1.300, 0.450, 1.100, 1.100]
index = [1.200, -0.080, 1.150, 1.100]
```

它的最长连续双接触仅 0.0333 s，平均法向夹角 86.91 deg，平均切向滑移 10.51 mm/s，最大 penetration 3.50 mm；它不是稳定 pinch。

## 最佳 opposition 近失解

最佳命令 joint target（rad）为：

```text
thumb = [1.127061, 0.560455, 0.877203, 0.947153]
index = [0.615979, 0.061319, 1.038071, 1.362935]
```

由于当前 controller 从实际 `q_current` 施加每步 rate limit，且接触载荷阻止若干 joint 继续运动，命令 target 并未被精确达到。2.0 s 结束时：

```text
actual thumb = [0.789197, 0.575864, 0.022574, 0.714395]
actual index = [0.025834, 0.075618, 0.780305, 1.372346]

applied PD thumb target = [0.855981, 0.560455, 0.090834, 0.779245]
applied PD index target = [0.096069, 0.061319, 0.833136, 1.362935]
```

这是保留当前 PD 与 rate limit 后的实际可达平衡，不应把 nominal target 冒充为真实到达姿态。

最佳近失解的接触与运动指标：

| metric | value |
|---|---:|
| longest continuous thumb+index contact | 0.1333 s |
| thumb contact time in 1 s window | 0.5667 s |
| index contact time in 1 s window | 0.3333 s |
| thumb cube face | `-Y` |
| index cube face | `+Z` |
| thumb normal on cube, cube frame | `[0.0008, 0.9999, 0.0133]` |
| index normal on cube, cube frame | `[-0.1924, 0.0067, -0.9813]` |
| mean normal opposition angle | 90.36 deg |
| mean / max tangential slip | 13.99 / 47.17 mm/s |
| relative translation drift | 0.103 mm |
| relative rotation drift | 0.039 deg |
| cube displacement | 0.182 mm |
| cube rotation | 0.064 deg |
| deepest penetration | 1.382 mm |
| penetration exploit | false |
| strong pinch | false |

低 cube motion 和低 SE(3) drift 不能覆盖接触持续性失败：此时 cube 仍由桌面支撑，双指 opposition 只短暂存在。

## Seeds 7–11 固定 target 验证

下表对所有 seed 使用完全相同的最佳命令 target：

| seed | longest dual contact (s) | normal angle (deg) | mean slip (mm/s) | SE(3) drift (mm / deg) | cube motion (mm / deg) | penetration (mm) | strong |
|---:|---:|---:|---:|---:|---:|---:|:---:|
| 7 | 0.1333 | 90.36 | 13.99 | 0.103 / 0.039 | 0.182 / 0.064 | 1.382 | no |
| 8 | 0.0333 | 91.28 | 11.57 | 0.102 / 0.043 | 0.177 / 0.064 | 1.199 | no |
| 9 | 0.0667 | 92.29 | 13.66 | 0.103 / 0.037 | 0.182 / 0.063 | 1.225 | no |
| 10 | 0.0333 | 92.96 | 12.32 | 0.105 / 0.040 | 0.182 / 0.065 | 1.163 | no |
| 11 | 0.0667 | 90.02 | 14.20 | 0.103 / 0.042 | 0.185 / 0.073 | 1.341 | no |

结果为 `0/5` strong success，平均最长双接触 0.0667 s，最大 0.1333 s。因此 seeds 7–11 没有表现出稳定 pinch 鲁棒性。

## 对五个问题的回答

1. **fixed palm 下是否存在稳定 thumb-index pinch？** 本轮 bounded search 没有找到；在已检查的 1961 个候选内答案是否定的，不能进一步声称数学上绝对不存在。
2. **最佳 posture 是什么？** 上述 nominal thumb/index target 是最佳具有 opposition 的近失解；实际终态也已单独列出。它不是成功姿态。
3. **能稳定保持多久？** seed 7 最长 0.1333 s，远低于 0.5 s。
4. **seeds 7–11 是否有一定鲁棒性？** 没有，strong success 为 0/5；最长双接触范围 0.0333–0.1333 s。
5. **下一步是否值得训练 two-finger SAC？** 目前不值得。应先改变并验证 palm-object geometry 或 rigid-contact setup，使无训练搜索至少出现一个 0.5 s stable pinch；之后 two-finger SAC 才有明确可学习目标。

## 可复现产物

- `scripts/two_finger_pinch_reachability.py`：无训练 8D bounded search；
- `outputs/two_finger_pinch/candidates.parquet`：1961 个候选及连续数值指标；
- `outputs/two_finger_pinch/top_candidates.json`：搜索配置、manual reference、前 30 个长窗口复核结果，前 10 个附逐帧 telemetry；
- `outputs/two_finger_pinch/multi_seed_validation.json`：固定最佳 target 的 seeds 7–11 结果。
