# Reward V3 contact-quality 5K 实验

## 结论

本轮只改变 reward，保持 `expert_pca5_absolute`、5D latent action、228D observation、fixed palm、rigid cube、reset、termination、success gate、SAC 超参数、PD 与 rate limit 不变。

Reward V3 通过了训练前 ranking sanity，但 fresh 5K 没有让策略从“瞬时多指接触”转向“持续、低滑移、非边缘接触”。它保住了 3–4 指探索，却没有提高平均 edge margin，也没有降低 contact-level tangential slip；`>=3` 指连续保持反而缩短。因此按预先约定的 stop rule，本实验不继续到 10K。

## 实现

Reward V3 在完整 Reward V2 上只增加：

```text
+ 0.30 * contact_interior
+ 0.30 * contact_persistence
- 0.20 * contact_slip
```

- `contact_interior`：每根手指先按 normal force 聚合同指多个 contact，再使用 `1-exp(-d_edge/0.003)`；最后除以 3 并裁剪到 `[0,1]`。一个 finger 最多贡献 `1/3`。
- `contact_persistence`：每根手指连续有效接触时间除以 `0.30 s` 后裁剪，再求和除以 3。接触丢失即清零；一个 finger 最多贡献 `1/3`。
- `contact_slip`：接触点 finger-body 相对 cube 的速度移除法向分量后得到切向速度，使用 `1-exp(-(v_tan/0.01)^2)`，跨有效手指平均并裁剪到 `[0,1]`。
- valid contact 沿用原定义：normal force `>=0.10 N`，没有重新限制指定 fingertip、指定 cube face，也没有禁止合理 phalanx contact。
- 所有权重与尺度均在 V3 config 中；contact-quality 只进入 `info` telemetry 和 reward，不进入 228D policy observation。

新增 per-step telemetry：每根手指的 valid contact、连续时长、normal force、切向速度、最近 face、edge margin、region、geom、contact point；以及 aggregate contact count、interior count/fraction、edge margin、slip、persistence 和三个 Reward V3 原始项。

## 训练前 reward ranking

使用相同 reset/physics，把当前 V2 5K deterministic policy、scripted closing、raw expert、PCA5 expert reconstruction、EP74 以及 no-contact/thumb-only probe 分别按 V2/V3 重算。这里的 V2 checkpoint 是用完全相同 config、seed 7、5000 steps 重建此前未成功落盘的 artifact，不是继续训练。

全部 gate 通过：

- contact trajectory 高于 no-contact；
- quality increment 最优轨迹高于 EP74 unstable reference；
- matched 3-finger interior+low-slip score 为 `+0.5574`，matched edge+high-slip 为 `-0.1366`；
- matched 3-finger quality 高于 thumb-only 的 `+0.1845`；
- scripted multi-contact 高于 thumb-only；
- V3/V2 absolute return 中位比为 `1.0261`，仍处于同量级。

当前 V2 policy seed 7 的同一轨迹由 V2 `35.371` 变为 V3 `34.642`，说明 V3 没有无条件抬高分数，而是在扣除现有 edge/slip 行为。

## Fresh 5K 训练

两次训练均为 seed 7、41 个完整 episode、5000 environment steps：

| 指标 | PCA5 + V2 | PCA5 + V3 |
|---|---:|---:|
| max contact fingers | 4 | 4 |
| any-contact episode fraction | 1.000 | 1.000 |
| multi-contact episode fraction | 1.000 | 1.000 |
| `>=3` contact episode fraction | 0.732 | 0.732 |
| success events | 0 | 0 |
| final entropy alpha | 0.23022 | 0.23008 |
| last 250-step mean reward | 0.1433 | 0.0983 |

V3 训练 episode 中三个新增 weighted component 的均值分别为：interior `+2.390`、persistence `+0.712`、contact slip `-6.761`。slip 项是有影响的，但没有导致 contact avoidance：any/multi/three-contact 频率与 V2 相同。

## Deterministic seeds 7–11 配对结果

下表中的 edge/slip 是所有有效 finger-step contact 的统计。`interior fraction` 使用 telemetry 的 contact-region 分类；连续保持在 30 Hz 下按完整帧计数。

| 指标（5 seeds aggregate） | V2 | V3 | 变化 |
|---|---:|---:|---:|
| seeds with `>=2` contacts | 5/5 | 5/5 | 保持 |
| seeds with `>=3` contacts | 5/5 | 5/5 | 保持 |
| mean max contacts | 3.6 | 3.6 | 0 |
| mean edge margin | 3.111 mm | 3.021 mm | -2.9%（变差） |
| interior contact fraction | 34.94% | 35.37% | +0.43 pp（近似不变） |
| mean tangential slip | 7.821 mm/s | 7.995 mm/s | +2.2%（变差） |
| mean max-contiguous `>=2` | 0.153 s | 0.160 s | +0.007 s |
| mean max-contiguous `>=3` | 0.073 s | 0.047 s | -36.4% |
| `>=2` hold 0.1 / 0.3 / 0.5 s | 5 / 0 / 0 | 5 / 0 / 0 | 无新 milestone |
| `>=3` hold 0.1 / 0.3 / 0.5 s | 1 / 0 / 0 | 0 / 0 / 0 | 0.1 s 退化 |
| stable grasp success | 0/5 | 0/5 | 无改善 |

每个 seed：

| Ver. | Seed | max C | `>=2` contig | `>=3` contig | edge mean | slip mean | first loss after first `>=3` |
|---|---:|---:|---:|---:|---:|---:|---|
| V2 | 7 | 3 | 0.133 s | 0.067 s | 2.982 mm | 7.868 mm/s | finger3 |
| V2 | 8 | 4 | 0.200 s | 0.100 s | 2.923 mm | 7.705 mm/s | finger1 |
| V2 | 9 | 3 | 0.100 s | 0.067 s | 3.200 mm | 7.944 mm/s | finger1+finger3+finger4 |
| V2 | 10 | 4 | 0.133 s | 0.067 s | 3.080 mm | 7.687 mm/s | finger3 |
| V2 | 11 | 4 | 0.200 s | 0.067 s | 3.368 mm | 7.899 mm/s | finger3 |
| V3 | 7 | 4 | 0.200 s | 0.067 s | 3.158 mm | 8.490 mm/s | finger1+finger2 |
| V3 | 8 | 4 | 0.133 s | 0.033 s | 3.097 mm | 7.521 mm/s | finger1+finger3 |
| V3 | 9 | 4 | 0.133 s | 0.033 s | 2.807 mm | 7.917 mm/s | finger2+finger3 |
| V3 | 10 | 3 | 0.200 s | 0.067 s | 3.037 mm | 8.634 mm/s | finger2 |
| V3 | 11 | 3 | 0.133 s | 0.033 s | 3.006 mm | 7.415 mm/s | finger2+finger4 |

所有 seed 的最小 edge margin 均为 0 mm：edge/corner contact 仍然存在。V3 把 corner-region finger-frames 从 37 降到 22，但 edge-region frames 从 436 增至 444，不能据此宣称 edge/corner 问题显著减少。

## Finger3 / finger4

| 指标（5-seed mean） | Finger | V2 | V3 |
|---|---|---:|---:|
| total contact duration | finger3 | 0.960 s | 1.020 s |
| max contiguous contact | finger3 | 0.087 s | 0.100 s |
| total contact duration | finger4 | 1.053 s | 1.220 s |
| max contiguous contact | finger4 | 0.140 s | 0.140 s |

finger3/4 的累计接触量有轻微改善；finger3 首次掉指参与率从 4/5 降到 2/5，finger4 仍是 1/5。然而失稳拓扑迁移到了 finger2：V3 中 finger2 参与首次掉落 4/5，V2 为 0/5。由于 `>=3` 指整体持续时间下降，这不是稳定抓取改善，只是 failure topology 发生转移。

## Stability 与 safety

| 指标（5-seed mean） | V2 | V3 |
|---|---:|---:|
| relative linear speed during `>=2` | 1.063 mm/s | 1.013 mm/s |
| relative angular speed during `>=2` | 1.365 deg/s | 1.270 deg/s |
| max window translation drift | 0.0435 mm | 0.0406 mm |
| max window rotation drift | 0.0601 deg | 0.0406 deg |
| deepest penetration | 0.770 mm | 0.610 mm |
| penetration exploit count | 0 | 0 |
| max cube displacement | 0.460 mm | 0.485 mm |

物体-掌心整体运动与 penetration 略有改善，但它们没有转化为 0.3 s multi-contact hold。当前问题依然是 contact topology 快速 collapse，而不是 cube/palm 的宏观 SE(3) 爆炸。

## 对六个问题的回答

1. **是否转向持续低滑移接触？** 否。`>=2` 连续时长只增加 7 ms，`>=3` 连续时长下降，平均切向滑移上升 2.2%。
2. **edge/corner 是否显著减少？** 否。corner frame 减少，但平均 edge margin 下降 2.9%，edge frame 增加，所有 seed 都仍出现 0 mm margin。
3. **finger3/finger4 persistence 是否改善？** 单指累计时长有小幅改善，finger3 contiguous 从 87 ms 到 100 ms，finger4 contiguous 不变；没有形成更稳定的三指 topology。
4. **0.3/0.5 s hold 是否出现？** 没有，V2/V3 均为 0/5。
5. **当前瓶颈是什么？** 20D 独立探索瓶颈已经由 PCA5 解决；现在证据指向 PCA5 manifold 中可用稳定姿态不足和/或 fixed-palm rigid-cube 几何可达性。仅凭本次 reward ablation 无法在两者中二选一，但继续 reward tuning 的优先级已经很低。
6. **是否值得继续到 10K？** 否。明确触发 stop rule。

## 下一步（不在本轮实现）

先做无训练的 reachability/manifold audit：在 PCA5 latent space 内搜索可达 posture，比较其最佳三指 edge margin、切向速度和连续接触；再用同一 fixed palm 的 unrestricted 20D/expert postures 做上界。如果 unrestricted 能稳定而 PCA5 不能，瓶颈是 manifold；两者都不能，则优先处理 fixed-palm geometry / rigid contact dynamics，而不是再调 reward。

## 产物

- `outputs/reward_v3_contact_quality/reward_ranking.json`
- `outputs/reward_v3_contact_quality/reward_component_stats.json`
- `outputs/reward_v3_contact_quality/sac_5k_metrics.json`
- `outputs/reward_v3_contact_quality/deterministic_seed_7_11.json`
- `outputs/reward_v3_contact_quality/contact_quality_timeseries/{v2,v3}/seed_7..11.json`

SAC checkpoints 是实验二进制 artifact，保存在本机 Codex visualization workspace；上述仓库 JSON 已记录 checkpoint 来源、配置和完整评估数据。
