# Expert palm–hand coordination replay

本轮没有训练，也没有修改 SAC、Reward V2、stable-grasp gate、penetration rule、Lift 或 cube material。目标是验证“固定掌心破坏 expert arm–hand coordination”这一假设。

## 先说结论

当前公开 `wuji-pick-and-place` 数据不能合法恢复 expert palm/wrist SE(3)。Parquet 只有：

```text
observation.state: 54D
action:           54D
timestamp/frame/task metadata
```

数据说明只给出 `left_arm[0:7] + right_arm[7:14] + left_hand[14:34] + right_hand[34:54]`，并注明 arm 已从 degree 转成 rad、action 是 absolute joint target；`info.json` 的 `names` 为 `null`。数据包不包含 expert arm joint names、URDF/MJCF、base/tool transform、palm pose 或 FK recorder 字段。

所以本轮没有把 OpenArm 7D 当成 expert 7D，也没有把 expert arm 数值直接复制给 OpenArm。没有 expert FK 时，C/D 和 translation-only/rotation-only/full-6D 必须标记为 blocked，不能伪造 palm 位移或旋转统计。

## 数据审计

- `left_arm`: global index `[0:7]`，0-based inclusive `0–6`；joint names 未提供。
- `left_hand`: global index `[14:34]`，0-based inclusive `14–33`。
- hand 顺序沿用已有 Wuji flat-array convention：finger1/thumb、finger2/index、finger3/middle、finger4/ring、finger5/little，每指 joint1–4。
- arm state/action 单位：rad；state 是绝对 encoder position，action 是绝对 target。
- recorder pose 字段：无。

完整审计写在 [`outputs/expert_palm_replay/expert_palm_motion_stats.json`](../outputs/expert_palm_replay/expert_palm_motion_stats.json)，phase 边界写在 [`selected_episode_phase_boundaries.json`](../outputs/expert_palm_replay/selected_episode_phase_boundaries.json)。

## 当前能测到的是什么

对 episode `69, 75, 87, 77, 84` 的 grasp phase，能直接测到的是 expert arm joint 变化，不是 palm 变化。7D arm state 变化范数的均值为约 `0.529 rad`，各 episode 约 `0.305–0.682 rad`。这不能换算成毫米，也不能推断 wrist rotation；不同 embodiment 下同样的 joint delta 可以对应完全不同的末端 SE(3)。

这五条轨迹的 hand closing 与 arm joint excursion 是同时发生的，但“palm 是否靠近方块、是否 lateral correction、是否 wrist rotation”在缺少 FK 前均保持未知。

## A/B fixed-palm baseline

这部分使用相同 seed 7、相同 reset、相同 120-step horizon 和当前 PCA5 controller，只比较：

- A：fixed palm + raw expert 20D hand；
- B：fixed palm + PCA5 reconstructed hand。

| episode | A max contacts | B max contacts | A 连续 3-contact 最长 | B 连续 3-contact 最长 | A finger3/finger4 最大滑动 | B finger3/finger4 最大滑动 |
|---:|---:|---:|---:|---:|---:|---:|
| 69 | 3 | 3 | 0.067 s | 0.033 s | 23.4 / 53.3 mm/s | 34.1 / 57.5 mm/s |
| 75 | 3 | 2 | 0.033 s | 0 s | 6.7 / 33.9 mm/s | — / 37.0 mm/s |
| 87 | 3 | 4 | 0.033 s | 0.033 s | 35.4 / 70.7 mm/s | 43.7 / 36.4 mm/s |
| 77 | 3 | 4 | 0.067 s | 0.067 s | 60.7 / 46.0 mm/s | 112.7 / 60.5 mm/s |
| 84 | 3 | 3 | 0.033 s | 0.033 s | 50.0 / 51.8 mm/s | 35.2 / 57.1 mm/s |

所有 A/B 条件的 `stable_grasp_success` 都是 false，且没有任何条件达到 `0.3 s` 的连续双指窗口。A/B 只提供 fixed-palm 基线，不能回答 expert palm motion 是否改善接触；PCA5 的 max contact 在不同 episode 有升有降，不能据此宣称掌心运动有效或无效。

指标明细在 [`outputs/expert_palm_replay/replay_comparison.json`](../outputs/expert_palm_replay/replay_comparison.json)。本次保留了接触拓扑转移和 finger3/finger4 峰值滑动的关键时序 [`key_timeseries.json`](../outputs/expert_palm_replay/key_timeseries.json)；脚本运行时还会生成逐帧 contact count、normal force、sliding、edge margin、相对速度、cube displacement 和 penetration timeseries，字段清单见 [`timeseries_manifest.json`](../outputs/expert_palm_replay/timeseries_manifest.json)。

## C/D 与 translation/rotation ablation

以下实验已经在脚本中实现接口，但当前运行状态为 `blocked_without_expert_fk`：

- C：expert-relative palm SE(3) + raw20 hand；
- D：expert-relative palm SE(3) + PCA5 hand；
- translation-only；
- rotation-only；
- full 6D。

脚本只有在明确提供以下三项后才会执行：

```text
--expert-fk-model <expert robot URDF/MJCF/MJB>
--expert-arm-joint-names name1,...,name7
--expert-palm-site <expert palm/wrist site>
```

它会先用 expert 模型计算 `T_palm,expert(t)`，再构造：

```text
T_target_OpenArm(t) = T_OpenArm_start · ΔT_palm,expert(t)
```

随后通过当前 OpenArm 6D IK 跟踪，并记录 desired/actual palm pose、IK residual、orientation residual、joint target、qpos、unreachable/joint-limit events。没有提供这些输入时不会 fallback 到 OpenArm FK。

相关机器结果：

- [`translation_vs_rotation_ablation.json`](../outputs/expert_palm_replay/translation_vs_rotation_ablation.json)
- [`palm_tracking_metrics.json`](../outputs/expert_palm_replay/palm_tracking_metrics.json)

## 对四个问题的当前回答

### Q1：expert grasp 阶段 palm/wrist 移动多少？

目前只能确认 expert arm joint 在移动；不能从当前数据确认 palm 平移毫米数或 wrist 旋转角度。任何这类数值都需要 expert FK 和 base/tool 标定。

### Q2：恢复 expert palm 后是否改善 PCA5 的 edge/corner contact 和 tangential slip？

尚未可判定。C/D 尚未执行；A/B 是 fixed-palm 对照，不是 palm ablation。

### Q3：translation、rotation 还是两者更重要？

尚未可判定。当前 `translation_vs_rotation_ablation.json` 明确标记 blocked。

### Q4：下一步 SAC action space 是 5D、8D 还是 11D？

现在继续保持 **5D PCA hand**，不扩大 SAC action space。原因不是证明 palm 不重要，而是 expert palm SE(3) 尚未被可靠观测/恢复。只有拿到 expert FK 并完成 paired deterministic replay 后，才按结果决定 3D residual + 5D 或 6D residual + 5D。

## 如何解除阻塞

需要从数据发布者或采集系统补齐：expert arm 型号/URDF 或 MJCF、7 个 joint 的名字和顺序、palm/wrist link 或 site 名称，以及 base/tool transform。拿到后直接运行：

```powershell
$env:PYTHONPATH='src'
& '.venvs/lerobot-policy/Scripts/python.exe' `
  scripts/expert_palm_hand_coordination_replay.py `
  --expert-fk-model <expert-model.mjcf-or-mjb> `
  --expert-arm-joint-names name1,name2,name3,name4,name5,name6,name7 `
  --expert-palm-site <palm-site>
```
