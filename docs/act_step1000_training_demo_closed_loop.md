# ACT D step1000：20 条训练场景纯闭环复现评测

## 结论

D step1000 **不能在 closed loop 中复现它真正训练过的 20 个场景**。
策略在每帧只读取 27D actual qpos、front RGB、wrist RGB，并输出 27D absolute controller target；H_exec=1。rollout 阶段完全没有加载或执行 expert action。

- Reach：0/20
- Approach：0/20
- Task success：0/20
- 最小 pregrasp error：mean 13.35 mm，median 13.99 mm，best 4.79 mm
- 进入 12 mm：9/20；总计 24 帧；最长连续 3 帧（gate 要求 5 帧）
- cube 最大位移：mean 15.23 mm，median 0.000 mm，max 78.78 mm；>25 mm 为 4/20
- early contact：5/20

这不是新场景泛化失败，而是训练场景上的 closed-loop reproduction failure。9 条轨迹能短暂进入 12 mm basin，但没有一条保持到 5 帧；同时 4 条把 cube 推动超过 25 mm。主要 blocker 是闭环时序稳定性/误差恢复，而不是单纯没有见过这些初始条件。

## 初始条件与评测契约

训练 demo seeds：`0, 1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 17, 18, 19, 20, 21, 22`。raw expert frame-0 与 ACT rollout frame-0 的 27D actual state 最大绝对差为 `5.628e-08` rad，确认使用相同 reset robot state；cube reset 由同一 seed 的任务 reset 生成。

为了让统计和视频严格对应，最终 run 一次性保存了全部 20 条 rollout 的双相机输入帧。额外重复性检查发现 Windows/MuJoCo 渲染跨进程偶有极少像素 ±1 灰度差，微小输入差会被当前闭环长期放大；这与‘缺少稳定吸引域’的主结论一致。

## 每条训练场景

| seed | Reach | Approach | Task | min pregrasp mm | 12mm frames | longest dwell | cube max mm | early contact | max contacts |
|---:|:---:|:---:|:---:|---:|---:|---:|---:|:---:|---:|
| 0 | 0 | 0 | 0 | 10.73 | 1 | 1 | 0.34 | 1 | 1 |
| 1 | 0 | 0 | 0 | 15.56 | 0 | 0 | 0.00 | 0 | 0 |
| 2 | 0 | 0 | 0 | 11.05 | 1 | 1 | 0.00 | 0 | 0 |
| 3 | 0 | 0 | 0 | 10.72 | 3 | 2 | 2.00 | 0 | 0 |
| 5 | 0 | 0 | 0 | 25.46 | 0 | 0 | 0.00 | 0 | 0 |
| 6 | 0 | 0 | 0 | 11.15 | 1 | 1 | 0.00 | 0 | 0 |
| 7 | 0 | 0 | 0 | 15.93 | 0 | 0 | 0.00 | 0 | 0 |
| 8 | 0 | 0 | 0 | 16.80 | 0 | 0 | 0.00 | 0 | 0 |
| 9 | 0 | 0 | 0 | 18.15 | 0 | 0 | 0.00 | 0 | 0 |
| 10 | 0 | 0 | 0 | 8.63 | 5 | 2 | 74.32 | 1 | 3 |
| 11 | 0 | 0 | 0 | 4.79 | 9 | 3 | 0.03 | 0 | 0 |
| 12 | 0 | 0 | 0 | 15.88 | 0 | 0 | 0.00 | 0 | 0 |
| 14 | 0 | 0 | 0 | 10.37 | 2 | 1 | 78.78 | 1 | 3 |
| 16 | 0 | 0 | 0 | 7.45 | 1 | 1 | 0.00 | 0 | 0 |
| 17 | 0 | 0 | 0 | 15.59 | 0 | 0 | 73.29 | 1 | 3 |
| 18 | 0 | 0 | 0 | 13.41 | 0 | 0 | 0.00 | 0 | 0 |
| 19 | 0 | 0 | 0 | 15.52 | 0 | 0 | 0.00 | 0 | 0 |
| 20 | 0 | 0 | 0 | 9.92 | 1 | 1 | 0.00 | 0 | 0 |
| 21 | 0 | 0 | 0 | 15.29 | 0 | 0 | 75.85 | 1 | 3 |
| 22 | 0 | 0 | 0 | 14.57 | 0 | 0 | 0.05 | 0 | 0 |

## 选中对照

按“最长连续 12 mm dwell 优先，其次更低 pregrasp error，再其次更小 cube displacement”选择：best = seed 11，worst = seed 5；另固定展示 seed 7。

### Seed 7（requested_seed_7）

ACT：min pregrasp 15.93 mm，longest dwell 0 帧，cube max displacement 0.00 mm。
同一 30 Hz 时间轴前 140 帧的 arm-state RMSE=0.194 rad，hand-state RMSE=0.401 rad，27D action RMSE=0.465 rad。

- 动画：`seed_000007_expert_vs_act.gif`
- 数值轨迹图：`seed_000007_expert_vs_act_trajectory.png`

### Seed 11（best）

ACT：min pregrasp 4.79 mm，longest dwell 3 帧，cube max displacement 0.03 mm。
同一 30 Hz 时间轴前 145 帧的 arm-state RMSE=0.262 rad，hand-state RMSE=0.389 rad，27D action RMSE=0.454 rad。

- 动画：`seed_000011_expert_vs_act.gif`
- 数值轨迹图：`seed_000011_expert_vs_act_trajectory.png`

### Seed 5（worst）

ACT：min pregrasp 25.46 mm，longest dwell 0 帧，cube max displacement 0.00 mm。
同一 30 Hz 时间轴前 143 帧的 arm-state RMSE=0.292 rad，hand-state RMSE=0.387 rad，27D action RMSE=0.469 rad。

- 动画：`seed_000005_expert_vs_act.gif`
- 数值轨迹图：`seed_000005_expert_vs_act_trajectory.png`

## 判断

ACT 已学到能够把手臂送入 pregrasp 邻域的部分映射，但没有学到在观测误差和自身动作引起的状态偏移下持续留在 basin、再推进 Approach/Grasp/Lift 的闭环策略。因为连 20 条训练初始条件都为 0/20 Reach，当前不能把失败归因于测试 seed 泛化；应优先处理 demonstration 的闭环覆盖/恢复数据与 temporal stabilization，而不是继续用 training loss 证明策略已复现 expert。
