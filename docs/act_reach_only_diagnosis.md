# Reach-only ACT diagnosis

## 结论

**Case A：Reach-only ACT reaches high training-seed success; full-task phase/temporal ambiguity is the primary failure source.**

本实验只执行 `reset → Reach → stable pregrasp hold → END`。rollout 每帧只向 ACT 输入双相机和 27D actual qpos，执行其 27D absolute target；未读取或执行 expert action，也没有进入 Approach/Grasp/Preload/Lift/Hold。
三条对照的 expert 与 ACT 初始 27D state 完全一致；GIF/trajectory 的 error 已对齐到 observation timestamp。成功 gate 则按动作执行后的真实 MuJoCo 状态判定。

## 数据与配置

- 20 条训练 seed，427 frames：267 个真实 Reach 帧 + 160 个终点 hold 帧。
- 每条终点 hold 8 帧，重复 first post-Reach observation，并保持最后一个已经稳定的 Reach controller target。
- Wuji 20D target 在所有 Reach 段中恒定；数据不含 grasp close。
- 27D state、27D action、front+wrist 240×320、30 Hz；ImageNet ResNet-18；chunk20；normal sampling；fresh training；H_exec=1。

## Checkpoint 对比

| step | Reach/20 | timeout | min error mean/median/best mm | enter 12mm | first entry mean frame | total dwell | max consecutive | early contact | cube max mean/max mm | Reach arm H1 MAE | Reach hand H1 MAE | clipped values | frame0 visual/expert arm spread |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 500 | 11/20 | 9/20 | 7.18/6.33/2.19 | 20/20 | 1.70 | 102 | 5 | 0 | 0.10/1.89 | 0.00609 | 0.00000 | 0 | 1.425 |
| 1000 | 16/20 | 4/20 | 5.41/5.39/1.62 | 20/20 | 1.00 | 128 | 5 | 0 | 0.09/1.78 | 0.00437 | 0.00000 | 0 | 1.515 |
| 2000 | 19/20 | 1/20 | 4.36/4.54/0.52 | 20/20 | 1.00 | 134 | 5 | 0 | 0.09/1.76 | 0.00371 | 0.00000 | 0 | 1.527 |

## Full-task baseline 对照

相同训练 reset 的 full-task D step1000 为 Reach 0/20，mean/median/best pregrasp error = 13.35/13.99/4.79 mm。
它只有 9/20 进入过 12 mm basin、最长 dwell 3 帧；Reach-only step2000 则为 20/20 进入、19/20 连续保持满 5 帧。

## 诊断解释

- step 500 → 1000 → 2000 的 Reach 为 11/20 → 16/20 → 19/20，同时 real-Reach arm H=1 MAE 为 0.00609 → 0.00437 → 0.00371 rad；二值与连续指标一致改善。
- 三档均为 20/20 至少进入一次 12 mm basin，且没有 action clipping、early cube contact 或 >25 mm cube displacement；失败不是 controller limit 或碰撞造成的。
- step2000 唯一失败 seed 8：minimum error 8.27 mm、共 8 帧在 gate 内，但最长连续 dwell=4，距离正式成功只差第 5 个连续帧；随后停在 gate 外侧而超时。
- seed 7、best seed 14、worst seed 8 在共同前缀上均未出现 `>0.1 rad 且连续 3 帧` 的 gross arm/action divergence。这不表示轨迹完全相同，而是 seed 8 的失败更像终点小偏差/稳定保持问题。
- frame-0 visual-only arm prediction spread / expert spread 在三档均 >1，说明双相机条件化没有塌缩为同一个动作；该指标只证明视觉响应存在，不等同于未见初始条件上的泛化。
- 这是 training-seed 复现诊断，足以回答 phase ambiguity 假设，但不能据此声称对 unseen seeds 泛化。

## Clipping by joint

- step 500: j0 count=0 max=0.0000 rad, j1 count=0 max=0.0000 rad, j2 count=0 max=0.0000 rad, j3 count=0 max=0.0000 rad, j4 count=0 max=0.0000 rad
- step 1000: j0 count=0 max=0.0000 rad, j1 count=0 max=0.0000 rad, j2 count=0 max=0.0000 rad, j3 count=0 max=0.0000 rad, j4 count=0 max=0.0000 rad
- step 2000: j0 count=0 max=0.0000 rad, j1 count=0 max=0.0000 rad, j2 count=0 max=0.0000 rad, j3 count=0 max=0.0000 rad, j4 count=0 max=0.0000 rad

## Error-vs-time 与 expert 对照

每个 checkpoint 的 best/median/worst pregrasp 曲线已保存。最终 step2000 对 seed 7、best、worst 保存了 7D arm qpos、27D action heatmap、pregrasp error、持续偏离帧和 front/wrist GIF。

- step 500: `pregrasp_error_step_000500.png`（seeds {'best': 7, 'median': 14, 'worst': 2}）
- step 1000: `pregrasp_error_step_001000.png`（seeds {'best': 21, 'median': 1, 'worst': 8}）
- step 2000: `pregrasp_error_step_002000.png`（seeds {'best': 14, 'median': 1, 'worst': 8}）
- seed 7 (seed7): arm 首次持续偏离 frame=None，action 首次持续偏离 frame=None；初始 27D state 最大差=0.00e+00；`seed_000007_expert_vs_act_trajectory.png` / `seed_000007_expert_vs_act.gif`
- seed 14 (best): arm 首次持续偏离 frame=None，action 首次持续偏离 frame=None；初始 27D state 最大差=0.00e+00；`seed_000014_expert_vs_act_trajectory.png` / `seed_000014_expert_vs_act.gif`
- seed 8 (worst): arm 首次持续偏离 frame=None，action 首次持续偏离 frame=None；初始 27D state 最大差=0.00e+00；`seed_000008_expert_vs_act_trajectory.png` / `seed_000008_expert_vs_act.gif`

## 下一步

Case A 下不要立刻恢复原 full-task 训练。首选 staged/FSM policies（先让 Reach-only policy 到位并稳定，再显式切换下一阶段）；随后分别验证给 full-task ACT 显式 phase/progress，以及 previous-action/short-history。这里只给出建议，本轮未执行。
