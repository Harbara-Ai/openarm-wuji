# 从机会式抓取到显式约束：Wuji Reach–Grasp–Lift 工程复盘

本文记录我们为什么为 Wuji 灵巧手引入显式抓取约束、每一阶段暴露了什么问题、采取了什么解决方法，以及哪些结论已经成立、哪些仍待验证。

当前实验分支为 `experiment/grasp-settle-gate`。本文是工程决策记录，不把单个 seed 的调参结果写成真实硬件结论，也不把外部数据集阈值宣称为 Wuji 的最终标准。

## 1. 最初的抓取是怎么工作的

最初的控制器没有指定每根手指的接触点。它采用三维 hand synergy action，其中第一个标量控制整体闭合程度，并映射到 Wuji 的 20 个独立手指关节目标：

```text
Reach
  -> 机械臂 IK 把 grasp center 移到方块附近
Approach
  -> 掌心下降到抓取高度
Grasp close
  -> synergy 每帧增加 0.03，所有手指沿预设协同轨迹闭合
Contact gate
  -> synergy >= 0.7 且至少两个 finger groups 有接触
Preload / settle
  -> 保持机械臂目标并继续轻微闭合
Lift
  -> 保持 synergy，机械臂抬升
```

seed 7 通常在 `synergy ≈ 0.72` 形成多指接触，之后预载到 `0.76`。方块是 MuJoCo free joint，没有被 weld 到手上，Lift 也没有直接修改方块位姿。因此它确实依靠仿真接触力、摩擦和包络/卡持效应把方块抬起来，不是“粘住方块”的假动作。

但该方法属于机会式 power grasp：它只要求“闭合量足够 + 接触手指数足够”，没有约束：

- 哪个 fingertip 接触哪个 cube face；
- 接触点是否远离 edge/corner；
- designated fingertip 是否先于 distal pad 或 proximal link 接触；
- 接触法向是否形成对置；
- 从 pregrasp 到 endpoint 的完整路径是否无碰撞；
- 物体相对掌心是否在 Lift 全程保持稳定。

所以“MuJoCo 中真实受力并抬起来”和“稳定、可复现、可迁移的抓取”不是同一个结论。

## 2. 为什么必须引入显式约束

旧成功条件主要是抬升高度和多指接触。seed 7 能达到高度目标，但后续 SE(3) 诊断表明：

- `task_success = true`；
- `grasp_stable = false`；
- outcome 为 `settled_after_slip`；
- Lift 期间物体相对掌心最大平移漂移约 `52.5 mm`；
- 最大相对旋转漂移约 `17.7°`；
- establishment translation 约 `24.7 mm`；
- 前 0.5 秒实际掌心抬升约 `25.0 mm`，没有完整跟踪外部 `50 mm / 0.5 s` 参考。

这说明方块虽然最后被带到了高处，但在手里发生了明显滑移和重新定位。若只看最终高度，它会被误报成完整抓取成功，并可能被错误地录入 ACT expert demonstrations。

显式约束的目的不是让原抓取“从假变真”，而是让成功条件具备以下性质：

1. **可解释**：知道是哪一个 geom、在哪个 face、以什么法向接触。
2. **可执行**：endpoint 可达之外，完整关节路径也必须无 non-tip collision。
3. **可评测**：任务完成、抓取稳定性和接触诊断彼此独立。
4. **可复现**：不同 seed 与 replay 使用同一程序化定义。
5. **可迁移**：未来能把 MuJoCo 指标与真实 Wuji 硬件标定对应起来。

## 3. 评测层：先把“抬起来”和“抓稳了”分开

### 遇到的问题

“抬升高度 + 多指接触”无法识别手内滑移；接触数量也不能证明接触分布形成了稳定对置或 force closure。

### 解决方法

episode recorder 和 task telemetry 增加：

- cube 世界坐标位置与四元数；
- grasp center 世界坐标位置与四元数；
- cube 相对 grasp center 的完整 SE(3)；
- 每个接触点的位置、世界力向量、所属 finger、geom 和 cube face；
- 合力、关于 cube center 的合力矩、edge/corner 分类。

这些字段只属于 task telemetry，明确不进入 LeRobot/ACT policy observation。

结果被拆成：

- `task_success`：达到目标抬升高度并保持；
- `grasp_stable`：Lift 稳定窗口内物体相对掌心的平移和旋转漂移合格；
- `contact_diagnostics`：接触手指、方向、合力和合力矩，只用于解释与评测。

同时建立互斥 outcome taxonomy：

- `rigid_success`
- `settled_after_slip`
- `persistent_slip`
- `drop`
- `never_lift`
- `approach_push`

### 外部基准的使用边界

CD-WM 的公开条件被明确标记为 external baseline：

- 0.5 秒内夹爪抬升 50 mm；
- 物体实际抬升至少 25 mm；
- 稳定窗口相对平移漂移 `< 8 mm`；
- 相对旋转漂移 `< 6°`。

8 mm / 6° 目前只是对照 gate，不是经过 Wuji 灵巧手或真实传感噪声标定的最终阈值。报告始终保留连续漂移指标，避免阈值掩盖趋势。

## 4. 状态机层：停止边闭合边评估稳定性

### 遇到的问题

早期实现把仍在闭合的帧放进稳定窗口。手指持续加大闭合量时，物体可能仍在重排，因此该窗口不能代表冻结后的抓取稳定性。过高的 preload 还产生了非常大的瞬态接触力，甚至更容易掉落。

### 解决方法

状态机调整为：

```text
grasp_close
  -> synergy >= 0.7 且形成多指接触（seed 7 约 0.72）
freeze_synergy
  -> 记录首次满足条件的闭合量
preload
  -> 机械臂目标固定，synergy 每帧最多增加 0.01 到 0.76
preload_settle
  -> 手臂和手指目标保持，重新开始独立稳定窗口
lift_s_curve 或 regrasp_required
```

`preload_settle` 要求：

- 接触持续；
- 相对 SE(3) 漂移通过外部对照 gate；
- 接触合力与合力矩趋势不继续发散。

仍在 close/preload ramp 的帧不再进入稳定窗口。如果在预算内不能稳定，则拒绝 Lift，而不是带着不确定抓取继续抬升。

### 实验结论

seed 7 的 `0.76` 是当前较温和的 provisional preload。更大的闭合并不等于更稳：历史 sweep 中 `0.82/0.85/0.93` 出现 drop，`0.93` 产生约 `698 N` 的峰值合力。该 sweep 只用于否定“继续加力一定更好”，不能作为多 seed 成功率。

## 5. Lift 层：从猛抬改为有界参考轨迹

### 遇到的问题

旧 Lift 在前 0.5 秒实际抬升约 87 mm，速度变化过猛，容易激发手内滑移或掉落。仅限制每帧关节增量不能保证 Cartesian 速度、加速度和 jerk 有界。

### 解决方法

Lift 参考改为两段 quintic minimum-jerk S-curve：

- 第一段：50 mm / 0.5 s；
- 第二段：70 mm / 0.7 s；
- 显式配置参考速度、加速度与 jerk 上限；
- 保持 preload synergy，不在 Lift 起始帧突然继续闭合。

### 剩余问题

参考轨迹有界不等于实际机械臂运动有界。MuJoCo 位置执行器、重力 sag 和 IK 跟踪误差使实际掌心仍可能超出参考导数，且 6D 姿态会漂移。因此报告同时记录 reference 和 measured trajectory，不能把参考上限写成真实硬件保证。

## 6. 6D 掌心控制与接触几何诊断

### 遇到的问题

仅控制掌心位置时，Lift 中掌心姿态变化会改变手指与方块的相对几何。与此同时，多指接触并不代表正确对置：固定 seed 实验显示 thumb 经常落在 `-Y`，其他手指主要分布在 `+X/+Y/-Y`，而不是预期的 `-X/+X`。

### 解决方法

- DLS IK 从三维位置升级为位置 + quaternion rotation-vector 的 6D pose IK；
- 同时记录 desired、simulation-compensated IK command 和 actual palm orientation；
- 所有 hand/cube contacts 转换到 cube frame；
- 输出 dominant face、edge/corner rate、接触质心、水平合力与合力矩。

### 实验结论

运动学 IK 可以达到很小的姿态残差，但真实执行器在 Lift 下仍有约 5° 的姿态漂移。五个固定 seeds 中，四个达到高度，零个达到 strict stable grasp。三例最终窗口能重新 settle，但此前已经发生 52–57 mm 的瞬态相对位移，仍不合格。

## 7. 从 synergy endpoint 转向 contact-region optimization

### 遇到的问题

统一 synergy 无法独立决定每根手指的接触面和接触点。我们需要先回答 Wuji 的关节几何是否能表达目标拓扑，而不是直接调闭合力。

### 解决方法

新增独立的 whole-hand kinematic optimizer，决策变量包含 7 个 arm joints 和 20 个 finger joints。第一版目标拓扑为：

- thumb/finger1 -> cube `-X`；
- finger2、middle/finger3、ring/finger4 -> cube `+X`；
- little/finger5 optional。

约束与代价包含：

- contact region，而不是固定死单点；
- fingertip position 与 normal alignment；
- edge clearance；
- fingertip penetration；
- 所有 non-tip/cube clearance；
- hand self-collision；
- joint limits 与偏离 nominal 的幅度。

### 遇到的新问题

Stage-1 可以求出合法 endpoint，但独立执行后接触拓扑被破坏：thumb 丢失，其他手指落到错误 face 或 edge；取消桌面支撑后很快失效。

由此得到关键结论：

> endpoint feasible 不等于 contact acquisition feasible，更不等于 stable grasp。

## 8. 从 joint-space retreat 转向 Cartesian-normal pregrasp

### 遇到的问题

直接从 open pose 或沿关节空间线性插值到 q_star，会让 distal pad、link3 或其他 non-tip geom 抢先碰到 cube。仅检查最终 endpoint 无 penetration 无法发现路径碰撞。

### 解决方法

- 为每个 required designated fingertip 沿目标 face normal 构造约 5 mm precontact；
- 使用单指 point-Jacobian IK 求解 q_pre；
- 在 q_pre -> q_star 上采样所有 finger collision geoms 的 signed distance；
- 接触 gate 只接受 designated fingertip、正确 face、非 edge/corner、法向与力均合格的连续确认；
- distal non-tip contact 保留为 telemetry，但不能冒充 fingertip contact。

### seed 7 暴露的问题

finger2 的最终 designated fingertip geom57 endpoint 可达，但路径先被 geom53/geom55 阻挡。最初的 first contact 实际是 distal non-tip geom55，而不是 geom57。

## 9. finger2 path-aware contact planning

### 解决方法

只固定当前 palm pose、cube pose 和其他手指，针对 finger2 的 `+X` face 搜索五个有明确语义的 contact regions：

- center
- upper
- lower
- +Y-side
- -Y-side

每个候选分阶段求解：

```text
q_pre -> q_mid -> q_star
```

q_mid 目标包含 non-tip path clearance、平滑性、joint-limit margin 和 self-collision。每段采样 21 点，遍历 finger2 全部 collision geoms，不只 hardcode geom53/55。

### 结果

运动学上 center、upper 和 -Y-side 均找到正 signed-distance 路径。upper 最好：

- endpoint error `0.041 mm`；
- geom53 最小余量 `0.980 mm`；
- geom55 最小余量 `0.994 mm`；
- 运动学预测 first contact 为 geom57。

真实 MuJoCo dynamics 却显示：

- upper：geom55 在约 38 ms 首触；
- center：geom55 在约 38 ms 首触；
- -Y-side：geom55 在约 36 ms 首触；
- geom57 要到约 3.0–3.6 s 后才接触，而且已经是 edge contact。

原因是四个位置控制关节的实际动态响应不同步，真实轨迹偏离同步 joint interpolation。将执行放慢到每段 3 秒仍不能改变首触顺序。

### 决策

finger2 被判为 `finger2_required_contact_rejected`。它可以保留为 optional/support，但不再强迫为当前 palm pose 下的 required `+X` 对置手指。

## 10. 三指主对置拓扑与 clean-state 问题

### 新拓扑

- required：thumb/finger1 -> `-X`；
- required：middle/finger3 -> `+X`；
- required：ring/finger4 -> `+X`；
- optional/open：finger2、finger5。

### 遇到的问题一：pregrasp 实验相互污染

早期 isolated-finger test 仍同时构造其他 required fingers 的 pregrasp，导致一个手指的碰撞被错误归因到另一个手指。

修正：`build_pregrasp` 接受本次测试的 required set，单指实验只规划目标手指；其他 fingers 锁定。

### 遇到的问题二：retreat off-by-one

报告显示 adaptive retreat 到 12 mm，但旧循环最后实际只求解到 9 mm。

修正：显式配置五次尝试并真正执行 `0/3/6/9/12 mm`。

### 遇到的问题三：inactive fingers 没有真正固定

保持相同 position ctrl 不等于物理锁定。在 `thumb_first` 中，未激活的 finger4 仍可能因执行器动力学移动并先接触 cube。

修正：诊断 executor 可在每个 2 ms MuJoCo physics step 数值固定 optional 和 strategy-inactive fingers；已激活或已 acquired fingers 保留真实动力学。

### 遇到的问题四：旧抓取状态污染 clean acquisition

最初每次优化前调用 `run_grasp(seed7)`。旧抓取已经改变了 cube pose/velocity；随后瞬移到 pregrasp 并释放时，方块会在桌面上重新落稳，这种自然运动被误判成新 acquisition 的 unilateral push。

修正：

- pregrasp teleport 后清零全部 generalized velocities；
- 三指几何实验改从 `run_reach(seed7)` 开始，使 palm 到位但 cube 尚未被旧抓取扰动；
- 30 Hz 目标可在线性分配到 2 ms physics substeps，避免把控制阶跃和几何失败混在一起。

### clean-state 结果

从干净 `run_reach(seed7)` 状态测试三个有限语义初始化：

| initialization | satisfied | thumb error | middle error | ring error |
| --- | --- | ---: | ---: | ---: |
| current_grasp | finger1 | 5.31 mm | 20.97 mm | 20.06 mm |
| three_finger_opposition_same_palm | finger1 | 4.14 mm | 6.05 mm | 8.45 mm |
| three_finger_power_small_yaw | finger1 | 4.16 mm | 7.49 mm | 7.72 mm |

三个初始化都没有同时满足 thumb、middle、ring endpoint，结论是 `kinematic_candidate_invalid`。由于 endpoint 未通过，正确行为是停止，不运行 acquisition、squeeze 或 Lift。

旧 post-grasp 状态中曾出现 3/3 endpoint 成功，但那依赖已经被旧抓取改变的 cube/palm 相对状态，不能作为 clean tabletop 抓取 seed。

## 11. 哪些尝试有效，哪些没有解决根因

### 已确认有效

- SE(3) telemetry 揭示了“抬起来但手内滑移”；
- task/stability/contact 三层评测避免误报；
- freeze -> preload -> settle 避免用闭合帧评估稳定性；
- S-curve 明显降低旧 Lift 的激烈参考跳变；
- 6D IK 与 contact-face telemetry 让掌心姿态和接触布局可解释；
- designated fingertip 分类阻止 distal pad 冒充指尖；
- swept-path sampling 揭示 endpoint/path 不等价；
- q_mid 证明 finger2 存在运动学路径，但动力学首触仍失败；
- clean `run_reach` 初始化消除了旧 grasp 对新几何实验的状态污染。

### 没有解决根因

- 单纯增加 preload：高 preload 会产生冲击甚至 drop；
- 只降低 Lift 速度：能减少掉落，但不能修复错误抓取几何；
- 只做 endpoint IK：无法保证 non-tip-free path 和动态 first contact；
- 只要求多指接触：无法证明接触对置、稳定或 force closure；
- 在 post-grasp cube pose 上继续优化：会得到不可用于 clean tabletop task 的假 seed；
- 立即训练 ACT/RL：当前 expert 行为仍包含明显 slip 和错误接触，会把缺陷写进数据集。

## 12. 当前状态与下一步

目前可以诚实地说：

1. 原 synergy controller 能在 MuJoCo 中通过真实接触把 cube 抬起来；
2. 它不是 strict stable grasp，也尚未证明可迁移到真实 Wuji；
3. 显式约束已经定位出主要瓶颈是 grasp geometry、pregrasp/path feasibility 和 actuator tracking，而不是缺少 ACT/RL；
4. finger2 不适合作为当前 palm pose 下的 required `+X` 对置手指；
5. thumb-middle-ring 拓扑合理，但当前三个 clean palm seeds 仍不能同时覆盖 `-X/+X` 接触区域。

下一步是从 clean tabletop state 生成 palm reachable workspace map：

- 在 cube frame 中可视化 palm translation/yaw；
- 对每个 palm pose 分别计算 thumb、middle、ring 的 contact-region 最小误差；
- 标出 arm IK reachable、无 non-tip penetration、三指同时覆盖的区域；
- 从连续可行区域中心选少量 palm pose seeds，而不是恢复大规模盲目 yaw/XY brute-force；
- 只有 clean endpoint 和 swept path 都通过，才重新进入 contact acquisition。

在此之前继续调 gate、squeeze、Lift、ACT 或 RL 都不是优先事项。

## 13. 相关实现与报告

- 原状态机：`src/openarm_wuji/tasks/reach_grasp_lift.py`
- MuJoCo synergy adapter：`src/openarm_wuji/simulation/mujoco_backend.py`
- SE(3) 与 outcome：`src/openarm_wuji/tasks/se3.py`、`src/openarm_wuji/tasks/outcomes.py`
- contact optimizer / acquisition：`src/openarm_wuji/tasks/contact_grasp.py`
- finger2 path planner：`src/openarm_wuji/tasks/finger_path_planner.py`
- 显式约束配置：`configs/contact_grasp_optimization.json`
- 当前 Lift 指标：`outputs/reach_grasp_lift/lift_report.json`
- preload/S-curve 记录：`docs/grasp_settle_experiment.md`
- 几何诊断：`docs/grasp_geometry_diagnostics.md`
- Stage-1 contact optimization：`docs/contact_grasp_optimization_seed7.md`
- contact acquisition：`docs/contact_acquisition_seed7.md`
- finger2 q_mid 实验：`docs/finger2_path_planning_seed7.md`
- 三指 clean-state 实验：`docs/three_finger_topology_seed7.md`

