# OpenArm + Wuji 项目实验总账

更新日期：2026-09-15（Asia/Singapore）
覆盖范围：从环境搭建、OpenArm/Wuji 集成、scripted Reach–Grasp–Lift、抓取几何与 RL，到 coordinated demonstrations、ACT、staged policies 和最新 scripted Lift 验证。  
当前仓库：`openarm-wuji-learning`。`d29299e`（tag `staged-act-router-v1`）冻结了 Reach/Approach/Recovery/router；本次正式收口进一步冻结 GraspSecure step-1500 与 scripted Lift，并以 `configs/staged_pipeline.json` 作为唯一 manifest。

> 这是一份可持续维护的实验账本，不是聊天记录摘要。文中区分“已验证事实”“当时假设”“负结果”“被后续结果推翻的结论”和“尚未执行的计划”。更细的配置、逐 seed 表格和机器可读结果保留在链接的专题报告与 `outputs/` 中。

## 1. 当前结论一页版

当前系统已经从早期 monolithic controller/monolithic ACT，演进为：

```text
reset
→ Reach ACT
→ Reach gate
→ Approach ACT
→ selective_hysteresis router
→ 必要时 Recovery ACT
→ GraspSecure ACT
→ Grasp/Preload gate
→ scripted Lift
→ 1 s hold
```

目前已确认：

- OpenArm + Wuji MuJoCo 组合模型、双相机、30 Hz 同步记录、确定性 seeded replay 可用。
- 行为克隆统一接口已经跑通：`27D actual qpos + front/wrist RGB → 27D absolute controller target`。
- telemetry（cube pose、palm-relative SE(3)、contact、force、slip、success label、timestamp）与 policy observation 严格分离。
- monolithic full-task ACT 在训练 loss 持续下降时仍无法完成 Reach；把任务拆为单阶段后，Reach-only ACT 在训练 seeds 上达到 `19/20`，证明主要问题之一是多阶段 phase/temporal ambiguity，而不是 27D 接口本身损坏。
- frozen Reach→Approach 系统在早期 20-seed 链式评测中达到 `16/20`；经 selective Recovery router 的独立 200-seed 匹配评测，Approach-stage conditional success 达到 `129/155 = 83.2%`。
- GraspSecure ACT 最佳 step-1500：standalone `14/20 = 70%`，接在真实 staged handoff 后 `9/12 = 75%` conditional success。
- 但是 GraspSecure 的静态 gate 并不等价于 load-bearing grasp。条件测试中 scripted Lift 只成功 `14/30 = 46.7%`；早期 full staged fresh 100-run 为 `17/100 = 17%`。本次使用正式 runner 和新的 seeds `9000–9099` 得到最终 frozen benchmark：

| 阶段 | 结果 |
|---|---:|
| `P(Reach)` | `78/100 = 78.0%` |
| `P(Approach-stage success | Reach)` | `60/78 = 76.9%` |
| `P(finite GraspSecure terminal | Approach)` | `60/60 = 100.0%` |
| `P(Lift | finite terminal)`（含 safety reject） | `21/60 = 35.0%` |
| `P(Lift | actual Lift attempt)` | `21/44 = 47.7%` |
| `P(full success)` | `21/100 = 21.0%` |

当前最小条件成功率仍是 `Lift | GraspSecure terminal`。本里程碑不再继续 gate/retry/probe 优化，而是：

> 接受约 40–50% 的实际 Lift conditional success，冻结并完整记录当前系统。GraspSecure 静态 gate 仅作诊断；finite terminal 通过原 25 mm pre-Lift safety guard 后即可进入 scripted Lift。

详见 [最终 staged ACT 报告](FINAL_STAGED_ACT_REPORT.md)。

## 2. 全程保持的接口与判据纪律

### 2.1 两个 action contract 不应混淆

项目中出现过两个合法但用途不同的接口：

| 用途 | Observation | Action | 说明 |
|---|---|---|---|
| 初期在线 LeRobot robot plugin | 27D actual qpos + 双 RGB | 10D：7D arm target + 3D hand synergy | 设备级简洁控制接口；仍保留 |
| coordinated demo / ACT | 27D actual qpos + 双 RGB | 27D：7D arm target + 20D hand actuator target | 直接保存 position controller 真正收到的 target，保留 hand target–actual preload |

ACT 的 `action` 从来不应使用 next-frame actual qpos 代替。当前所有 learned staged policies 都采用第二行的 27D absolute-target 语义。

### 2.2 Policy observation 与 telemetry 分离

policy observation 默认只有：

```text
observation.state = [7D OpenArm actual qpos, 20D Wuji actual qpos]
observation.images.front
observation.images.wrist
```

以下字段只用于诊断、标注和评测，不输入 ACT：

- timestamp / frame index；
- cube 世界坐标位置与四元数；
- grasp center/palm 世界坐标位置与四元数；
- cube 相对 palm 的完整 SE(3)；
- contact point、world force vector、finger/body/geom；
- contact topology、slip、net wrench、preload；
- phase、success、outcome taxonomy。

### 2.3 `task_success`、`grasp_stable` 与 diagnostics 分离

早期仅用“抬升高度 + 多指接触”会把明显滑移的 episode 误报为完整稳定抓取。修正后：

- `task_success`：达到目标抬升高度并保持；
- `grasp_stable`：Lift 稳定窗口内 cube 相对 palm 的平移和旋转漂移通过诊断要求；
- `contact_diagnostics`：接触手指、方向、合力、合力矩、slip，只负责解释；
- outcome taxonomy：`rigid_success`、`settled_after_slip`、`persistent_slip`、`drop`、`never_lift`、`approach_push`，互斥记录 episode 结果。

CD-WM 的公开条件只作为 external baseline：`0.5 s` 内参考抬升 `50 mm`、实际至少 `25 mm`、稳定窗口相对平移 `<8 mm`、旋转 `<6°`。这些阈值没有被宣称为 Wuji 灵巧手的最终标准，也没有在最新 load-bearing 评测中重新变成 hard gate。

## 3. 实验时间线

以下编号是本总账为了追踪方便补加的，不是运行时原有编号。

## 3.1 基础环境、模型与接口

### E00 — 开发环境与上游版本固定

- Windows + WSL2 Ubuntu 22.04；RTX 2000 Ada 16 GB。
- OpenArm MuJoCo、Wuji retargeting 与相关 submodules/LFS 固定到明确 commit。
- Wuji retargeting 需要 ABI pin；不受约束的最新版 `cmeel` 依赖曾导致运行时不兼容。
- 结论：使用隔离虚拟环境，仿真、LeRobot/ACT 和 WSL retargeting 分开管理。

详见 [environment_report.md](environment_report.md)、[setup_decisions.md](setup_decisions.md)、[upstream_versions.md](upstream_versions.md)。

### E01 — OpenArm 官方模型 smoke

- 加载官方 OpenArm v2 MJCF，做 headless 关节/执行器清点、受限关节运动和渲染。
- 观测到有限的 `0.095249 rad` 程序化关节变化，模型稳定。
- 结论：OpenArm MuJoCo 与 joint position control 可作为后续基线。

详见 [milestone_1_openarm.md](milestone_1_openarm.md)。

### E02 — Wuji 官方 retargeting smoke

- 将预录 `21×3` hand keypoints 转为 `2751×20` Wuji joint trajectory。
- 全部数值 finite，吞吐约 `211.82 FPS`。
- 结论：官方 retargeting 路线可用，但这只证明运动学重定向，不证明物理抓取。

详见 [milestone_2_wuji.md](milestone_2_wuji.md)。

### E03 — OpenArm + Wuji 组合 MJCF

- 移除 OpenArm 原夹爪，将官方 Wuji left hand 挂到 `left_ee_control_point`。
- 组合模型：`nq=36, nv=36, nu=35`，稳定运行。
- 安装 transform 当时来自仿真视觉估计；真实硬件 tool-to-palm transform 尚未标定。
- 结论：统一 embodiment 建立，但不对真实硬件标定作声明。

详见 [milestone_3_4_combined.md](milestone_3_4_combined.md)。

### E04 — 双相机同步记录与 seeded replay

- 增加 fixed front camera 与 wrist camera。
- 30 Hz 同步记录 state/action/image/timestamp；90-frame replay 可复现。
- 后续 recorder 继续沿用 observation-at-`t` 与 action-at-`t` 的因果对齐。

### E05 — LeRobot robot plugin

- 独立注册 `openarm_wuji_follower`，不修改 LeRobot 本体。
- Mock/MuJoCo adapter contract 通过；RealBackend 保持未实现，避免在未知协议下误控硬件。
- 初期在线 action 是 10D；timestamp 未放入 policy observation。

详见 [lerobot_integration.md](lerobot_integration.md)。

### E06 — deterministic Reach task

- 加入桌面、seeded random cube、固定相机和 6D palm-pose IK。
- Reach 后续大样本 scripted/learned 结果表明 OpenArm 本身通常可到 pregrasp；早期项目估计 Reach 成功约 96%，后续 frozen ACT 在不同 fresh-seed 集上为 74%–80% 左右，不能把不同分布的比例直接混为一个固定常数。
- 后来最关键的分解实验是 Reach-only ACT `19/20`，详见 E35。

## 3.2 Scripted Reach–Grasp–Lift 与稳定性诊断

### E07 — 最早的自动 scripted grasp

实现方式不是调用一个 MuJoCo “grasp API”。控制器自己生成 OpenArm position targets 和 Wuji synergy/20D targets；MuJoCo 只负责刚体、接触、摩擦和重力。

因此它是物理仿真中的真实动态抓取尝试，不是把 cube weld 到手上。但“发生接触并抬高”不等于“稳定、居中、可复现的抓取”。原方案后来在历史快照 `36cbb5b` 上得到 54% load-bearing，而所有成功都属于 off-center，且有明显 palm-frame 漂移，故只保留为 historical scripted candidate。

### E08 — Causal episode recorder 与成功定义修正

- 从 `reset → reach/approach → grasp close → lift → hold` 全程记录 `observation_t → bounded action_t → observation_t+1`。
- 增加 cube/palm 世界位姿、relative SE(3)、每个 contact 点的 world force 和 finger identity。
- seeded replay 对物理字段可精确复现；早期图像重渲染存在最多约 `2/255` 的像素差。
- seed 7 被重新标为：`task_success=true`、`grasp_stable=false`、`outcome=settled_after_slip`。后续 schema-v3 结果的最大 relative drift 约 `52.5 mm / 17.7°`，最终窗口又收敛到 `1.39 mm / 1.60°`。

这一步推翻了“seed 7 是完整稳定抓取成功”的早期说法。详见 [episode_recording.md](episode_recording.md)、[reach_grasp_lift.md](reach_grasp_lift.md)。

### E09 — `freeze_synergy → grasp_settle → Lift`

- 首次满足 `synergy ≥ 0.7` 且多指接触时冻结；seed 7 约为 `0.72`。
- closing frames 不进入 stable window。
- 新增 `grasp_settle`；如果在 `max_close_steps` 内无法稳定，则拒绝 Lift。
- 目的：区分“闭合正在推动 cube”与“闭合完成后形成稳定 contact”。

结果：能正确拒绝部分不稳抓取，但没有解决 grasp geometry。

### E10 — Lift 速度重新对齐 external baseline

- 旧轨迹前 `0.5 s` 实际约抬升 `87 mm`，过猛并会诱发掉落。
- 改成两段 quintic minimum-jerk reference：前 `0.5 s` 参考 `50 mm`，再用 `0.7 s` 参考 `70 mm`。
- 参考峰值约 `0.1875 m/s`、`1.1547 m/s²`、`24 m/s³`；实际 tracking 仍可能有偏差。
- 结论：减小冲击是必要控制改进，但不能替代稳定 grasp geometry。

### E11 — `grasp_close → preload → preload_settle → lift_s_curve`

- arm 在 preload 完全固定；synergy 从冻结值缓慢增加，限制每帧增量。
- preload_settle 要求 contact 保持、relative SE(3) 稳定、合力/合力矩不继续发散。
- Lift 期间保持 preload hand target，不突然继续闭合。

seed-7 preload sweep：

| preload target | outcome | peak force | max relative rotation |
|---:|---|---:|---:|
| 0.74 | settled_after_slip | 9.2 N | 20.8° |
| 0.76 | settled_after_slip | 9.2 N | 16.8° |
| 0.78 | settled_after_slip | 11.7 N | 20.0° |
| 0.80 | settled_after_slip | 9.3 N | 18.7° |
| 0.82 | drop | 22.3 N | 40.9° |
| 0.85 | drop | 10.7 N | 37.7° |
| 0.88 | never_lift / gate reject | 8.4 N | N/A |
| 0.90 | settled_after_slip | 50.7 N | 88.1° |
| 0.93 | drop | 697.9 N | 43.2° |

没有任何 strict stable success。`0.76` 只是 seed-7 provisional choice，不是多 seed 最优值。高 preload 甚至会产生极端 force/moment transient；“挤得更紧”不是单调改进。

详见 [grasp_settle_experiment.md](grasp_settle_experiment.md)。

### E12 — 五 seed grasp geometry diagnostic

seeds `0, 7, 11, 19, 29`：

- height task success `4/5`；
- strict stable grasp `0/5`；
- post-settle stable `3/5`；
- Lift 全窗口 transient drift 约 `52–57 mm`；
- thumb 常在 cube `-Y` 边缘，除 seed 7 外其他成功轨迹的其他手指常由 `+X` 面主导，opposition 不可靠；
- commanded 6D palm orientation 在 Lift 中仍有约 `6.2–6.9°` 最大跟踪误差。

结论：根因优先指向 grasp geometry/opposition 与 controller tracking，不应继续只调 preload gate。详见 [grasp_geometry_diagnostics.md](grasp_geometry_diagnostics.md)。

### E13 — 显式接触约束的 kinematic grasp optimization

目标：thumb 覆盖 `-X`，index/middle/ring 覆盖 `+X`。Stage-1 endpoint optimization 得到一个看似很好的候选：thumb/index/middle/ring position error 分别为 `2.08/3.30/0.88/0.80 mm`，零 tip penetration。

动态 squeeze/static test 却失败：

- intended topology 没有被保持，thumb 失去接触；
- index 落到 `-Y`，middle/ring 落到 `+Z`，多处是 edge contact；
- complete contact loss `0.066 s`；
- relative drift `405.13 mm / 110.39°`；
- static success false。

这是重要负结果：**kinematic endpoint feasibility 不等于 collision-aware executable contact path，也不等于 force-closure**。详见 [contact_grasp_optimization_seed7.md](contact_grasp_optimization_seed7.md)。

### E14 — contact-event acquisition state machine

尝试 `synchronized`、`thumb_first`、`fingers_first` 三种 acquisition；三者都在严格 path feasibility 下失败。主要发现：finger2 指定 fingertip geom57 之前，distal non-tip geom55 会先碰到 cube 并推块。

结果不再被误报为“tip contact acquired”，而被正确标为 `non_tip_contact/path_infeasible`。详见 [contact_acquisition_seed7.md](contact_acquisition_seed7.md)。

### E15 — finger2 contact-region/q_mid path planning

对 center、upper、lower、±Y-side 做 contact-region 搜索，并为可行点加入 q_mid：

- kinematic endpoint/path 对 center、upper、-Y-side 可通过；
- dynamics 中三者均在 `36–38 ms` 先由 geom55 非 tip 碰撞；
- cube 被推动 `4.37–4.61 mm`、旋转 `5.46–5.74°`；
- finger2 被拒绝作为当前 palm pose 下 required `+X` contact，可降为 optional/support。

详见 [finger2_path_planning_seed7.md](finger2_path_planning_seed7.md)。

### E16 — thumb + middle + ring 三指拓扑

从 clean `run_reach(seed=7)` 开始，尝试 current grasp、same-palm opposition、小 yaw power 三个初始化。三者 endpoint 都无效；只有 thumb 满足，middle/ring error 仍约 `6–21 mm`。

这一步还修复了“从 legacy 已扰动状态开始会得到假可行结果”的 executor 问题。结论：应从 clean tabletop state 搜 palm workspace seeds，而不是继续调 contact gate。

详见 [three_finger_topology_seed7.md](three_finger_topology_seed7.md)。

### E17 — clean-state palm workspace map：提出但未完成

当时提出：扫描 arm-IK-reachable palm poses，寻找 middle/ring 覆盖 `+X` 内部、thumb 覆盖 `-X`，并从连续可行区域中心选择 seeds。

当前仓库有后续 `scripts/palm_object_geometry_diagnosis.py`（面向 thumb-index 的小范围 geometry diagnosis），但没有 clean-table thumb-middle-ring workspace map 的正式输出或专题报告。因此此项必须记为 **planned / not verified**，不能当作已经找到可行 palm seeds。

### E18 — fixed-palm thumb-index bounded reachability

- 只搜索 thumb + index 8D targets，其他三指保持 open。
- `1 manual + 1600 LHS + 360 local = 1961` candidates。
- `1658` 曾出现双接触，`56` 保持至少 `0.1 s`，但没有任何候选达到 `0.2/0.5 s`。
- 最佳连续双接触 `0.1333 s`；seeds 7–11 同一 target 为 `0/5` strong success。

结论：当前 fixed-palm geometry 下尚未证明最简单 stable pinch 存在，不值得直接训练 two-finger SAC。详见 [two_finger_pinch_reachability.md](two_finger_pinch_reachability.md)。

### E19 — cube size 70 mm → 54 mm 单变量消融

- 同样 `1961` candidates。
- 最长双接触只从 `0.1333 s` 增至 `0.1667 s`；仍无 `≥0.2 s` 或 stable pinch。
- best mean slip 反而从 `13.99` 增至 `32.47 mm/s`；normal angle 从 `90.36°` 到 `95.37°`，仍远非 antipodal。

结论：较小 cube 有轻微 reachability 改善，但不是解决方案。详见 [two_finger_pinch_cube54_ablation.md](two_finger_pinch_cube54_ablation.md)。

### E20 — 历史 `36cbb5b` scripted grasp 的 100-run Lift baseline

为了回答“最早自动抓取到底是不是有效”，从 commit `36cbb5b` 建独立 worktree `openarm-wuji-lift-baseline`，保留原 reset、IK、synergy、gain、friction、cube 和 joint-space Lift，只将 load-bearing success 定义清楚。

| 结果 | 数量 |
|---|---:|
| load-bearing PASS | 54/100 |
| acquisition failure | 22/100 |
| failure during Lift | 0/100 |
| failure during 1 s hold | 24/100 |
| drop | 24/100 |
| centered success | 0 |
| off-center success | 54 |

同 seed 7 重复 replay 轨迹完全一致。结论：它确实能在物理仿真中偶尔承重，不是假的 weld grasp；但成功率、偏心和漂移都不足以称为可靠 baseline，只作为 historical candidate 保存。

详细报告位于 sibling worktree：`D:/yl/embodied ai/openarm-wuji-lift-baseline/LIFT_BASELINE_100_REPORT.md`。

### E21 — 36cbb5b success-vs-failure causal diagnosis

固定原 100-run outcome，只重放 T0–T5 诊断字段：

- initial-pose-only 5-fold CV `accuracy/AUC = 0.710/0.780`，说明存在空间 success basin；
- 但小于 4 mm 的 matched initial-pose pairs 仍能产生不同 outcome，初始 XY 不是完整解释；
- `20/22` acquisition failures 在 pre-Lift 缺少 thumb，全部五指 topology 成功率 `44/61 = 72.1%`，无 thumb 的 fingers2–5 topology 仅 `2/17 = 11.8%`；
- total preload 对 hold success 仅弱/中等区分（Cliff's delta `0.202`），joint-specific preload/force 更有信息；
- hold failure 常从卸载后的 slip/rotation/drift 开始，完整 force/contact collapse 较晚；
- 综合结论 Case C：initial-pose coverage 与 post-contact grasp quality 都是主要瓶颈。

提出但没有在本项目主线继续执行的最小 A/B：grasp-reference Y、thumb joint4、index closing amplitude，各自单变量 100-seed 测试。详见 sibling worktree `SUCCESS_FAILURE_CAUSAL_REPORT.md`。

## 3.3 固定 palm 的 RL、action representation 与 expert prior

### E22 — Stage-1 SAC / 20D independent joint delta

- 固定 7D arm/palm；只学 20D Wuji joint delta。
- privileged observation 为 228D：hand q/qvel/targets/previous action、palm/cube pose、relative SE(3)、tip/pad geometry 和 aggregated contact features。
- horizon 120；成功要求持续 multi-contact、低 relative motion/drift 和有限 cube displacement。
- 256-step smoke：replay/gradient/actor update 均可工作，但 max contact 0，不能解释为已学会抓取。
- V1 5K：`6/41` episode 有任意 contact，max simultaneous contact `1`，success `0`；deterministic seeds 7–11 全部零 contact。

详见 [rl_grasp_stage1.md](rl_grasp_stage1.md)。

### E23 — Reward V2 5K

移除 persistent absolute proximity 和 thumb-specific bonus，改成 coverage progress、third-closest finger、contact transition/maintain；其他环境/action/success 不变。

| 指标 | V1 5K | V2 5K |
|---|---:|---:|
| any-contact episodes | 6/41 | 10/41 |
| max simultaneous contact | 1 | 1 |
| `≥2` contact episodes | 0 | 0 |
| stable success | 0 | 0 |

20D action mean absolute action 基本均匀，policy std 仍大，因此不是简单的单维 action collapse。Reward V2 消除了明显刷分方式，但没有产生 coordinated closing。停止继续 reward tuning，转向 action representation。详见 [reward_v2_5k_report.md](reward_v2_5k_report.md)。

### E24 — 手工 5D per-finger structured action

- 每根 finger 一个 scalar，沿 scripted closing direction 生成该 finger 的 4D joint delta。
- all-ones sanity test 达到 5 simultaneous contacts；五个 one-hot 测试确认不会污染其他 finger。
- fresh SAC 5K：any-contact `20/41`，但 max simultaneous learned contact 仍为 `1`；无 multi-contact、无 hold、无 success；seeds 7–11 仍零 contact。

结论：结构化 5D 提高 isolated contact 频率，但没有打破 stochastic exploration bottleneck；不继续 10K，也不提前加 20D residual。详见 [structured_action_ablation_5k.md](structured_action_ablation_5k.md)。

### E25 — 真实 Wuji cube teleop Parquet 数值分析

下载 `yeeeiii111/wuji-pick-and-place` 固定 revision，分析 60 条 cube episodes、`20,769` frames。所有 joint coordination 结论来自 54D state/action Parquet，不靠视频猜关节。

- left hand global indices `14:34`，right hand `34:54`；每手按五指 × 四关节排列。
- names 在数据元信息中为 null，具体命名按 Wuji 官方 flat-array convention 推断并明确标注。
- 每指在各自 closing phase 内近似一维，但不是固定全手 scalar。
- thumb 明显区别于另外四指，而且左右手符号不同：left thumb J3 为负，right thumb J2/J3 为负，J4 接近零；其余手指大多数 material directions 为正。
- finger onset 不同步，存在 staged closing；左右聚合 median onset spread 约 `0.617 s`。
- centered PCA：pooled state closing delta 达到 80/90/95% variance 约需 `2/4/6` 维；side-specific action 约 5D 可接近 95%。
- 手工 `[+,+,+,+]` closing direction 的 thumb 符号/比例明显错误。

导出 5 条典型 trajectory 和 machine-readable summary。详见 [wuji_cube_teleop_analysis.md](wuji_cube_teleop_analysis.md)。

### E26 — `expert_pca5_absolute` action prior + replay gate

- 从真实 teleop closing phases 构建 side-specific 5D PCA absolute posture action。
- PCA5 解释约 `94.83%` phase variance，20D reconstruction RMSE 约 `0.038 rad`。
- deterministic replay seeds 7–11：`5/5` 达到至少 2 contacts，`4/5` 达到至少 3 contacts。
- fresh SAC 5K：training max contact `4`；deterministic seeds 7–11 达到 3–4 contacts，全部曾 `≥3` 且部分维持 `0.1 s`，但没有 `0.3 s`，stable success `0/5`。

关键结论：expert PCA5 明确解决了原 20D/手工5D 的 coordinated exploration bottleneck，但新瓶颈变成 contact topology persistence，而不是“碰不到”。详见 [expert_pca5_action_prior.md](expert_pca5_action_prior.md)。

### E27 — contact topology stability diagnosis

对达到三 contact 的时刻逐帧分析：

- 没有固定的“三根手指组合”；
- 三 contact 通常只存在一个 30 Hz frame；
- 一个清晰案例中 finger4 位于 edge/corner、force 弱，最先掉；其他案例会多指近乎同时 collapse；
- 掉成两个 contacts 后也没有形成长期稳定的二指子拓扑。

结论：不能只针对某一固定手指增加 bonus；问题是整体 contact establishment 太脆弱。详见 [contact_topology_stability_diagnosis.md](contact_topology_stability_diagnosis.md)。

### E28 — contact latch / freeze-after-contact

假设：达到 multi-contact 后继续执行 SAC action 把已形成的抓取破坏。实现 contact-triggered posture latch 后，只有 seed 8 在约定条件下真正触发；drift 有所降低，但 `≥0.3 s` contact hold 仍为 `0/5`。

结论：post-contact continued motion 不是主要根因；latch 可作为诊断/未来 hybrid controller 元件，但不应冒充有效 Stage-1 baseline。详见 [contact_latch_5k_ablation.md](contact_latch_5k_ablation.md)。

### E29 — Reward V3 contact quality 5K

只在 Reward V2 上增加 interior-contact、contact persistence 和 tangential-slip 项；其余保持不变。

| 5-seed deterministic aggregate | V2 | V3 |
|---|---:|---:|
| max contacts mean | 3.6 | 3.6 |
| mean edge margin | 3.111 mm | 3.021 mm |
| mean tangential slip | 7.821 mm/s | 7.995 mm/s |
| `≥3` contact contiguous | 0.073 s | 0.047 s |
| stable success | 0/5 | 0/5 |

V3 保住 multi-contact exploration，却没有改善 edge/slip，三指持续性反而下降。按预设 stop rule 不继续 10K，停止 reward tuning。详见 [reward_v3_contact_quality_5k.md](reward_v3_contact_quality_5k.md)。

### E30 — PCA5 fixed-palm no-training reachability search

- `2500` global LHS + `720` local refinement + `50` full-hold candidates。
- full candidates：formal stable success `0/50`，Level-2 `0/50`，Level-1（三指 `0.1 s`）仅 `3/50`。
- 最长 `≥2 contacts = 0.267 s`，最长 `≥3 = 0.100 s`。
- 所有 50 个候选的 SE(3)、cube motion、penetration 条件基本都过，唯一一致失败项是持续 contact topology。
- finger5 在 full candidates 中仅 `2/50` 有有效接触；PC perturbation 常同时破坏多根手指。
- seeds 7–11 同一最佳 latent：formal success `0/5`。

结论：有限搜索不是数学不存在性证明，但足以停止 fixed-palm 下盲目堆 RL steps。首要怀疑 palm-object geometry/rigid contact domain，其次是 PCA5 缺少 independent local correction。详见 [pca5_fixed_palm_reachability_analysis.md](pca5_fixed_palm_reachability_analysis.md)。

### E31 — expert palm–hand coordination replay：数据不足而阻塞

公开数据只有 54D arrays 与 left/right arm/hand slice；缺少可确认的 arm joint names、机器人模型、base/tool transform、palm pose 与 FK recorder fields。因此不能诚实重建 expert palm SE(3)。

已完成的 fixed-palm A/B（raw expert 20D hand vs PCA5 reconstruction）在 5 条典型 episode 中都没有 `0.3 s` 双指保持或 stable success。该结果只说明当前 MuJoCo fixed-palm replay 不稳，不能回答“真实 expert palm movement 是否能救活抓取”。

结论：保持 5D PCA hand，不凭猜测扩成 8D/11D palm residual；先取得 expert robot model/FK transform。详见 [expert_palm_hand_coordination_replay.md](expert_palm_hand_coordination_replay.md)。

## 3.4 Coordinated demonstrations 与 LeRobotDataset

### E32 — 统一 successful trajectory recorder

现有 scripted pipeline 不改控制逻辑，每个 control step 同步保存：

```text
front_t + wrist_t
+ [7 arm actual q, 20 hand actual q]_t
→ [7 arm position target, 20 hand position target]_t
```

- state/action 均为 27D；`raw_script_action` 另存 10D 供审计/replay。
- 完整保存 reset 到 hold，不只截 grasp 后半段。
- 成功轨迹与 failure diagnostics 分目录；failure 不进入 BC dataset。
- cube poses at reset/before approach/before grasp/before Lift、first contact phase/body 全部作为 metadata。
- native LeRobotDataset v3.0 export、reload、DataLoader 与 exact state/action round trip 已通过。

详见 [coordinated_demonstrations.md](coordinated_demonstrations.md)。

### E33 — 20 successful demonstrations collection

- 23 attempts 得到 20 successes；失败 seeds 4、13、15 留在 diagnostics。
- 训练 seeds：`0,1,2,3,5,6,7,8,9,10,11,12,14,16,17,18,19,20,21,22`。
- 总 frames `2804`，30 Hz，front/wrist 均 `240×320 RGB`。
- DataLoader：state `[8,27]`、action `[8,27]`；ACT chunk 化后为 `[B,H,27]`。

这批 demo 足以做 pipeline smoke 和训练场景复现诊断，不足以声称广泛泛化。

## 3.5 Monolithic full-task ACT

### E34 — ACT 50-step end-to-end smoke

- LeRobot 0.6.2 ACT：ResNet-18、model dim 512、chunk100、27D output。
- loss 从 `68.460` 降到约 `5.237`；checkpoint save/reload 通过。
- seeds 0/7/11 全部 Reach 失败，hand 从第一个 frame 就过早闭合并推 cube。
- 没有 action-interface 或 normalization 断裂；失败属于行为未学会/训练不足。

结论：`MuJoCo expert → LeRobotDataset → ACT → MuJoCo controller` 软件链路跑通，但“复现 expert”尚未跑通。详见 [act_e2e_smoke.md](act_e2e_smoke.md)。

### E35 — 原 random-backbone、chunk100 续训到 500

- step 100–500 loss 持续降至 `2.111`，offline first-action MAE 降到约 `0.084 rad`。
- 20-seed step500：Reach/Approach/task 全部 `0/20`；mean min pregrasp `31.9 mm`；cube displacement mean `122.0 mm`。
- contact 偶尔很多，但全发生在有效 Reach 前，不能算 grasp。
- clipping 开始集中于 `wuji_left_finger1_joint1`。

重要教训：training loss / contact count 与 ordered closed-loop success 脱钩。详见 [act_500_step_fixed_seed_rollouts.md](act_500_step_fixed_seed_rollouts.md)。

### E36 — monolithic ACT 系统诊断与 execution-horizon ablation

完成：phase-wise full-chunk reconstruction、H_exec `1/5/10/20/50/100`、train/rollout preprocessing parity、temporal alignment、padding/mask、normalization 和 approximate expert-state reset。

关键结果：

- step500 Reach arm H=1 MAE `0.11599 rad`，是最差 phase；
- Reach 原始 frames 占 `9.52%`，但 chunk100 有效 target slots 只占 `1.10%`，Lift 占 `39.05%`；
- padding `35.31%`，但 mask 正确，没有把 padding 算入 loss；
- H_exec=1 仍 `0/3` Reach，说明 chunk100 open-loop execution 是放大器，不是唯一根因；
- state/action order、camera preprocessing、one-frame alignment、controller semantics 均未发现 bug；
- teacher-forced/near-expert reset 显示偏离后 error 快速累积，但 expert-state Reach prediction 本身也 underfit。

结论：主要问题是 visually conditioned Reach underfit + phase objective imbalance，而不是接口错位。详见 [act_closed_loop_failure_diagnosis.md](act_closed_loop_failure_diagnosis.md)。

### E37 — ImageNet ResNet-18 initialization ablation

只把 backbone 从 random 改为 `IMAGENET1K_V1`，其余 chunk100/data/seeds/interface 不变。

- frame-0 arm spread/expert：random step500 `2.97%`，ImageNet step500 `67.49%`，step1000 `93.50%`；视觉 conditioning 明显恢复，主要来自 front camera。
- matched step1000：random 与 ImageNet 都是 Reach `0/20`；ImageNet mean min pregrasp 从 `25.46` 改善到 `20.17 mm`，典型 cube displacement 更小，但 seed11 有 `804 mm` workspace-ejection outlier。
- ImageNet 产生更多早期 multi-contact，但没有先通过 ordered Reach。

结论：ImageNet beneficial but insufficient；后续保留 ImageNet 初始化。详见 [act_imagenet_backbone_ablation.md](act_imagenet_backbone_ablation.md)。

### E38 — Phase-balanced + chunk20，step200/500

只改变两个变量：phase-aware anchor sampling（25/25/20/10/10/10%）和 chunk `100→20`；H_exec 保持 1。

step200：Reach `0/20`，但 mean min pregrasp `17.08 mm`、Reach arm MAE `0.06992 rad`，比旧 step500 更高样本效率；同时 `15/20` cube displacement >25 mm，过早 contact 很严重。

step500：

- Reach/Approach/task 仍 `0/20`；
- Reach arm MAE 降至 `0.02369 rad`；mean/best pregrasp `15.45/2.49 mm`；
- >25 mm 推块从 `15/20` 降至 `1/20`；
- 但没有任何 seed 连续 5 frames 通过 Reach；
- clipped values 增至 `2920`，全部为 thumb joint1 lower-bound mismatch。

结论：连续指标改善但 binary Reach 仍零，必须做 2×2 因果消融。详见 [act_phase_balanced_chunk20_step200.md](act_phase_balanced_chunk20_step200.md)、[act_phase_balanced_chunk20_step500.md](act_phase_balanced_chunk20_step500.md)。

### E39 — phase balancing × chunk size 2×2

所有格都在 step500、seeds 0–19、H_exec=1：

| Sampling | chunk100 | chunk20 |
|---|---|---|
| Normal | A: Reach 0/20；pregrasp 28.49 mm；Reach-arm MAE 0.11419；cube median 139.02 mm | B: Reach 0/20；pregrasp 16.28 mm；MAE 0.04003；cube median 18.26 mm |
| Phase-balanced | C: Reach 0/20；pregrasp 20.12 mm；MAE 0.05815；cube median 26.88 mm | D: Reach 0/20；pregrasp 15.45 mm；MAE 0.02369；cube median约0 mm |

- shorter chunk 和 phase balancing 都独立改善连续 Reach metrics；
- 存在 diminishing-return interaction，但 D 仍是整体最优失败模型；
- 四格全部 binary Reach 0/20，因此强制结论 Case E：没有配置解决 Reach；保留 D 继续诊断。
- clipping 四格都只在 thumb joint1，主要是 absolute-target lower-bound calibration 问题，不是 chunk20 单独制造。

详见 [act_phase_chunk_2x2_ablation.md](act_phase_chunk_2x2_ablation.md)。

### E40 — 最佳 D 从 step500 精确 resume 到 1000/2000

- step1000：Reach `0/20`，mean min pregrasp `13.06 mm`，`7/20` 进入 position-only 12 mm basin，但最长连续只有 3 frames。
- step2000：Reach `0/20`，mean min pregrasp 退化到 `19.64 mm`，`0/20` 进入 basin。
- Reach arm H=1 MAE `0.02369 → 0.02434 → 0.02914 rad`，没有继续下降。
- target delta magnitude 在 step2000 更平滑，但轨迹带 bias；不是“已到 basin 只因 jitter 无法 hold”。
- thumb joint1 clipping 在 step2000 每一帧发生。

强制结论 Case D：binary 和 continuous Reach 在 step2000 plateau/regress，停止在同一 20-demo full-task 数据上堆 steps。保留 step1000 作为该曲线最佳比较 checkpoint。详见 [act_best_config_d_to_step2000.md](act_best_config_d_to_step2000.md)。

### E41 — D step1000 对真正 training initial conditions 的纯 closed-loop 复现

- 使用 20 个 training demo seeds/相同 frame-0 robot state，不读取 expert action。
- Reach/Approach/task 均 `0/20`。
- mean/median/best pregrasp `13.35/13.99/4.79 mm`；`9/20` 进入 12 mm，最长连续 3 frames。
- cube displacement >25 mm `4/20`；early contact `5/20`。

这排除了“只是不泛化到新 seed”的解释：monolithic ACT 连训练场景都无法稳定 closed-loop reproduce。详见 [act_step1000_training_demo_closed_loop.md](act_step1000_training_demo_closed_loop.md)。

### E42 — Reach-only ACT

从同 20 条 successful demos 截取 `reset → Reach → 8-frame stable pregrasp hold`：20 episodes、427 frames；不含 Approach/Grasp/Lift，Wuji target 恒定。fresh ImageNet ACT、chunk20、normal sampling。

| step | Reach | mean min error | enter 12 mm | timeout | clipping | early contact |
|---:|---:|---:|---:|---:|---:|---:|
| 500 | 11/20 | 7.18 mm | 20/20 | 9 | 0 | 0 |
| 1000 | 16/20 | 5.41 mm | 20/20 | 4 | 0 | 0 |
| 2000 | 19/20 | 4.36 mm | 20/20 | 1 | 0 | 0 |

唯一失败 seed8 已进入 gate 8 frames，但最长连续 4，差一帧。结论 Case A：移除 phase ambiguity 后，ACT 可以学会单阶段 Reach；monolithic full-task 的主要问题是 phase/temporal ambiguity。由此正式转向 staged policy，而不是继续 full-task ACT。详见 [act_reach_only_diagnosis.md](act_reach_only_diagnosis.md)。

## 3.6 Staged ACT：Reach、Approach 与 Recovery

### E43 — Reach ACT → Approach ACT

- 冻结 Reach-only step2000。
- 构建 Approach-only dataset：20 episodes、957 frames（797 real Approach + terminal hold）；Wuji 保持 open/pre-shape。
- fresh Approach ACT step500/1000/1500/2000 的 exact-start success 为 `0/20, 5/20, 12/20, 11/20`；但真实 Reach handoff 上 step2000 更好。
- chained 20 training seeds：Reach `19/20`；Approach|Reach `16/19`；joint `16/20`。
- 失败：seed8 Reach；seed2 cube-motion safety；seeds6/16 terminal timeout。

结论：real Reach endpoint 对 Approach 可用，瓶颈在 Approach terminal correction/stop，不在 handoff distribution distance。详见 [staged_act_pipeline.md](staged_act_pipeline.md)。

### E44 — 200-rollout Approach near-failure mining

冻结两 policy，运行 seeds 1000–1199：

- Reach `160/200`；Approach success `115/160`，failure `45`（18 cube displacement、27 timeout）。
- detector 得到 499 triggers，覆盖 150 rollouts。
- 按 terminal error direction/magnitude、arm q、cube displacement、failure mode 去重为 50 snapshots。
- snapshot 保存 MuJoCo integration state、qpos/qvel/act/ctrl、warm-start、controller bookkeeping、reference、图像与 observation。
- scripted Approach expert 成功恢复 `30/50`，生成 30 episodes / 559 frames correction dataset；失败 correction 不入 BC。
- correction directions 覆盖六个轴向和 8 个 error octants，最常见是 `-Z`。

详见 [approach_near_failure_mining.md](approach_near_failure_mining.md)。

### E45 — 原 Approach + 30% correction 混合重训

fresh ACT，实际 mix 70% original / 30% correction：

- 最佳 step500 unseen Approach|Reach `111/156 = 71.2%`；原 baseline 为 `116/157 = 73.9%`。
- cube safety failures `20→8`，但 timeout `21→37`。
- overall recovery after trigger `72.5%→71.2%`，没有提升。

结论：correction data 让 push 更小、timeout 更多，安全改善但 nominal/recovery success 退化。按预案降到 20%，不修改其他变量。详见 [approach_act_with_correction_demos.md](approach_act_with_correction_demos.md)。

### E46 — 20% correction matched retry

- 最佳 step1500 unseen Approach|Reach `101/156 = 64.7%`；cube failures 降到 `2`，timeouts 增至 `53`。
- recovery after trigger 降到 `62.8%`。
- 安全/成功 trade-off 比 30% 更明显，未解决 target conflict。

结论：保留原 Approach ACT，不再继续调 mixture ratio；correction trajectory 更适合作为独立 Recovery policy。详见 [approach_act_with_correction20.md](approach_act_with_correction20.md)。

### E47 — 独立 Recovery ACT + first-trigger router

- Recovery 只训练 30 successful corrections / 559 frames；不混 original Approach。
- exact correction starts：step1500 最佳 `19/30 = 63.3%`，无 cube failure。
- 但在 matched unseen seeds 1400–1599，第一 trigger 立即切 Recovery：Approach-stage success 从 baseline `119/154 = 77.3%` 降到 `96/154 = 62.3%`；rescued 20、regressed 43，McNemar `p=0.005152`。
- cube safety failures `17→4`，但 timeouts `18→54`。

结论：Recovery policy 本身有用，错误在 router 的 switch timing；single noisy trigger 不能证明 Approach 会失败。详见 [staged_act_with_recovery.md](staged_act_with_recovery.md)。

### E48 — Recovery router ablation

不重训任何 policy，用同一 trigger snapshot fork 比较 Approach future 与 Recovery future，并测试 persistence/hysteresis rules。

`selective_hysteresis` closed-loop matched result：

- 155 Reach successes 中 switch 74 次；Recovery 成功 48；
- Approach-stage success `112→129 = 83.2%`；
- rescued 22、regressed 5，net `+17`；
- timeout `20→20`，不增加；
- cube safety failures `23→6`。

结论 Case A：简单持久性/迟滞 router 已经 net-positive，无需 learned router。冻结 Reach step2000、Approach step2000、Recovery step1500 与 `configs/recovery_router.json`；推送到 main 的 checkpoint 为 `d29299e`。详见 [recovery_router_ablation.md](recovery_router_ablation.md)、[frozen_staged_act_v1.md](frozen_staged_act_v1.md)。

## 3.7 GraspSecure ACT 与 scripted Lift

### E49 — 独立 Grasp + Preload / GraspSecure ACT

- 上游 Reach/Approach/Recovery/router 全部冻结。
- dataset：50 successful episodes、2107 frames；20 nominal scripted starts + 30 real frozen staged handoffs。
- policy input/output 保持双 RGB +27D state →27D absolute target；contact/force/cube/preload telemetry 不输入 policy。
- fresh ImageNet ACT、chunk20、H_exec=1。

standalone 20-seed：

| step | Grasp+Preload success | Grasp | Preload | >25 mm cube motion |
|---:|---:|---:|---:|---:|
| 500 | 9/20 | 19/20 | 9/20 | 5 |
| 1000 | 0/20 | 14/20 | 1/20 | 6 |
| 1500 | 14/20 | 20/20 | 15/20 | 1 |
| 2000 | 6/20 | 17/20 | 7/20 | 1 |

选 step1500。success terminal preload L2 mean `1.353 rad`，failure `0.726 rad`；这是关联，不是因果。真实 staged handoff：upstream `12/20`，Grasp+Preload `9/12 = 75%`，joint `9/20 = 45%`。

当时结论 Case A 是“GraspSecure stage 可冻结并接 scripted Lift”，并不表示已经证明 load bearing。详见 [grasp_preload_act.md](grasp_preload_act.md)。

### E50 — Conditional GraspSecure → scripted Lift

- 使用 frozen full pipeline 跑 seeds 4000–4076；77 rollouts 才积累 30 个 safe GraspSecure terminal states。
- 50 个 Approach-stage success 中有 32 个 formal GraspSecure passes；2 个被 pre-Lift 25 mm safety guard 拒绝；30 个实际 Lift。
- handoff 不 reset、不注入 expert state，不把 hand target 改成 actual qpos；20D hand controller target 全程最大变化为 0。
- scripted Lift trajectory、gain、grasp 参数保持不变，Lift 后 hold 30 frames = 1 s。

结果：`14/30 = 46.7%`；12 drops、2 height-not-held、2 environment-support failures。按所有 formal GraspSecure passes 计为 `14/32 = 43.8%`。

### E51 — Fresh 100-run full staged load-bearing evaluation

seeds 5000–5099：full success `17/100`。互斥 first failure stage：

| failure stage | count |
|---|---:|
| Reach | 26 |
| Approach | 0 |
| Recovery | 13 |
| grasp formation | 3 |
| preload formation | 15 |
| terminal hold / pre-Lift safety | 9 |
| Lift | 8 |
| Lift hold | 9 |

34 个实际 Lift attempts 中成功 17；失败包括 8 drop、7 environment-support、2 height-not-held；另有 5 formal GraspSecure passes 被 safety guard 拒绝。

preload 与 load outcome：full success/failure median `1.360/1.294 rad`、Cliff's delta `+0.260`；pooled median `1.360/1.291`、delta `+0.357`。有正相关但分布明显重叠，所以不能只提高 scalar threshold。

fresh success/failure 的 median max palm-frame translation drift 为 `42.8/54.2 mm`，rotation drift `15.8/79.6°`；最新评测没有把 8 mm/6° 重新设成 hard gate。

最终强制决策 Case B：GraspSecure passes 中仍有大量 load failure，static gate 不足；保持 Lift scripted，先改进 load-bearing grasp criterion。详见 [staged_act_with_scripted_lift.md](staged_act_with_scripted_lift.md)。

本轮实现后的完整测试集为 `90` 项通过；另对 `64` 个实际 Lift NPZ 做了结构检查，所有成功轨迹都满足 30-frame（1 s）unsupported hold，且 state/action 均为 `[T,27]`。这证明最新失败率不是 recorder 或 handoff 文件损坏造成的。

### E52 — GraspSecure gate confusion-matrix diagnostic（forced Lift）

冻结 Reach/Approach/Recovery、`selective_hysteresis` router、GraspSecure step1500、27D absolute controller-target semantics 和原 scripted Lift。唯一诊断变化是：GraspSecure gate 与 25 mm pre-Lift safety reference 只保留为标签；只要 simulator finite 且 cube 未越出宽松 hard workspace guard，PASS 与 FAIL 均执行完全相同的 scripted Lift。本轮无训练、无 expert action、无 terminal-state reset，也没有修改正式 gate 或 Lift success definition。

fresh seeds `6100–6261` 共运行 162 个完整 staged rollouts；131 Reach success，101 个到达 GraspSecure terminal，PASS/FAIL=`63/38`。其中 1 个状态因真正 workspace invalid 被硬保护排除，最终得到整 100 次实际 Lift：37 success、36 environment-support、21 drop、6 height-hold failure。43 次为 diagnostic forced Lift，23 个 terminal 被 formal 25 mm safety reference 拒绝。

关键 confusion matrix：`TP=27, FP=36, FN=10, TN=27`。因此 `P(Lift success | PASS)=42.9%`、`P(Lift success | FAIL)=27.0%`、recall `73.0%`、specificity `42.9%`、accuracy `54.0%`。25 mm safety 标签也几乎不分离：`P(success|safety PASS)=37.2%`，`P(success|safety reject)=36.4%`。

preload-only ROC AUC 仅 `0.621`。当前 `1.1775 rad` 的 preload-only precision/recall 为 `41.9%/70.3%`；同样本 best-F1 threshold `0.880 rad` 虽将 recall 提至 `94.6%`，precision 仍仅 `47.3%`，不能作为新 gate。FP 与 TP 的 preload median (`1.309/1.283 rad`)、五指 topology 和 terminal slip 高度重叠；10 个 FN 中 9 个仍是五指 topology，contact persistence median `2.30 s`，却因 preload/terminal-hold conjunction 被拒绝。

只使用 pre-Lift features 的 class-balanced L2 logistic 在 stratified 5-fold OOF 上得到 AUC `0.714`、balanced accuracy `0.685`、precision `55.3%`、recall `70.3%`、specificity `66.7%`。这只说明 per-joint preload pattern、contact persistence/coverage、force distribution、cube-palm geometry 与 relative motion 比单一 L2 更有信息；51D feature、100 samples 下不能部署或宣称因果。

强制结论 **Case B**：现有 gate 同时存在实质 false positives 与 false negatives，定义本身需要重做；但 preload scan 也呈现 Case D 的机制证据，因此下一步应在 held-out seeds 上预注册验证 geometry/persistence/小幅 physical load probe，而不是直接移动 `1.1775 rad`。报告见 [graspsecure_gate_confusion_matrix.md](graspsecure_gate_confusion_matrix.md)，机器可读结果和 100 条完整 Lift telemetry 位于 `outputs/graspsecure_gate_confusion/`。新增 evaluator 与测试后，全套 `93` tests 通过；Parquet 100 rows/60 columns、101 terminal snapshots、100 Lift NPZ、共 5600 Lift frames 均通过 shape/finite 检查。

### E53 — Micro-lift verification + one frozen GraspSecure retry

保持 Reach/Approach/Recovery/router/GraspSecure、27D absolute controller target、PD/rate limit 和原 scripted Lift 全部冻结。新增的 deployable 分支只做：GraspSecure terminal 后复用原 Lift 的 15 mm reference prefix，保留 live 20D hand target，等 rate-limited arm 实际进入至少 10 mm micro-lift 后 hold 0.3 s；若 probe FAIL，则沿相同 prefix 最小回落/settle，并仅重跑一次 frozen GraspSecure。为了测 `P(full Lift success | probe FAIL)`，conditional evaluator 另开了明确标注的 counterfactual snapshot branch；实际 retry branch 从未恢复原 grasp-start。

正式评估复用 E52 保存的 60 个 exact terminal snapshots。首次 smoke 发现固定 9 帧并不是实际 hold：arm target 尚在 rate-limit 爬升，actual palm 只移动 1–3 mm；这是 evaluator 时序 bug，正式 60 条前修成“实际 palm 到位后才开始计 0.3 s”，控制曲线与阈值不因结果调优。

正式结果：probe1 PASS/FAIL=`0/60`。虽然所有状态都被 probe 拒绝，FAIL 后继续原 Lift 仍有 `18/60=30.0%` 成功，2×2 为 `TP=0, FP=0, FN=18, TN=42`，所以 precision 无法定义、recall 为 0。criterion pass counts：finite 60、physical hold complete 55、palm maintained 46、cube followed 14、support released 6、contact retained 52、no-fast-escape 6。这说明 contact collapse 不是主因；在 15 mm reference 插入静态 micro-hold 改变了原连续上抬动力学，无法作为无损 load-bearing predictor。Post-hoc 仅移除 drift/speed 两项仍只有 1/60 满足核心条件（TP/FP/FN/TN=`1/0/17/42`），所以结论不依赖这两个阈值。

60 条全部执行一次 regrasp；return 60/60 完成，57 条回落后物理有效，57 条形成可比较 attempt2 state，56 条可执行第二 probe，但 probe2 PASS 仍为 `0/56`，因此 rescue=`0`、absolute retry gain=`0`。Regrasp 确实不是数值重复：57/57 被 continuous-change diagnostic 标为不同，median target/actual 20D L2 change=`0.112/0.754 rad`，median cube-palm translation/rotation change=`24.51 mm/42.28°`，28/57 topology changed；然而这些变化过大且没有转化成更好的 load-bearing state。

强制结论 **Case C**：当前 probe 自身不能分离好/坏 grasp，主要瓶颈是 probe precision/动力学设计，不能归咎于 regrasp quality。按预注册流程，conditional 未证明有效，因此没有启动 100–200 fresh-seed full staged comparison，也没有冻结/部署此机制。报告见 [grasp_probe_regrasp_retry.md](grasp_probe_regrasp_retry.md)，机器可读 summary、60 条 baseline/probe/forced-continuation/return/regrasp/probe2 轨迹位于 `outputs/grasp_probe_regrasp_retry/`。最终全套 `99` tests 通过；353 个新 NPZ 均通过 JSON、27D state/target、finite 和 probe hand-target-preservation 检查。

### E54 — 正式 frozen staged ACT 项目收口与 fresh 100-seed benchmark

停止 retry、micro-lift probe、GraspSecure gate 和 policy 优化。新增唯一正式 manifest `configs/staged_pipeline.json`，固定 Reach step2000、Approach step2000、Recovery step1500、`selective_hysteresis` router、GraspSecure step1500、30 Hz 27D absolute-target contract、原 scripted Lift 与 1 s hold。manifest 只使用 repo-relative paths，并保存四个约 206.5 MB policy weights 和约 89.7 MB MuJoCo model 的 size/SHA-256；大文件继续由 `.gitignore` 排除。

正式入口 `scripts/run_staged_pickup.py` 支持单 seed、批量、断点续跑、manifest/artifact `--check-only`、每 episode 明确 phase timeline、`failure_stage`/`failure_reason`、compact/full JSON，以及成功、上游失败、Lift 失败各一条双相机 GIF 和 phase timeline。正式配置明确关闭 retry、probe、Lift ACT 和 online expert action。

经过 E52/E53 的诊断，GraspSecure static gate 在正式 pipeline 中降为 telemetry：只要 GraspSecure terminal 的 qpos/qvel/ctrl/cube state finite，且原 25 mm pre-Lift cube-motion safety guard 通过，就从真实 closed-loop terminal state 进入原 scripted Lift；hand target 保留 controller target，绝不改为 actual qpos。正式 success 仍要求实际完成 Lift、无环境支撑并 hold 1 s。

fresh seeds `9000–9099` 的 100-run 正式结果：Reach `78/100`；Approach-stage `60/78=76.9%`；finite GraspSecure terminal `60/60`；pre-Lift safety pass `44/60`；actual Lift success `21/44=47.7%`；按全部 finite terminal 计 `21/60=35.0%`；full success `21/100=21.0%`。Recovery trigger `34/78=43.6%`，trigger 后成功 `16/34=47.1%`。互斥 failure stages：Reach 22、Recovery/Approach 18、pre-Lift terminal safety 16、Lift drop 10、Lift hold 13。

静态 gate 的 60-terminal 交叉表为：PASS success/fail=`17/21`，FAIL success/fail=`4/18`；seeds `9030,9032,9033,9097` 是 gate FAIL 但完整 Lift+hold success。它再次证明 static gate 不能硬阻止所有 Lift。代表性 deterministic replay 为：success seed9002、Reach failure seed9000、Lift drop seed9003，三者的 outcome/failure reason 与 benchmark 一致。

机器可读摘要：[staged_pickup_fresh_seed9000_100.json](../benchmarks/staged_pickup_fresh_seed9000_100.json)。统一说明、复现命令、模块/判据和限制见 [FINAL_STAGED_ACT_REPORT.md](FINAL_STAGED_ACT_REPORT.md)。

最终全套 `103/103` tests 通过；额外 formal-runner smoke（seed8999 作为独立 inference stream）完整完成 Reach→Recovery→GraspSecure→Lift→Hold。manifest 的 5 个大资产 size/SHA-256 全部匹配，三个 representative replay 的 benchmark outcome、failure stage 和 failure reason 全部一致。

## 4. 走过的弯路，以及为什么它们仍然有价值

| 弯路/早期假设 | 为什么当时合理 | 后来看到的反证 | 保留下来的价值 |
|---|---|---|---|
| 用“抬升高度 + 多指接触”宣布成功 | 最容易自动化 | seed7 可抬高但 relative SE(3) 大幅漂移 | 促成 task/stability/diagnostic 三分法和 outcome taxonomy |
| 围绕 seed7 调 freeze/preload/Lift | 有稳定复现样本，便于调试 | 0.74–0.93 没有 strict success，高 preload 可产生 698 N transient | 建立 rate limit、settle、S-curve、连续 telemetry |
| kinematic endpoint contact optimization | 能直接编码目标 face/topology | 动态路径先碰 non-tip，free cube 被推走，endpoint topology 崩溃 | 明确必须做 collision-aware execution 与 contact-event validation |
| finger2 必须覆盖 +X | 直觉上 index 最适合对置 thumb | geom55 总在 fingertip 前碰撞并推 cube | finger2 改为 optional，暴露 collision-geometry 语义问题 |
| 20D independent SAC + reward tuning | 最通用、表达力最大 | V1/V2 5K 都只有 1 contact；action std 并未 collapse | 证明瓶颈是 coordinated exploration，不只是 reward |
| 手工 5D 全指 closing | scripted all-ones 能到 5 contacts | learned policy 仍 max1；teleop 显示 thumb 符号/比例错误且 staged | 引导转向真实 expert-derived PCA |
| PCA5 解决全部抓取 | replay/训练可到 3–4 contacts | topology 通常 1–3 frames 内 collapse，0/5 stable | 证明 PCA5 解决 exploration，但不解决 stable geometry |
| contact latch 能防止策略破坏抓取 | 继续动作确实可能扰动 contact | latch 不产生 0.3 s hold | 排除 post-contact action 作为主因 |
| Reward V3 interior/slip 会稳定 contact | 指标物理含义合理 | edge/slip 不改善，三指 duration 退化 | 停止 reward tuning，转向 reachability/geometry |
| fixed-palm 下继续长训 | 可能只是 SAC sample 不够 | 3220-candidate search、1961 pinch search 都没有稳定解 | 避免用 RL 搜索未证明存在的解 |
| monolithic ACT loss 下降会带来闭环成功 | BC 常用指标 | full-task 0/20，连 training seeds 也 0/20 | 建立 offline-vs-closed-loop、gate/dwell、alignment audits |
| random ResNet 可由 20 demos 学视觉 | end-to-end 简洁 | frame0 visual spread 仅 2.97% expert | ImageNet 成为后续固定初始化 |
| chunk20/phase balance 单独就能修好 Reach | 连续 MAE 明显改善 | 2×2 四格仍 0/20，D 到2k退化 | 证明 phase ambiguity 比单纯 chunk 更关键 |
| 把 correction 混进 Approach BC | 可直接教 recovery | 30%/20% 都以更多 timeout 换更少 push，success 下降 | correction 数据改用于独立 Recovery ACT |
| 第一个 near-failure trigger 就切 Recovery | 反应快 | rescued20、regressed43，显著负效应 | 促成 matched counterfactual router 与 hysteresis |
| GraspSecure static gate 代表可承重 | contact/preload/hold 都看似合格 | conditional Lift 仅46.7%，full `Lift|Grasp=43.6%` | 生成带 load outcome 的 terminal-state 数据，下一步可校准 load-bearing gate |
| preload threshold 可以校准 GraspSecure gate | preload 与历史 Lift success 有弱正相关 | forced-Lift confusion 中 PASS precision 42.9%、FAIL success 27.0%；preload AUC 0.621 | 证明要联合 geometry、persistence、force distribution 或 physical load probe |
| 15 mm static micro-lift hold 可以验证承载 | 小幅 support release 物理直观，且可复用 Lift prefix | 0/60 PASS，但 FAIL 后 continuation 仍 18/60 success；probe2 0/56 | 发现 hold 会改变当前 rate-limited continuous-Lift 动力学，verification 必须做到无损或改用在线连续信号 |

这些负结果不应删除。它们缩小了假设空间，也构成项目最有价值的工程/研究叙事：每次转向都有受控实验，而不是凭感觉换算法。

## 5. 当前被冻结与保留的资产

### 5.1 已冻结并推送的 staged v1

- Reach ACT：`outputs/act_reach_only/act_train/checkpoints/002000/pretrained_model`
- Approach ACT：`outputs/act_staged/approach_only/act_train/checkpoints/002000/pretrained_model`
- Recovery ACT：`outputs/staged_act_with_recovery/recovery_act_train/checkpoints/001500/pretrained_model`
- Router：`configs/recovery_router.json`，`selective_hysteresis`
- Git checkpoint：`d29299e` / `staged-act-router-v1`

### 5.2 正式 frozen staged pickup 资产

- GraspSecure config：`configs/grasp_secure_stage.json`
- GraspSecure checkpoint：`outputs/grasp_preload_act/act_train/checkpoints/001500/pretrained_model`
- Conditional/full Lift records：`outputs/staged_act_with_scripted_lift/`
- Gate diagnostic：`outputs/graspsecure_gate_confusion/`
- Probe/regrasp conditional diagnostic：`outputs/grasp_probe_regrasp_retry/`
- Formal manifest：`configs/staged_pipeline.json`
- Formal runner：`scripts/run_staged_pickup.py`
- Fresh 100-seed compact result：`benchmarks/staged_pickup_fresh_seed9000_100.json`
- 最新报告：[FINAL_STAGED_ACT_REPORT.md](FINAL_STAGED_ACT_REPORT.md)、[grasp_preload_act.md](grasp_preload_act.md)、[staged_act_with_scripted_lift.md](staged_act_with_scripted_lift.md)、[graspsecure_gate_confusion_matrix.md](graspsecure_gate_confusion_matrix.md)、[grasp_probe_regrasp_retry.md](grasp_probe_regrasp_retry.md)

### 5.3 历史 scripted baseline

- commit `36cbb5b`
- independent worktree：`D:/yl/embodied ai/openarm-wuji-lift-baseline`
- 100-run raw/telemetry/causal outputs：该 worktree 的 `outputs/lift_baseline_100/`

## 6. 尚未完成、不可误报为完成的事项

- clean-table thumb-middle-ring palm reachable workspace map 没有正式结果产物。
- public teleop dataset 的 expert palm SE(3) 无法从现有字段可靠恢复；需要 arm names/model/FK/base-tool transform。
- 未完成任何真实硬件通信、tool-to-palm 标定、真实 contact-force/tactile 校准或 sim-to-real 验证。
- 未训练 Lift ACT；当前 Lift 仍是 scripted reference。
- 未训练 SmolVLA。
- semantic 11-action hand interface 只是一份设计基线，未实现/未验证，见 [dexterous_semantic_action_space_v0_1.md](dexterous_semantic_action_space_v0_1.md)。
- ACT 的 training-seed 成功不能当成 unseen distribution 泛化；Reach-only 的 19/20 只证明 phase ambiguity 假设。
- CD-WM 8 mm/6° 不是 Wuji 最终稳定性阈值。

## 7. 未来研究问题（当前暂停，仅记录）

如果未来恢复算法优化，当前最干净的问题是：

> 在保持 frozen policies、scripted Lift、controller 和 safety reference 不变时，什么 pre-Lift measurement 能预测真正的 load-bearing outcome？

建议利用已经保存的 64 个 actual Lift attempts 和 exact GraspSecure terminal states，比较：

- terminal preload 的完整 20D pattern，而不只是 L2；
- per-finger normal-force distribution、force balance/HHI 与 thumb opposition；
- short active load probe / very small support-release transient；
- 不插入静态 hold 的 continuous-prefix online verification，或对 probe 后续 Lift 做 matched calibration；
- palm-frame translation/rotation velocity、incipient slip；
- contact topology persistence 与 zero-contact runs；
- 预注册的 held-out seed split，避免在同一 64 条上同时挑 threshold 又报性能。

本次已经按现状冻结完整 staged pipeline；未来若训练 Lift policy 或部署 retry，应建立新版本 manifest，并始终与本 frozen benchmark 对照。

## 8. 专题报告索引

### 基础与接口

- [environment_report.md](environment_report.md)
- [setup_decisions.md](setup_decisions.md)
- [upstream_versions.md](upstream_versions.md)
- [milestone_1_openarm.md](milestone_1_openarm.md)
- [milestone_2_wuji.md](milestone_2_wuji.md)
- [milestone_3_4_combined.md](milestone_3_4_combined.md)
- [lerobot_integration.md](lerobot_integration.md)
- [episode_recording.md](episode_recording.md)
- [reach_grasp_lift.md](reach_grasp_lift.md)

### Scripted grasp、几何与 reachability

- [grasp_settle_experiment.md](grasp_settle_experiment.md)
- [grasp_geometry_diagnostics.md](grasp_geometry_diagnostics.md)
- [explicit_grasp_constraints_journey.md](explicit_grasp_constraints_journey.md)
- [contact_grasp_optimization_seed7.md](contact_grasp_optimization_seed7.md)
- [contact_acquisition_seed7.md](contact_acquisition_seed7.md)
- [finger2_path_planning_seed7.md](finger2_path_planning_seed7.md)
- [three_finger_topology_seed7.md](three_finger_topology_seed7.md)
- [two_finger_pinch_reachability.md](two_finger_pinch_reachability.md)
- [two_finger_pinch_cube54_ablation.md](two_finger_pinch_cube54_ablation.md)

### RL 与 expert hand prior

- [rl_grasp_stage1.md](rl_grasp_stage1.md)
- [reward_v2_5k_report.md](reward_v2_5k_report.md)
- [structured_action_ablation_5k.md](structured_action_ablation_5k.md)
- [wuji_cube_teleop_analysis.md](wuji_cube_teleop_analysis.md)
- [expert_pca5_action_prior.md](expert_pca5_action_prior.md)
- [contact_topology_stability_diagnosis.md](contact_topology_stability_diagnosis.md)
- [contact_latch_5k_ablation.md](contact_latch_5k_ablation.md)
- [reward_v3_contact_quality_5k.md](reward_v3_contact_quality_5k.md)
- [pca5_fixed_palm_reachability_analysis.md](pca5_fixed_palm_reachability_analysis.md)
- [expert_palm_hand_coordination_replay.md](expert_palm_hand_coordination_replay.md)

### Demonstrations 与 monolithic ACT

- [coordinated_demonstrations.md](coordinated_demonstrations.md)
- [act_e2e_smoke.md](act_e2e_smoke.md)
- [act_500_step_fixed_seed_rollouts.md](act_500_step_fixed_seed_rollouts.md)
- [act_closed_loop_failure_diagnosis.md](act_closed_loop_failure_diagnosis.md)
- [act_imagenet_backbone_ablation.md](act_imagenet_backbone_ablation.md)
- [act_phase_balanced_chunk20_step200.md](act_phase_balanced_chunk20_step200.md)
- [act_phase_balanced_chunk20_step500.md](act_phase_balanced_chunk20_step500.md)
- [act_phase_chunk_2x2_ablation.md](act_phase_chunk_2x2_ablation.md)
- [act_best_config_d_to_step2000.md](act_best_config_d_to_step2000.md)
- [act_step1000_training_demo_closed_loop.md](act_step1000_training_demo_closed_loop.md)
- [act_reach_only_diagnosis.md](act_reach_only_diagnosis.md)

### Staged ACT

- [FINAL_STAGED_ACT_REPORT.md](FINAL_STAGED_ACT_REPORT.md)
- [staged_act_pipeline.md](staged_act_pipeline.md)
- [approach_near_failure_mining.md](approach_near_failure_mining.md)
- [approach_act_with_correction_demos.md](approach_act_with_correction_demos.md)
- [approach_act_with_correction20.md](approach_act_with_correction20.md)
- [staged_act_with_recovery.md](staged_act_with_recovery.md)
- [recovery_router_ablation.md](recovery_router_ablation.md)
- [frozen_staged_act_v1.md](frozen_staged_act_v1.md)
- [grasp_preload_act.md](grasp_preload_act.md)
- [staged_act_with_scripted_lift.md](staged_act_with_scripted_lift.md)
- [graspsecure_gate_confusion_matrix.md](graspsecure_gate_confusion_matrix.md)
- [grasp_probe_regrasp_retry.md](grasp_probe_regrasp_retry.md)

## 9. Git 里程碑

| 日期 | Commit | 内容 |
|---|---|---|
| 2026-09-02 | `de1d9dc` | Initial commit |
| 2026-09-02 | `18df7d3` | OpenArm v2 + Wuji combined simulation |
| 2026-09-03 | `acb18f2` | synchronized camera recording and replay |
| 2026-09-03 | `e6d7e27` | LeRobot OpenArm-Wuji plugin |
| 2026-09-04 | `f11e409` | deterministic Reach task |
| 2026-09-04 | `36cbb5b` | causal grasp stability evaluation；后用于历史 100-run baseline |
| 2026-09-04 | `4746d74` | frozen-synergy grasp-settle gate |
| 2026-09-04 | `3b1189a` | Lift pacing aligned to external protocol |
| 2026-09-05 | `a8a14dc` | preload + bounded S-curve Lift |
| 2026-09-08 | `639e22a` | grasp-settle checkpoint |
| 2026-09-10 | `49e9b65` | preserve grasp experiment outputs |
| 2026-09-10 | `d2b8754` | coordinated OpenArm-Wuji demonstrations |
| 2026-09-14 | `d29299e` | freeze Reach/Approach/Recovery/router on main |
| 2026-09-15 | 本次 milestone commit | freeze and package full staged ACT + scripted Lift pipeline |

## 10. 维护规则

以后每个新实验至少补充：

1. 只改变了什么、明确没改变什么；
2. dataset/seeds/steps/checkpoint 与 action semantics；
3. binary success、连续诊断和 safety 指标；
4. 结果是否推翻先前假设；
5. stop/continue 决策；
6. 代码、报告、JSON/Parquet/视频路径；
7. 未验证限制，尤其是 training-seed vs unseen-seed、simulation vs hardware。

这样本项目以后不会再用“loss 下降”“接触更多”或“某一个 seed 看起来成功”替代真正的 closed-loop、load-bearing、可复现实验结论。
