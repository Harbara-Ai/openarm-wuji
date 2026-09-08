# Wuji expert-PCA5 action prior

本轮只改变 action representation。没有改动 228D observation、Reward V2、reset、episode horizon、success gate、Lift 或 ACT 接口。

## 数据和 joint mapping

输入是 `outputs/wuji_teleop_analysis/cube_60_hand_trajectories.parquet` 中的真实 LeRobot Parquet 数值轨迹，不是视频估计。使用 `cube_left` 的 30 条 episode（episode 60–89）；每行保存 20D `q_hand`、20D `a_hand`、frame index 和 timestamp。数据集原始 `info.json` 的 `names` 为 `null`，因此 mapping 的通道顺序按公开 Wuji flat-array convention 与当前 MuJoCo 模型逐项核对：dataset 54D 的 left slice 是 `[14:34]`，当前模型的 20 个 `wuji_left_finger{1..5}_joint{1..4}` 名称逐项匹配，符号为 `+1`、单位为 rad。这个模型侧 mapping 是 proven 的，但外部 recorder 如何构造无名字的 flat array 仍保留记录级不确定性；完整表在 `outputs/expert_pca5/joint_mapping.json`。

## phase segmentation

每条轨迹取 `frame 0 -> early hold`，不使用 transport/release/reopen。`close_start` 是 Parquet action excursion 首次持续越过 12% 且累计 excursion 至少 0.08 rad 的帧；`grasp_established` 是该阶段后 3 帧；随后保留 1 s early-hold 窗口。数据集没有 tactile/contact 字段，因此 phase 是明确标注的数值 heuristic，不把视频当作 joint 数值来源。元数据见 `phase_metadata.json`。

## PCA posture manifold

对 phase 内的绝对姿态 `q_t` 做 pooled SVD：

```text
q_target = clip(mu + B @ z, joint_limits)
z = latent_center + latent_half_range * a
a in [-1, 1]^5
```

`B` 是 20×5 basis，`mu` 是 20D phase mean。第一步只用 absolute posture prior，避免把 PCA 轴误当作手工 closing direction。文件为 `mean.npy`、`basis.npy`、`latent_scaling.json`。

| component | explained variance |
|---|---:|
| PC1 | 70.3275% |
| PC2 | 10.0148% |
| PC3 | 9.2453% |
| PC4 | 3.3989% |
| PC5 | 1.8412% |

前 5 个 PC 累计解释约 94.83%。累计 80%、90%、95% 的维数分别为 2、4、6（前 5 个已经接近 95%，完整 reconstruction 指标仍保留 20D reference）。phase posture 的 PCA5 reconstruction RMSE 为 0.03803 rad；thumb 四关节 RMSE 为 0.04137 rad，其余 16D 为 0.03715 rad；单条轨迹 RMSE 的均值/中位数/最大值为 0.03656/0.03456/0.05475 rad。代表性 raw-vs-PCA5 图在 `outputs/expert_pca5/episode_*_raw_vs_pca5.png`。

## 环境接口

新增配置 `configs/rl/grasp_stage1_expert_pca5.json`，representation 名称是 `expert_pca5_absolute`。SAC 输出 5D latent action；环境把它映射为 20D absolute target，随后保留关节限位、PD stepping 和每关节速度限幅。`step_absolute_target()` 仅用于同一 Reward V2 路径的 deterministic replay，不向 observation 注入 dataset telemetry。

## deterministic replay gate

脚本：`scripts/replay_expert_pca5.py`。选择的 5 条代表轨迹是 77、71、70、84、74；raw 和 PCA5 使用同一个 seed 7、同一个 reset、同一个 120-step horizon。当前 horizon 比部分专家 phase 短，所以 replay 明确记录为 truncated；这不是 stable grasp success。

PCA5 replay 的 gate 结果：

- 最大同时接触：4 指（要求至少 2）；
- 最大穿透：0.000571 m（0.571 mm，限制 0.008 m）；
- 最大方块位移：0.000431 m（0.431 mm，限制 0.120 m）；
- 方块最大旋转约 0.382°；
- 5/5 条 replay 都至少出现 2 指接触，4/5 条出现 3 指接触；
- replay 阶段没有 static stable success，因为它只覆盖 phase + early hold，且固定 120 步窗口未达到当前 0.5 s 稳定判定。

完整 compact 结果见 `outputs/expert_pca5/replay_metrics.json`，奖励排序见 `reward_ranking.json`。manual 3D 与 structured5 baseline 在这里使用相同的 open-to-close absolute target probe，因此它们的数值相同；这是一项可比性诊断，不把它宣称为新的独立训练结果。现有 deterministic structured5 SAC（Reward V2、5000 steps）历史基线仍是 max contact 1、seed 7–11 deterministic evaluation 未形成 multi-contact。

作为 reward ranking 的量级 sanity check，PCA5 replay 的 5 条 return 为 13.24–37.92，而历史 deterministic structured5 SAC seed 7–11 约为 0.068–0.078、max contact 为 0；这支持“expert-like latent target 比当前 policy 更接近 contact-rich manifold”的判断，但不能把 replay return 直接当成 SAC success rate。

## fresh SAC 5K

由于 deterministic PCA5 replay gate 通过，启动了全新的 SAC（seed 7、5000 steps），没有加载 20D V1/V2 或 structured5 checkpoint。结果保存在 `outputs/expert_pca5/sac_5k_metrics.json`：

- training max contact = 4，success events = 0；
- deterministic seeds 7–11 的 max contact 为 3、4、3、4、4；
- 5/5 seed 达到至少 2 指、5/5 达到至少 3 指、5/5 保持 0.1 s 接触；
- 5/5 尚未保持 0.3 s，因此 stable grasp success 仍为 0；
- mean return 约 40.39；mean absolute latent action 约 0.520；final entropy coefficient 约 0.230；
- 方块位移均约 0.43–0.50 mm，没有通过推动方块获取接触的明显迹象。

桌面沙箱拒绝了二进制 SB3 checkpoint 和完整 JSON report 的运行时写入；因此本轮保留了可复核的 compact metrics，而没有虚报 checkpoint 已保存。

## 结论和下一步

真实 Wuji cube grasp 的绝对 posture phase 明显位于低维 manifold：PC1–PC2 已超过 80%，PC1–PC4 约 93%，PCA5 约 94.8%。这打破了原先 20D 独立 joint exploration 的主要瓶颈：deterministic policy 的最大同时接触从旧 structured5 结果的 1 提升到 PCA5 SAC 的 3–4。

因此值得继续到 10K，但下一步应先保留当前 action prior 与 Reward V2 不变，检查 0.3 s hold、relative SE(3) drift 和多 seed 稳定性；不要马上加入 20D residual。只有 PCA5 在更长训练和多 seed 下持续稳定后，才考虑小幅 residual（初始 λ 约 0.05–0.10）作为独立 ablation。
