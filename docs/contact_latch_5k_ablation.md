# PCA5 post-contact target-latch ablation

## Scope

本轮只验证一个假设：PCA5 policy 在已经形成接触后继续修改 latent/absolute hand target，是否是 `hold 0.1 s` 无法延长到 `0.3/0.5 s` 的主要原因。

保持不变：`expert_pca5_absolute`、5D latent、228D observation、Reward V2、formal `stable_grasp_success`、reset、termination、penetration rule、PD/velocity limit、SAC 超参数、fixed-palm Stage1 和 rigid cube。没有改 PCA basis、latent scaling、reward 权重、success threshold、residual、PPO、BC/demo replay 或 cube material。

## latch definition

新增配置 `configs/rl/grasp_stage1_expert_pca5_contact_latch.json`，只打开：

```text
valid_contact_finger_count >= 3
continuously for ceil(0.10 s * control_hz) = 3 frames
```

这个触发器独立于正式 success criterion（正式 gate 仍是原来的 2 指条件）。首次触发时锁存实际已发送的 20D absolute `q_target`：

```text
latched_q_target = current_q_target.copy()
```

后续 SAC 仍产生 latent action，并记录 `requested_hand_target_rad`，但 MuJoCo 收到的 `desired_hand_target_rad` 保持 `latched_q_target`。没有把 latent 置零，因为 PCA latent 0 是 mean posture，不是当前抓姿。reset 会清除 latch 状态。

环境新增 telemetry：`post_contact_latched`、`latched_q_target`、`contact_established_time_s`、`contact_count_at_established`、`requested_hand_target_rad`。这些字段不进入 228D policy observation。

## deterministic comparison

脚本：`scripts/evaluate_contact_latch_ablation.py`。

由于上一轮 SB3 zip checkpoint 在 managed desktop sandbox 中没有成功写出，本轮用完全相同的 seed 7、SAC 超参数和 5000 steps 在内存中重建了当前 PCA5 policy；同一个 policy、同一个 reset seed 分别跑 baseline 和 latch。它不是加载 checkpoint 的声称，而是可复现的 same-seed reconstruction。

结果（seeds 7–11）：

| metric | PCA5 baseline | PCA5 + latch |
|---|---:|---:|
| any contact | 5/5 | 5/5 |
| >=2 contacts | 5/5 | 5/5 |
| >=3 contacts | 5/5 | 5/5 |
| contact established (3 fingers × 0.10 s) | 1/5 | 1/5 |
| hold >=0.1 s | 5/5 | 5/5 |
| hold >=0.3 s | 0/5 | 0/5 |
| hold >=0.5 s | 0/5 | 0/5 |
| stable_grasp_success | 0/5 | 0/5 |
| max simultaneous contacts | [3,4,3,4,4] | [3,4,3,4,4] |

唯一触发的 seed 是 seed 8，established time 为 0.6667 s、contact count 为 3。latch 后它的最大相对平移漂移从 0.0527 mm 降到 0.0257 mm，最大相对旋转漂移从 0.0815° 降到 0.0443°，方块位移从 0.475 mm 降到 0.386 mm；但 contact 很快降到 2 指，仍没有达到 0.3 s hold。其余 4 个 seed 根本没有满足连续 3 指窗口，因此 latch 没有机会触发。

所有 seed 的最大相对速度、方块位移和穿透仍很小，说明本轮不是 drop/penetration exploit；问题是稳定多指状态本身没有持续形成。

逐 seed 的 event-window（latent action、requested/latched q_target、qpos、contact count）见：

- `outputs/contact_latch/seed_7_11_timeseries/seed_8.json`：唯一触发 latch 的 seed；
- `seed_7.json`、`seed_9.json`、`seed_10.json`、`seed_11.json`：没有满足触发条件，因此没有 post-contact window。

总 machine-readable 对照见 `outputs/contact_latch/deterministic_baseline_vs_latch.json`。

## reward semantics check

本轮没有修改 reward。seed 8 的 latch 后 reward component 仍会计算 policy 输出的 latent：post-latch `weighted_action` 和 `weighted_delta_action` 仍为负值（分别约 -0.0137、-0.0093），尽管 latent 已不再改变实际 q_target。这正是预期的 action/reward semantics mismatch；本轮只报告，不擅自改 Reward V2。

跨 5 个 seed 汇总时，baseline/latch 的 pre-latch component 基本一致；唯一触发 seed 的 post-latch weighted total reward 分别约为 35.67（baseline 的“established 后继续控制”区间）和 34.74（latch），没有出现 reward 方面的虚假改善。

## stop-rule conclusion

问题 1：在同一个已重建的 PCA5 5K policy 上，冻结当前 20D q_target 能否显著提高 0.3/0.5 s hold？

不能。结果是 `0/5 -> 0/5`，因为只有 1/5 episode 达到 latch 触发条件，而且触发后仍快速失去 3 指持续接触。

问题 2：当前 Stage1 是否应采用 `SAC acquisition + contact-triggered posture latch` 作为 hybrid controller baseline？

可以保留这个实现作为诊断工具和后续 expert-trajectory generation 的候选接口，但当前证据不足以把它作为有效 Stage1 baseline，也不应继续训练 latch 版 SAC。本轮不创建 `sac_5k_metrics.json`，因为 stop rule 明确要求 latch diagnostic 先出现非零 `hold >=0.3 s` 改善。

当前结论转向：主要瓶颈更可能是 `contact-established` 本身太脆弱，而不是 established 后的继续运动。下一步应检查 grasp geometry、thumb opposition、接触力平衡/摩擦和 latent posture quality；不要用 latch 掩盖未形成稳定三指拓扑的问题。
