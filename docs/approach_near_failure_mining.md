# Approach near-failure targeted data mining

## 结论

冻结的 Reach ACT 与 Approach ACT 在新 seeds 上完成了
200 条纯 closed-loop rollout。Reach 成功
160/200；在
160 次真实 Approach 中成功
115，失败 45。

检测器产生 499 个候选 snapshot，覆盖
150 条 rollout。按 terminal error 向量/幅值、
7D arm configuration、cube displacement 和 failure mode 去重后保留
50 个；scripted Approach expert 从中成功
恢复 30 条。失败 correction 不进入数据集。

本轮没有修改或重训 Reach/Approach policy，也没有进入 Grasp、Preload、Lift、
RL 或 SmolVLA。

## Mining 设置

- Seeds: 1000–1199
- 流程：reset → frozen Reach ACT → Reach gate → frozen Approach ACT
- ACT rollout 不读取 expert action
- Near-failure trigger：terminal plateau、进入后远离、cube 位移 >3 mm、
  terminal-region near-timeout、目标附近 gate stall
- Snapshot 保存 `mjSTATE_INTEGRATION`、qpos/qvel/act/ctrl、solver warm-start、
  外力/mocap、controller bookkeeping、task reference、双相机与 27D observation
- 恢复检查：physics/state/ctrl 精确；离屏重渲染保持亚灰度级一致

Trigger 总分布：`{'cube_displacement': 58, 'moving_away': 148, 'terminal_plateau': 148, 'gate_stall': 122, 'near_timeout_terminal': 23}`

Approach failure 分布：`{'cube_displacement': 18, 'timeout': 27}`。
最常见 failure mode 是 `timeout`。

## 去重后 snapshot

- 数量：50
- 来源 outcome：`{'cube_displacement': 14, 'success': 23, 'timeout': 13}`
- Trigger：`{'cube_displacement': 9, 'moving_away': 25, 'terminal_plateau': 5, 'gate_stall': 8, 'near_timeout_terminal': 3}`
- Terminal error：median 20.12 mm，p10–p90
  6.72–39.75 mm，range
  2.71–46.01 mm
- Cube displacement：median 5.67 mm，p10–p90
  0.09–21.81 mm，range
  0.02–24.07 mm

## Scripted correction

Expert 从完整恢复的 near-failure state 重新对准当前 cube 的 grasp-start，
Wuji 保持 snapshot 中的 open/pre-shape target，并在 Approach gate 后额外 hold
8 帧。动作仍是 27D absolute position-controller target。

- 成功：30/50
- 失败 correction：`{'cube_displacement': 17, 'correction_timeout': 3}`（只保留诊断）
- Recovery length：median 19.0 frames，p10–p90
  13.0–21.0
- Correction 造成的额外 cube 位移：median 1.712 mm，
  p90 14.255 mm
- 主导 correction direction：`{'-X': 5, '+Y': 2, '-Z': 18, '-Y': 2, '+X': 2, '+Z': 1}`
- 最常见 correction direction：`-Z`
- 覆盖的 dominant directions：`['+X', '+Y', '+Z', '-X', '-Y', '-Z']`
- 覆盖 error octants：8

## 数据集

- Raw correction trajectories：`outputs/approach_correction_demos/raw`
- Native LeRobotDataset v3.0：`outputs/approach_correction_demos/lerobot_dataset`
- Episodes / frames：30 /
  559
- Observation：front+wrist RGB 240×320，27D actual qpos
- Action：27D absolute controller target
- State/action exact round-trip：
  `{'state': True, 'action': True}`
- Episode boundary / timestamp / image alignment：`True`
- ACT chunk20 DataLoader：
  state `[8, 27]`，
  action `[8, 20, 27]`，pad
  `[8, 20]`
- Manifest：`outputs/approach_correction_demos/manifest.json`
- Validation：`outputs/approach_correction_demos/validation.json`

## 下一轮混合建议（本轮不执行）

第一组 matched experiment 建议从头训练，不 resume：每个 batch/window 以
**70% 原 Approach-only + 30% correction** 采样。不要直接按 raw frame 拼接，
因为 correction episode 较短；先按 source episode 平衡，再在 correction 内按
`trigger_type` 和 `source_outcome` 分层。保持 ImageNet ResNet-18、chunk20、
H_exec=1、27D absolute action 和现有 gate 不变，并与 original-only baseline
使用相同 seeds 对照。若安全位移改善但 nominal Approach 退化，再把 correction
比例降到 20%，不要同时改模型或 gate。

Machine-readable analysis：`outputs/approach_correction_demos/analysis_summary.json`
