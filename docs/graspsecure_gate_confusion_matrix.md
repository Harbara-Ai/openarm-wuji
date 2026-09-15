# GraspSecure gate confusion-matrix diagnostic

## 结论

**Case B: The current gate has material false positives and false negatives.**

本实验没有训练或修改 Reach、Approach、Recovery、GraspSecure、router 或 scripted Lift。GraspSecure gate 与原 25 mm pre-Lift safety reference 只作为标签；只要状态 finite 且 cube 未超出宽松的 diagnostic workspace guard，PASS 和 FAIL 都执行同一个 Lift。

最关键结果：

- `P(Lift success | GraspSecure PASS) = 42.9%`；
- `P(Lift success | GraspSecure FAIL) = 27.0%`；
- recall = `73.0%`，specificity = `42.9%`；
- false positives = `36`，false negatives = `10`。

## Protocol 与样本

- fresh seeds：`6100–6261`；
- full staged rollouts：`162`；
- 到达 GraspSecure terminal evaluation：`101`；
- GraspSecure PASS / FAIL：`63 / 38`；
- formal 25 mm safety rejects：`23`；
- diagnostic forced Lift：`43`；
- physical invalid/workspace exclusions：`1`；
- actual Lift attempts：`100`。

GraspSecure FAIL 统一在 ACT 跑满 80-frame terminal evaluation 后取状态；PASS 在原 gate 首次通过时取状态。没有 reset、snapshot restore 或 expert action 插入 GraspSecure→Lift handoff。Lift 保持 terminal 20D hand target，轨迹、IK、gain、rate limit 和 1 s unsupported hold definition 均未修改。

## 2×2 confusion matrix

| | Lift SUCCESS | Lift FAIL |
|---|---:|---:|
| GraspSecure PASS | 27 | 36 |
| GraspSecure FAIL | 10 | 27 |

| Metric | Result |
|---|---:|
| precision / `P(success | PASS)` | 42.9% |
| false-positive rate | 57.1% |
| false-positive fraction among PASS | 57.1% |
| false-negative count | 10 |
| `P(success | FAIL)` | 27.0% |
| recall | 73.0% |
| specificity | 42.9% |
| accuracy | 54.0% |

25 mm formal pre-Lift safety reference 也只按标签评估：

- `P(Lift success | safety PASS) = 37.2%`；
- `P(Lift success | safety REJECT) = 36.4%`。

## False-positive / false-negative terminal features

数值格式为 median `[Q25, Q75]`；slip 与 cube motion 单位为 mm/s、mm。

| Group | n | preload L2 rad | active fingers | total force N | persistence s | contact slip mm/s | cube motion mm |
|---|---:|---:|---:|---:|---:|---:|---:|
| TP | 27 | 1.283 [1.228, 1.374] | 5.000 [5.000, 5.000] | 5.166 [4.415, 6.627] | 0.867 [0.633, 1.133] | 1.140 [0.816, 1.913] | 7.873 [5.598, 17.374] |
| FP | 36 | 1.309 [1.232, 1.361] | 5.000 [5.000, 5.000] | 4.446 [3.976, 4.784] | 0.900 [0.767, 1.075] | 0.738 [0.640, 1.112] | 5.819 [4.226, 7.958] |
| FN | 10 | 1.055 [0.889, 1.079] | 5.000 [5.000, 5.000] | 3.300 [2.687, 4.060] | 2.300 [1.983, 2.483] | 0.635 [0.565, 1.001] | 18.802 [11.608, 26.976] |
| TN | 27 | 0.458 [0.297, 0.770] | 3.000 [1.500, 4.000] | 1.900 [1.279, 2.900] | 2.233 [1.317, 2.533] | 1.321 [0.812, 2.055] | 24.049 [7.788, 55.330] |

完整 per-joint preload、per-finger force、topology、relative pose/velocity 与 outcome 分布在 `summary.json` 和 `samples.parquet`。FP 表示静态 gate 通过但卸载失败；FN 表示静态 gate 拒绝但实际可承载，是本实验用于消除选择偏差的核心样本。

### Error-mode interpretation

- **FP 不是 preload 不足造成。** FP preload median 为 `1.309 rad`，与 TP 的 `1.283 rad` 重叠；两组也几乎都是五指接触。FP 的 terminal slip median 甚至低于 TP，因此单帧低 slip、五指 topology 和高 preload 都不足以证明可承重。
- **FP 的真实 Lift failure** 为 `{'drop': 20, 'environment_support': 10, 'height_hold_failure': 6}`。这说明 gate 主要漏掉的是卸载后支撑、掉落和高度保持，而非形式上的 GraspSecure terminal 条件。
- **FN 是结构化漏检。** FN 中 `9/10` 仍为五指 topology；其 contact persistence median `2.300 s`，明显长于 TP 的 `0.867 s`，但 preload median 仅 `1.055 rad`。这些是真正能 Lift、却因 preload/terminal-hold conjunction 被拒绝的 grasp。
- **TN 与 FN 的主要差别不只是 preload。** TN 的 active-finger median 为 `3.0`、total force median `1.900 N`，而 FN 分别为 `5.0` 和 `3.300 N`；需要把接触覆盖、force distribution、relative pose/motion 与 persistence 联合考虑。

每指 terminal normal-force median `[Q25, Q75]`，顺序固定为 finger1/thumb → finger5/little：

| Group | f1 / f2 / f3 / f4 / f5 (N) |
|---|---|
| TP | 0.444 [0.376, 0.545] / 0.619 [0.453, 0.927] / 1.254 [0.708, 1.635] / 1.732 [1.486, 2.372] / 0.558 [0.475, 0.850] |
| FP | 0.418 [0.381, 0.461] / 0.574 [0.480, 1.020] / 0.834 [0.703, 1.266] / 1.688 [1.189, 1.771] / 0.532 [0.419, 0.709] |
| FN | 0.467 [0.342, 0.594] / 0.518 [0.468, 0.636] / 0.610 [0.512, 0.673] / 0.727 [0.586, 1.168] / 0.730 [0.475, 0.918] |
| TN | 0.000 [0.000, 0.156] / 0.103 [0.000, 0.448] / 0.435 [0.000, 0.564] / 0.418 [0.141, 0.606] / 0.453 [0.050, 0.895] |

| Group | terminal contact topology distribution |
|---|---|
| TP | finger1+finger2+finger3+finger4+finger5: 24; finger1+finger2+finger3+finger4: 1; finger1+finger3+finger4+finger5: 1; finger3+finger4+finger5: 1 |
| FP | finger1+finger2+finger3+finger4+finger5: 35; finger2+finger3+finger4+finger5: 1 |
| FN | finger1+finger2+finger3+finger4+finger5: 9; finger2+finger3+finger4+finger5: 1 |
| TN | finger2+finger3+finger4+finger5: 6; finger1+finger2+finger3+finger4+finger5: 4; finger2+finger3+finger4: 4; none: 3; finger5: 3; finger1+finger3+finger4+finger5: 2; finger3+finger4+finger5: 2; finger4+finger5: 2; finger1: 1 |

| Group | scripted Lift outcome distribution |
|---|---|
| TP | success: 27 |
| FP | drop: 20, environment_support: 10, height_hold_failure: 6 |
| FN | success: 10 |
| TN | environment_support: 26, drop: 1 |

## Preload threshold scan

这是 **preload-only diagnostic scan**，没有修改正式 conjunction gate。

- 当前 threshold：`1.177496 rad`；preload-only precision/recall/FNR = `41.9%` / `70.3%` / `29.7%`。
- best-F1 threshold：`0.8802 rad`；precision/recall/FNR = `47.3%` / `94.6%` / `5.4%`。
- preload-only ROC AUC：`0.6212`。

结论：当前 threshold 并非简单“太宽松”或“太严格”。降低阈值会减少 FN，但同时保留大量 FP；提高阈值会迅速损失 recall。即使本批样本内选择 best-F1 点，precision 也只有 `47.3%`，所以 preload 本身不是充分的 load-bearing 判据。

![Preload threshold scan](../outputs/graspsecure_gate_confusion/preload_threshold_scan.svg)

## Pre-Lift feature analysis

模型只使用 Lift 前信息；不包含 drop time、Lift drift、terminal Lift contact 或任何 future label feature。NumPy class-balanced L2 logistic regression 使用 stratified 5-fold out-of-fold evaluation。

| CV metric | Result |
|---|---:|
| ROC AUC | 0.7139 |
| balanced accuracy | 0.6847 |
| precision | 55.3% |
| recall | 70.3% |
| specificity | 66.7% |

标准化 full-fit coefficient 只用于解释关联，不是部署 classifier：

| Feature | coefficient toward Lift success |
|---|---:|
| preload_finger5_joint1_rad | +0.9437 |
| contact_persistence_s | +0.8639 |
| active_finger_count | +0.8187 |
| cube_palm_x_m | -0.8154 |
| force_finger2_n | +0.7941 |
| force_finger4_n | +0.7517 |
| preload_finger5_joint4_rad | -0.6838 |
| preload_finger4_joint4_rad | +0.6231 |
| relative_linear_speed_m_s | -0.5739 |
| preload_finger4_joint3_rad | -0.5686 |

![Pre-Lift feature coefficients](../outputs/graspsecure_gate_confusion/pre_lift_feature_coefficients.svg)

## 解释与限制

PASS 的 load-bearing precision 只有 42.9%，同时 FAIL 中仍有 27.0% 能 Lift；两种方向的错误都存在。如果 preload AUC/scan 同时有限，应重做 gate 定义而非只移动一个阈值，优先加入 geometry、persistence 或小幅 physical load probe。

- 本实验是 deterministic fresh-seed observational diagnostic；它量化 gate 判别能力，但不把 logistic coefficient 写成干预因果。
- threshold scan 在同一批样本上选择 best F1，只用于说明 preload 的分离上限；任何正式 gate 修改都必须在 held-out seeds 上预注册验证。
- `8 mm / 6°` 仍只是 GraspSecure static gate 中的 external diagnostic，不被宣称为 Wuji 最终标准。
- forced Lift 是仿真诊断路径，不得复制到真实机器人部署安全逻辑。

## Artifacts

- `D:/yl/embodied ai/openarm-wuji-learning/outputs/graspsecure_gate_confusion/summary.json`
- `D:/yl/embodied ai/openarm-wuji-learning/outputs/graspsecure_gate_confusion/samples.parquet`
- `D:/yl/embodied ai/openarm-wuji-learning/outputs/graspsecure_gate_confusion/progress.jsonl`
- `D:/yl/embodied ai/openarm-wuji-learning/outputs/graspsecure_gate_confusion/terminal_states/`
- `D:/yl/embodied ai/openarm-wuji-learning/outputs/graspsecure_gate_confusion/grasp/`
- `D:/yl/embodied ai/openarm-wuji-learning/outputs/graspsecure_gate_confusion/lift/`
- `D:/yl/embodied ai/openarm-wuji-learning/outputs/graspsecure_gate_confusion/preload_threshold_scan.svg`
- `D:/yl/embodied ai/openarm-wuji-learning/outputs/graspsecure_gate_confusion/pre_lift_feature_coefficients.svg`
