# Contact topology stability diagnosis

本轮只做诊断，不训练，不修改 SAC、Reward V2、success gate、Lift 或 LeRobot observation。

## 诊断范围

当前没有可加载的 `expert_pca5` SAC checkpoint。因此这里不能声称是在分析 learned policy。脚本重放的是公开 `wuji-pick-and-place` 已提取的专家 `q_hand` 数值轨迹：将每帧 20D 手部姿态映射为已有 PCA5 target，再送入固定掌心的 MuJoCo 环境，记录每根手指的接触、法向力、滑动速度和方块接触几何。视频没有用于推断关节运动。

本次使用之前选出的 5 条典型 `cube_left` episode（77、71、70、84、74），30 Hz。接触有效定义仍只是环境配置中的 `normal_force_n >= success.min_normal_force_n`，没有为本诊断发明新的稳定阈值。

机器可读结果：[`outputs/contact_topology/contact_topology_diagnosis.json`](../outputs/contact_topology/contact_topology_diagnosis.json)。可复现实验脚本：[`scripts/diagnose_contact_topology.py`](../scripts/diagnose_contact_topology.py)。

## 1. 达到 3 contacts 时是哪三根手指？

| episode | 首次 3-contact 拓扑 | 首次接触时间 |
|---|---|---:|
| 77 | finger1 + finger3 + finger4 | 1.067 s |
| 71 | 未达到 3 contacts，最大为 2 | — |
| 70 | finger1 + finger2 + finger4 | 1.700 s |
| 84 | finger1 + finger2 + finger3 | 3.400 s |
| 74 | finger1 + finger2 + finger4 | 1.333 s |

因此没有一个固定的“三指组合”。在这 4 条达到三指的轨迹中，finger1 出现 4/4，finger2 2/4，finger3 2/4，finger4 3/4，finger5 0/4。三指形态本身已经说明接触拓扑不稳定、且主要由 finger1 与不同的中指/无名指组合构成。

## 2. 哪一根先掉？掉之前发生了什么？

这里必须区分“唯一先掉”和“同一控制帧同时丢失”：

- episode 74 有明确的唯一先掉手指：finger4。掉落前它的法向力合计 `0.814 N`，滑动速度 `16.37 mm/s`，位于 `+Z` 面边缘，距最近边约 `0.15 µm`；finger1 为 `11.93 mm/s`，finger2 为 `4.23 mm/s`。因此 finger4 是这条轨迹中最明确的几何/滑动薄弱点。
- episode 77 的 finger1、finger3、finger4 在同一控制帧从 3 变为 0，不能诚实地指定唯一第一根。掉落前 finger4 的滑动速度最高（`21.49 mm/s`），同时是 edge+corner contact（距边 `1.33 µm`、距角 `4.35 µm`）；finger3 滑动仅 `0.33 mm/s`，说明它更像是被整体拓扑崩溃一起带掉。
- episode 70 的三根也在同一帧全部丢失。掉落前 finger1 滑动最高（`22.87 mm/s`），finger4 次之（`18.37 mm/s`），finger2 只有 `5.74 mm/s`；但没有单独的先后事件。
- episode 84 的三根同样在同一帧全部丢失，随后只看到 finger4 新出现。掉落前滑动最高的是 finger3（`48.70 mm/s`），其次 finger2（`38.40 mm/s`）、finger1（`32.60 mm/s`）。这说明该次接触已经明显处于滑移/边缘几何恶化状态，但不能把 finger3 宣称为程序上唯一先掉者。

共同模式是：三指建立后只持续了一个控制间隔（约 `33 ms`）就发生拓扑变化；掉落前高滑动速度通常出现在 `+Z` 面的 edge/corner 接触，或 finger1 的侧面接触。相对掌心的速度也在转移帧上上升：例如 episode 77 的线速度由 `0.50` 增至 `1.42 mm/s`、角速度由 `0.927` 增至 `2.063 deg/s`。

## 3. 掉成 2 contacts 后，剩余两个是否很稳定？

这批 replay 没有支持“非常稳定”的结论。唯一观测到明确 3→2 转换的是 episode 74：剩余 finger1 + finger2 只保持了 1 帧，按连续时间定义稳定窗口为 `0 s`。该帧的 finger1/finger2 法向力最大值分别为 `0.367/0.552 N`，滑动速度为 `13.77/14.35 mm/s`，方块相对掌心角速度为 `1.027 deg/s`。因此它们仍在滑动，且没有足够时间证明是稳定双指抓取。

其它三条轨迹是 3→0（episode 77、70）或 3→1（episode 84），没有可评价的连续双指窗口。结论是：当前数据更像“多指瞬时接触后拓扑崩溃”，而不是“先掉一指、剩余双指稳定承载”。

## 当前可执行结论

1. 不应再用 `max simultaneous contact = 3` 作为稳定抓取代理；三指只持续一个 control step。
2. finger4 的 edge/corner 几何和滑动是最清楚的可疑点，但在 episode 77/70/84 中存在同帧整体丢失，不能把它概括成唯一根因。
3. 后续若要改策略，应继续输出逐指 topology、法向力、切向力、滑动速度、edge/corner 距离和相对 SE(3) 漂移；先解决接触几何和滑动，再讨论是否增加 squeeze 或 lift 控制。
4. 本轮没有新增 gate，也没有把任何连续指标宣称为适合 Wuji 灵巧手的最终阈值。
