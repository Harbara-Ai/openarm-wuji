# Grasp verification + one regrasp retry

## 结论

**Case C: Probe itself cannot yet cleanly separate good and bad grasps.**

本实验没有训练或修改任何 ACT，也没有修改原 scripted Lift、IK、gain、rate limit、controller 或 27D action semantics。micro-lift 使用原 120 mm S-curve 的前缀，Wuji 始终保持 live 20D terminal controller target。

## Conditional probe experiment

- terminal states：`60`；
- first probe PASS/FAIL：`0/60`；
- first-probe pass rate：`0.0%`；
- `P(full Lift success | probe PASS) = N/A`；
- `P(full Lift success | probe FAIL) = 30.0%`。

| | Full Lift success | Full Lift fail |
|---|---:|---:|
| Probe PASS | 0 | 0 |
| Probe FAIL | 18 | 42 |

这里 probe FAIL 后的 full-Lift label 只在 conditional counterfactual branch 中测量；deployable retry branch 不执行这次失败后的 full Lift，而是从真实 post-probe state 回落并 regrasp。

同一批 60 个 terminal states 上，旧 GraspSecure label 对同一个 post-probe continuation outcome 的 precision 为 `38.9%`，`P(success | old gate FAIL)` 为 `16.7%`；虽然旧 gate 本身也弱，但 v1 probe 的 0 个 PASS 明显更差。

Probe criterion pass counts（分母 60）：

| Criterion | Pass |
|---|---:|
| finite | 60 |
| physical trajectory + hold completed | 55 |
| palm stayed above minimum lift | 46 |
| cube followed palm | 14 |
| environment support released | 6 |
| contact retained | 52 |
| no fast escape | 6 |

Post-hoc sensitivity（不作为新 formal gate）：完全移除 translation-drift 与 terminal-speed checks 后，核心 palm/cube/support/contact 条件仍仅 `1/60` PASS；对应 TP/FP/FN/TN=`1/0/17/42`。因此 0 PASS 不能归因于 15 mm/30 mm/s 阈值本身。

## One regrasp retry

- first-probe failures / regrasp attempts：`60`；
- return succeeded / physically valid before regrasp：`60/57`；
- frozen GraspSecure gate passes on retry：`18`；
- physically valid after regrasp / second probes attempted / passed：`56/56/0`；
- rescued to full success：`0`；
- rescue rate given probe1 FAIL：`0.0%`；
- single-attempt verified success：`0/60 = 0.0%`；
- one-retry success：`0/60 = 0.0%`；
- absolute retry gain：`0.0%`。

Regrasp attempt1→attempt2 continuous state changes：

- meaningfully different：`57/57`；
- median target/actual 20D L2 change：`0.1123/0.7540 rad`；
- median cube-palm translation/rotation change：`24.51 mm / 42.28°`；
- topology changed：`28/57`。

“meaningfully different” 仅是 diagnostic label：topology 改变，或 target/actual 20D L2 ≥0.05 rad，或 cube-palm translation ≥3 mm，或 rotation ≥3°。连续指标才是主要结果。

## Fresh-seed full staged matched comparison

完整 fresh-seed matched evaluation 尚未运行。

因为 conditional experiment 没有证明 probe 有效，按预先协议没有启动 100–200 fresh-seed full evaluation；这不是缺失结果，而是防止把已证伪的 verification gate 接入完整系统。

## Probe v1 definition

- 原 S-curve prefix target：15 mm（实际为首个超过该值的原 reference frame）；先等待 rate-limited arm 进入 physical micro-lift，再计 hold 0.3 s；
- physical settle 最多 1.0 s；terminal palm lift ≥10 mm；cube lift ≥5 mm 且至少为 palm lift 的 50%；
- hold 最后一帧无环境支撑，supported frames ≤1/3；
- hold 内 zero-contact longest run ≤3 frames；
- relative translation drift ≤15 mm，terminal relative speed ≤30 mm/s；
- rotation、finger count/topology、antipodal、edge margin 均不作为 hard gate。

15 mm drift 只是本轮预注册的 diagnostic bound，不宣称为 Wuji 最终稳定标准。正式 60 条结果出来后没有为提高数字而重调。

## Interpretation

v1 probe 的主要问题不是 contact collapse：`52/60` 保持了 contact。真正卡住的是 cube-follow（`14/60`）、环境支撑释放（`6/60`）以及静态 micro-hold 下的 relative escape（`6/60`）。与此同时，probe FAIL 后继续原上抬仍有 `18/60` 次完成 full Lift。这表明当前连续 scripted Lift 可以靠继续增加 arm target 进入承载状态，但在 15 mm reference 处插入静态 hold 会改变动力学，并不能作为无损的早期 load-bearing test。

Regrasp 确实不是简单重复：57 个可比状态全部触发了“不同 grasp”diagnostic，median actual-hand L2 变化 0.7540 rad，median cube-palm 平移/旋转变化 24.51 mm / 42.28°。但这种变化往往过大且没有转化成任何 probe2 PASS，因此“不同”不等于“更好”。

## Final decision

Redesign or recalibrate the probe before relying on retry logic.

## Artifacts

- `outputs/grasp_probe_regrasp_retry/summary.json`
- `outputs/grasp_probe_regrasp_retry/manifest.json`
- `outputs/grasp_probe_regrasp_retry/conditional/`
- `outputs/grasp_probe_regrasp_retry/full/` was intentionally not produced because the conditional prerequisite failed.
