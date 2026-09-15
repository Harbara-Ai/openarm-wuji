# 面向 20DoF 灵巧手的分层语义动作空间 v0.1

状态：设计基线，尚未实现，尚未经过仿真或真机验证。

适用范围：OpenArm + Wuji 项目中的 Wuji 20DoF 灵巧手。当前实验优先覆盖自由空间展示手势、手型切换和视觉驱动的终态修正；暂不包含触觉、接触力控制、稳定抓取、滑移恢复和机械臂协同。

本文档定义模型可见的动作语义、两层结构、接口数量、物理依据和验证标准。具体关节向量、步长、速度、起点包络和视觉阈值需要后续根据 Wuji 实测数据标定。

## 1. 设计目标

20DoF 灵巧手的模型接口需要避免两个极端：

- 只暴露 `GRASP` 或完整手势轨迹：容易调用，但表达能力不足，失败后无法局部修正。
- 暴露 `THUMB_J1+`、`INDEX_J2-` 等关节增量：表达完整，但退化为 joint-space control，搜索空间大，VLM 难以理解动作的物理效果。

本设计选择中间层：

> 宏技能负责快速复用，全手协同与局部协同负责组合、状态适配和纠错；20DoF 关节控制由设备解释器保留，不作为 agent 的常规动作空间。

目标性质：

1. **低维**：正常任务不要求 agent 同时决定 20 个关节。
2. **可解释**：动作名称对应可观察的手型变化，而不是给关节向量换名字。
3. **可组合**：宏技能可以由协同动作和参考轨迹构成。
4. **可纠错**：失败后能够从最新实测状态执行局部残差动作。
5. **设备隔离**：模型语义保持稳定，Wuji 关节映射和标定由解释器承担。
6. **可验证**：每个动作声明预期效果，执行后比较预期与实际效果。

## 2. 物理与研究依据

### 2.1 姿态协同不是 20 个独立关节

Santello、Flanders 和 Soechting 在 15 个手部关节上研究大量模拟工具抓握姿态，发现前两个主成分平均解释约 84% 的姿态方差，前三个约 90%。第一主成分主要包含全手 MCP 屈曲、较小的 PIP 屈曲、手指内收和拇指姿态变化；第二主成分主要包含 MCP 伸展与 PIP 屈曲的相反变化，可理解为平直与钩状手型之间的变化。

但高阶成分并不是纯噪声，仍携带区分具体对象和精细手型的信息。该工作提出的合理解释是：手型控制同时存在少量全手粗协同和更细粒度、分布式的控制。

参考：[Postural Hand Synergies for Tool Use](https://pmc.ncbi.nlm.nih.gov/articles/PMC6793309/)

### 2.2 Eigengrasp 适合预成形，但不能代替全部精细控制

Ciocarlie 和 Allen 将低维 eigengrasp 用作复杂机器人手的预抓取和规划空间。低维空间能显著降低搜索复杂度，但精确终态可能需要偏离 eigengrasp 子空间。因此两维空间适合作为 coarse pre-shape，不应被理解为完整的灵巧手控制空间。

参考：[Hand Posture Subspaces for Dexterous Robotic Grasping](https://www.cs.columbia.edu/~allen/PAPERS/ciocarlieallenijrr.pdf)

### 2.3 Synergy 数量依赖任务、数据与 embodiment

更大规模的日常抓握数据研究表明，需要更多协同成分才能覆盖精细动作变化；其中高阶成分常用于拇指和食指独立性等精细调节。这说明 PCA 维数和方向不是跨数据集、跨机器人固定不变的物理常数。

参考：[Kinematic synergies of hand grasps](https://pmc.ncbi.nlm.nih.gov/articles/PMC6540541/)

### 2.4 人体协同不能直接等同于机器人关节向量

机器人手领域通常采用两类方法：把人体协同映射到机器人，或者直接从目标机器人自身的动作数据中重新提取协同。人体到机器人的 joint-to-joint 映射可以作为仿人手的初始化，但最终仍应依据机器人自身的运动学、关节范围和成功姿态重新标定。

参考：[Replicating Human Hand Synergies Onto Robotic Hands](https://pmc.ncbi.nlm.nih.gov/articles/PMC6001282/)

### 2.5 当前只定义 kinematic/postural synergy

本文档不加入触觉，因此只定义自由空间手型和预抓取阶段的姿态协同。接触后的力分配、顺应性、稳定抓握和滑移属于 soft synergy 或 force synergy 范畴，不应由当前接口的成功结果代替。

参考：[Hand synergies: Integration of robotics and neuroscience](https://pmc.ncbi.nlm.nih.gov/articles/PMC5839666/)

## 3. 总体结构与接口数量

模型可见的运动接口共 11 个：

- Level 2：3 个宏技能接口。
- Level 1：8 个语义协同微动作接口。
- 20DoF joint control：保留在解释器和诊断层，不计入模型可见语义空间。

```text
Level 2：宏技能层（3）
    ↓ 快速复用已有能力
Level 1：语义协同微动作层（8）
    ↓ 组合、适配与纠错
Wuji Interpreter
    ↓ 设备映射、安全投影和平滑轨迹
20DoF joint control
```

此外定义 `OBSERVE`、`VERIFY`、`HOLD`、`STOP` 四个执行控制命令。它们不描述手型变化，不计入 11 个运动接口。

## 4. Level 2：宏技能层（3 个接口）

### 4.1 `SET_HANDSHAPE`

用于终态手型是主要任务语义的动作。

```text
SET_HANDSHAPE(
    skill_id,
    transition,
    duration,
    hold
)
```

示例：

```text
SET_HANDSHAPE("digit_3")
SET_HANDSHAPE("ok")
SET_HANDSHAPE("open_hand")
SET_HANDSHAPE("power_grasp_preshape")
SET_HANDSHAPE("tripod_preshape")
```

适用于：

- 数字 1—5；
- OK、V、握拳、张开手等静态手型；
- 未来抓取动作的接触前预成形。

技能可以保存目标 synergy 坐标、稀疏局部残差、参考关节姿态、参考轨迹、起点条件、保持行为和成功标准。

### 4.2 `PLAY_HAND_MOTION`

用于路径、顺序或节奏本身具有任务语义的动态动作。

```text
PLAY_HAND_MOTION(
    skill_id,
    tempo,
    repetitions
)
```

示例：

```text
PLAY_HAND_MOTION("wave")
PLAY_HAND_MOTION("finger_count_sequence")
PLAY_HAND_MOTION("beckon")
```

与 `SET_HANDSHAPE` 的区别：

- `SET_HANDSHAPE` 主要验收终态。
- `PLAY_HAND_MOTION` 还要验收阶段顺序、路径和节奏。

### 4.3 `RECOVER_HAND`

用于退出失败状态或返回经过验证的安全姿态。

```text
RECOVER_HAND(safe_pose_id)
```

示例：

```text
RECOVER_HAND("relaxed_open")
RECOVER_HAND("verified_neutral")
```

恢复必须从最新实际状态生成新过渡，并取消旧 execution ID 的剩余动作，不能假定失败后机械手仍在原技能起点。

## 5. Level 1：语义协同微动作层（8 个接口）

Level 1 分为四个全手粗协同和四个局部精细协同。

### 5.1 全手粗协同（4）

#### 5.1.1 `MODULATE_APERTURE`

```text
MODULATE_APERTURE(
    direction = open | close,
    amount = coarse | fine
)
```

物理效果：沿全手主要开合协同方向移动。典型地包含四指 MCP/IP 协同屈伸、一定程度的手指内收和拇指随动。它对应 Santello 第一主协同的核心物理趋势，但不是直接暴露数据相关的 `PC1`。

不应实现成所有关节增加相同角度。

#### 5.1.2 `MODULATE_CURL_PROFILE`

```text
MODULATE_CURL_PROFILE(
    direction = toward_hook | toward_flat,
    amount = coarse | fine
)
```

物理效果：调节 MCP 与 PIP/DIP 之间的屈曲分配。

- `toward_hook`：相对增加指间关节屈曲，形成钩状轮廓。
- `toward_flat`：减少远端卷曲，使手指趋于平直。

它对应 Santello 第二主协同中 MCP 与 PIP 相反变化的物理规律。

#### 5.1.3 `MODULATE_SPREAD`

```text
MODULATE_SPREAD(
    direction = spread | gather,
    amount = coarse | fine
)
```

物理效果：调节四指之间的整体外展或内收。

虽然手指内收常与全手闭合共同出现在主协同中，这里仍单独暴露，因为展示手势对指间间距敏感，而且不同机器人手的开合—外展机械耦合不同。语义协同不要求数学正交，可以存在可解释的重叠。

#### 5.1.4 `MODULATE_THUMB_OPPOSITION`

```text
MODULATE_THUMB_OPPOSITION(
    direction = oppose | reposition,
    amount = coarse | fine
)
```

物理效果：在拇指自然外展/侧方位置与掌面、其他手指方向之间执行粗粒度对掌或复位。底层通常组合拇指旋转、外展/内收和屈曲。

拇指运动学显著区别于其他四指，因此不完全依赖全手开合协同顺带控制。

### 5.2 局部精细协同（4）

#### 5.2.1 `ADJUST_DIGIT_FLEX`

```text
ADJUST_DIGIT_FLEX(
    target,
    direction = flex | extend,
    amount = coarse | fine
)
```

允许的目标：

```text
index
middle
ring
little
radial_pair = [index, middle]
ulnar_pair  = [ring, little]
four_fingers
```

物理效果：沿选定手指或指组的自然屈伸协同方向移动，同一手指的 MCP、PIP、DIP 按 Wuji 标定比例联动。

典型纠错：

```text
ADJUST_DIGIT_FLEX(little, flex, fine)
ADJUST_DIGIT_FLEX([ring, little], extend, fine)
```

#### 5.2.2 `ADJUST_DIGIT_CURL`

```text
ADJUST_DIGIT_CURL(
    target,
    direction = curl | flatten,
    amount = coarse | fine
)
```

物理效果：改变选定手指 PIP/DIP 相对于 MCP 的卷曲分布。

与 `ADJUST_DIGIT_FLEX` 的区别：

- `FLEX` 改变整根手指总体屈曲程度。
- `CURL` 改变近端与远端屈曲比例。

例如无名指总体弯曲程度足够，但指尖仍明显伸出时，应优先使用 `CURL`，而不是继续增加整体 `FLEX`。

#### 5.2.3 `ADJUST_DIGIT_LATERAL`

```text
ADJUST_DIGIT_LATERAL(
    target,
    direction = abduct | adduct,
    amount = coarse | fine
)
```

物理效果：调节指定手指或指间间隙的侧向位置，并尽量保持当前屈曲程度。

适用于：

- V 手势两指分得不够开；
- 数字 4 的四根手指排列过散；
- 单根手指发生明显侧向偏斜；
- 指间遮挡影响视觉验收。

如果 Wuji 的目标手指没有对应独立侧向能力，解释器必须拒绝或返回受限结果，不能假装完整实现了语义效果。

#### 5.2.4 `ADJUST_THUMB_REACH`

```text
ADJUST_THUMB_REACH(
    target = index | middle | ring | little | palm | repose,
    direction = toward | away,
    amount = coarse | fine
)
```

物理效果：根据拇指指尖相对目标手指或掌面的关系执行精细多关节调整。

- `MODULATE_THUMB_OPPOSITION`：拇指整体粗调。
- `ADJUST_THUMB_REACH`：面向具体目标的精调。

## 6. 统一动作参数

微动作统一采用结构化参数：

```json
{
  "action": "ADJUST_DIGIT_FLEX",
  "target": ["ring", "little"],
  "direction": "flex",
  "amount": "fine",
  "speed": "normal",
  "reference": "current_measured_state",
  "stop": "one_step_then_stabilize"
}
```

v0.1 仅提供两档幅度：

```text
coarse
fine
```

不让 VLM 直接输出任意关节角度。具体角度由 Wuji 解释器根据实测限位、当前姿态、速度限制和标定结果决定。

后续如需要连续幅值，可增加归一化参数 `alpha ∈ [-1, 1]`；它仍表示沿协同方向的幅度，不表示某个关节角度。

## 7. 执行控制命令（不计入动作空间）

```text
OBSERVE
VERIFY
HOLD
STOP
```

- `OBSERVE`：获取新图像和最新关节状态。
- `VERIFY`：运行关节到位、稳定性和视觉验收。
- `HOLD`：保持当前状态。
- `STOP`：取消当前动作和旧 execution ID 的剩余动作。

## 8. 数学表示与 Wuji Interpreter

宏技能目标可以表示为：

```text
q_goal = q_base + B_G s_goal + B_L r_goal
```

其中：

- `q_goal ∈ R^20`：目标关节姿态。
- `B_G ∈ R^(20×4)`：四个全手粗协同的 Wuji 映射。
- `s_goal ∈ R^4`：宏技能的全局协同坐标。
- `B_L r_goal`：稀疏的单指或指组残差。

一次微动作应从最新实际状态出发：

```text
q_cmd = project_constraints(
    q_actual + alpha * b_action(q_actual)
)
```

解释器负责：

1. 获取新鲜的 `q_actual`，避免使用旧 `joint_states`。
2. 将语义动作映射成当前状态下的 20 维动作方向。
3. 执行实测关节限位、速度、加速度和安全约束投影。
4. 生成平滑短轨迹。
5. 返回目标、实际效果、clamp 和稳定性信息。

协同向量允许是状态相关的，不要求所有姿态下使用同一个固定 20 维增量。

## 9. 为什么不直接暴露 `PC1+`、`PC2-`

原始 PCA 分量不适合作为模型语义：

1. PCA 分量的正负号可以任意翻转。
2. 接近的主成分可能发生旋转，跨被试、跨数据集不保持同一个轴。
3. PC1 常同时混合屈曲、内收和拇指运动，不利于 VLM 根据视觉误差选择动作。
4. 人体协同映射到 Wuji 时还受关节结构、活动范围和耦合关系影响。

因此 PCA/eigengrasp 用于解释器内部初始化和数据分析；模型面对的是具有稳定效果定义的 `APERTURE`、`CURL_PROFILE`、`SPREAD` 和 `THUMB_OPPOSITION`。

## 10. Action skill 表示

一个宏技能至少包含：

```yaml
skill_id: digit_3
kind: static_handshape

semantic_goal:
  label: digit_3

global_synergy:
  aperture: TBD
  curl_profile: TBD
  spread: TBD
  thumb_opposition: TBD

local_residual:
  index: TBD
  middle: TBD
  ring: TBD
  little: TBD
  thumb: TBD

reference:
  joint_pose: TBD
  trajectory: TBD

preconditions:
  embodiment: wuji_left_20dof
  calibration_version: TBD
  allowed_start_envelope: TBD

execution:
  duration: TBD
  hold_behavior: TBD

success:
  joint_tolerance: TBD
  stable_duration: TBD
  visual_label: digit_3
  visual_confidence_min: TBD

validation:
  status: unvalidated
  real_hardware_trials: 0
```

`TBD` 表示需要从当前 Wuji 真机、已有 1—5 手势脚本和后续视觉实验中确定。语法检查不能替代具体技能版本的真机验证。

## 11. 执行与纠错流程

```text
获取最新关节状态
→ 选择并检查宏技能
→ 从当前状态生成安全过渡
→ 执行 SET_HANDSHAPE
→ 检查关节跟踪
→ 等待姿态稳定
→ 获取新图像
→ 视觉验收
→ 将视觉误差映射到一个语义微动作
→ 作废原轨迹剩余部分
→ 从最新实际状态执行微动作
→ 再次验收
```

示例：

```text
SET_HANDSHAPE("digit_3")
→ VERIFY
→ 视觉诊断：小拇指偏直
→ ADJUST_DIGIT_FLEX(little, flex, fine)
→ VERIFY
```

视觉判断不确定时应执行 `OBSERVE`，不能直接修改手型。

## 12. 与仓库现有三维 synergy mapper 的关系

当前仓库已经存在：

- `src/openarm_wuji/teleop/synergies.py`
- `configs/wuji_hand_left_synergies.json`

现有实现将动作表示为：

```text
open_close
pinch
spread
```

并通过固定 `open_pose`、`close_pose`、`pinch_pose` 和 `spread_vector` 映射到 20 个关节。

该实现是较早的三维结构化动作 mapper，不是本文档中 11 个接口的实现。二者关系如下：

- `open_close` 可作为 `MODULATE_APERTURE` 的初始数据来源。
- `spread` 可作为 `MODULATE_SPREAD` 的初始数据来源。
- `pinch_pose` 更适合作为 `SET_HANDSHAPE("pinch_preshape")` 的参考姿态，而不是一个通用全局协同轴。
- 现有 mapper 尚未表示 `CURL_PROFILE`、拇指对掌、单指局部残差、当前状态相关映射和宏技能验证契约。

在后续实施前，不应声称现有代码已经实现了本文档定义的语义空间。

## 13. v0.1 扩展规则

只有同时满足以下条件，才增加新的模型可见语义原语：

1. 多次任务中出现现有八个微动作无法表示的稳定残差。
2. 该残差具有明确、可观察的物理效果。
3. 从多个起点调用时，动作效果方向一致。
4. 它不是某个具体关节编号的临时别名。
5. 增加后能明显改善覆盖率、纠错率或动作效率。

否则优先调整：

- Wuji 解释器的映射；
- synergy 向量或步长；
- 宏技能参数；
- 隐藏的关节级诊断工具。

## 14. 验证计划

比较以下四种动作表示：

| 动作空间 | 作用 |
|---|---|
| 纯宏技能 | 测试快速复用，但缺少局部纠错 |
| 两维 PCA/eigengrasp | 测试极低维空间的覆盖上限 |
| 4 个全局 + 4 个局部协同 | 本文档的核心方案 |
| 20DoF joint 增量 | 表达能力上界和 agent 可用性下界 |

任务集：

1. 数字 1—5。
2. OK、V、握拳、张开手。
3. 不同已验证起点之间的手势切换。
4. 单指总体屈曲偏差。
5. 指尖卷曲偏差。
6. 指间间距偏差。
7. 拇指位置偏差。
8. 无历史的新 agent 复用技能库。

主要指标：

- 最终手型成功率；
- 不同起点切换成功率；
- 受控偏差修正率；
- 误修率；
- 平均 agent 决策步数；
- 平均完成时间；
- 振荡和方向反转次数；
- 非法动作与 clamp 次数；
- 关节总行程；
- 语义动作的预期效果一致性；
- 现有八个微动作无法解释的失败比例；
- 人工干预次数；
- 新 agent 无历史复用成功率。

## 15. v0.1 冻结清单

### Level 2：宏技能

```text
SET_HANDSHAPE
PLAY_HAND_MOTION
RECOVER_HAND
```

### Level 1：全手粗协同

```text
MODULATE_APERTURE
MODULATE_CURL_PROFILE
MODULATE_SPREAD
MODULATE_THUMB_OPPOSITION
```

### Level 1：局部精细协同

```text
ADJUST_DIGIT_FLEX
ADJUST_DIGIT_CURL
ADJUST_DIGIT_LATERAL
ADJUST_THUMB_REACH
```

### 执行控制命令

```text
OBSERVE
VERIFY
HOLD
STOP
```

v0.1 的模型可见运动接口固定为：

```text
3 个宏技能接口
+ 4 个全手 postural synergies
+ 4 个稀疏局部 residual synergies
= 11 个运动接口
```

接口语义在 v0.1 内保持稳定；具体 joint mapping、synergy basis、步长、速度和验证阈值由后续 Wuji 标定版本定义。
