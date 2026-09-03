# LeRobot integration contract

Validated on 2026-09-03 against LeRobot commit
`fa048804d05c1b965b0a5801cf205f97a9e8a3e8` (package version 0.6.2).

## Plugin

The installable distribution and import package are both named
`lerobot_robot_openarm_wuji`, so LeRobot's third-party plugin discovery imports it
without modifying upstream source. It registers robot type `openarm_wuji_follower`.

`OpenArmWujiFollower` composes a Mock, MuJoCo, or deliberately inert Real backend.
The adapter owns LeRobot naming and serialization; the backends retain NumPy arrays,
simulation timestamps, and diagnostic state.

## Policy-visible features

- `observation.state`: 27 float32 values: 7 arm positions followed by 20 measured hand positions.
- `observation.images.front_rgb`: HWC uint8 RGB.
- `observation.images.wrist_rgb`: HWC uint8 RGB.
- `action`: 10 float32 values: 7 arm position targets followed by `open_close`, `pinch`, and `spread`.

Host timestamp, simulation timestamp, frame index, velocities, and the 20-D hand target
remain backend telemetry. They are intentionally excluded from the first ACT policy input.
LeRobotDataset supplies its own timestamp, frame index, episode index, and task index.

The canonical policy unit for arm positions is radians. A future Real backend must convert
the official OpenArm follower's degree representation at the adapter boundary.

## Validation

The contract test verifies plugin import, exact observation/action key equality, feature
conversion to state `(27,)` and action `(10,)`, exclusion of custom timestamps, two equal
camera shapes, actual-action return values, and a real MuJoCo step through the LeRobot adapter.

```powershell
./scripts/setup_lerobot_policy.ps1
./scripts/run_lerobot_contract_smoke.ps1
```

Validated native contract environment: Python 3.12.13, LeRobot 0.6.2 at the
commit above, plugin 0.1.0, PyTorch 2.11.0 CPU, MuJoCo 3.12.0, and NumPy 2.2.6.
This CPU environment validates integration; it is not the later CUDA ACT training environment.
Its resolved package set is recorded in
`envs/lerobot-policy/requirements.contract-windows.lock.txt`.

The custom NPZ recorder remains a simulation diagnostic tool. Training data will use
LeRobot's observe-then-act recording loop and LeRobotDataset writer.
