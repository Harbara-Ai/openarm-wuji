# OpenArm + Wuji Learning

Pre-hardware integration project for OpenArm MuJoCo, Wuji retargeting, data collection,
ACT, and a fail-safe asynchronous policy bridge.

## Current status

- Project interfaces, mock backend, hand synergies, action validation, and tests: ready.
- Official repositories are pinned in `docs/upstream_versions.md`.
- OpenArm v2 headless smoke: passed on native Windows with MuJoCo 3.12.0; `outputs/openarm/` is generated locally.
- Wuji Hand full prerecorded retargeting: passed under WSL2 Ubuntu 22.04; 2751 frames of 21×3 keypoints converted to finite 20-D trajectories.
- Combined OpenArm v2 + left Wuji Hand model: compiled and stable; 20-DoF hand synergies are config-driven and safety limited.
- Reach–Grasp–Lift scene: deterministic table/cube reset and fixed task camera are ready; the Reach-only Jacobian expert reaches a randomized pre-grasp target within 12 mm. Grasp and Lift are the next milestone.
- Real robot protocol: intentionally unimplemented until the customized arm specification arrives.

![OpenArm v2 left arm with Wuji Hand](outputs/combined/combined_smoke.png)

## Upstream model setup

Local upstream checkouts are intentionally excluded from Git. Clone and pin them
before building the combined model:

```powershell
git clone https://github.com/enactic/openarm_mujoco.git vendor/openarm_mujoco
git -C vendor/openarm_mujoco checkout a8c979629f2591ad035d99d338ce114969e6cddc
git clone --recursive https://github.com/wuji-technology/wuji-retargeting.git vendor/wuji-retargeting
git -C vendor/wuji-retargeting checkout 531f6ed4250b475d2e9231f54e988fc9b1c5b4ea
```

OpenArm MuJoCo is Apache-2.0 licensed. Wuji retargeting and its hand description
are MIT licensed; their copyright and license files remain in the upstream checkouts.

Run the dependency-free smoke test from PowerShell:

```powershell
./scripts/smoke_mock.ps1
./scripts/run_openarm_headless.ps1
./scripts/run_wuji_retarget_wsl.ps1 -WslPython /path/to/wuji-retarget/bin/python -WujiSource /path/to/wuji-retargeting
./scripts/run_combined_smoke.ps1
./scripts/run_combined_control_smoke.ps1
./scripts/setup_lerobot_policy.ps1
./scripts/run_lerobot_contract_smoke.ps1
./scripts/run_reach_only_smoke.ps1
# GUI (interactive; close the MuJoCo window to exit)
./scripts/run_openarm_gui.ps1
```

See `docs/environment_report.md`, `docs/setup_decisions.md`, and `docs/known_gaps.md`.
The evidence-backed status audit and next seven days are in `docs/status_and_7_day_plan.md`.

The combined-control smoke exposes one stable policy action vector: 7 left-arm joint
targets followed by three hand synergies (`open_close`, `pinch`, `spread`). It drives
both subsystems in the same MuJoCo step loop. Each observation contains front and wrist
RGB (`uint8`, HWC), arm and hand positions/velocities, the 20-D bounded hand target, the
3-D synergy command, the actual bounded 10-D policy action, contiguous frame index,
high-resolution host timestamp, and simulation timestamp. The smoke writes a 90-frame
episode to `outputs/combined/combined_control_recording.npz`, replays it from a clean
reset, and rejects state error above `1e-4`.

![OpenArm + Wuji front and wrist camera demo](outputs/combined/combined_control_demo.gif)

## LeRobot plugin contract

The installable package `lerobot_robot_openarm_wuji` registers
`openarm_wuji_follower` without modifying LeRobot. The policy-visible contract is two
same-size RGB cameras, 27-D measured position state, and a 10-D action consisting of
seven arm targets plus three hand synergies. Backend timestamps and diagnostic fields
are deliberately excluded from the first ACT input. See `docs/lerobot_integration.md`.

## Reach–Grasp–Lift task

`run_reach_only_smoke.ps1` rebuilds a task model from the pinned official assets,
resets a free cube with an explicit seed, solves position-only damped least-squares IK,
and drives the first `Reach` phase using the same 10-D robot action contract. The seed-7
reference reaches the pre-grasp point in 24 control frames with 8.9 mm final error.
This is not yet an ACT training dataset: Grasp, Lift, outcome labels, and correct
`observation_t -> action_t -> observation_t+1` episode recording come next. See
`docs/reach_grasp_lift.md`.

![Reach-only scripted expert](outputs/reach_grasp_lift/reach_demo.gif)

## License

This project is licensed under the Apache License 2.0. See [LICENSE](LICENSE).
Third-party components retain their respective upstream licenses.
