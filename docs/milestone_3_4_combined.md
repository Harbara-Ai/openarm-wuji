# Milestones 3–4 — project interface and combined model

Status: first simulation version passed on 2026-09-01.

- Unified interface: `OpenArmWujiRobot` with `MockOpenArmWuji`, `MujocoOpenArmWuji`, and deliberately inert `RealOpenArmWuji`.
- Combined model: official OpenArm v2 bimanual model, stock left gripper removed, official left Wuji Hand attached to `left_ee_control_point` using MuJoCo `MjSpec`.
- Mount transform: `[0, 0, -0.02]` m and XYZ Euler `[π, 0, 0]` rad. This is a visually checked simulation initial guess, not a measured real tool transform.
- Dimensions: `nq=36`, `nv=36`, `nu=35`, including 20 prefixed Wuji joints.
- Stability: 250 steps / 0.5 seconds, finite state. Settled contact inspection found no Wuji-to-arm collision; the only contact was the untouched right stock gripper's two fingers at negligible penetration (~`1e-7` m).
- Synergies: configuration-backed `open/close`, `pinch`, and `spread` map to the official 20-joint order, with joint clipping, 2 rad/s velocity limiting, and non-finite input rejection.
- Observation contract: synchronized front RGB, left-wrist RGB, 7-D arm position/velocity, 20-D hand position/velocity/target, 3-D synergy, bounded 10-D action, frame index, high-resolution host timestamp, and simulation timestamp.
- Replay: the 90-frame smoke episode is replayed from a clean reset; current maximum arm and hand position errors are both zero under the deterministic MuJoCo configuration.
- Timing: the 30 Hz controller follows an absolute simulation-time schedule and alternates 16/17 physics steps where needed, avoiding cumulative drift with the model's 2 ms timestep.
- Evidence: relocatable compiled `outputs/combined/openarm_v2_wuji_left.mjb`, `combined_report.json`, and `combined_smoke.png`. The MJB is used because a flattened XML cannot preserve mesh paths relative to both upstream repositories.
- Reproduce: `scripts/run_combined_smoke.ps1`.

Before real-hardware use, replace the mount transform with a measured tool-to-palm transform and validate side/orientation at low speed.
