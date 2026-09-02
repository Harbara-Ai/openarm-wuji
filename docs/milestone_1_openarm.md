# Milestone 1 — OpenArm MuJoCo

Status: headless acceptance passed on 2026-09-01.

- Official model: OpenArm MuJoCo v2.2.0 bimanual MJCF.
- Runtime: MuJoCo 3.12.0, isolated `.venvs/openarm-sim`.
- Dimensions: `nq=18`, `nv=18`, `nu=16`; 14 arm joints, four finger joints, 16 position actuators.
- Test: loaded `v2/openarm_bimanual.xml`, reset `home`, changed actuator 0 by a bounded 0.1 rad target, stepped 500 iterations / 1.0 seconds.
- Result: maximum qpos change 0.095249 rad; all state finite.
- Evidence: `outputs/openarm/model_inventory.json` and `outputs/openarm/headless_smoke_v2.png`.
- Reproduce: `scripts/run_openarm_headless.ps1`.
- GUI command is wrapped by `scripts/run_openarm_gui.ps1`; interactive verification remains pending because automated GUI launch cannot be treated as proof of keyboard interaction.

