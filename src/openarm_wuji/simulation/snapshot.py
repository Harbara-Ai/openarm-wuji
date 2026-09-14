"""Exact-enough MuJoCo state capture for closed-loop recovery mining."""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np


SNAPSHOT_VERSION = 1


def capture_simulator_snapshot(robot, task, *, observation: dict | None = None
                               ) -> dict[str, np.ndarray]:
    """Capture physics, controller, task, and synchronized visual state.

    ``mjSTATE_INTEGRATION`` preserves the physics integration state including
    solver warm-start data. Controller targets and the small amount of Python
    bookkeeping used by the 30 Hz wrapper are stored separately.
    """
    import mujoco

    if observation is None:
        observation = robot.get_observation()
    state_spec = mujoco.mjtState.mjSTATE_INTEGRATION
    integration_state = np.empty(
        mujoco.mj_stateSize(robot.model, state_spec), dtype=np.float64
    )
    mujoco.mj_getState(robot.model, robot.data, integration_state, state_spec)
    result = {
        "snapshot_version": np.asarray(SNAPSHOT_VERSION, dtype=np.int64),
        "mujoco_state_spec": np.asarray(int(state_spec), dtype=np.int64),
        "mujoco_integration_state": integration_state,
        "qpos": robot.data.qpos.copy(),
        "qvel": robot.data.qvel.copy(),
        "act": robot.data.act.copy(),
        "ctrl": robot.data.ctrl.copy(),
        "qacc_warmstart": robot.data.qacc_warmstart.copy(),
        "mocap_pos": robot.data.mocap_pos.copy(),
        "mocap_quat": robot.data.mocap_quat.copy(),
        "userdata": robot.data.userdata.copy(),
        "qfrc_applied": robot.data.qfrc_applied.copy(),
        "xfrc_applied": robot.data.xfrc_applied.copy(),
        "sim_time": np.asarray(robot.data.time, dtype=np.float64),
        "robot_frame_index": np.asarray(robot._frame_index, dtype=np.int64),
        "robot_control_start_time": np.asarray(
            robot._control_start_time, dtype=np.float64
        ),
        "robot_hand_target": robot._hand_target.copy(),
        "robot_last_synergy": robot._last_synergy.copy(),
        "robot_last_sent_action": robot._last_sent_action.copy(),
        "task_initial_cube_position": task.initial_cube_position.copy(),
        "task_target_position": task.target_position.copy(),
        "task_arm_goal": task.arm_goal.copy(),
        "task_arm_command": task.arm_command.copy(),
        "task_arm_command_initialized": np.asarray(
            task._arm_command_initialized, dtype=bool
        ),
        "observation.state": np.concatenate([
            observation["arm_joint_position"],
            observation["hand_joint_position"],
        ]).astype(np.float64),
        "observation.images.front": np.asarray(
            observation["front_rgb"], dtype=np.uint8
        ),
        "observation.images.wrist": np.asarray(
            observation["wrist_rgb"], dtype=np.uint8
        ),
        "observation.controller_target": np.asarray(
            observation["controller_joint_target"], dtype=np.float64
        ),
    }
    if hasattr(robot.data, "plugin_state"):
        result["plugin_state"] = robot.data.plugin_state.copy()
    if hasattr(robot.data, "eq_active"):
        result["eq_active"] = robot.data.eq_active.copy()
    return result


def restore_simulator_snapshot(robot, task, snapshot: Mapping[str, Any]
                               ) -> dict[str, float | bool]:
    """Restore a snapshot and report exact state/visual identity checks."""
    import mujoco

    version = int(np.asarray(snapshot["snapshot_version"]))
    if version != SNAPSHOT_VERSION:
        raise ValueError(f"unsupported snapshot version: {version}")
    state_spec = mujoco.mjtState(int(np.asarray(snapshot["mujoco_state_spec"])))
    mujoco.mj_setState(
        robot.model,
        robot.data,
        np.asarray(snapshot["mujoco_integration_state"], dtype=np.float64),
        state_spec,
    )
    robot.data.ctrl[:] = snapshot["ctrl"]
    robot.data.mocap_pos[:] = snapshot["mocap_pos"]
    robot.data.mocap_quat[:] = snapshot["mocap_quat"]
    robot.data.userdata[:] = snapshot["userdata"]
    robot.data.qfrc_applied[:] = snapshot["qfrc_applied"]
    robot.data.xfrc_applied[:] = snapshot["xfrc_applied"]
    if "plugin_state" in snapshot and hasattr(robot.data, "plugin_state"):
        robot.data.plugin_state[:] = snapshot["plugin_state"]
    if "eq_active" in snapshot and hasattr(robot.data, "eq_active"):
        robot.data.eq_active[:] = snapshot["eq_active"]
    mujoco.mj_forward(robot.model, robot.data)
    # mj_forward recomputes derived quantities; retain the original integration
    # warm start for the next mj_step.
    robot.data.qacc_warmstart[:] = snapshot["qacc_warmstart"]

    robot._frame_index = int(np.asarray(snapshot["robot_frame_index"]))
    robot._control_start_time = float(
        np.asarray(snapshot["robot_control_start_time"])
    )
    robot._hand_target = np.asarray(
        snapshot["robot_hand_target"], dtype=float
    ).copy()
    robot._last_synergy = np.asarray(
        snapshot["robot_last_synergy"], dtype=float
    ).copy()
    robot._last_sent_action = np.asarray(
        snapshot["robot_last_sent_action"], dtype=float
    ).copy()
    robot.records.clear()

    task.initial_cube_position = np.asarray(
        snapshot["task_initial_cube_position"], dtype=float
    ).copy()
    task.target_position = np.asarray(
        snapshot["task_target_position"], dtype=float
    ).copy()
    task.arm_goal = np.asarray(snapshot["task_arm_goal"], dtype=float).copy()
    task.arm_command = np.asarray(
        snapshot["task_arm_command"], dtype=float
    ).copy()
    task._arm_command_initialized = bool(
        np.asarray(snapshot["task_arm_command_initialized"])
    )
    task._phase_start_telemetry = {}
    task._phase_samples = {}
    task._transition_telemetry = task.task_telemetry()
    task._transition_observation = None

    # Render once to refresh the offscreen renderer after the discontinuous
    # state restore, then compare a second render. The first frame can contain
    # stale shadow/depth-buffer pixels even though physics state is exact.
    robot.get_observation()
    restored = robot.get_observation()
    restored_state = np.concatenate([
        restored["arm_joint_position"], restored["hand_joint_position"]
    ])
    state_error = float(np.max(np.abs(
        restored_state - np.asarray(snapshot["observation.state"])
    )))
    qpos_error = float(np.max(np.abs(
        robot.data.qpos - np.asarray(snapshot["qpos"])
    )))
    qvel_error = float(np.max(np.abs(
        robot.data.qvel - np.asarray(snapshot["qvel"])
    )))
    ctrl_error = float(np.max(np.abs(
        robot.data.ctrl - np.asarray(snapshot["ctrl"])
    )))
    front_error = int(np.max(np.abs(
        restored["front_rgb"].astype(np.int16)
        - np.asarray(snapshot["observation.images.front"]).astype(np.int16)
    )))
    wrist_error = int(np.max(np.abs(
        restored["wrist_rgb"].astype(np.int16)
        - np.asarray(snapshot["observation.images.wrist"]).astype(np.int16)
    )))
    front_mean_error = float(np.mean(np.abs(
        restored["front_rgb"].astype(np.int16)
        - np.asarray(snapshot["observation.images.front"]).astype(np.int16)
    )))
    wrist_mean_error = float(np.mean(np.abs(
        restored["wrist_rgb"].astype(np.int16)
        - np.asarray(snapshot["observation.images.wrist"]).astype(np.int16)
    )))
    physics_passed = bool(
        state_error <= 1e-12
        and qpos_error <= 1e-12
        and qvel_error <= 1e-12
        and ctrl_error <= 1e-12
    )
    visual_exact = bool(front_error == 0 and wrist_error == 0)
    # Offscreen OpenGL shadow/edge pixels are not bit-deterministic after a
    # discontinuous restore. Keep exactness visible, while accepting only a
    # sub-one-gray-level mean difference as the same rendered observation.
    visual_consistent = bool(
        front_mean_error <= 1.0 and wrist_mean_error <= 1.0
    )
    return {
        "passed": bool(physics_passed and visual_consistent),
        "physics_passed": physics_passed,
        "visual_exact": visual_exact,
        "visual_consistent": visual_consistent,
        "state_max_abs_error": state_error,
        "qpos_max_abs_error": qpos_error,
        "qvel_max_abs_error": qvel_error,
        "ctrl_max_abs_error": ctrl_error,
        "front_max_pixel_error": front_error,
        "wrist_max_pixel_error": wrist_error,
        "front_mean_pixel_error": front_mean_error,
        "wrist_mean_pixel_error": wrist_mean_error,
    }
