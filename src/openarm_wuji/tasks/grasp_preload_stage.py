"""Scripted Grasp+Preload expert and shared static-grasp diagnostics."""
from __future__ import annotations

from typing import Any

import numpy as np

from .reach_grasp_lift import preload_wrench_trends
from .se3 import pose_drift, rotation_geodesic_angle_deg


FINGER_NAMES = tuple(f"finger{index}" for index in range(1, 6))


def _state(observation: dict[str, Any]) -> np.ndarray:
    return np.concatenate([
        observation["arm_joint_position"],
        observation["hand_joint_position"],
    ]).astype(np.float32)


def finger_force_vector(telemetry: dict[str, Any]) -> np.ndarray:
    return np.asarray([
        telemetry["finger_normal_forces_n"].get(name, 0.0)
        for name in FINGER_NAMES
    ], dtype=np.float64)


def grasp_window_metrics(samples: list[dict[str, Any]], *,
                         minimum_fingers: int,
                         minimum_force_n: float) -> dict[str, Any]:
    """Evaluate the existing static Grasp/Preload window without topology rules."""
    if not samples:
        return {
            "contacts_sustained": False,
            "max_translation_drift_m": None,
            "max_rotation_drift_deg": None,
            "force_slope_n_per_frame": None,
            "moment_slope_nm_per_frame": None,
            "wrench_stable": False,
        }
    anchor = samples[0]
    translations = []
    rotations = []
    contacts_sustained = True
    for sample in samples:
        translation, rotation = pose_drift(
            anchor["object_relative_position_m"],
            anchor["object_relative_quaternion_wxyz"],
            sample["object_relative_position_m"],
            sample["object_relative_quaternion_wxyz"],
        )
        translations.append(float(translation))
        rotations.append(float(rotation))
        active = np.count_nonzero(
            finger_force_vector(sample) >= minimum_force_n
        )
        contacts_sustained = contacts_sustained and active >= minimum_fingers
    force_slope = moment_slope = None
    wrench_stable = False
    if len(samples) >= 2:
        force_slope, moment_slope, wrench_stable = preload_wrench_trends(samples)
    return {
        "contacts_sustained": bool(contacts_sustained),
        "max_translation_drift_m": max(translations),
        "max_rotation_drift_deg": max(rotations),
        "force_slope_n_per_frame": force_slope,
        "moment_slope_nm_per_frame": moment_slope,
        "wrench_stable": bool(wrench_stable),
    }


def window_passes(metrics: dict[str, Any], *, baseline: dict[str, Any],
                  require_wrench_stable: bool = True) -> bool:
    translation = metrics["max_translation_drift_m"]
    rotation = metrics["max_rotation_drift_deg"]
    return bool(
        metrics["contacts_sustained"]
        and (metrics["wrench_stable"] or not require_wrench_stable)
        and translation is not None
        and rotation is not None
        and translation < float(baseline["max_relative_translation_drift_m"])
        and rotation < float(baseline["max_relative_rotation_drift_deg"])
    )


def initialize_task_from_handoff(task) -> None:
    """Initialize task bookkeeping around the robot's current staged state."""
    telemetry = task.task_telemetry()
    cube = np.asarray(telemetry["cube_position_m"], dtype=float)
    task.initial_cube_position = cube.copy()
    task.target_position = cube + np.asarray(
        task.config["grasp"]["target_offset_m"], dtype=float
    )
    arm_target = task.robot.data.ctrl[task.robot.arm_actuator_ids]
    task.arm_command = np.asarray(arm_target, dtype=float).copy()
    task.arm_goal = task.arm_command.copy()
    task._arm_command_initialized = True
    task._phase_start_telemetry = {}
    task._phase_samples = {}
    task._transition_telemetry = task.task_telemetry()
    task._transition_observation = None


def _longest_true_run(values: list[bool]) -> int:
    best = current = 0
    for value in values:
        current = current + 1 if value else 0
        best = max(best, current)
    return best


def run_scripted_grasp_preload(task, *, seed: int,
                               source_kind: str,
                               terminal_hold_frames: int = 8
                               ) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Run only the existing close->preload->settle expert from current state.

    The OpenArm target is frozen at the staged handoff target. The recorded
    policy action is always the exact 27-D position-actuator target, not a
    next-frame state and not the expert's 3-D synergy command.
    """
    if not 5 <= terminal_hold_frames <= 10:
        raise ValueError("terminal_hold_frames must be between 5 and 10")
    robot = task.robot
    config = task.config
    grasp = config["grasp"]
    baseline = config["external_baseline"]
    minimum_fingers = int(grasp["min_finger_groups"])
    minimum_force = float(grasp["min_normal_force_n"])
    settle_window = int(grasp["contact_hold_frames"])
    arm_target = robot.data.ctrl[robot.arm_actuator_ids].copy()
    stage_start_cube = np.asarray(task.task_telemetry()["cube_position_m"])

    rows: dict[str, list[Any]] = {
        "observation.state": [],
        "next_observation.state": [],
        "action": [],
        "raw_script_action": [],
        "observation.images.front": [],
        "observation.images.wrist": [],
        "phase": [],
        "telemetry.cube_pose_world": [],
        "telemetry.cube_pose_relative_to_palm": [],
        "telemetry.active_finger_mask": [],
        "telemetry.finger_normal_force_n": [],
        "telemetry.total_normal_force_n": [],
        "telemetry.resultant_force_world_n": [],
        "telemetry.resultant_moment_world_nm": [],
        "telemetry.hand_target_actual_preload_rad": [],
        "telemetry.cube_displacement_m": [],
        "sim_time": [],
    }
    telemetry_samples: list[dict[str, Any]] = []
    first_contact_fingers: list[str] | None = None
    max_simultaneous = 0
    multi_contact_flags: list[bool] = []
    peak_forces = np.zeros(5, dtype=float)

    def take_step(synergy: float, phase: str) -> dict[str, Any]:
        nonlocal first_contact_fingers, max_simultaneous, peak_forces
        observation = robot.get_observation()
        raw_action = np.r_[arm_target, [synergy, 0.0, 0.0]]
        robot.send_action(raw_action)
        post_observation = robot.latest_record
        controller_target = np.asarray(
            post_observation["controller_joint_target"], dtype=np.float32
        )
        post_state = _state(post_observation)
        telemetry = task.task_telemetry()
        forces = finger_force_vector(telemetry)
        active = forces >= minimum_force
        active_names = [
            name for name, value in zip(FINGER_NAMES, active, strict=True)
            if value
        ]
        if active_names and first_contact_fingers is None:
            first_contact_fingers = active_names
        max_simultaneous = max(max_simultaneous, len(active_names))
        multi_contact_flags.append(len(active_names) >= minimum_fingers)
        peak_forces = np.maximum(peak_forces, forces)
        cube_pose = np.r_[
            telemetry["cube_position_m"], telemetry["cube_quaternion_wxyz"]
        ]
        relative_pose = np.r_[
            telemetry["object_relative_position_m"],
            telemetry["object_relative_quaternion_wxyz"],
        ]
        rows["observation.state"].append(_state(observation))
        rows["next_observation.state"].append(post_state)
        rows["action"].append(controller_target)
        rows["raw_script_action"].append(raw_action.astype(np.float32))
        rows["observation.images.front"].append(
            np.asarray(observation["front_rgb"], dtype=np.uint8)
        )
        rows["observation.images.wrist"].append(
            np.asarray(observation["wrist_rgb"], dtype=np.uint8)
        )
        rows["phase"].append(phase)
        rows["telemetry.cube_pose_world"].append(cube_pose)
        rows["telemetry.cube_pose_relative_to_palm"].append(relative_pose)
        rows["telemetry.active_finger_mask"].append(active)
        rows["telemetry.finger_normal_force_n"].append(forces)
        rows["telemetry.total_normal_force_n"].append(float(np.sum(forces)))
        rows["telemetry.resultant_force_world_n"].append(
            telemetry["contact_resultant_force_world_n"]
        )
        rows["telemetry.resultant_moment_world_nm"].append(
            telemetry["contact_resultant_moment_about_cube_world_nm"]
        )
        rows["telemetry.hand_target_actual_preload_rad"].append(
            controller_target[7:] - post_state[7:]
        )
        rows["telemetry.cube_displacement_m"].append(float(np.linalg.norm(
            np.asarray(telemetry["cube_position_m"]) - stage_start_cube
        )))
        rows["sim_time"].append(float(observation["sim_time"]))
        telemetry_samples.append(telemetry)
        return telemetry

    synergy = 0.0
    frozen_synergy = None
    close_steps = 0
    for close_steps in range(1, int(grasp["max_close_steps"]) + 1):
        synergy = min(1.0, synergy + float(grasp["close_synergy_step"]))
        telemetry = take_step(synergy, "grasp_close")
        active = np.count_nonzero(
            finger_force_vector(telemetry) >= minimum_force
        )
        if active >= minimum_fingers and synergy >= float(
            grasp["min_synergy_for_success"]
        ):
            frozen_synergy = synergy
            break

    preload_target = max(
        frozen_synergy or 0.0, float(grasp["preload_target_synergy"])
    )
    preload_steps = 0
    preload_reached = False
    if frozen_synergy is not None:
        for preload_steps in range(1, int(grasp["preload_max_steps"]) + 1):
            synergy = min(
                preload_target,
                synergy + float(grasp["preload_synergy_step"]),
            )
            take_step(synergy, "preload")
            if synergy >= preload_target:
                preload_reached = True
                break

    settle_samples: list[dict[str, Any]] = []
    settle_metrics = grasp_window_metrics(
        [], minimum_fingers=minimum_fingers,
        minimum_force_n=minimum_force,
    )
    settle_stable = False
    settle_steps = 0
    if preload_reached:
        for settle_steps in range(
            1, int(grasp["preload_settle_max_steps"]) + 1
        ):
            settle_samples.append(take_step(synergy, "preload_settle"))
            if len(settle_samples) < settle_window:
                continue
            settle_metrics = grasp_window_metrics(
                settle_samples[-settle_window:],
                minimum_fingers=minimum_fingers,
                minimum_force_n=minimum_force,
            )
            settle_stable = window_passes(settle_metrics, baseline=baseline)
            if settle_stable:
                break

    hold_samples: list[dict[str, Any]] = []
    if settle_stable:
        for _ in range(terminal_hold_frames):
            hold_samples.append(take_step(synergy, "terminal_hold"))
    hold_metrics = grasp_window_metrics(
        hold_samples,
        minimum_fingers=minimum_fingers,
        minimum_force_n=minimum_force,
    )
    # The original preload-settle gate above remains unchanged. The extra
    # terminal hold is a persistence check: keep contact and SE(3) stable,
    # while retaining wrench slopes as diagnostics. Requiring the sign of a
    # near-zero moment slope a second time rejects physically settled holds.
    hold_stable = bool(
        len(hold_samples) == terminal_hold_frames
        and window_passes(
            hold_metrics, baseline=baseline, require_wrench_stable=False
        )
    )
    success = bool(
        frozen_synergy is not None
        and preload_reached
        and settle_stable
        and hold_stable
    )
    if frozen_synergy is None:
        failure_stage = "contact_acquisition"
    elif not preload_reached:
        failure_stage = "preload_formation"
    elif not settle_stable:
        failure_stage = "grasp_formation"
    elif not hold_stable:
        failure_stage = "terminal_hold"
    else:
        failure_stage = None

    arrays = {key: np.asarray(value) for key, value in rows.items()}
    relative = arrays["telemetry.cube_pose_relative_to_palm"]
    translation_speed = np.zeros(len(relative), dtype=float)
    rotation_speed = np.zeros(len(relative), dtype=float)
    if len(relative) > 1:
        translation_speed[1:] = np.linalg.norm(
            np.diff(relative[:, :3], axis=0), axis=1
        ) * float(robot.control_hz)
        rotation_speed[1:] = np.asarray([
            rotation_geodesic_angle_deg(left, right)
            for left, right in zip(relative[:-1, 3:], relative[1:, 3:], strict=True)
        ]) * float(robot.control_hz)
    arrays["telemetry.relative_translation_speed_m_s"] = translation_speed
    arrays["telemetry.relative_rotation_speed_deg_s"] = rotation_speed
    preload = arrays["telemetry.hand_target_actual_preload_rad"]
    terminal_mask = arrays["phase"] == "terminal_hold"
    terminal_preload_l2 = np.linalg.norm(preload[terminal_mask], axis=1)
    terminal_preload_abs = np.abs(preload[terminal_mask])
    result = {
        "success": success,
        "grasp_success": frozen_synergy is not None,
        "preload_success": preload_reached,
        "terminal_hold_success": hold_stable,
        "failure_stage": failure_stage,
        "seed": int(seed),
        "source_kind": source_kind,
        "frames": len(arrays["action"]),
        "close_steps": close_steps,
        "preload_steps": preload_steps,
        "settle_steps": settle_steps,
        "terminal_hold_frames": len(hold_samples),
        "frozen_synergy": frozen_synergy,
        "preload_target_synergy": preload_target,
        "first_contact_fingers": first_contact_fingers or [],
        "max_simultaneous_contacts": max_simultaneous,
        "longest_persistent_multifinger_frames": _longest_true_run(
            multi_contact_flags
        ),
        "peak_per_finger_normal_force_n": peak_forces,
        "final_per_finger_normal_force_n": (
            arrays["telemetry.finger_normal_force_n"][-1]
        ),
        "settle_metrics": settle_metrics,
        "terminal_hold_metrics": hold_metrics,
        "terminal_hand_preload_l2_mean_rad": (
            float(np.mean(terminal_preload_l2)) if len(terminal_preload_l2) else None
        ),
        "terminal_hand_preload_l2_min_rad": (
            float(np.min(terminal_preload_l2)) if len(terminal_preload_l2) else None
        ),
        "terminal_hand_preload_abs_mean_rad": (
            np.mean(terminal_preload_abs, axis=0)
            if len(terminal_preload_abs) else np.zeros(20)
        ),
        "maximum_cube_displacement_m": float(
            np.max(arrays["telemetry.cube_displacement_m"])
        ),
        "maximum_relative_translation_speed_m_s": float(
            np.max(translation_speed)
        ),
        "maximum_relative_rotation_speed_deg_s": float(np.max(rotation_speed)),
        "load_bearing_ready_static_proxy": success,
        "unsupported_lift_tested": False,
    }
    return result, arrays
