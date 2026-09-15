"""Run the existing scripted Lift from an arbitrary 27-D controller handoff.

This module deliberately does not contain a new Lift controller.  It reuses
the calibrated S-curve, Cartesian IK target, joint-rate limit, and position
controller from :mod:`reach_grasp_lift`.  The only adapter is the control
interface: the arm follows that scripted reference while the exact 20-D Wuji
target present at the GraspSecure terminal state is held unchanged.
"""
from __future__ import annotations

import json
import math
from typing import Any, Sequence

import numpy as np

from .grasp_preload_stage import FINGER_NAMES, finger_force_vector
from .reach_grasp_lift import lift_trajectory_duration_s, s_curve_lift_waypoint
from .se3 import pose_drift


def compose_controller_target(arm_target: Sequence[float],
                              preserved_hand_target: Sequence[float]
                              ) -> np.ndarray:
    """Compose a 27-D target without remapping the ACT hand target."""
    arm = np.asarray(arm_target, dtype=float)
    hand = np.asarray(preserved_hand_target, dtype=float)
    if arm.shape != (7,) or hand.shape != (20,):
        raise ValueError("arm_target must be 7-D and hand_target must be 20-D")
    if not np.isfinite(np.r_[arm, hand]).all():
        raise ValueError("controller targets must be finite")
    return np.r_[arm, hand]


def load_bearing_success(*, trajectory_complete: bool,
                         terminal_heights_m: Sequence[float],
                         terminal_environment_support: Sequence[bool],
                         required_hold_frames: int,
                         success_height_m: float,
                         dropped: bool) -> bool:
    """Evaluate physical Lift+hold without imposing contact topology gates."""
    heights = np.asarray(terminal_heights_m, dtype=float)
    support = np.asarray(terminal_environment_support, dtype=bool)
    if required_hold_frames < 1 or success_height_m <= 0:
        raise ValueError("hold length and height threshold must be positive")
    if heights.shape != support.shape:
        raise ValueError("terminal height/support arrays must align")
    return bool(
        trajectory_complete
        and len(heights) >= required_hold_frames
        and np.all(heights[-required_hold_frames:] >= success_height_m)
        and not np.any(support[-required_hold_frames:])
        and not dropped
    )


def cliffs_delta(success_values: Sequence[float],
                 failure_values: Sequence[float]) -> float | None:
    """Return Cliff's delta (success minus failure), or None if a group is empty."""
    success = np.asarray(success_values, dtype=float)
    failure = np.asarray(failure_values, dtype=float)
    if not len(success) or not len(failure):
        return None
    comparisons = success[:, None] - failure[None, :]
    return float((np.count_nonzero(comparisons > 0)
                  - np.count_nonzero(comparisons < 0)) / comparisons.size)


def _cube_environment_contacts(task) -> list[dict[str, Any]]:
    """Return cube contacts with non-robot bodies (table/world/environment)."""
    import mujoco

    model = task.model
    data = task.data
    result: list[dict[str, Any]] = []
    for index in range(data.ncon):
        contact = data.contact[index]
        body1 = int(model.geom_bodyid[contact.geom1])
        body2 = int(model.geom_bodyid[contact.geom2])
        if task.cube_body_id not in (body1, body2):
            continue
        other_geom = int(contact.geom2 if body1 == task.cube_body_id
                         else contact.geom1)
        other_body = int(model.geom_bodyid[other_geom])
        body_name = mujoco.mj_id2name(
            model, mujoco.mjtObj.mjOBJ_BODY, other_body
        ) or "world"
        # Robot support is allowed; the requirement is that the object is no
        # longer supported by the table or another environment object.
        if body_name.startswith(("openarm_", "wuji_")):
            continue
        result.append({
            "other_body": body_name,
            "other_geom": mujoco.mj_id2name(
                model, mujoco.mjtObj.mjOBJ_GEOM, other_geom
            ) or f"geom_{other_geom}",
            "distance_m": float(contact.dist),
        })
    return result


def _contact_payload(telemetry: dict[str, Any]) -> str:
    contacts = []
    for contact in telemetry["contacts"]:
        contacts.append({
            "finger": contact["finger"],
            "cube_face": contact["cube_face"],
            "normal_force_n": float(contact["normal_force_n"]),
            "position_world_m": np.asarray(
                contact["position_world_m"], dtype=float
            ).tolist(),
            "force_on_cube_world_n": np.asarray(
                contact["force_on_cube_world_n"], dtype=float
            ).tolist(),
        })
    return json.dumps(contacts, separators=(",", ":"))


def run_scripted_lift_from_handoff(task, *, seed: int,
                                   terminal_hold_s: float = 1.0,
                                   frame_callback=None,
                                   ) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Run calibrated scripted Lift from the current live simulator state.

    No reset, state restore, expert state, or hand remapping occurs.  The 20-D
    target currently held by MuJoCo is copied once at handoff and asserted to
    remain bit-identical (up to floating point clipping) on every Lift step.
    """
    robot = task.robot
    config = task.config
    lift = config["lift"]
    baseline = config["external_baseline"]
    if not np.isfinite(terminal_hold_s) or terminal_hold_s < 1.0:
        raise ValueError("terminal_hold_s must be at least 1.0 second")

    current_target = np.r_[
        robot.data.ctrl[robot.arm_actuator_ids],
        robot.data.ctrl[robot.hand_actuator_ids],
    ].astype(float)
    preserved_hand_target = current_target[7:].copy()
    task.arm_command = current_target[:7].copy()
    task.arm_goal = task.arm_command.copy()
    task._arm_command_initialized = True

    initial = task.task_telemetry()
    lift_start_palm = np.asarray(
        initial["grasp_center_position_m"], dtype=float
    )
    lift_start_cube = np.asarray(initial["cube_position_m"], dtype=float)
    initial_relative_position = np.asarray(
        initial["object_relative_position_m"], dtype=float
    )
    initial_relative_quaternion = np.asarray(
        initial["object_relative_quaternion_wxyz"], dtype=float
    )
    preload_before = float(np.linalg.norm(
        preserved_hand_target - robot.data.qpos[robot.hand_qpos_ids]
    ))

    lift_delta = np.asarray(lift["grasp_center_delta_m"], dtype=float)
    lift_target = lift_start_palm + lift_delta
    _, final_residual, final_orientation_residual = task._solve_arm_goal(
        lift_target, phase_config=lift
    )
    ik_feasible = bool(
        final_residual <= float(lift["ik_solve_tolerance_m"])
        and final_orientation_residual <= float(
            config["reach"]["ik_orientation_solve_tolerance_deg"]
        )
    )

    trajectory_frames = int(round(lift_trajectory_duration_s(
        total_delta=lift_delta, baseline=baseline
    ) * robot.control_hz))
    terminal_hold_frames_required = int(math.ceil(
        terminal_hold_s * robot.control_hz
    ))
    max_steps = int(lift["max_steps"])
    required_steps = trajectory_frames + terminal_hold_frames_required
    if required_steps > max_steps:
        raise ValueError(
            "existing Lift max_steps cannot contain the scripted trajectory "
            "and requested terminal hold"
        )

    rows: dict[str, list[Any]] = {
        "phase": [],
        "sim_time": [],
        "controller_target": [],
        "actual_state": [],
        "cube_pose_world": [],
        "cube_pose_relative_to_palm": [],
        "cube_height_m": [],
        "palm_lift_m": [],
        "relative_translation_drift_m": [],
        "relative_rotation_drift_deg": [],
        "finger_normal_force_n": [],
        "hand_contact_points": [],
        "contact_finger_count": [],
        "contact_topology": [],
        "contacts_json": [],
        "environment_support": [],
        "environment_contacts_json": [],
        "resultant_force_world_n": [],
        "resultant_moment_world_nm": [],
        "hand_preload_l2_rad": [],
    }
    ik_failure_frame = None
    trajectory_complete = False
    crossed_minimum_lift = False
    dropped = False
    drop_frame = None
    maximum_hand_target_error = 0.0
    zero_contact_run = maximum_zero_contact_run = 0

    if ik_feasible:
        for frame in range(1, required_steps + 1):
            reference_frame = min(frame, trajectory_frames)
            waypoint = s_curve_lift_waypoint(
                start_position=lift_start_palm,
                total_delta=lift_delta,
                step=reference_frame,
                control_hz=robot.control_hz,
                baseline=baseline,
                limits=lift["s_curve_limits"],
            )
            arm_goal, residual, orientation_residual = task._solve_arm_goal(
                waypoint, phase_config=lift
            )
            if (residual > float(lift["ik_solve_tolerance_m"])
                    or orientation_residual > float(
                        config["reach"]["ik_orientation_solve_tolerance_deg"]
                    )):
                ik_failure_frame = frame
                break
            max_step = float(lift["max_action_step_rad"])
            task.arm_command += np.clip(
                arm_goal - task.arm_command, -max_step, max_step
            )
            sent = robot.send_controller_joint_target(
                compose_controller_target(task.arm_command, preserved_hand_target)
            )
            telemetry = task.task_telemetry()
            relative_translation, relative_rotation = pose_drift(
                initial_relative_position,
                initial_relative_quaternion,
                telemetry["object_relative_position_m"],
                telemetry["object_relative_quaternion_wxyz"],
            )
            cube_height = float(
                np.asarray(telemetry["cube_position_m"])[2]
                - lift_start_cube[2]
            )
            palm_lift = float(
                np.asarray(telemetry["grasp_center_position_m"])[2]
                - lift_start_palm[2]
            )
            forces = finger_force_vector(telemetry)
            topology = [
                name for name, force in zip(FINGER_NAMES, forces, strict=True)
                if force >= float(lift["min_normal_force_n"])
            ]
            raw_contact_count = len(telemetry["contacts"])
            zero_contact_run = zero_contact_run + 1 if raw_contact_count == 0 else 0
            maximum_zero_contact_run = max(
                maximum_zero_contact_run, zero_contact_run
            )
            environment_contacts = _cube_environment_contacts(task)
            preload = float(np.linalg.norm(
                preserved_hand_target - robot.data.qpos[robot.hand_qpos_ids]
            ))
            target_error = float(np.max(np.abs(
                sent[7:] - preserved_hand_target
            )))
            maximum_hand_target_error = max(
                maximum_hand_target_error, target_error
            )
            phase = "lift_s_curve" if frame <= trajectory_frames else "lift_hold"
            rows["phase"].append(phase)
            rows["sim_time"].append(float(robot.data.time))
            rows["controller_target"].append(sent)
            rows["actual_state"].append(np.r_[
                robot.data.qpos[robot.arm_qpos_ids],
                robot.data.qpos[robot.hand_qpos_ids],
            ])
            rows["cube_pose_world"].append(np.r_[
                telemetry["cube_position_m"],
                telemetry["cube_quaternion_wxyz"],
            ])
            rows["cube_pose_relative_to_palm"].append(np.r_[
                telemetry["object_relative_position_m"],
                telemetry["object_relative_quaternion_wxyz"],
            ])
            rows["cube_height_m"].append(cube_height)
            rows["palm_lift_m"].append(palm_lift)
            rows["relative_translation_drift_m"].append(relative_translation)
            rows["relative_rotation_drift_deg"].append(relative_rotation)
            rows["finger_normal_force_n"].append(forces)
            rows["hand_contact_points"].append(raw_contact_count)
            rows["contact_finger_count"].append(len(topology))
            rows["contact_topology"].append("+".join(topology))
            rows["contacts_json"].append(_contact_payload(telemetry))
            rows["environment_support"].append(bool(environment_contacts))
            rows["environment_contacts_json"].append(json.dumps(
                environment_contacts, separators=(",", ":")
            ))
            rows["resultant_force_world_n"].append(
                telemetry["contact_resultant_force_world_n"]
            )
            rows["resultant_moment_world_nm"].append(
                telemetry["contact_resultant_moment_about_cube_world_nm"]
            )
            rows["hand_preload_l2_rad"].append(preload)
            if frame_callback is not None:
                frame_callback(
                    "LIFT" if phase == "lift_s_curve" else "HOLD",
                    robot.get_observation(), telemetry,
                )

            crossed_minimum_lift = bool(
                crossed_minimum_lift
                or cube_height >= float(baseline["min_object_lift_m"])
            )
            if (crossed_minimum_lift
                    and cube_height < float(baseline["min_object_lift_m"])):
                dropped = True
                drop_frame = frame
                break
            if frame == trajectory_frames:
                trajectory_complete = True

    arrays = {key: np.asarray(value) for key, value in rows.items()}
    terminal_mask = (
        arrays["phase"] == "lift_hold" if len(arrays["phase"])
        else np.zeros(0, dtype=bool)
    )
    terminal_heights = arrays["cube_height_m"][terminal_mask]
    terminal_support = arrays["environment_support"][terminal_mask]
    success = load_bearing_success(
        trajectory_complete=trajectory_complete,
        terminal_heights_m=terminal_heights,
        terminal_environment_support=terminal_support,
        required_hold_frames=terminal_hold_frames_required,
        success_height_m=float(lift["success_height_m"]),
        dropped=dropped,
    )
    if not ik_feasible or ik_failure_frame is not None:
        failure_stage = "lift"
        failure_reason = "ik_infeasible"
    elif dropped:
        failure_stage = "lift"
        failure_reason = "drop"
    elif not trajectory_complete:
        failure_stage = "lift"
        failure_reason = "trajectory_incomplete"
    elif len(terminal_heights) < terminal_hold_frames_required:
        failure_stage = "lift_hold"
        failure_reason = "hold_incomplete"
    elif np.any(terminal_support):
        failure_stage = "lift_hold"
        failure_reason = "environment_support"
    elif np.any(terminal_heights < float(lift["success_height_m"])):
        failure_stage = "lift_hold"
        failure_reason = "height_not_held"
    else:
        failure_stage = failure_reason = None

    peak_height = float(np.max(arrays["cube_height_m"])) if len(
        arrays["cube_height_m"]
    ) else 0.0
    final_height = float(arrays["cube_height_m"][-1]) if len(
        arrays["cube_height_m"]
    ) else 0.0
    metrics: dict[str, Any] = {
        "seed": int(seed),
        "success": success,
        "failure_stage": failure_stage,
        "failure_reason": failure_reason,
        "trajectory_profile": "two_segment_quintic_minimum_jerk",
        "trajectory_modified": False,
        "controller_gain_modified": False,
        "lift_grasp_parameter_modified": False,
        "hand_target_source": "live_graspsecure_terminal_controller_target",
        "hand_target_remapped": False,
        "state_reset_or_restored_at_handoff": False,
        "trajectory_frames": trajectory_frames,
        "trajectory_complete": trajectory_complete,
        "terminal_hold_frames_required": terminal_hold_frames_required,
        "terminal_hold_frames_completed": int(np.count_nonzero(terminal_mask)),
        "terminal_hold_duration_s": float(
            np.count_nonzero(terminal_mask) / robot.control_hz
        ),
        "ik_final_target_feasible": ik_feasible,
        "ik_final_residual_m": float(final_residual),
        "ik_final_orientation_residual_deg": float(final_orientation_residual),
        "ik_failure_frame": ik_failure_frame,
        "preload_before_lift_l2_rad": preload_before,
        "preload_during_lift_l2_rad": {
            "mean": float(np.mean(arrays["hand_preload_l2_rad"])) if len(
                arrays["hand_preload_l2_rad"]
            ) else None,
            "median": float(np.median(arrays["hand_preload_l2_rad"])) if len(
                arrays["hand_preload_l2_rad"]
            ) else None,
            "min": float(np.min(arrays["hand_preload_l2_rad"])) if len(
                arrays["hand_preload_l2_rad"]
            ) else None,
            "max": float(np.max(arrays["hand_preload_l2_rad"])) if len(
                arrays["hand_preload_l2_rad"]
            ) else None,
        },
        "preserved_hand_target_rad": preserved_hand_target,
        "maximum_hand_target_change_rad": maximum_hand_target_error,
        "target_height_m": float(lift["success_height_m"]),
        "peak_cube_height_m": peak_height,
        "final_cube_height_m": final_height,
        "minimum_terminal_cube_height_m": (
            float(np.min(terminal_heights)) if len(terminal_heights) else None
        ),
        "cube_unsupported_for_entire_terminal_hold": bool(
            len(terminal_support) >= terminal_hold_frames_required
            and not np.any(terminal_support[-terminal_hold_frames_required:])
        ),
        "drop": dropped,
        "drop_frame": drop_frame,
        "drop_time_s": (
            float(drop_frame / robot.control_hz) if drop_frame is not None else None
        ),
        "max_cube_palm_translation_drift_m": (
            float(np.max(arrays["relative_translation_drift_m"])) if len(
                arrays["relative_translation_drift_m"]
            ) else 0.0
        ),
        "max_cube_palm_rotation_drift_deg": (
            float(np.max(arrays["relative_rotation_drift_deg"])) if len(
                arrays["relative_rotation_drift_deg"]
            ) else 0.0
        ),
        "maximum_consecutive_zero_hand_contact_frames": (
            maximum_zero_contact_run
        ),
        "max_simultaneous_contact_fingers": (
            int(np.max(arrays["contact_finger_count"])) if len(
                arrays["contact_finger_count"]
            ) else 0
        ),
        "terminal_contact_topology": (
            str(arrays["contact_topology"][-1]) if len(
                arrays["contact_topology"]
            ) else ""
        ),
        "peak_per_finger_normal_force_n": (
            np.max(arrays["finger_normal_force_n"], axis=0) if len(
                arrays["finger_normal_force_n"]
            ) else np.zeros(5)
        ),
        "terminal_per_finger_normal_force_n": (
            arrays["finger_normal_force_n"][-1] if len(
                arrays["finger_normal_force_n"]
            ) else np.zeros(5)
        ),
        "contact_topology_is_diagnostic_only": True,
        "force_is_diagnostic_only": True,
        "relative_drift_is_diagnostic_only": True,
    }
    return metrics, arrays


__all__ = [
    "cliffs_delta",
    "compose_controller_target",
    "load_bearing_success",
    "run_scripted_lift_from_handoff",
]
