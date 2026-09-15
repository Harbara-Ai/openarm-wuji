"""Physical micro-lift verification and one-regrasp support.

The probe is a prefix of the existing scripted Lift reference.  It holds the
exact live 20-D Wuji controller target and inserts a short dwell after the
first reference frame that reaches the configured probe height.  A passing
probe resumes the original reference at the next frame; it does not reset the
simulator or start a second 120 mm Lift.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import math
from typing import Any, Mapping, Sequence

import numpy as np

from .grasp_preload_stage import FINGER_NAMES, finger_force_vector
from .reach_grasp_lift import lift_trajectory_duration_s, s_curve_lift_waypoint
from .scripted_lift_handoff import (
    _contact_payload,
    _cube_environment_contacts,
    compose_controller_target,
    load_bearing_success,
)
from .se3 import pose_drift


@dataclass(frozen=True)
class MicroLiftProbeSpec:
    """Simple, interpretable v1 verification thresholds."""

    height_m: float = 0.015
    hold_s: float = 0.3
    maximum_settle_s: float = 1.0
    minimum_palm_lift_m: float = 0.010
    minimum_cube_lift_m: float = 0.005
    minimum_cube_to_palm_lift_ratio: float = 0.5
    maximum_supported_hold_fraction: float = 1.0 / 3.0
    maximum_zero_contact_hold_frames: int = 3
    maximum_relative_translation_drift_m: float = 0.015
    maximum_terminal_relative_speed_m_s: float = 0.03

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "MicroLiftProbeSpec":
        return cls(**{
            field: value[field]
            for field in cls.__dataclass_fields__
            if field in value
        })

    def validate(self) -> None:
        numeric = np.asarray([
            self.height_m,
            self.hold_s,
            self.maximum_settle_s,
            self.minimum_palm_lift_m,
            self.minimum_cube_lift_m,
            self.minimum_cube_to_palm_lift_ratio,
            self.maximum_supported_hold_fraction,
            self.maximum_relative_translation_drift_m,
            self.maximum_terminal_relative_speed_m_s,
        ], dtype=float)
        if not np.isfinite(numeric).all() or np.any(numeric <= 0.0):
            raise ValueError("probe thresholds must be positive and finite")
        if not 0.0 < self.maximum_supported_hold_fraction < 1.0:
            raise ValueError("supported hold fraction must lie in (0, 1)")
        if self.maximum_zero_contact_hold_frames < 0:
            raise ValueError("zero-contact allowance cannot be negative")


def longest_true_run(values: Sequence[bool]) -> int:
    best = current = 0
    for value in values:
        current = current + 1 if value else 0
        best = max(best, current)
    return best


def probe_reference_frame(*, full_delta_m: Sequence[float],
                          probe_height_m: float, control_hz: float,
                          baseline: Mapping[str, Any],
                          limits: Mapping[str, Any]) -> int:
    """Return the first unmodified full-Lift frame reaching probe height."""
    delta = np.asarray(full_delta_m, dtype=float)
    distance = float(np.linalg.norm(delta))
    if not 0.0 < probe_height_m < distance:
        raise ValueError("probe height must be inside the full Lift distance")
    frames = int(round(lift_trajectory_duration_s(
        total_delta=delta, baseline=dict(baseline)
    ) * control_hz))
    start = np.zeros(3, dtype=float)
    for frame in range(1, frames + 1):
        waypoint = s_curve_lift_waypoint(
            start_position=start,
            total_delta=delta,
            step=frame,
            control_hz=control_hz,
            baseline=dict(baseline),
            limits=dict(limits),
        )
        if float(np.linalg.norm(waypoint - start)) >= probe_height_m:
            return frame
    raise RuntimeError("full Lift reference never reached the probe height")


def evaluate_probe_arrays(arrays: Mapping[str, np.ndarray], *,
                          spec: MicroLiftProbeSpec,
                          hold_frames: int,
                          trajectory_complete: bool,
                          ik_failure_frame: int | None) -> dict[str, Any]:
    """Apply the v1 physical gate to recorded probe telemetry."""
    spec.validate()
    count = len(arrays["phase"])
    if count == 0:
        return {
            "pass": False,
            "criteria": {"trajectory_complete": False},
            "failure_reasons": ["trajectory_incomplete"],
        }
    hold_indices = np.flatnonzero(
        np.asarray(arrays["phase"]) == "micro_lift_probe_hold"
    )
    if len(hold_indices):
        tail_indices = hold_indices[-min(3, len(hold_indices)):]
    else:
        tail_indices = np.arange(max(0, count - min(3, count)), count)
    palm_lift = float(np.median(arrays["palm_lift_m"][tail_indices]))
    cube_lift = float(np.median(arrays["cube_height_m"][tail_indices]))
    ratio = cube_lift / max(palm_lift, 1e-12)
    supported_fraction = (
        float(np.mean(arrays["environment_support"][hold_indices]))
        if len(hold_indices) else 1.0
    )
    final_supported = bool(arrays["environment_support"][-1])
    zero_contact_run = longest_true_run(
        (arrays["hand_contact_points"][hold_indices] == 0).tolist()
    )
    translation_drift = float(np.max(
        arrays["relative_translation_drift_m"]
    ))
    rotation_drift = float(np.max(arrays["relative_rotation_drift_deg"]))
    terminal_speed = float(np.median(
        arrays["relative_linear_speed_m_s"][tail_indices]
    ))
    finite = bool(all(np.isfinite(arrays[key]).all() for key in (
        "controller_target", "actual_state", "cube_pose_world",
        "cube_pose_relative_to_palm", "cube_height_m", "palm_lift_m",
        "relative_translation_drift_m", "relative_rotation_drift_deg",
    )))
    criteria = {
        "finite": finite,
        "trajectory_complete": bool(
            trajectory_complete and ik_failure_frame is None
            and len(hold_indices) == hold_frames
        ),
        "palm_moved": palm_lift >= spec.minimum_palm_lift_m,
        "cube_followed": bool(
            cube_lift >= spec.minimum_cube_lift_m
            and ratio >= spec.minimum_cube_to_palm_lift_ratio
        ),
        "environment_support_released": bool(
            not final_supported
            and supported_fraction <= spec.maximum_supported_hold_fraction
        ),
        "contact_retained": (
            zero_contact_run <= spec.maximum_zero_contact_hold_frames
        ),
        "no_fast_escape": bool(
            translation_drift <= spec.maximum_relative_translation_drift_m
            and terminal_speed <= spec.maximum_terminal_relative_speed_m_s
        ),
    }
    failed = [name for name, passed in criteria.items() if not passed]
    return {
        "pass": not failed,
        "criteria": criteria,
        "failure_reasons": failed,
        "terminal_palm_lift_m": palm_lift,
        "terminal_cube_lift_m": cube_lift,
        "cube_to_palm_lift_ratio": ratio,
        "supported_hold_fraction": supported_fraction,
        "final_environment_supported": final_supported,
        "maximum_zero_contact_hold_frames": zero_contact_run,
        "maximum_relative_translation_drift_m": translation_drift,
        # Rotation is deliberately diagnostic-only in v1.
        "maximum_relative_rotation_drift_deg": rotation_drift,
        "terminal_relative_linear_speed_m_s": terminal_speed,
        "maximum_contact_slip_m_s": float(np.nanmax(
            arrays["contact_slip_max_m_s"]
        )) if np.isfinite(arrays["contact_slip_max_m_s"]).any() else None,
    }


def _contact_slip_speeds(task, contacts: list[dict[str, Any]]) -> list[float]:
    import mujoco

    model, data = task.model, task.data
    cube_velocity = np.zeros(6)
    mujoco.mj_objectVelocity(
        model, data, mujoco.mjtObj.mjOBJ_BODY,
        task.cube_body_id, cube_velocity, 0,
    )
    cube_angular, cube_linear = cube_velocity[:3], cube_velocity[3:]
    cube_com = np.asarray(data.xpos[task.cube_body_id], dtype=float)
    result = []
    for contact in contacts:
        body_id = int(contact["other_body_id"])
        finger_velocity = np.zeros(6)
        mujoco.mj_objectVelocity(
            model, data, mujoco.mjtObj.mjOBJ_BODY,
            body_id, finger_velocity, 0,
        )
        point = np.asarray(contact["position_world_m"], dtype=float)
        finger_point = finger_velocity[3:] + np.cross(
            finger_velocity[:3], point - np.asarray(data.xpos[body_id])
        )
        cube_point = cube_linear + np.cross(cube_angular, point - cube_com)
        relative = finger_point - cube_point
        normal = np.asarray(contact["normal_on_cube_world"], dtype=float)
        normal /= max(float(np.linalg.norm(normal)), 1e-12)
        tangential = relative - float(np.dot(relative, normal)) * normal
        result.append(float(np.linalg.norm(tangential)))
    return result


def _empty_rows() -> dict[str, list[Any]]:
    return {
        "phase": [], "reference_frame": [], "sim_time": [],
        "controller_target": [], "actual_state": [],
        "cube_pose_world": [], "cube_pose_relative_to_palm": [],
        "cube_height_m": [], "palm_lift_m": [],
        "relative_translation_drift_m": [],
        "relative_rotation_drift_deg": [],
        "relative_linear_speed_m_s": [],
        "relative_angular_speed_rad_s": [],
        "finger_normal_force_n": [], "hand_contact_points": [],
        "contact_finger_count": [], "contact_topology": [],
        "contacts_json": [], "environment_support": [],
        "environment_contacts_json": [],
        "contact_slip_mean_m_s": [], "contact_slip_max_m_s": [],
        "resultant_force_world_n": [], "resultant_moment_world_nm": [],
        "hand_preload_l2_rad": [],
    }


def _to_arrays(rows: dict[str, list[Any]]) -> dict[str, np.ndarray]:
    return {key: np.asarray(value) for key, value in rows.items()}


def concatenate_trajectories(*parts: Mapping[str, np.ndarray]
                             ) -> dict[str, np.ndarray]:
    keys = tuple(parts[0]) if parts else ()
    if any(tuple(part) != keys for part in parts):
        raise ValueError("trajectory schemas do not match")
    return {
        key: np.concatenate([np.asarray(part[key]) for part in parts], axis=0)
        for key in keys
    }


def _record_step(task, rows: dict[str, list[Any]], *, phase: str,
                 reference_frame: int, sent: np.ndarray,
                 reference_palm: np.ndarray, reference_cube: np.ndarray,
                 reference_relative_position: np.ndarray,
                 reference_relative_quaternion: np.ndarray,
                 preserved_hand_target: np.ndarray) -> None:
    robot = task.robot
    telemetry = task.task_telemetry()
    relative_translation, relative_rotation = pose_drift(
        reference_relative_position,
        reference_relative_quaternion,
        telemetry["object_relative_position_m"],
        telemetry["object_relative_quaternion_wxyz"],
    )
    cube_position = np.asarray(telemetry["cube_position_m"], dtype=float)
    palm_position = np.asarray(
        telemetry["grasp_center_position_m"], dtype=float
    )
    forces = finger_force_vector(telemetry)
    minimum_force = float(task.config["lift"]["min_normal_force_n"])
    topology = [
        name for name, force in zip(FINGER_NAMES, forces, strict=True)
        if force >= minimum_force
    ]
    environment = _cube_environment_contacts(task)
    contacts = telemetry["contacts"]
    slips = _contact_slip_speeds(task, contacts)
    cube_linear = np.asarray(
        telemetry["cube_linear_velocity_world_m_s"], dtype=float
    )
    palm_linear = np.asarray(
        telemetry["grasp_center_linear_velocity_world_m_s"], dtype=float
    )
    cube_angular = np.asarray(
        telemetry["cube_angular_velocity_world_rad_s"], dtype=float
    )
    palm_angular = np.asarray(
        telemetry["grasp_center_angular_velocity_world_rad_s"], dtype=float
    )
    rows["phase"].append(phase)
    rows["reference_frame"].append(reference_frame)
    rows["sim_time"].append(float(robot.data.time))
    rows["controller_target"].append(sent)
    rows["actual_state"].append(np.r_[
        robot.data.qpos[robot.arm_qpos_ids],
        robot.data.qpos[robot.hand_qpos_ids],
    ])
    rows["cube_pose_world"].append(np.r_[
        cube_position, telemetry["cube_quaternion_wxyz"]
    ])
    rows["cube_pose_relative_to_palm"].append(np.r_[
        telemetry["object_relative_position_m"],
        telemetry["object_relative_quaternion_wxyz"],
    ])
    rows["cube_height_m"].append(float(cube_position[2] - reference_cube[2]))
    rows["palm_lift_m"].append(float(palm_position[2] - reference_palm[2]))
    rows["relative_translation_drift_m"].append(relative_translation)
    rows["relative_rotation_drift_deg"].append(relative_rotation)
    rows["relative_linear_speed_m_s"].append(float(np.linalg.norm(
        cube_linear - palm_linear
    )))
    rows["relative_angular_speed_rad_s"].append(float(np.linalg.norm(
        cube_angular - palm_angular
    )))
    rows["finger_normal_force_n"].append(forces)
    rows["hand_contact_points"].append(len(contacts))
    rows["contact_finger_count"].append(len(topology))
    rows["contact_topology"].append("+".join(topology))
    rows["contacts_json"].append(_contact_payload(telemetry))
    rows["environment_support"].append(bool(environment))
    rows["environment_contacts_json"].append(json.dumps(
        environment, separators=(",", ":")
    ))
    rows["contact_slip_mean_m_s"].append(
        float(np.mean(slips)) if slips else np.nan
    )
    rows["contact_slip_max_m_s"].append(
        float(np.max(slips)) if slips else np.nan
    )
    rows["resultant_force_world_n"].append(
        telemetry["contact_resultant_force_world_n"]
    )
    rows["resultant_moment_world_nm"].append(
        telemetry["contact_resultant_moment_about_cube_world_nm"]
    )
    rows["hand_preload_l2_rad"].append(float(np.linalg.norm(
        preserved_hand_target - robot.data.qpos[robot.hand_qpos_ids]
    )))


def _send_reference(task, *, start_palm: np.ndarray,
                    full_delta: np.ndarray, reference_frame: int,
                    preserved_hand_target: np.ndarray
                    ) -> tuple[np.ndarray | None, dict[str, float]]:
    robot = task.robot
    lift = task.config["lift"]
    waypoint = s_curve_lift_waypoint(
        start_position=start_palm,
        total_delta=full_delta,
        step=reference_frame,
        control_hz=robot.control_hz,
        baseline=task.config["external_baseline"],
        limits=lift["s_curve_limits"],
    )
    arm_goal, residual, orientation_residual = task._solve_arm_goal(
        waypoint, phase_config=lift
    )
    feasible = bool(
        residual <= float(lift["ik_solve_tolerance_m"])
        and orientation_residual <= float(
            task.config["reach"]["ik_orientation_solve_tolerance_deg"]
        )
    )
    diagnostics = {
        "residual_m": float(residual),
        "orientation_residual_deg": float(orientation_residual),
        "feasible": feasible,
    }
    if not feasible:
        return None, diagnostics
    max_step = float(lift["max_action_step_rad"])
    task.arm_command += np.clip(
        arm_goal - task.arm_command, -max_step, max_step
    )
    sent = robot.send_controller_joint_target(compose_controller_target(
        task.arm_command, preserved_hand_target
    ))
    return sent, diagnostics


def _new_context(task, spec: MicroLiftProbeSpec) -> dict[str, Any]:
    robot = task.robot
    lift = task.config["lift"]
    baseline = task.config["external_baseline"]
    current = np.r_[
        robot.data.ctrl[robot.arm_actuator_ids],
        robot.data.ctrl[robot.hand_actuator_ids],
    ].astype(float)
    task.arm_command = current[:7].copy()
    task.arm_goal = task.arm_command.copy()
    task._arm_command_initialized = True
    telemetry = task.task_telemetry()
    full_delta = np.asarray(lift["grasp_center_delta_m"], dtype=float)
    frames = int(round(lift_trajectory_duration_s(
        total_delta=full_delta, baseline=baseline
    ) * robot.control_hz))
    probe_frame = probe_reference_frame(
        full_delta_m=full_delta,
        probe_height_m=spec.height_m,
        control_hz=robot.control_hz,
        baseline=baseline,
        limits=lift["s_curve_limits"],
    )
    return {
        "start_palm": np.asarray(
            telemetry["grasp_center_position_m"], dtype=float
        ),
        "start_cube": np.asarray(telemetry["cube_position_m"], dtype=float),
        "start_relative_position": np.asarray(
            telemetry["object_relative_position_m"], dtype=float
        ),
        "start_relative_quaternion": np.asarray(
            telemetry["object_relative_quaternion_wxyz"], dtype=float
        ),
        "preserved_hand_target": current[7:].copy(),
        "full_delta": full_delta,
        "full_trajectory_frames": frames,
        "probe_reference_frame": probe_frame,
    }


def run_micro_lift_probe(task, *, spec: MicroLiftProbeSpec
                         ) -> tuple[dict[str, Any], dict[str, np.ndarray], dict[str, Any]]:
    """Run the first 10–20 mm of the existing Lift and hold 0.2–0.3 s."""
    spec.validate()
    context = _new_context(task, spec)
    robot = task.robot
    hold_frames = int(round(spec.hold_s * robot.control_hz))
    if hold_frames < 1:
        raise ValueError("probe hold must contain at least one control frame")
    rows = _empty_rows()
    ik_failure_frame = None
    last_ik = None
    max_settle_frames = int(round(spec.maximum_settle_s * robot.control_hz))
    acquisition_complete = False
    local_frame = 0

    def execute(reference_frame: int, phase: str) -> bool:
        nonlocal ik_failure_frame, last_ik, local_frame
        local_frame += 1
        sent, last_ik = _send_reference(
            task,
            start_palm=context["start_palm"],
            full_delta=context["full_delta"],
            reference_frame=reference_frame,
            preserved_hand_target=context["preserved_hand_target"],
        )
        if sent is None:
            ik_failure_frame = local_frame
            return False
        _record_step(
            task, rows,
            phase=phase,
            reference_frame=reference_frame,
            sent=sent,
            reference_palm=context["start_palm"],
            reference_cube=context["start_cube"],
            reference_relative_position=context["start_relative_position"],
            reference_relative_quaternion=context["start_relative_quaternion"],
            preserved_hand_target=context["preserved_hand_target"],
        )
        return True

    # Execute the unmodified Lift reference prefix once.
    for reference_frame in range(1, context["probe_reference_frame"] + 1):
        if not execute(reference_frame, "micro_lift_probe"):
            break
    if len(rows["palm_lift_m"]):
        acquisition_complete = bool(
            rows["palm_lift_m"][-1] >= spec.minimum_palm_lift_m
        )

    # Repeating the last prefix reference is required because the existing
    # arm target has a per-frame rate limit.  These frames are settling toward
    # the same small IK goal and are deliberately not counted as probe hold.
    settle_completed = 0
    while (ik_failure_frame is None and not acquisition_complete
           and settle_completed < max_settle_frames):
        if not execute(
            context["probe_reference_frame"], "micro_lift_probe_settle"
        ):
            break
        settle_completed += 1
        acquisition_complete = bool(
            rows["palm_lift_m"][-1] >= spec.minimum_palm_lift_m
        )

    hold_completed = 0
    while (ik_failure_frame is None and acquisition_complete
           and hold_completed < hold_frames):
        if not execute(
            context["probe_reference_frame"], "micro_lift_probe_hold"
        ):
            break
        hold_completed += 1
    arrays = _to_arrays(rows)
    trajectory_complete = bool(
        ik_failure_frame is None
        and acquisition_complete
        and hold_completed == hold_frames
    )
    metrics = evaluate_probe_arrays(
        arrays,
        spec=spec,
        hold_frames=hold_frames,
        trajectory_complete=trajectory_complete,
        ik_failure_frame=ik_failure_frame,
    )
    metrics.update({
        "probe_height_requested_m": spec.height_m,
        "hold_frames": hold_frames,
        "hold_duration_s": hold_frames / robot.control_hz,
        "settle_frames_before_hold": settle_completed,
        "maximum_settle_frames": max_settle_frames,
        "actual_micro_lift_acquired": acquisition_complete,
        "probe_reference_frame": context["probe_reference_frame"],
        "full_trajectory_frames": context["full_trajectory_frames"],
        "trajectory_prefix_reused": True,
        "hand_target_preserved": bool(
            len(arrays["controller_target"]) > 0
            and np.max(np.abs(
                arrays["controller_target"][:, 7:]
                - context["preserved_hand_target"][None, :]
            )) <= 1e-12
        ),
        "ik_failure_frame": ik_failure_frame,
        "last_ik": last_ik,
        "rotation_is_diagnostic_only": True,
        "fixed_finger_topology_required": False,
        "antipodal_gate_used": False,
    })
    return metrics, arrays, context


def continue_scripted_lift_after_probe(
        task, *, context: Mapping[str, Any],
        probe_arrays: Mapping[str, np.ndarray], seed: int,
        terminal_hold_s: float = 1.0,
        diagnostic_forced_after_probe_fail: bool = False,
        ) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Resume the original Lift at the frame immediately after the probe."""
    robot = task.robot
    lift = task.config["lift"]
    baseline = task.config["external_baseline"]
    hold_frames = int(math.ceil(terminal_hold_s * robot.control_hz))
    if terminal_hold_s < 1.0:
        raise ValueError("full Lift terminal hold must remain at least 1 second")
    rows = _empty_rows()
    start_frame = int(context["probe_reference_frame"]) + 1
    final_frame = int(context["full_trajectory_frames"])
    ik_failure_frame = None
    trajectory_complete = False
    crossed_minimum = bool(
        len(probe_arrays["cube_height_m"])
        and np.max(probe_arrays["cube_height_m"])
        >= float(baseline["min_object_lift_m"])
    )
    dropped = False
    drop_frame = None
    local_frame = 0
    for reference_frame in list(range(start_frame, final_frame + 1)) + [final_frame] * hold_frames:
        local_frame += 1
        sent, _ = _send_reference(
            task,
            start_palm=np.asarray(context["start_palm"]),
            full_delta=np.asarray(context["full_delta"]),
            reference_frame=reference_frame,
            preserved_hand_target=np.asarray(context["preserved_hand_target"]),
        )
        if sent is None:
            ik_failure_frame = local_frame
            break
        phase = (
            "lift_s_curve" if reference_frame < final_frame
            or local_frame <= final_frame - start_frame + 1
            else "lift_hold"
        )
        _record_step(
            task, rows,
            phase=phase,
            reference_frame=reference_frame,
            sent=sent,
            reference_palm=np.asarray(context["start_palm"]),
            reference_cube=np.asarray(context["start_cube"]),
            reference_relative_position=np.asarray(
                context["start_relative_position"]
            ),
            reference_relative_quaternion=np.asarray(
                context["start_relative_quaternion"]
            ),
            preserved_hand_target=np.asarray(context["preserved_hand_target"]),
        )
        cube_height = rows["cube_height_m"][-1]
        crossed_minimum = bool(
            crossed_minimum
            or cube_height >= float(baseline["min_object_lift_m"])
        )
        if crossed_minimum and cube_height < float(baseline["min_object_lift_m"]):
            dropped = True
            drop_frame = local_frame
            break
        if reference_frame == final_frame:
            trajectory_complete = True
    arrays = _to_arrays(rows)
    hold_mask = arrays["phase"] == "lift_hold" if len(
        arrays["phase"]
    ) else np.zeros(0, dtype=bool)
    heights = arrays["cube_height_m"][hold_mask]
    support = arrays["environment_support"][hold_mask]
    success = load_bearing_success(
        trajectory_complete=trajectory_complete,
        terminal_heights_m=heights,
        terminal_environment_support=support,
        required_hold_frames=hold_frames,
        success_height_m=float(lift["success_height_m"]),
        dropped=dropped,
    )
    if ik_failure_frame is not None:
        reason = "ik_infeasible"
    elif dropped:
        reason = "drop"
    elif not trajectory_complete:
        reason = "trajectory_incomplete"
    elif len(heights) < hold_frames:
        reason = "hold_incomplete"
    elif np.any(support[-hold_frames:]):
        reason = "environment_support"
    elif np.any(heights[-hold_frames:] < float(lift["success_height_m"])):
        reason = "height_not_held"
    else:
        reason = None
    combined = concatenate_trajectories(probe_arrays, arrays)
    metrics = {
        "seed": seed,
        "success": success,
        "failure_reason": reason,
        "trajectory_complete": trajectory_complete,
        "terminal_hold_frames_required": hold_frames,
        "terminal_hold_frames_completed": int(np.count_nonzero(hold_mask)),
        "terminal_hold_duration_s": float(np.count_nonzero(hold_mask) / robot.control_hz),
        "drop": dropped,
        "drop_frame_after_probe": drop_frame,
        "peak_cube_height_m": float(np.max(combined["cube_height_m"])),
        "final_cube_height_m": float(combined["cube_height_m"][-1]),
        "maximum_relative_translation_drift_m": float(np.max(
            combined["relative_translation_drift_m"]
        )),
        "maximum_relative_rotation_drift_deg": float(np.max(
            combined["relative_rotation_drift_deg"]
        )),
        "preserved_hand_target_rad": np.asarray(
            context["preserved_hand_target"]
        ),
        "maximum_hand_target_change_rad": float(np.max(np.abs(
            combined["controller_target"][:, 7:]
            - np.asarray(context["preserved_hand_target"])[None, :]
        ))),
        "probe_prefix_reference_frames": int(context["probe_reference_frame"]),
        "remaining_reference_started_at_frame": start_frame,
        "full_reference_final_frame": final_frame,
        "original_scripted_lift_target_preserved": True,
        "scripted_lift_trajectory_modified": False,
        "diagnostic_forced_after_probe_fail": diagnostic_forced_after_probe_fail,
    }
    return metrics, combined


def return_probe_to_grasp(task, *, context: Mapping[str, Any],
                          settle_frames: int = 6
                          ) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Reverse the executed prefix, then hold before the one allowed regrasp."""
    if settle_frames < 1:
        raise ValueError("regrasp settle must contain at least one frame")
    rows = _empty_rows()
    reverse = list(range(int(context["probe_reference_frame"]) - 1, -1, -1))
    references = reverse + [0] * settle_frames
    ik_failure_frame = None
    for local_frame, reference_frame in enumerate(references, start=1):
        sent, _ = _send_reference(
            task,
            start_palm=np.asarray(context["start_palm"]),
            full_delta=np.asarray(context["full_delta"]),
            reference_frame=reference_frame,
            preserved_hand_target=np.asarray(context["preserved_hand_target"]),
        )
        if sent is None:
            ik_failure_frame = local_frame
            break
        _record_step(
            task, rows,
            phase=("probe_return" if local_frame <= len(reverse)
                   else "regrasp_settle"),
            reference_frame=reference_frame,
            sent=sent,
            reference_palm=np.asarray(context["start_palm"]),
            reference_cube=np.asarray(context["start_cube"]),
            reference_relative_position=np.asarray(
                context["start_relative_position"]
            ),
            reference_relative_quaternion=np.asarray(
                context["start_relative_quaternion"]
            ),
            preserved_hand_target=np.asarray(context["preserved_hand_target"]),
        )
    arrays = _to_arrays(rows)
    return {
        "success": bool(len(arrays["phase"]) == len(references)),
        "ik_failure_frame": ik_failure_frame,
        "frames": len(arrays["phase"]),
        "settle_frames": settle_frames,
        "maximum_cube_displacement_from_probe_start_m": float(np.max(
            np.linalg.norm(
                arrays["cube_pose_world"][:, :3]
                - np.asarray(context["start_cube"])[None, :], axis=1
            )
        )) if len(arrays["phase"]) else 0.0,
    }, arrays


def grasp_state_delta(first: Mapping[str, Any], second: Mapping[str, Any]
                      ) -> dict[str, Any]:
    """Continuous attempt-1 versus attempt-2 state change diagnostics."""
    from .se3 import rotation_geodesic_angle_deg

    first_actual = np.asarray(first["actual_hand_qpos_rad"], dtype=float)
    second_actual = np.asarray(second["actual_hand_qpos_rad"], dtype=float)
    first_target = np.asarray(first["target_hand_qpos_rad"], dtype=float)
    second_target = np.asarray(second["target_hand_qpos_rad"], dtype=float)
    first_pose = np.asarray(first["cube_pose_relative_to_palm"], dtype=float)
    second_pose = np.asarray(second["cube_pose_relative_to_palm"], dtype=float)
    first_force = np.asarray(first["per_finger_normal_force_n"], dtype=float)
    second_force = np.asarray(second["per_finger_normal_force_n"], dtype=float)
    topology_changed = first["contact_topology"] != second["contact_topology"]
    return {
        "target_hand_l2_change_rad": float(np.linalg.norm(
            second_target - first_target
        )),
        "actual_hand_l2_change_rad": float(np.linalg.norm(
            second_actual - first_actual
        )),
        "preload_l2_change_rad": float(
            second["preload_l2_rad"] - first["preload_l2_rad"]
        ),
        "force_distribution_l2_change_n": float(np.linalg.norm(
            second_force - first_force
        )),
        "cube_palm_translation_change_m": float(np.linalg.norm(
            second_pose[:3] - first_pose[:3]
        )),
        "cube_palm_rotation_change_deg": float(
            rotation_geodesic_angle_deg(first_pose[3:], second_pose[3:])
        ),
        "topology_changed": topology_changed,
    }


__all__ = [
    "MicroLiftProbeSpec",
    "concatenate_trajectories",
    "continue_scripted_lift_after_probe",
    "evaluate_probe_arrays",
    "grasp_state_delta",
    "longest_true_run",
    "probe_reference_frame",
    "return_probe_to_grasp",
    "run_micro_lift_probe",
]
