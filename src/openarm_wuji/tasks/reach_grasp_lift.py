from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from ..simulation.mujoco_backend import MujocoOpenArmWuji
from .ik import DampedLeastSquaresIK
from .metrics import FingerContactMonitor, summarize_contact_geometry
from .outcomes import evaluate_lift_outcome
from .se3 import (
    normalize_quaternion,
    pose_drift,
    quaternion_error_rotvec,
    quaternion_multiply,
    relative_pose,
    rotation_vector_to_quaternion,
    rotation_geodesic_angle_deg,
)

if TYPE_CHECKING:
    from ..dataset.episode_recorder import CausalEpisodeRecorder


def _minimum_jerk(progress: float) -> float:
    progress = float(np.clip(progress, 0.0, 1.0))
    return progress ** 3 * (
        10.0 - 15.0 * progress + 6.0 * progress ** 2
    )


def _segment_peak_kinematics(distance_m: float,
                             duration_s: float) -> tuple[float, float, float]:
    """Analytic velocity, acceleration, and jerk peaks of minimum jerk."""
    return (
        1.875 * distance_m / duration_s,
        (10.0 / np.sqrt(3.0)) * distance_m / duration_s ** 2,
        60.0 * distance_m / duration_s ** 3,
    )


def lift_trajectory_duration_s(*, total_delta: np.ndarray,
                               baseline: dict) -> float:
    """Duration of the two-segment minimum-jerk reference."""
    distance = float(np.linalg.norm(total_delta))
    protocol_duration = float(baseline["lift_duration_s"])
    protocol_distance = min(float(baseline["gripper_lift_m"]), distance)
    if distance <= protocol_distance:
        return protocol_duration
    nominal_speed = protocol_distance / protocol_duration
    return protocol_duration + (distance - protocol_distance) / nominal_speed


def s_curve_lift_waypoint(*, start_position: np.ndarray,
                          total_delta: np.ndarray, step: int,
                          control_hz: float, baseline: dict,
                          limits: dict) -> np.ndarray:
    """Return a bounded minimum-jerk lift waypoint.

    Segment one covers the published CD-WM 50 mm / 0.5 s motion. Any remaining
    distance uses the same mean speed in a second minimum-jerk segment. The
    analytic peaks are checked against explicit Cartesian limits before use.
    """
    start = np.asarray(start_position, dtype=float)
    delta = np.asarray(total_delta, dtype=float)
    if start.shape != (3,) or delta.shape != (3,):
        raise ValueError("lift positions must be 3-D")
    if not np.all(np.isfinite(np.r_[start, delta, control_hz])) or control_hz <= 0:
        raise ValueError("lift positions and control frequency must be finite and valid")
    distance = float(np.linalg.norm(delta))
    if distance == 0.0:
        return start.copy()
    protocol_duration = float(baseline["lift_duration_s"])
    protocol_distance = min(float(baseline["gripper_lift_m"]), distance)
    if (not np.isfinite(protocol_duration + protocol_distance)
            or protocol_duration <= 0.0 or protocol_distance <= 0.0):
        raise ValueError("external lift protocol must have positive distance/time")
    remaining_distance = distance - protocol_distance
    nominal_speed = protocol_distance / protocol_duration
    remaining_duration = remaining_distance / nominal_speed
    segments = [(protocol_distance, protocol_duration)]
    if remaining_distance > 0.0:
        segments.append((remaining_distance, remaining_duration))
    peaks = [_segment_peak_kinematics(*segment) for segment in segments]
    planned_peaks = np.max(np.asarray(peaks), axis=0)
    configured_limits = np.asarray([
        limits["max_velocity_m_s"],
        limits["max_acceleration_m_s2"],
        limits["max_jerk_m_s3"],
    ], dtype=float)
    if not np.all(np.isfinite(configured_limits)) or np.any(configured_limits <= 0):
        raise ValueError("Cartesian limits must be positive and finite")
    if np.any(planned_peaks > configured_limits + 1e-12):
        raise ValueError(
            "minimum-jerk lift exceeds configured velocity/acceleration/jerk limits"
        )

    elapsed = max(step, 0) / control_hz
    if elapsed <= protocol_duration:
        commanded_distance = protocol_distance * _minimum_jerk(
            elapsed / protocol_duration
        )
    else:
        commanded_distance = protocol_distance
        if remaining_distance > 0.0:
            commanded_distance += remaining_distance * _minimum_jerk(
                (elapsed - protocol_duration) / remaining_duration
            )
    commanded_distance = min(commanded_distance, distance)
    return start + delta * (commanded_distance / distance)


def _linear_slope(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    x = np.arange(len(values), dtype=float)
    x -= np.mean(x)
    return float(np.dot(x, np.asarray(values) - np.mean(values)) / np.dot(x, x))


def preload_wrench_trends(samples: list[dict]) -> tuple[float, float, bool]:
    """Return per-frame resultant wrench slopes and a non-divergence flag."""
    force = [
        float(np.linalg.norm(sample["contact_resultant_force_world_n"]))
        for sample in samples
    ]
    moment = [
        float(np.linalg.norm(
            sample["contact_resultant_moment_about_cube_world_nm"]
        ))
        for sample in samples
    ]
    force_slope = _linear_slope(force)
    moment_slope = _linear_slope(moment)
    return force_slope, moment_slope, bool(
        len(samples) >= 2 and force_slope <= 0.0 and moment_slope <= 0.0
    )


@dataclass(frozen=True)
class ReachResult:
    success: bool
    failure_reason: str | None
    seed: int
    steps: int
    hold_frames: int
    initial_error_m: float
    final_error_m: float
    minimum_error_m: float
    ik_residual_m: float
    ik_orientation_residual_deg: float
    cube_position_m: list[float]
    target_position_m: list[float]
    target_quaternion_wxyz: list[float]
    final_grasp_center_m: list[float]
    final_grasp_center_quaternion_wxyz: list[float]
    final_orientation_error_deg: float

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class GraspResult:
    success: bool
    failure_reason: str | None
    seed: int
    reach: ReachResult
    approach_steps: int
    close_steps: int
    frozen_synergy: float
    preload_steps: int
    preload_settle_steps: int
    preload_target_synergy: float
    preload_reached: bool
    preload_wrench_stable: bool
    preload_force_slope_n_per_frame: float | None
    preload_moment_slope_nm_per_frame: float | None
    settle_steps: int
    final_synergy: float
    synergy_frozen: bool
    settle_stable: bool
    settle_window_frames: int
    settle_max_translation_drift_m: float | None
    settle_max_rotation_drift_deg: float | None
    regrasp_required: bool
    contact_hold_frames: int
    max_contact_groups: int
    final_contact_groups: list[str]
    final_normal_forces_n: dict[str, float]
    peak_normal_forces_n: dict[str, float]
    approach_cube_displacement_m: float
    total_cube_displacement_m: float
    final_cube_position_m: list[float]

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class LiftResult:
    success: bool
    failure_reason: str | None
    task_success: bool
    grasp_stable: bool
    post_settle_stable: bool
    continuous_slip: bool
    outcome: str
    seed: int
    grasp: GraspResult
    steps: int
    height_hold_frames: int
    contact_loss_frames: int
    max_contact_loss_frames: int
    max_contact_groups: int
    final_contact_groups: list[str]
    final_normal_forces_n: dict[str, float]
    target_height_m: float
    peak_height_m: float
    final_height_m: float
    final_cube_position_m: list[float]
    max_relative_translation_drift_m: float
    max_relative_rotation_drift_deg: float
    final_window_translation_drift_m: float
    final_window_rotation_drift_deg: float
    post_settle_translation_drift_m: float
    post_settle_rotation_drift_deg: float
    post_settle_max_relative_translation_speed_m_s: float
    post_settle_max_relative_angular_speed_deg_s: float
    post_settle_rms_relative_translation_speed_m_s: float
    post_settle_rms_relative_angular_speed_deg_s: float
    post_settle_translation_drift_slope_m_s: float
    post_settle_rotation_drift_slope_deg_s: float
    establishment_translation_m: float
    establishment_rotation_deg: float
    gripper_lift_within_baseline_window_m: float
    external_lift_protocol_reached: bool
    external_object_lifted: bool
    external_baseline: dict
    contact_diagnostics: dict
    trajectory_diagnostics: dict

    def to_dict(self) -> dict:
        return asdict(self)


class ReachGraspLiftTask:
    """Deterministic Reach-Grasp-Lift state machine with optional recording."""

    def __init__(self, robot: MujocoOpenArmWuji, config: dict, *,
                 recorder: CausalEpisodeRecorder | None = None):
        import mujoco

        self.robot = robot
        self.config = config
        grasp = config["grasp"]
        target = float(grasp["preload_target_synergy"])
        increment = float(grasp["preload_synergy_step"])
        if not np.isfinite(target) or not 0.0 <= target <= 1.0:
            raise ValueError("preload target must be in [0, 1]")
        if not np.isfinite(increment) or increment <= 0.0:
            raise ValueError("preload synergy step must be positive and finite")
        if (int(grasp["contact_hold_frames"]) < 2
                or int(grasp["preload_max_steps"]) < 1
                or int(grasp["preload_settle_max_steps"]) < 1):
            raise ValueError("preload budgets must be positive and window >= 2")
        # Reject infeasible trajectories before resetting or moving the robot.
        s_curve_lift_waypoint(
            start_position=np.zeros(3),
            total_delta=np.asarray(config["lift"]["grasp_center_delta_m"]),
            step=0, control_hz=robot.control_hz,
            baseline=config["external_baseline"],
            limits=config["lift"]["s_curve_limits"],
        )
        self.recorder = recorder
        self.model = robot.model
        self.data = robot.data
        self.arm_side = config.get("arm_side", "left")
        self.arm_joint_names = [
            f"openarm_{self.arm_side}_joint{index}" for index in range(1, 8)
        ]
        reach = config["reach"]
        self.ik = DampedLeastSquaresIK(
            self.model,
            self.data,
            site_name=config["scene"]["grasp_site_name"],
            joint_names=self.arm_joint_names,
            damping=reach["ik_damping"],
            max_joint_step=reach["max_joint_step_rad"],
        )
        self.cube_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, config["scene"]["cube_name"]
        )
        self.cube_joint_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_JOINT, config["scene"]["cube_joint_name"]
        )
        if self.cube_body_id < 0 or self.cube_joint_id < 0:
            raise KeyError("task model is missing the configured cube body or free joint")
        self.contact_monitor = FingerContactMonitor(
            self.model,
            self.data,
            cube_body_id=self.cube_body_id,
            cube_half_size_m=config["scene"]["cube_half_size_m"],
            hand_side=self.arm_side,
        )
        self.commanded_palm_quaternion = normalize_quaternion(
            config["scene"]["palm_target_quaternion_wxyz"]
        )
        orientation_compensation = rotation_vector_to_quaternion(
            config["scene"].get(
                "palm_ik_orientation_compensation_rotvec_rad", [0.0, 0.0, 0.0]
            )
        )
        self.ik_command_palm_quaternion = quaternion_multiply(
            orientation_compensation, self.commanded_palm_quaternion
        )
        self.target_position = np.zeros(3)
        self.initial_cube_position = np.zeros(3)
        self.arm_goal = np.zeros(7)
        self.arm_command = np.zeros(7)
        self._arm_command_initialized = False
        self.ik_residual_m = float("inf")
        self.ik_orientation_residual_deg = float("inf")
        self._transition_observation: dict | None = None
        self._transition_telemetry: dict | None = None
        self._phase_start_telemetry: dict[str, dict] = {}
        self._phase_samples: dict[str, list[dict]] = {}

    @classmethod
    def from_json(cls, robot: MujocoOpenArmWuji, path: str | Path):
        return cls(robot, json.loads(Path(path).read_text(encoding="utf-8")))

    def _object_position(self) -> np.ndarray:
        return self.data.xpos[self.cube_body_id].copy()

    def _solve_arm_goal(self, target_position, *, phase_config
                        ) -> tuple[np.ndarray, float, float]:
        import mujoco

        scratch_data = mujoco.MjData(self.model)
        scratch_data.qpos[:] = self.data.qpos
        if self._arm_command_initialized:
            scratch_data.qpos[self.robot.arm_qpos_ids] = self.arm_command
        scratch_ik = DampedLeastSquaresIK(
            self.model,
            scratch_data,
            site_name=self.config["scene"]["grasp_site_name"],
            joint_names=self.arm_joint_names,
            damping=self.config["reach"]["ik_damping"],
            max_joint_step=self.config["reach"]["max_joint_step_rad"],
        )
        compensated_target = np.asarray(target_position, dtype=float) + np.asarray(
            phase_config["gravity_compensation_offset_m"], dtype=float
        )
        goal, residual, orientation_residual, _ = scratch_ik.solve_pose(
            compensated_target,
            self.ik_command_palm_quaternion,
            max_iterations=int(phase_config["ik_max_iterations"]),
            position_tolerance=float(phase_config["ik_solve_tolerance_m"]),
            orientation_tolerance_deg=float(
                self.config["reach"]["ik_orientation_solve_tolerance_deg"]
            ),
            orientation_weight=float(self.config["reach"]["ik_orientation_weight"]),
        )
        return goal, residual, orientation_residual

    def reset(self, seed: int) -> dict[str, np.ndarray | int]:
        import mujoco

        rng = np.random.default_rng(seed)
        reset = self.config["reset"]
        mujoco.mj_resetData(self.model, self.data)

        home = np.asarray(reset["home_arm_joint_position_rad"], dtype=float)
        if home.shape != (7,):
            raise ValueError("home_arm_joint_position_rad must be 7-D")
        for side in ("left", "right"):
            for index, value in enumerate(home, start=1):
                joint_name = f"openarm_{side}_joint{index}"
                actuator_name = f"{side}_joint{index}_ctrl"
                joint_id = mujoco.mj_name2id(
                    self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name
                )
                actuator_id = mujoco.mj_name2id(
                    self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_name
                )
                if joint_id >= 0:
                    self.data.qpos[self.model.jnt_qposadr[joint_id]] = value
                if actuator_id >= 0:
                    self.data.ctrl[actuator_id] = value

        hand_open = self.robot.mapper.open_pose.copy()
        self.data.qpos[self.robot.hand_qpos_ids] = hand_open
        self.data.ctrl[self.robot.hand_actuator_ids] = hand_open

        nominal = np.asarray(reset["cube_nominal_position_m"], dtype=float)
        noise = np.asarray(reset["cube_xy_noise_m"], dtype=float)
        if nominal.shape != (3,) or noise.shape != (2,) or np.any(noise < 0):
            raise ValueError("cube reset position/noise has an invalid shape or range")
        cube_position = nominal.copy()
        cube_position[:2] += rng.uniform(-noise, noise)
        cube_qpos_address = self.model.jnt_qposadr[self.cube_joint_id]
        self.data.qpos[cube_qpos_address:cube_qpos_address + 7] = np.r_[
            cube_position, [1.0, 0.0, 0.0, 0.0]
        ]
        self.data.qvel.fill(0.0)
        if self.data.act.size:
            self.data.act.fill(0.0)
        mujoco.mj_forward(self.model, self.data)
        for _ in range(int(reset["settle_physics_steps"])):
            mujoco.mj_step(self.model, self.data)
        self.robot.synchronize_after_reset(hand_open)

        self.initial_cube_position = self._object_position()
        self.target_position = self.initial_cube_position + np.asarray(
            self.config["reach"]["pregrasp_offset_m"], dtype=float
        )
        # Solve against a kinematic copy. The configured Cartesian bias
        # compensates measured gravity sag; it is explicit task calibration.
        self._arm_command_initialized = False
        (self.arm_goal, self.ik_residual_m,
         self.ik_orientation_residual_deg) = self._solve_arm_goal(
            self.target_position, phase_config=self.config["reach"]
        )
        self.arm_command = self.data.qpos[self.robot.arm_qpos_ids].copy()
        self._arm_command_initialized = True
        self._phase_start_telemetry = {}
        self._phase_samples = {}
        initial_telemetry = self.task_telemetry()
        self._transition_telemetry = initial_telemetry
        self._transition_observation = None
        if self.recorder is not None:
            self.recorder.start(seed)
            self._transition_observation = self.robot.get_observation()
        return {
            "seed": seed,
            "cube_position_m": self.initial_cube_position.copy(),
            "target_position_m": self.target_position.copy(),
            "grasp_center_position_m": self.ik.position,
            "grasp_center_quaternion_wxyz": self.ik.quaternion,
            "commanded_palm_quaternion_wxyz": self.commanded_palm_quaternion.copy(),
            "ik_command_palm_quaternion_wxyz": (
                self.ik_command_palm_quaternion.copy()
            ),
        }

    def task_telemetry(self) -> dict:
        """Return task diagnostics kept outside the policy observation."""
        import mujoco

        mujoco.mj_forward(self.model, self.data)
        cube_position = self._object_position()
        cube_quaternion = self.data.xquat[self.cube_body_id].copy()
        grasp_position = self.data.site_xpos[self.ik.site_id].copy()
        grasp_quaternion = np.zeros(4)
        mujoco.mju_mat2Quat(
            grasp_quaternion, self.data.site_xmat[self.ik.site_id]
        )
        relative_position, relative_quaternion = relative_pose(
            grasp_position,
            grasp_quaternion,
            cube_position,
            cube_quaternion,
        )
        cube_velocity = np.zeros(6)
        grasp_velocity = np.zeros(6)
        mujoco.mj_objectVelocity(
            self.model, self.data, mujoco.mjtObj.mjOBJ_BODY,
            self.cube_body_id, cube_velocity, 0,
        )
        mujoco.mj_objectVelocity(
            self.model, self.data, mujoco.mjtObj.mjOBJ_SITE,
            self.ik.site_id, grasp_velocity, 0,
        )
        contacts = self.contact_monitor.contacts(0.0)
        contact_geometry = summarize_contact_geometry(contacts)
        finger_forces: dict[str, float] = {}
        for contact in contacts:
            finger = contact["finger"]
            finger_forces[finger] = max(
                finger_forces.get(finger, 0.0), contact["normal_force_n"]
            )
        resultant_force = np.sum(
            [contact["force_on_cube_world_n"] for contact in contacts],
            axis=0,
        ) if contacts else np.zeros(3)
        resultant_moment = np.sum(
            [contact["moment_about_cube_center_world_nm"] for contact in contacts],
            axis=0,
        ) if contacts else np.zeros(3)
        return {
            "cube_position_m": cube_position,
            "cube_quaternion_wxyz": cube_quaternion,
            "cube_height_m": float(
                cube_position[2] - self.initial_cube_position[2]
            ),
            "grasp_center_position_m": grasp_position,
            "grasp_center_quaternion_wxyz": grasp_quaternion,
            "commanded_palm_quaternion_wxyz": self.commanded_palm_quaternion.copy(),
            "ik_command_palm_quaternion_wxyz": (
                self.ik_command_palm_quaternion.copy()
            ),
            "palm_orientation_error_rotvec_rad": quaternion_error_rotvec(
                self.commanded_palm_quaternion, grasp_quaternion
            ),
            "palm_orientation_error_deg": rotation_geodesic_angle_deg(
                self.commanded_palm_quaternion, grasp_quaternion
            ),
            "cube_angular_velocity_world_rad_s": cube_velocity[:3].copy(),
            "cube_linear_velocity_world_m_s": cube_velocity[3:].copy(),
            "grasp_center_angular_velocity_world_rad_s": grasp_velocity[:3].copy(),
            "grasp_center_linear_velocity_world_m_s": grasp_velocity[3:].copy(),
            "object_relative_position_m": relative_position,
            "object_relative_quaternion_wxyz": relative_quaternion,
            "finger_normal_forces_n": finger_forces,
            "contacts": contacts,
            "contact_resultant_force_world_n": resultant_force,
            "contact_resultant_moment_about_cube_world_nm": resultant_moment,
            "contact_geometry": contact_geometry,
        }

    def _send_action(self, action, *, phase: str) -> np.ndarray:
        """Send one action and optionally record its exact causal transition."""
        if self._transition_telemetry is None:
            raise RuntimeError("task must be reset before sending actions")
        self._phase_start_telemetry.setdefault(
            phase, self._transition_telemetry
        )
        sent_action = self.robot.send_action(action)
        telemetry_tp1 = self.task_telemetry()
        self._phase_samples.setdefault(phase, []).append(telemetry_tp1)
        if self.recorder is not None:
            if (self._transition_observation is None
                    or self._transition_telemetry is None):
                raise RuntimeError("task must be reset before recording actions")
            observation_tp1 = self.robot.latest_record
            self.recorder.record_transition(
                phase=phase,
                observation_t=self._transition_observation,
                action_t=sent_action,
                observation_tp1=observation_tp1,
                telemetry_t=self._transition_telemetry,
                telemetry_tp1=telemetry_tp1,
            )
            self._transition_observation = observation_tp1
        self._transition_telemetry = telemetry_tp1
        return sent_action

    def reach_action(self) -> np.ndarray:
        max_step = float(self.config["reach"]["max_action_step_rad"])
        self.arm_command += np.clip(
            self.arm_goal - self.arm_command, -max_step, max_step
        )
        return np.r_[self.arm_command, [0.0, 0.0, 0.0]]

    def run_reach(self, seed: int) -> ReachResult:
        self.reset(seed)
        return self._run_reach_from_reset(seed)

    def _run_reach_from_reset(self, seed: int) -> ReachResult:
        reach = self.config["reach"]
        tolerance = float(reach["position_tolerance_m"])
        required_hold = int(reach["hold_frames"])
        max_steps = int(reach["max_steps"])
        initial_error = float(np.linalg.norm(self.target_position - self.ik.position))
        initial_orientation_error = rotation_geodesic_angle_deg(
            self.commanded_palm_quaternion, self.ik.quaternion
        )
        minimum_error = initial_error
        hold = 0
        steps = 0
        if (self.ik_residual_m > float(reach["ik_solve_tolerance_m"])
                or self.ik_orientation_residual_deg
                > float(reach["ik_orientation_solve_tolerance_deg"])):
            position = self.ik.position
            quaternion = self.ik.quaternion
            return ReachResult(
                success=False,
                failure_reason="ik_unreachable",
                seed=seed,
                steps=0,
                hold_frames=0,
                initial_error_m=initial_error,
                final_error_m=initial_error,
                minimum_error_m=initial_error,
                ik_residual_m=self.ik_residual_m,
                ik_orientation_residual_deg=self.ik_orientation_residual_deg,
                cube_position_m=self.initial_cube_position.tolist(),
                target_position_m=self.target_position.tolist(),
                target_quaternion_wxyz=self.commanded_palm_quaternion.tolist(),
                final_grasp_center_m=position.tolist(),
                final_grasp_center_quaternion_wxyz=quaternion.tolist(),
                final_orientation_error_deg=initial_orientation_error,
            )
        for steps in range(1, max_steps + 1):
            self._send_action(self.reach_action(), phase="reach")
            error = float(np.linalg.norm(self.target_position - self.ik.position))
            orientation_error = rotation_geodesic_angle_deg(
                self.commanded_palm_quaternion, self.ik.quaternion
            )
            minimum_error = min(minimum_error, error)
            hold = hold + 1 if (
                error <= tolerance
                and orientation_error <= float(reach["orientation_tolerance_deg"])
            ) else 0
            if hold >= required_hold:
                break
        final_position = self.ik.position
        final_error = float(np.linalg.norm(self.target_position - final_position))
        final_quaternion = self.ik.quaternion
        final_orientation_error = rotation_geodesic_angle_deg(
            self.commanded_palm_quaternion, final_quaternion
        )
        success = hold >= required_hold
        return ReachResult(
            success=success,
            failure_reason=None if success else "reach_timeout",
            seed=seed,
            steps=steps,
            hold_frames=hold,
            initial_error_m=initial_error,
            final_error_m=final_error,
            minimum_error_m=minimum_error,
            ik_residual_m=self.ik_residual_m,
            ik_orientation_residual_deg=self.ik_orientation_residual_deg,
            cube_position_m=self.initial_cube_position.tolist(),
            target_position_m=self.target_position.tolist(),
            target_quaternion_wxyz=self.commanded_palm_quaternion.tolist(),
            final_grasp_center_m=final_position.tolist(),
            final_grasp_center_quaternion_wxyz=final_quaternion.tolist(),
            final_orientation_error_deg=final_orientation_error,
        )

    def run_grasp(self, seed: int) -> GraspResult:
        self.reset(seed)
        reach_result = self._run_reach_from_reset(seed)
        if not reach_result.success:
            return self._grasp_failure(
                seed, reach_result, reach_result.failure_reason or "reach_failed"
            )

        grasp = self.config["grasp"]
        approach_start_cube = self._object_position()
        grasp_target = approach_start_cube + np.asarray(
            grasp["target_offset_m"], dtype=float
        )
        grasp_goal, grasp_residual, grasp_orientation_residual = self._solve_arm_goal(
            grasp_target, phase_config=grasp
        )
        if (grasp_residual > float(grasp["ik_solve_tolerance_m"])
                or grasp_orientation_residual
                > float(self.config["reach"]["ik_orientation_solve_tolerance_deg"])):
            return self._grasp_failure(
                seed, reach_result, "grasp_ik_unreachable"
            )

        approach_steps = 0
        settled = 0
        max_action_step = float(grasp["max_action_step_rad"])
        for approach_steps in range(1, int(grasp["approach_max_steps"]) + 1):
            self.arm_command += np.clip(
                grasp_goal - self.arm_command, -max_action_step, max_action_step
            )
            self._send_action(
                np.r_[self.arm_command, [0.0, 0.0, 0.0]], phase="approach"
            )
            at_goal = bool(np.max(np.abs(grasp_goal - self.arm_command)) <= 1e-9)
            settled = settled + 1 if at_goal else 0
            if settled >= int(grasp["approach_settle_frames"]):
                break
        approach_displacement = float(np.linalg.norm(
            self._object_position() - approach_start_cube
        ))
        if approach_displacement > float(grasp["max_approach_cube_displacement_m"]):
            return self._grasp_failure(
                seed,
                reach_result,
                "approach_collision",
                approach_steps=approach_steps,
                approach_cube_displacement_m=approach_displacement,
            )
        if settled < int(grasp["approach_settle_frames"]):
            return self._grasp_failure(
                seed,
                reach_result,
                "approach_timeout",
                approach_steps=approach_steps,
                approach_cube_displacement_m=approach_displacement,
            )

        synergy = 0.0
        synergy_frozen = False
        max_contact_groups = 0
        final_contacts: dict[str, float] = {}
        peak_forces: dict[str, float] = {}
        close_steps = 0
        min_contact_groups = int(grasp["min_finger_groups"])
        min_normal_force = float(grasp["min_normal_force_n"])
        minimum_synergy = float(grasp["min_synergy_for_success"])
        for close_steps in range(1, int(grasp["max_close_steps"]) + 1):
            synergy = min(1.0, synergy + float(grasp["close_synergy_step"]))
            self._send_action(
                np.r_[self.arm_command, [synergy, 0.0, 0.0]],
                phase="grasp_close",
            )
            final_contacts = self.contact_monitor.sample(min_normal_force)
            max_contact_groups = max(max_contact_groups, len(final_contacts))
            for finger, force in final_contacts.items():
                peak_forces[finger] = max(peak_forces.get(finger, 0.0), force)
            enough_contacts = len(final_contacts) >= min_contact_groups
            enough_closure = synergy >= minimum_synergy
            if enough_contacts and enough_closure:
                synergy_frozen = True
                break

        frozen_synergy = synergy if synergy_frozen else 0.0
        # A later first contact may freeze above the configured target. Never
        # reopen the grasp at the transition into preload.
        preload_target = max(frozen_synergy, float(grasp["preload_target_synergy"]))
        preload_steps = 0
        preload_reached = False
        if synergy_frozen:
            for preload_steps in range(1, int(grasp["preload_max_steps"]) + 1):
                synergy = min(
                    preload_target,
                    synergy + float(grasp["preload_synergy_step"]),
                )
                self._send_action(
                    np.r_[self.arm_command, [synergy, 0.0, 0.0]],
                    phase="preload",
                )
                final_contacts = self.contact_monitor.sample(min_normal_force)
                max_contact_groups = max(max_contact_groups, len(final_contacts))
                for finger, force in final_contacts.items():
                    peak_forces[finger] = max(peak_forces.get(finger, 0.0), force)
                if synergy >= preload_target:
                    preload_reached = True
                    break

        settle_window_frames = int(grasp["contact_hold_frames"])
        preload_settle_steps = 0
        settle_stable = False
        settle_max_translation: float | None = None
        settle_max_rotation: float | None = None
        preload_wrench_stable = False
        preload_force_slope: float | None = None
        preload_moment_slope: float | None = None
        settle_samples: list[dict] = []
        if preload_reached:
            for preload_settle_steps in range(
                1, int(grasp["preload_settle_max_steps"]) + 1
            ):
                self._send_action(
                    np.r_[self.arm_command, [synergy, 0.0, 0.0]],
                    phase="preload_settle",
                )
                telemetry = self._phase_samples["preload_settle"][-1]
                settle_samples.append(telemetry)
                final_contacts = {
                    finger: force
                    for finger, force in telemetry["finger_normal_forces_n"].items()
                    if force >= min_normal_force
                }
                max_contact_groups = max(max_contact_groups, len(final_contacts))
                for finger, force in final_contacts.items():
                    peak_forces[finger] = max(peak_forces.get(finger, 0.0), force)
                window = settle_samples[-settle_window_frames:]
                anchor = window[0]
                translation_drifts = []
                rotation_drifts = []
                contacts_sustained = True
                for sample in window:
                    translation, rotation = pose_drift(
                        anchor["object_relative_position_m"],
                        anchor["object_relative_quaternion_wxyz"],
                        sample["object_relative_position_m"],
                        sample["object_relative_quaternion_wxyz"],
                    )
                    translation_drifts.append(translation)
                    rotation_drifts.append(rotation)
                    active_fingers = sum(
                        force >= min_normal_force
                        for force in sample["finger_normal_forces_n"].values()
                    )
                    contacts_sustained = (
                        contacts_sustained and active_fingers >= min_contact_groups
                    )
                settle_max_translation = max(translation_drifts)
                settle_max_rotation = max(rotation_drifts)
                if len(settle_samples) < settle_window_frames:
                    continue
                preload_force_slope, preload_moment_slope, preload_wrench_stable = (
                    preload_wrench_trends(window)
                )
                baseline = self.config["external_baseline"]
                settle_stable = (
                    contacts_sustained
                    and preload_wrench_stable
                    and settle_max_translation
                    < float(baseline["max_relative_translation_drift_m"])
                    and settle_max_rotation
                    < float(baseline["max_relative_rotation_drift_deg"])
                )
                if settle_stable:
                    break

        final_cube = self._object_position()
        failure_reason = None
        if not synergy_frozen:
            failure_reason = "grasp_empty"
        elif not preload_reached:
            failure_reason = "preload_timeout"
        elif not settle_stable:
            failure_reason = "preload_unstable"
        return GraspResult(
            success=settle_stable,
            failure_reason=failure_reason,
            seed=seed,
            reach=reach_result,
            approach_steps=approach_steps,
            close_steps=close_steps,
            frozen_synergy=frozen_synergy,
            preload_steps=preload_steps,
            preload_settle_steps=preload_settle_steps,
            preload_target_synergy=preload_target,
            preload_reached=preload_reached,
            preload_wrench_stable=preload_wrench_stable,
            preload_force_slope_n_per_frame=preload_force_slope,
            preload_moment_slope_nm_per_frame=preload_moment_slope,
            settle_steps=preload_settle_steps,
            final_synergy=synergy,
            synergy_frozen=synergy_frozen,
            settle_stable=settle_stable,
            settle_window_frames=settle_window_frames,
            settle_max_translation_drift_m=settle_max_translation,
            settle_max_rotation_drift_deg=settle_max_rotation,
            regrasp_required=bool(synergy_frozen and not settle_stable),
            contact_hold_frames=settle_window_frames if settle_stable else 0,
            max_contact_groups=max_contact_groups,
            final_contact_groups=sorted(final_contacts),
            final_normal_forces_n=final_contacts,
            peak_normal_forces_n=peak_forces,
            approach_cube_displacement_m=approach_displacement,
            total_cube_displacement_m=float(np.linalg.norm(
                final_cube - self.initial_cube_position
            )),
            final_cube_position_m=final_cube.tolist(),
        )

    def _grasp_failure(self, seed: int, reach: ReachResult, reason: str, *,
                       approach_steps: int = 0,
                       approach_cube_displacement_m: float = 0.0) -> GraspResult:
        final_cube = self._object_position()
        return GraspResult(
            success=False,
            failure_reason=reason,
            seed=seed,
            reach=reach,
            approach_steps=approach_steps,
            close_steps=0,
            frozen_synergy=0.0,
            preload_steps=0,
            preload_settle_steps=0,
            preload_target_synergy=float(
                self.config["grasp"]["preload_target_synergy"]
            ),
            preload_reached=False,
            preload_wrench_stable=False,
            preload_force_slope_n_per_frame=None,
            preload_moment_slope_nm_per_frame=None,
            settle_steps=0,
            final_synergy=0.0,
            synergy_frozen=False,
            settle_stable=False,
            settle_window_frames=int(self.config["grasp"]["contact_hold_frames"]),
            settle_max_translation_drift_m=None,
            settle_max_rotation_drift_deg=None,
            regrasp_required=False,
            contact_hold_frames=0,
            max_contact_groups=0,
            final_contact_groups=[],
            final_normal_forces_n={},
            peak_normal_forces_n={},
            approach_cube_displacement_m=approach_cube_displacement_m,
            total_cube_displacement_m=float(np.linalg.norm(
                final_cube - self.initial_cube_position
            )),
            final_cube_position_m=final_cube.tolist(),
        )

    def run_lift(self, seed: int) -> LiftResult:
        grasp_result = self.run_grasp(seed)
        if not grasp_result.success:
            return self._lift_result(
                seed=seed,
                grasp=grasp_result,
                approach_push=grasp_result.failure_reason == "approach_collision",
            )

        lift = self.config["lift"]
        lift_start_position = self.ik.position
        lift_delta = np.asarray(lift["grasp_center_delta_m"], dtype=float)
        lift_target = lift_start_position + lift_delta
        lift_goal, lift_residual, lift_orientation_residual = self._solve_arm_goal(
            lift_target, phase_config=lift
        )
        if (lift_residual > float(lift["ik_solve_tolerance_m"])
                or lift_orientation_residual
                > float(self.config["reach"]["ik_orientation_solve_tolerance_deg"])):
            return self._lift_result(
                seed=seed,
                grasp=grasp_result,
                approach_push=False,
            )

        success_height = float(lift["success_height_m"])
        required_height_hold = int(lift["success_hold_frames"])
        required_contact_groups = int(lift["min_finger_groups"])
        hold_synergy = grasp_result.final_synergy
        height_hold = 0
        contact_loss = 0
        max_contact_loss = 0
        max_contact_groups = 0
        peak_height = 0.0
        final_contacts: dict[str, float] = {}
        task_success = False
        trajectory_frames = int(round(lift_trajectory_duration_s(
            total_delta=lift_delta,
            baseline=self.config["external_baseline"],
        ) * self.robot.control_hz))
        required_post_trajectory_hold = int(round(
            float(self.config["post_settle"]["window_s"]) * self.robot.control_hz
        ))
        steps = 0
        minimum_external_lift = float(
            self.config["external_baseline"]["min_object_lift_m"]
        )
        crossed_minimum_lift = False

        for steps in range(1, int(lift["max_steps"]) + 1):
            waypoint = s_curve_lift_waypoint(
                start_position=lift_start_position,
                total_delta=lift_delta,
                step=steps,
                control_hz=self.robot.control_hz,
                baseline=self.config["external_baseline"],
                limits=lift["s_curve_limits"],
            )
            lift_goal, lift_residual, lift_orientation_residual = self._solve_arm_goal(
                waypoint, phase_config=lift
            )
            if (lift_residual > float(lift["ik_solve_tolerance_m"])
                    or lift_orientation_residual
                    > float(self.config["reach"]["ik_orientation_solve_tolerance_deg"])):
                steps -= 1  # Count only actions actually sent to the robot.
                break
            max_step = float(lift["max_action_step_rad"])
            self.arm_command += np.clip(
                lift_goal - self.arm_command, -max_step, max_step
            )
            self._send_action(
                np.r_[self.arm_command, [hold_synergy, 0.0, 0.0]],
                phase="lift_s_curve",
            )
            height = float(self._object_position()[2] - self.initial_cube_position[2])
            peak_height = max(peak_height, height)
            final_contacts = self.contact_monitor.sample(
                float(lift["min_normal_force_n"])
            )
            max_contact_groups = max(max_contact_groups, len(final_contacts))
            enough_contacts = len(final_contacts) >= required_contact_groups
            contact_loss = 0 if enough_contacts else contact_loss + 1
            max_contact_loss = max(max_contact_loss, contact_loss)
            crossed_minimum_lift = (
                crossed_minimum_lift or height >= minimum_external_lift
            )

            stable_height = height >= success_height
            height_hold = height_hold + 1 if stable_height else 0
            if height_hold >= required_height_hold:
                task_success = True
            if crossed_minimum_lift and height < minimum_external_lift:
                task_success = False
                break
            if (task_success and steps - trajectory_frames
                    >= required_post_trajectory_hold):
                break
        final_cube = self._object_position()
        final_height = float(final_cube[2] - self.initial_cube_position[2])
        return self._assemble_lift_result(
            seed=seed,
            grasp=grasp_result,
            task_success=task_success,
            approach_push=False,
            steps=steps,
            height_hold=height_hold,
            contact_loss=contact_loss,
            max_contact_loss=max_contact_loss,
            max_contact_groups=max_contact_groups,
            final_contacts=final_contacts,
            peak_height=peak_height,
            final_height=final_height,
            final_cube=final_cube,
        )

    def _lift_result(self, *, seed: int, grasp: GraspResult,
                     approach_push: bool) -> LiftResult:
        final_cube = self._object_position()
        final_height = float(final_cube[2] - self.initial_cube_position[2])
        return self._assemble_lift_result(
            seed=seed,
            grasp=grasp,
            task_success=False,
            approach_push=approach_push,
            steps=0,
            height_hold=0,
            contact_loss=0,
            max_contact_loss=0,
            max_contact_groups=0,
            final_contacts={},
            peak_height=max(0.0, final_height),
            final_height=final_height,
            final_cube=final_cube,
        )

    def _assemble_lift_result(self, *, seed: int, grasp: GraspResult,
                              task_success: bool, approach_push: bool,
                              steps: int, height_hold: int,
                              contact_loss: int, max_contact_loss: int,
                              max_contact_groups: int,
                              final_contacts: dict[str, float],
                              peak_height: float, final_height: float,
                              final_cube: np.ndarray) -> LiftResult:
        current = self._transition_telemetry or self.task_telemetry()
        lift_start = self._phase_start_telemetry.get("lift_s_curve", current)
        lift_samples = [lift_start, *self._phase_samples.get("lift_s_curve", [])]
        grasp_close_start = self._phase_start_telemetry.get(
            "grasp_close", lift_start
        )
        evaluation = evaluate_lift_outcome(
            task_success=task_success,
            approach_push=approach_push,
            grasp_close_start=grasp_close_start,
            lift_samples=lift_samples,
            control_hz=self.robot.control_hz,
            success_hold_frames=int(self.config["lift"]["success_hold_frames"]),
            baseline=self.config["external_baseline"],
            post_settle=self.config["post_settle"],
        )
        stable_success = bool(task_success and evaluation["grasp_stable"])
        contact_diagnostics = self._contact_diagnostics(lift_samples)
        return LiftResult(
            success=stable_success,
            failure_reason=None if stable_success else evaluation["outcome"],
            task_success=bool(evaluation["task_success"]),
            grasp_stable=bool(evaluation["grasp_stable"]),
            post_settle_stable=bool(evaluation["post_settle_stable"]),
            continuous_slip=bool(evaluation["continuous_slip"]),
            outcome=str(evaluation["outcome"]),
            seed=seed,
            grasp=grasp,
            steps=steps,
            height_hold_frames=height_hold,
            contact_loss_frames=contact_loss,
            max_contact_loss_frames=max_contact_loss,
            max_contact_groups=max_contact_groups,
            final_contact_groups=sorted(final_contacts),
            final_normal_forces_n=final_contacts,
            target_height_m=float(self.config["lift"]["success_height_m"]),
            peak_height_m=peak_height,
            final_height_m=final_height,
            final_cube_position_m=final_cube.tolist(),
            max_relative_translation_drift_m=float(
                evaluation["max_relative_translation_drift_m"]
            ),
            max_relative_rotation_drift_deg=float(
                evaluation["max_relative_rotation_drift_deg"]
            ),
            final_window_translation_drift_m=float(
                evaluation["final_window_translation_drift_m"]
            ),
            final_window_rotation_drift_deg=float(
                evaluation["final_window_rotation_drift_deg"]
            ),
            post_settle_translation_drift_m=float(
                evaluation["post_settle_metrics"]["max_translation_drift_m"]
            ),
            post_settle_rotation_drift_deg=float(
                evaluation["post_settle_metrics"]["max_rotation_drift_deg"]
            ),
            post_settle_max_relative_translation_speed_m_s=float(
                evaluation["post_settle_metrics"][
                    "max_relative_translation_speed_m_s"
                ]
            ),
            post_settle_max_relative_angular_speed_deg_s=float(
                evaluation["post_settle_metrics"][
                    "max_relative_angular_speed_deg_s"
                ]
            ),
            post_settle_rms_relative_translation_speed_m_s=float(
                evaluation["post_settle_metrics"][
                    "rms_relative_translation_speed_m_s"
                ]
            ),
            post_settle_rms_relative_angular_speed_deg_s=float(
                evaluation["post_settle_metrics"][
                    "rms_relative_angular_speed_deg_s"
                ]
            ),
            post_settle_translation_drift_slope_m_s=float(
                evaluation["post_settle_metrics"]["translation_drift_slope_m_s"]
            ),
            post_settle_rotation_drift_slope_deg_s=float(
                evaluation["post_settle_metrics"]["rotation_drift_slope_deg_s"]
            ),
            establishment_translation_m=float(
                evaluation["establishment_translation_m"]
            ),
            establishment_rotation_deg=float(
                evaluation["establishment_rotation_deg"]
            ),
            gripper_lift_within_baseline_window_m=float(
                evaluation["gripper_lift_within_baseline_window_m"]
            ),
            external_lift_protocol_reached=bool(
                evaluation["external_lift_protocol_reached"]
            ),
            external_object_lifted=bool(evaluation["external_object_lifted"]),
            external_baseline=evaluation["external_baseline"],
            contact_diagnostics=contact_diagnostics,
            trajectory_diagnostics=self._trajectory_diagnostics(lift_samples),
        )

    def _trajectory_diagnostics(self, samples: list[dict]) -> dict:
        hz = float(self.robot.control_hz)
        positions = np.asarray([s["grasp_center_position_m"] for s in samples])
        measured = {}
        for order, name in enumerate(("velocity_m_s", "acceleration_m_s2",
                                      "jerk_m_s3"), start=1):
            values = np.diff(positions, n=order, axis=0) * hz ** order
            measured["peak_" + name] = (
                float(np.max(np.linalg.norm(values, axis=1))) if len(values) else None
            )
        start = positions[0]
        desired = np.asarray([
            s_curve_lift_waypoint(
                start_position=start,
                total_delta=np.asarray(self.config["lift"]["grasp_center_delta_m"]),
                step=i, control_hz=hz, baseline=self.config["external_baseline"],
                limits=self.config["lift"]["s_curve_limits"],
            ) for i in range(len(samples))
        ])
        first_quaternion = samples[0]["grasp_center_quaternion_wxyz"]
        orientation_drift = [
            rotation_geodesic_angle_deg(
                first_quaternion, sample["grasp_center_quaternion_wxyz"]
            )
            for sample in samples
        ]
        orientation_error = [
            float(sample["palm_orientation_error_deg"]) for sample in samples
        ]
        return {
            "profile": "two_segment_quintic_minimum_jerk",
            "limit_scope": "Cartesian_reference_before_IK_and_actuator_dynamics",
            "configured_limits": dict(self.config["lift"]["s_curve_limits"]),
            "measured_finite_difference_hz": hz,
            "measured_peaks": measured,
            "max_position_tracking_error_m": float(np.max(
                np.linalg.norm(positions - desired, axis=1)
            )),
            "commanded_palm_quaternion_wxyz": (
                self.commanded_palm_quaternion.tolist()
            ),
            "ik_command_palm_quaternion_wxyz": (
                self.ik_command_palm_quaternion.tolist()
            ),
            "max_palm_orientation_drift_deg": max(orientation_drift),
            "max_palm_orientation_error_deg": max(orientation_error),
            "mean_palm_orientation_error_deg": float(np.mean(orientation_error)),
            "final_palm_orientation_error_deg": orientation_error[-1],
            "baseline_window_completed": len(samples) - 1 >= int(round(
                hz * self.config["external_baseline"]["lift_duration_s"]
            )),
            "trajectory_reference_duration_s": lift_trajectory_duration_s(
                total_delta=np.asarray(self.config["lift"]["grasp_center_delta_m"]),
                baseline=self.config["external_baseline"],
            ),
            "trajectory_reference_hold_frames": max(
                0,
                len(samples) - 1 - int(round(hz * lift_trajectory_duration_s(
                    total_delta=np.asarray(
                        self.config["lift"]["grasp_center_delta_m"]
                    ),
                    baseline=self.config["external_baseline"],
                ))),
            ),
        }

    @staticmethod
    def _contact_diagnostics(samples: list[dict]) -> dict:
        final = samples[-1]
        contacts = final["contacts"]
        resultant_force = np.asarray(
            final["contact_resultant_force_world_n"], dtype=float
        )
        resultant_moment = np.asarray(
            final["contact_resultant_moment_about_cube_world_nm"], dtype=float
        )
        faces = ("+X", "-X", "+Y", "-Y", "+Z", "-Z")
        all_contacts = [contact for sample in samples for contact in sample["contacts"]]
        thumb_contacts = [
            contact for contact in all_contacts if contact["finger"] == "finger1"
        ]
        other_contacts = [
            contact for contact in all_contacts if contact["finger"] != "finger1"
        ]
        geometry_samples = [sample["contact_geometry"] for sample in samples]
        opposition_defined = [
            item for item in geometry_samples
            if item["thumb_dominant_face"] is not None
            and item["other_finger_dominant_face"] is not None
        ]

        def face_distribution(contacts):
            return {
                face: sum(item["cube_face"] == face for item in contacts)
                for face in faces
            }

        horizontal_force = [
            float(item["horizontal_net_force_n"]) for item in geometry_samples
        ]
        moment_magnitude = [
            float(item["net_moment_magnitude_nm"]) for item in geometry_samples
        ]
        force_angles = [
            float(item["thumb_other_force_angle_deg"])
            for item in geometry_samples
            if item["thumb_other_force_angle_deg"] is not None
        ]
        return {
            "role": "evaluation_only_not_policy_observation",
            "final_contact_count": len(contacts),
            "final_contact_fingers": sorted({item["finger"] for item in contacts}),
            "final_resultant_force_world_n": resultant_force.tolist(),
            "final_resultant_force_magnitude_n": float(np.linalg.norm(resultant_force)),
            "final_resultant_moment_about_cube_world_nm": resultant_moment.tolist(),
            "final_resultant_moment_magnitude_nm": float(np.linalg.norm(resultant_moment)),
            "peak_resultant_force_magnitude_n": max(
                float(np.linalg.norm(sample["contact_resultant_force_world_n"]))
                for sample in samples
            ),
            "peak_resultant_moment_magnitude_nm": max(
                float(np.linalg.norm(
                    sample["contact_resultant_moment_about_cube_world_nm"]
                ))
                for sample in samples
            ),
            "thumb_face_distribution": face_distribution(thumb_contacts),
            "other_finger_face_distribution": face_distribution(other_contacts),
            "thumb_edge_contact_rate": (
                sum(item["edge_contact"] for item in thumb_contacts)
                / len(thumb_contacts) if thumb_contacts else None
            ),
            "thumb_corner_contact_rate": (
                sum(item["corner_contact"] for item in thumb_contacts)
                / len(thumb_contacts) if thumb_contacts else None
            ),
            "opposition_evaluable_frames": len(opposition_defined),
            "opposition_success_rate": (
                sum(item["opposition_faces"] for item in opposition_defined)
                / len(opposition_defined) if opposition_defined else None
            ),
            "peak_horizontal_net_force_n": max(horizontal_force),
            "mean_horizontal_net_force_n": float(np.mean(horizontal_force)),
            "peak_net_moment_magnitude_nm": max(moment_magnitude),
            "mean_net_moment_magnitude_nm": float(np.mean(moment_magnitude)),
            "mean_thumb_other_force_angle_deg": (
                float(np.mean(force_angles)) if force_angles else None
            ),
            "final_contact_geometry": final["contact_geometry"],
            "final_contacts": [
                {
                    key: value.tolist() if isinstance(value, np.ndarray) else value
                    for key, value in contact.items()
                }
                for contact in contacts
            ],
        }
