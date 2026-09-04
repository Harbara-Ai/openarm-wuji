from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from ..simulation.mujoco_backend import MujocoOpenArmWuji
from .ik import DampedLeastSquaresIK
from .metrics import FingerContactMonitor
from .outcomes import evaluate_lift_outcome
from .se3 import relative_pose

if TYPE_CHECKING:
    from ..dataset.episode_recorder import CausalEpisodeRecorder


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
    cube_position_m: list[float]
    target_position_m: list[float]
    final_grasp_center_m: list[float]

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
    final_synergy: float
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
    establishment_translation_m: float
    establishment_rotation_deg: float
    gripper_lift_within_baseline_window_m: float
    external_lift_protocol_reached: bool
    external_object_lifted: bool
    external_baseline: dict
    contact_diagnostics: dict

    def to_dict(self) -> dict:
        return asdict(self)


class ReachGraspLiftTask:
    """Deterministic Reach-Grasp-Lift state machine with optional recording."""

    def __init__(self, robot: MujocoOpenArmWuji, config: dict, *,
                 recorder: CausalEpisodeRecorder | None = None):
        import mujoco

        self.robot = robot
        self.config = config
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
            hand_side=self.arm_side,
        )
        self.target_position = np.zeros(3)
        self.initial_cube_position = np.zeros(3)
        self.arm_goal = np.zeros(7)
        self.arm_command = np.zeros(7)
        self.ik_residual_m = float("inf")
        self._transition_observation: dict | None = None
        self._transition_telemetry: dict | None = None
        self._phase_start_telemetry: dict[str, dict] = {}
        self._phase_samples: dict[str, list[dict]] = {}

    @classmethod
    def from_json(cls, robot: MujocoOpenArmWuji, path: str | Path):
        return cls(robot, json.loads(Path(path).read_text(encoding="utf-8")))

    def _object_position(self) -> np.ndarray:
        return self.data.xpos[self.cube_body_id].copy()

    def _solve_arm_goal(self, target_position, *, phase_config) -> tuple[np.ndarray, float]:
        import mujoco

        scratch_data = mujoco.MjData(self.model)
        scratch_data.qpos[:] = self.data.qpos
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
        goal, residual, _ = scratch_ik.solve(
            compensated_target,
            max_iterations=int(phase_config["ik_max_iterations"]),
            tolerance=float(phase_config["ik_solve_tolerance_m"]),
        )
        return goal, residual

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
        self.arm_goal, self.ik_residual_m = self._solve_arm_goal(
            self.target_position, phase_config=self.config["reach"]
        )
        self.arm_command = self.data.qpos[self.robot.arm_qpos_ids].copy()
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
        contacts = self.contact_monitor.contacts(0.0)
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
            "object_relative_position_m": relative_position,
            "object_relative_quaternion_wxyz": relative_quaternion,
            "finger_normal_forces_n": finger_forces,
            "contacts": contacts,
            "contact_resultant_force_world_n": resultant_force,
            "contact_resultant_moment_about_cube_world_nm": resultant_moment,
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
        minimum_error = initial_error
        hold = 0
        steps = 0
        if self.ik_residual_m > float(reach["ik_solve_tolerance_m"]):
            position = self.ik.position
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
                cube_position_m=self.initial_cube_position.tolist(),
                target_position_m=self.target_position.tolist(),
                final_grasp_center_m=position.tolist(),
            )
        for steps in range(1, max_steps + 1):
            self._send_action(self.reach_action(), phase="reach")
            error = float(np.linalg.norm(self.target_position - self.ik.position))
            minimum_error = min(minimum_error, error)
            hold = hold + 1 if error <= tolerance else 0
            if hold >= required_hold:
                break
        final_position = self.ik.position
        final_error = float(np.linalg.norm(self.target_position - final_position))
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
            cube_position_m=self.initial_cube_position.tolist(),
            target_position_m=self.target_position.tolist(),
            final_grasp_center_m=final_position.tolist(),
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
        grasp_goal, grasp_residual = self._solve_arm_goal(
            grasp_target, phase_config=grasp
        )
        if grasp_residual > float(grasp["ik_solve_tolerance_m"]):
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
        contact_hold = 0
        max_contact_groups = 0
        final_contacts: dict[str, float] = {}
        peak_forces: dict[str, float] = {}
        close_steps = 0
        success = False
        for close_steps in range(1, int(grasp["max_close_steps"]) + 1):
            synergy = min(1.0, synergy + float(grasp["close_synergy_step"]))
            self._send_action(
                np.r_[self.arm_command, [synergy, 0.0, 0.0]],
                phase="grasp_close",
            )
            final_contacts = self.contact_monitor.sample(
                float(grasp["min_normal_force_n"])
            )
            max_contact_groups = max(max_contact_groups, len(final_contacts))
            for finger, force in final_contacts.items():
                peak_forces[finger] = max(peak_forces.get(finger, 0.0), force)
            enough_contacts = len(final_contacts) >= int(grasp["min_finger_groups"])
            enough_closure = synergy >= float(grasp["min_synergy_for_success"])
            contact_hold = contact_hold + 1 if enough_contacts and enough_closure else 0
            if contact_hold >= int(grasp["contact_hold_frames"]):
                success = True
                break

        final_cube = self._object_position()
        return GraspResult(
            success=success,
            failure_reason=None if success else "grasp_empty",
            seed=seed,
            reach=reach_result,
            approach_steps=approach_steps,
            close_steps=close_steps,
            final_synergy=synergy,
            contact_hold_frames=contact_hold,
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
            final_synergy=0.0,
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
        lift_target = self.ik.position + np.asarray(
            lift["grasp_center_delta_m"], dtype=float
        )
        lift_goal, lift_residual = self._solve_arm_goal(
            lift_target, phase_config=lift
        )
        if lift_residual > float(lift["ik_solve_tolerance_m"]):
            return self._lift_result(
                seed=seed,
                grasp=grasp_result,
                approach_push=False,
            )

        success_height = float(lift["success_height_m"])
        required_height_hold = int(lift["success_hold_frames"])
        required_contact_groups = int(lift["min_finger_groups"])
        hold_synergy = max(
            grasp_result.final_synergy, float(lift["hold_synergy"])
        )
        height_hold = 0
        contact_loss = 0
        max_contact_loss = 0
        max_contact_groups = 0
        peak_height = 0.0
        final_contacts: dict[str, float] = {}
        task_success = False
        steps = 0
        minimum_external_lift = float(
            self.config["external_baseline"]["min_object_lift_m"]
        )
        crossed_minimum_lift = False

        for steps in range(1, int(lift["max_steps"]) + 1):
            max_step = float(lift["max_action_step_rad"])
            self.arm_command += np.clip(
                lift_goal - self.arm_command, -max_step, max_step
            )
            self._send_action(
                np.r_[self.arm_command, [hold_synergy, 0.0, 0.0]],
                phase="lift",
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
                break
            if crossed_minimum_lift and height < minimum_external_lift:
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
        lift_start = self._phase_start_telemetry.get("lift", current)
        lift_samples = [lift_start, *self._phase_samples.get("lift", [])]
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
        )
        stable_success = bool(task_success and evaluation["grasp_stable"])
        contact_diagnostics = self._contact_diagnostics(lift_samples)
        return LiftResult(
            success=stable_success,
            failure_reason=None if stable_success else evaluation["outcome"],
            task_success=bool(evaluation["task_success"]),
            grasp_stable=bool(evaluation["grasp_stable"]),
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
        )

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
            "final_contacts": [
                {
                    key: value.tolist() if isinstance(value, np.ndarray) else value
                    for key, value in contact.items()
                }
                for contact in contacts
            ],
        }
