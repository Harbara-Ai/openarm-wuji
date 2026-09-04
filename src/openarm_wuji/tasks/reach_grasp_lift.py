from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from ..simulation.mujoco_backend import MujocoOpenArmWuji
from .ik import DampedLeastSquaresIK


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


class ReachGraspLiftTask:
    """Deterministic task reset plus the Reach phase of a scripted expert."""

    def __init__(self, robot: MujocoOpenArmWuji, config: dict):
        import mujoco

        self.robot = robot
        self.config = config
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
        self.target_position = np.zeros(3)
        self.initial_cube_position = np.zeros(3)
        self.arm_goal = np.zeros(7)
        self.arm_command = np.zeros(7)
        self.ik_residual_m = float("inf")

    @classmethod
    def from_json(cls, robot: MujocoOpenArmWuji, path: str | Path):
        return cls(robot, json.loads(Path(path).read_text(encoding="utf-8")))

    def _object_position(self) -> np.ndarray:
        return self.data.xpos[self.cube_body_id].copy()

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
        # Solve against a kinematic copy. The small configured Cartesian bias
        # compensates the measured gravity sag of the position-controlled arm;
        # it is explicit task calibration, not hidden policy state.
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
        compensated_target = self.target_position + np.asarray(
            self.config["reach"]["gravity_compensation_offset_m"], dtype=float
        )
        self.arm_goal, self.ik_residual_m, _ = scratch_ik.solve(
            compensated_target,
            max_iterations=int(self.config["reach"]["ik_max_iterations"]),
            tolerance=float(self.config["reach"]["ik_solve_tolerance_m"]),
        )
        self.arm_command = self.data.qpos[self.robot.arm_qpos_ids].copy()
        return {
            "seed": seed,
            "cube_position_m": self.initial_cube_position.copy(),
            "target_position_m": self.target_position.copy(),
            "grasp_center_position_m": self.ik.position,
        }

    def reach_action(self) -> np.ndarray:
        max_step = float(self.config["reach"]["max_action_step_rad"])
        self.arm_command += np.clip(
            self.arm_goal - self.arm_command, -max_step, max_step
        )
        return np.r_[self.arm_command, [0.0, 0.0, 0.0]]

    def run_reach(self, seed: int) -> ReachResult:
        self.reset(seed)
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
            self.robot.send_action(self.reach_action())
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
