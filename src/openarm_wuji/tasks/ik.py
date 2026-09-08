from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from .se3 import normalize_quaternion, quaternion_error_rotvec


class DampedLeastSquaresIK:
    """Damped least-squares position or full-pose IK for a MuJoCo chain."""

    def __init__(self, model, data, *, site_name: str, joint_names: Sequence[str],
                 damping: float = 0.03, max_joint_step: float = 0.06,
                 joint_limit_margin: float = 1e-4):
        import mujoco

        if damping <= 0 or max_joint_step <= 0 or joint_limit_margin < 0:
            raise ValueError("invalid IK tuning")
        self.model = model
        self.data = data
        self.damping = float(damping)
        self.max_joint_step = float(max_joint_step)
        self.joint_limit_margin = float(joint_limit_margin)
        self.site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
        if self.site_id < 0:
            raise KeyError(f"site missing: {site_name}")
        self.joint_ids = np.asarray([
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            for name in joint_names
        ], dtype=int)
        missing = [name for name, index in zip(joint_names, self.joint_ids) if index < 0]
        if missing:
            raise KeyError(f"joints missing: {missing}")
        self.qpos_ids = model.jnt_qposadr[self.joint_ids]
        self.qvel_ids = model.jnt_dofadr[self.joint_ids]

    @property
    def position(self) -> np.ndarray:
        import mujoco

        mujoco.mj_forward(self.model, self.data)
        return self.data.site_xpos[self.site_id].copy()

    @property
    def quaternion(self) -> np.ndarray:
        import mujoco

        mujoco.mj_forward(self.model, self.data)
        quaternion = np.zeros(4)
        mujoco.mju_mat2Quat(quaternion, self.data.site_xmat[self.site_id])
        return normalize_quaternion(quaternion)

    def solve_step(self, target_position: Sequence[float], *,
                   target_quaternion: Sequence[float] | None = None,
                   orientation_weight: float = 1.0) -> np.ndarray:
        import mujoco

        target = np.asarray(target_position, dtype=float)
        if target.shape != (3,) or not np.isfinite(target).all():
            raise ValueError("target_position must be finite and 3-D")
        if orientation_weight <= 0 or not np.isfinite(orientation_weight):
            raise ValueError("orientation_weight must be positive and finite")
        mujoco.mj_forward(self.model, self.data)
        error = target - self.data.site_xpos[self.site_id]
        jacobian_position = np.zeros((3, self.model.nv))
        jacobian_rotation = np.zeros((3, self.model.nv))
        mujoco.mj_jacSite(
            self.model, self.data, jacobian_position, jacobian_rotation, self.site_id
        )
        jacobian = jacobian_position[:, self.qvel_ids]
        task_error = error
        if target_quaternion is not None:
            target_quaternion = normalize_quaternion(target_quaternion)
            rotation_error = quaternion_error_rotvec(
                target_quaternion, self.quaternion
            )
            jacobian = np.vstack([
                jacobian,
                orientation_weight * jacobian_rotation[:, self.qvel_ids],
            ])
            task_error = np.r_[error, orientation_weight * rotation_error]
        regularizer = self.damping ** 2 * np.eye(jacobian.shape[0])
        delta = jacobian.T @ np.linalg.solve(
            jacobian @ jacobian.T + regularizer, task_error
        )
        delta = np.clip(delta, -self.max_joint_step, self.max_joint_step)
        target_qpos = self.data.qpos[self.qpos_ids] + delta
        lower = self.model.jnt_range[self.joint_ids, 0] + self.joint_limit_margin
        upper = self.model.jnt_range[self.joint_ids, 1] - self.joint_limit_margin
        return np.clip(target_qpos, lower, upper)

    def solve(self, target_position: Sequence[float], *, max_iterations: int,
              tolerance: float) -> tuple[np.ndarray, float, int]:
        """Solve on this instance's data, returning joints, residual, and iterations."""
        if max_iterations <= 0 or tolerance <= 0:
            raise ValueError("IK iteration limit and tolerance must be positive")
        target = np.asarray(target_position, dtype=float)
        error = float("inf")
        for iteration in range(1, max_iterations + 1):
            self.data.qpos[self.qpos_ids] = self.solve_step(target)
            error = float(np.linalg.norm(target - self.position))
            if error <= tolerance:
                break
        return self.data.qpos[self.qpos_ids].copy(), error, iteration

    def solve_pose(self, target_position: Sequence[float],
                   target_quaternion: Sequence[float], *, max_iterations: int,
                   position_tolerance: float, orientation_tolerance_deg: float,
                   orientation_weight: float = 1.0,
                   ) -> tuple[np.ndarray, float, float, int]:
        """Solve a 6D site pose using a quaternion rotation-vector error."""
        if (max_iterations <= 0 or position_tolerance <= 0
                or orientation_tolerance_deg <= 0):
            raise ValueError("pose IK limits and tolerances must be positive")
        target_position = np.asarray(target_position, dtype=float)
        target_quaternion = normalize_quaternion(target_quaternion)
        position_error = float("inf")
        orientation_error_deg = float("inf")
        for iteration in range(1, max_iterations + 1):
            self.data.qpos[self.qpos_ids] = self.solve_step(
                target_position,
                target_quaternion=target_quaternion,
                orientation_weight=orientation_weight,
            )
            position_error = float(np.linalg.norm(target_position - self.position))
            orientation_error_deg = float(np.degrees(np.linalg.norm(
                quaternion_error_rotvec(target_quaternion, self.quaternion)
            )))
            if (position_error <= position_tolerance
                    and orientation_error_deg <= orientation_tolerance_deg):
                break
        return (
            self.data.qpos[self.qpos_ids].copy(),
            position_error,
            orientation_error_deg,
            iteration,
        )
