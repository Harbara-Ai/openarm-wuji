from __future__ import annotations

import time
from pathlib import Path
from typing import Sequence

import numpy as np

from ..robot.base import OpenArmWujiRobot
from ..teleop.synergies import HandSynergyMapper


class MujocoOpenArmWuji(OpenArmWujiRobot):
    """Position-control adapter exposing a 7-D arm + 3-D hand-synergy action."""

    ARM_DOF = 7
    SYNERGY_DOF = 3

    def __init__(self, model_path: str | Path, synergy_config: str | Path, *,
                 arm_side: str = "left", control_hz: float = 30.0):
        if control_hz <= 0:
            raise ValueError("control_hz must be positive")
        self.model_path = str(model_path)
        self.mapper = HandSynergyMapper.from_json(synergy_config)
        self.arm_side = arm_side
        self.control_dt = 1.0 / control_hz
        self._model = self._data = None
        self._arm_actuator_ids = self._arm_qpos_ids = self._arm_qvel_ids = None
        self._hand_actuator_ids = self._hand_qpos_ids = self._hand_qvel_ids = None
        self._last_synergy = np.zeros(self.SYNERGY_DOF)
        self._hand_target = self.mapper.open_pose.copy()
        self._last_sent_action = np.zeros(self.ARM_DOF + self.SYNERGY_DOF)
        self.records: list[dict[str, np.ndarray | float]] = []

    def _ids(self, mujoco, kind, names: Sequence[str]) -> np.ndarray:
        ids = np.asarray([mujoco.mj_name2id(self._model, kind, name) for name in names], dtype=int)
        missing = [name for name, idx in zip(names, ids) if idx < 0]
        if missing:
            raise KeyError(f"model objects missing: {missing}")
        return ids

    def connect(self) -> None:
        import mujoco
        if Path(self.model_path).suffix.lower() == ".mjb":
            self._model = mujoco.MjModel.from_binary_path(self.model_path)
        else:
            self._model = mujoco.MjModel.from_xml_path(self.model_path)
        self._data = mujoco.MjData(self._model)
        arm_joints = [f"openarm_{self.arm_side}_joint{i}" for i in range(1, 8)]
        arm_actuators = [f"{self.arm_side}_joint{i}_ctrl" for i in range(1, 8)]
        hand_joints = [f"wuji_{name}" for name in self.mapper.joint_names]
        hand_actuators = [f"{name}_actuator" for name in hand_joints]
        arm_joint_ids = self._ids(mujoco, mujoco.mjtObj.mjOBJ_JOINT, arm_joints)
        hand_joint_ids = self._ids(mujoco, mujoco.mjtObj.mjOBJ_JOINT, hand_joints)
        self._arm_actuator_ids = self._ids(mujoco, mujoco.mjtObj.mjOBJ_ACTUATOR, arm_actuators)
        self._hand_actuator_ids = self._ids(mujoco, mujoco.mjtObj.mjOBJ_ACTUATOR, hand_actuators)
        self._arm_qpos_ids = self._model.jnt_qposadr[arm_joint_ids]
        self._arm_qvel_ids = self._model.jnt_dofadr[arm_joint_ids]
        self._hand_qpos_ids = self._model.jnt_qposadr[hand_joint_ids]
        self._hand_qvel_ids = self._model.jnt_dofadr[hand_joint_ids]
        key = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_KEY, "home")
        if key >= 0:
            mujoco.mj_resetDataKeyframe(self._model, self._data, key)
        self._hand_target = self.mapper.open_pose.copy()
        self.records.clear()

    def _require_connected(self) -> None:
        if self._data is None or self._model is None:
            raise RuntimeError("simulation is disconnected")

    def get_observation(self):
        self._require_connected()
        return {
            "timestamp": time.monotonic(), "sim_time": float(self._data.time),
            "arm_joint_position": self._data.qpos[self._arm_qpos_ids].copy(),
            "arm_joint_velocity": self._data.qvel[self._arm_qvel_ids].copy(),
            "hand_joint_position": self._data.qpos[self._hand_qpos_ids].copy(),
            "hand_joint_velocity": self._data.qvel[self._hand_qvel_ids].copy(),
            "hand_synergy_action": self._last_synergy.copy(),
            "sent_action": self._last_sent_action.copy(),
        }

    def send_action(self, action: Sequence[float]) -> None:
        import mujoco
        self._require_connected()
        action = np.asarray(action, dtype=float)
        expected = (self.ARM_DOF + self.SYNERGY_DOF,)
        if action.shape != expected:
            raise ValueError(f"expected action shape {expected}, got {action.shape}")
        if not np.isfinite(action).all():
            raise ValueError("action contains NaN or infinity")
        arm = np.clip(action[:7], self._model.actuator_ctrlrange[self._arm_actuator_ids, 0],
                      self._model.actuator_ctrlrange[self._arm_actuator_ids, 1])
        synergy = np.clip(action[7:], [0.0, 0.0, -1.0], [1.0, 1.0, 1.0])
        self._hand_target = self.mapper.next(synergy, previous=self._hand_target, dt=self.control_dt)
        self._data.ctrl[self._arm_actuator_ids] = arm
        self._data.ctrl[self._hand_actuator_ids] = self._hand_target
        steps = max(1, round(self.control_dt / self._model.opt.timestep))
        mujoco.mj_step(self._model, self._data, nstep=steps)
        self._last_synergy = synergy.copy()
        self._last_sent_action = np.concatenate([arm, synergy])
        observation = self.get_observation()
        self.records.append({key: value.copy() if isinstance(value, np.ndarray) else value
                             for key, value in observation.items()})

    def save_recording(self, path: str | Path) -> None:
        if not self.records:
            raise RuntimeError("no control samples have been recorded")
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        keys = self.records[0].keys()
        np.savez_compressed(destination, **{key: np.asarray([row[key] for row in self.records])
                                           for key in keys})

    def disconnect(self) -> None:
        self._model = self._data = None
