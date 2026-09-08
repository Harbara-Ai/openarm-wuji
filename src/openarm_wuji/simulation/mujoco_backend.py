from __future__ import annotations

import time
from pathlib import Path
from typing import Sequence

import numpy as np

from ..robot.base import OpenArmWujiRobot
from ..teleop.synergies import HandSynergyMapper


class MujocoOpenArmWuji(OpenArmWujiRobot):
    """MuJoCo adapter with a 7-D arm + 3-D hand-synergy action contract."""

    ARM_DOF = 7
    SYNERGY_DOF = 3
    RECORDING_VERSION = 1

    def __init__(self, model_path: str | Path, synergy_config: str | Path, *,
                 arm_side: str = "left", control_hz: float = 30.0,
                 image_height: int = 240, image_width: int = 320,
                 front_camera: str | None = None, render: bool = True):
        if control_hz <= 0:
            raise ValueError("control_hz must be positive")
        if image_height <= 0 or image_width <= 0:
            raise ValueError("image dimensions must be positive")
        self.model_path = str(model_path)
        self.mapper = HandSynergyMapper.from_json(synergy_config)
        self.arm_side = arm_side
        self.control_hz = float(control_hz)
        self.control_dt = 1.0 / self.control_hz
        self.image_height = image_height
        self.image_width = image_width
        self.front_camera = front_camera
        self.render_enabled = bool(render)
        self._model = self._data = self._renderer = self._front_camera = None
        self._arm_actuator_ids = self._arm_qpos_ids = self._arm_qvel_ids = None
        self._hand_actuator_ids = self._hand_qpos_ids = self._hand_qvel_ids = None
        self._last_synergy = np.zeros(self.SYNERGY_DOF)
        self._hand_target = self.mapper.open_pose.copy()
        self._last_sent_action = np.zeros(self.ARM_DOF + self.SYNERGY_DOF)
        self._frame_index = 0
        self._control_start_time = 0.0
        self.records: list[dict[str, np.ndarray | float | int]] = []

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
        self._ids(mujoco, mujoco.mjtObj.mjOBJ_CAMERA, [f"camera_wrist_{self.arm_side}"])
        key = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_KEY, "home")
        if key >= 0:
            mujoco.mj_resetDataKeyframe(self._model, self._data, key)
        if self.render_enabled:
            self._renderer = mujoco.Renderer(
                self._model, height=self.image_height, width=self.image_width
            )
        if not self.render_enabled:
            self._front_camera = None
        elif self.front_camera is None:
            self._front_camera = mujoco.MjvCamera()
            self._front_camera.lookat[:] = self._model.stat.center
            self._front_camera.distance = self._model.stat.extent * 1.2
            self._front_camera.azimuth = 145
            self._front_camera.elevation = -20
        else:
            self._ids(mujoco, mujoco.mjtObj.mjOBJ_CAMERA, [self.front_camera])
            self._front_camera = self.front_camera
        self._last_synergy.fill(0.0)
        self._last_sent_action.fill(0.0)
        self._hand_target = self.mapper.open_pose.copy()
        self._frame_index = 0
        self._control_start_time = float(self._data.time)
        self.records.clear()

    @property
    def model(self):
        self._require_connected()
        return self._model

    @property
    def data(self):
        self._require_connected()
        return self._data

    @property
    def arm_qpos_ids(self) -> np.ndarray:
        self._require_connected()
        return self._arm_qpos_ids.copy()

    @property
    def hand_qpos_ids(self) -> np.ndarray:
        self._require_connected()
        return self._hand_qpos_ids.copy()

    @property
    def arm_qvel_ids(self) -> np.ndarray:
        self._require_connected()
        return self._arm_qvel_ids.copy()

    @property
    def hand_qvel_ids(self) -> np.ndarray:
        self._require_connected()
        return self._hand_qvel_ids.copy()

    @property
    def arm_actuator_ids(self) -> np.ndarray:
        self._require_connected()
        return self._arm_actuator_ids.copy()

    @property
    def hand_actuator_ids(self) -> np.ndarray:
        self._require_connected()
        return self._hand_actuator_ids.copy()

    @property
    def latest_record(self) -> dict[str, np.ndarray | float | int]:
        self._require_connected()
        if not self.records:
            raise RuntimeError("no post-action observation is available")
        return {
            key: value.copy() if isinstance(value, np.ndarray) else value
            for key, value in self.records[-1].items()
        }

    def synchronize_after_reset(self, hand_target: Sequence[float] | None = None) -> None:
        """Synchronize controller bookkeeping after a task resets MuJoCo state."""
        self._require_connected()
        if hand_target is None:
            hand_target = self.mapper.open_pose
        hand_target = np.asarray(hand_target, dtype=float)
        if hand_target.shape != (20,) or not np.isfinite(hand_target).all():
            raise ValueError("hand_target must be finite and 20-D")
        self._hand_target = hand_target.copy()
        self._last_synergy.fill(0.0)
        self._last_sent_action = np.concatenate([
            self._data.qpos[self._arm_qpos_ids].copy(),
            np.zeros(self.SYNERGY_DOF),
        ])
        self._frame_index = 0
        self._control_start_time = float(self._data.time)
        self.records.clear()

    def _require_connected(self) -> None:
        if self._data is None or self._model is None:
            raise RuntimeError("simulation is disconnected")

    def _render(self, camera) -> np.ndarray:
        if self._renderer is None:
            raise RuntimeError("rendering is disabled for this simulation")
        self._renderer.update_scene(self._data, camera=camera)
        return self._renderer.render().copy()

    def get_observation(self):
        self._require_connected()
        return {
            "frame_index": self._frame_index,
            # On Windows, monotonic() may resolve to the 15.625 ms GetTickCount64
            # clock. perf_counter() uses QueryPerformanceCounter and preserves
            # ordering for faster-than-real-time simulation frames.
            "timestamp": time.perf_counter(),
            "sim_time": float(self._data.time),
            "front_rgb": self._render(self._front_camera),
            "wrist_rgb": self._render(f"camera_wrist_{self.arm_side}"),
            "arm_joint_position": self._data.qpos[self._arm_qpos_ids].copy(),
            "arm_joint_velocity": self._data.qvel[self._arm_qvel_ids].copy(),
            "hand_joint_position": self._data.qpos[self._hand_qpos_ids].copy(),
            "hand_joint_velocity": self._data.qvel[self._hand_qvel_ids].copy(),
            "hand_joint_target": self._hand_target.copy(),
            "hand_synergy_action": self._last_synergy.copy(),
            "sent_action": self._last_sent_action.copy(),
        }

    def send_action(self, action: Sequence[float]) -> np.ndarray:
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
        # A 2 ms physics step does not divide a 30 Hz control period exactly.
        # Target the absolute control timeline so 16/17 physics steps alternate
        # instead of accumulating the error from always taking 17 steps.
        target_time = self._control_start_time + (self._frame_index + 1) * self.control_dt
        steps = max(1, round((target_time - self._data.time) / self._model.opt.timestep))
        mujoco.mj_step(self._model, self._data, nstep=steps)
        self._last_synergy = synergy.copy()
        self._last_sent_action = np.concatenate([arm, synergy])
        self._frame_index += 1
        observation = self.get_observation()
        self.records.append({key: value.copy() if isinstance(value, np.ndarray) else value
                             for key, value in observation.items()})
        return self._last_sent_action.copy()

    def save_recording(self, path: str | Path) -> None:
        if not self.records:
            raise RuntimeError("no control samples have been recorded")
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        arrays = {key: np.asarray([row[key] for row in self.records]) for key in self.records[0]}
        arrays["recording_version"] = np.asarray(self.RECORDING_VERSION)
        arrays["control_hz"] = np.asarray(self.control_hz)
        np.savez_compressed(destination, **arrays)

    def replay_recording(self, path: str | Path) -> dict[str, float | int]:
        """Replay bounded policy actions and compare every proprioceptive frame."""
        self._require_connected()
        with np.load(path) as episode:
            version = int(episode["recording_version"])
            recorded_hz = float(episode["control_hz"])
            if version != self.RECORDING_VERSION:
                raise ValueError(f"unsupported recording version: {version}")
            if not np.isclose(recorded_hz, self.control_hz):
                raise ValueError(f"control frequency mismatch: recording={recorded_hz}, robot={self.control_hz}")
            actions = episode["sent_action"].copy()
            expected_arm = episode["arm_joint_position"].copy()
            expected_hand = episode["hand_joint_position"].copy()
        arm_error = hand_error = 0.0
        for index, action in enumerate(actions):
            self.send_action(action)
            observation = self.records[-1]
            arm_error = max(arm_error, float(np.max(np.abs(
                observation["arm_joint_position"] - expected_arm[index]))))
            hand_error = max(hand_error, float(np.max(np.abs(
                observation["hand_joint_position"] - expected_hand[index]))))
        return {"samples": len(actions), "max_arm_position_error": arm_error,
                "max_hand_position_error": hand_error}

    def disconnect(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
        self._model = self._data = self._renderer = self._front_camera = None
