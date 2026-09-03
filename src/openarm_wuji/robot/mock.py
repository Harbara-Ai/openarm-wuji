from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from ..safety.actions import ActionSafety
from ..teleop.synergies import HandSynergyMapper
from .base import OpenArmWujiRobot


class MockOpenArmWuji(OpenArmWujiRobot):
    """Deterministic 10-D policy backend with the same schema as MuJoCo."""

    def __init__(self, synergy_config: str | Path, control_hz: float = 30.0,
                 image_height: int = 240, image_width: int = 320):
        if control_hz <= 0:
            raise ValueError("control_hz must be positive")
        self.mapper = HandSynergyMapper.from_json(synergy_config)
        self.dt = 1.0 / control_hz
        self.image_shape = (image_height, image_width, 3)
        self.arm_position = np.zeros(7, dtype=np.float64)
        self.arm_velocity = np.zeros(7, dtype=np.float64)
        self.hand_position = self.mapper.open_pose.copy()
        self.hand_velocity = np.zeros(20, dtype=np.float64)
        self.hand_target = self.mapper.open_pose.copy()
        self.last_synergy = np.zeros(3, dtype=np.float64)
        self.last_action = np.zeros(10, dtype=np.float64)
        self.arm_safety = ActionSafety(
            np.full(7, -np.pi), np.full(7, np.pi), np.full(7, 2.0)
        )
        self.connected = False
        self.frame_index = 0
        self.sim_time = 0.0

    def connect(self) -> None:
        self.connected = True

    def get_observation(self) -> Mapping[str, Any]:
        if not self.connected:
            raise RuntimeError("robot is disconnected")
        zeros = np.zeros(self.image_shape, dtype=np.uint8)
        return {
            "frame_index": self.frame_index,
            "timestamp": time.perf_counter(),
            "sim_time": self.sim_time,
            "front_rgb": zeros.copy(),
            "wrist_rgb": zeros.copy(),
            "arm_joint_position": self.arm_position.copy(),
            "arm_joint_velocity": self.arm_velocity.copy(),
            "hand_joint_position": self.hand_position.copy(),
            "hand_joint_velocity": self.hand_velocity.copy(),
            "hand_joint_target": self.hand_target.copy(),
            "hand_synergy_action": self.last_synergy.copy(),
            "sent_action": self.last_action.copy(),
        }

    def send_action(self, action: Sequence[float]) -> np.ndarray:
        if not self.connected:
            raise RuntimeError("robot is disconnected")
        action = np.asarray(action, dtype=float)
        if action.shape != (10,):
            raise ValueError(f"expected action shape (10,), got {action.shape}")
        if not np.isfinite(action).all():
            raise ValueError("action contains NaN or infinity")
        previous_arm = self.arm_position.copy()
        previous_hand = self.hand_position.copy()
        self.arm_position = self.arm_safety.validate(
            action[:7], previous=previous_arm, dt=self.dt
        )
        self.arm_velocity = (self.arm_position - previous_arm) / self.dt
        self.last_synergy = np.clip(action[7:], [0.0, 0.0, -1.0], [1.0, 1.0, 1.0])
        self.hand_target = self.mapper.next(
            self.last_synergy, previous=previous_hand, dt=self.dt
        )
        self.hand_position = self.hand_target.copy()
        self.hand_velocity = (self.hand_position - previous_hand) / self.dt
        self.last_action = np.concatenate([self.arm_position, self.last_synergy])
        self.frame_index += 1
        self.sim_time = self.frame_index * self.dt
        return self.last_action.copy()

    def disconnect(self) -> None:
        self.connected = False
