from __future__ import annotations

from functools import cached_property
from typing import Any

import numpy as np
from lerobot.lerobot_types import RobotAction, RobotObservation
from lerobot.robots import Robot

from openarm_wuji.robot.mock import MockOpenArmWuji
from openarm_wuji.robot.real import RealOpenArmWuji
from openarm_wuji.simulation.mujoco_backend import MujocoOpenArmWuji

from .config_openarm_wuji_follower import OpenArmWujiFollowerConfig


class OpenArmWujiFollower(Robot):
    """LeRobot adapter for the shared OpenArm + Wuji backend contract."""

    config_class = OpenArmWujiFollowerConfig
    name = "openarm_wuji_follower"
    arm_position_names = tuple(f"joint_{index}.pos" for index in range(1, 8))
    hand_position_names = tuple(f"hand_joint_{index:02d}.pos" for index in range(1, 21))
    synergy_names = ("hand.open_close", "hand.pinch", "hand.spread")

    def __init__(self, config: OpenArmWujiFollowerConfig):
        super().__init__(config)
        self.config = config
        self._is_connected = False
        common = {
            "synergy_config": config.synergy_config,
            "control_hz": config.control_hz,
            "image_height": config.image_height,
            "image_width": config.image_width,
        }
        if config.backend == "mock":
            self.backend = MockOpenArmWuji(**common)
        elif config.backend == "mujoco":
            self.backend = MujocoOpenArmWuji(
                config.model_path, arm_side=config.arm_side, **common
            )
        else:
            self.backend = RealOpenArmWuji()

    @cached_property
    def observation_features(self) -> dict[str, type | tuple[int, int, int]]:
        state = {name: float for name in self.arm_position_names + self.hand_position_names}
        cameras = {
            "front_rgb": (self.config.image_height, self.config.image_width, 3),
            "wrist_rgb": (self.config.image_height, self.config.image_width, 3),
        }
        return {**state, **cameras}

    @cached_property
    def action_features(self) -> dict[str, type]:
        return {name: float for name in self.arm_position_names + self.synergy_names}

    @property
    def is_connected(self) -> bool:
        return self._is_connected

    def connect(self, calibrate: bool = True) -> None:
        if self.is_connected:
            raise RuntimeError(f"{self} is already connected")
        self.backend.connect()
        self._is_connected = True

    @property
    def is_calibrated(self) -> bool:
        return True

    def calibrate(self) -> None:
        return None

    def configure(self) -> None:
        return None

    def get_observation(self) -> RobotObservation:
        if not self.is_connected:
            raise RuntimeError(f"{self} is not connected")
        raw = self.backend.get_observation()
        arm = np.asarray(raw["arm_joint_position"], dtype=float)
        hand = np.asarray(raw["hand_joint_position"], dtype=float)
        if arm.shape != (7,) or hand.shape != (20,):
            raise RuntimeError(f"invalid backend state shapes: arm={arm.shape}, hand={hand.shape}")
        observation: dict[str, Any] = {
            name: float(value)
            for name, value in zip(self.arm_position_names + self.hand_position_names,
                                   np.concatenate([arm, hand]), strict=True)
        }
        observation["front_rgb"] = raw["front_rgb"]
        observation["wrist_rgb"] = raw["wrist_rgb"]
        if set(observation) != set(self.observation_features):
            raise RuntimeError("observation keys do not match observation_features")
        return observation

    def send_action(self, action: RobotAction) -> RobotAction:
        if not self.is_connected:
            raise RuntimeError(f"{self} is not connected")
        expected = set(self.action_features)
        if set(action) != expected:
            missing = sorted(expected - set(action))
            extra = sorted(set(action) - expected)
            raise ValueError(f"action keys mismatch: missing={missing}, extra={extra}")
        vector = np.asarray(
            [action[name] for name in self.arm_position_names + self.synergy_names], dtype=float
        )
        if not np.isfinite(vector).all():
            raise ValueError("action contains NaN or infinity")
        sent = np.asarray(self.backend.send_action(vector), dtype=float)
        if sent.shape != (10,) or not np.isfinite(sent).all():
            raise RuntimeError(f"backend returned invalid sent action: shape={sent.shape}")
        return {
            name: float(value)
            for name, value in zip(self.arm_position_names + self.synergy_names, sent, strict=True)
        }

    def disconnect(self) -> None:
        if not self.is_connected:
            return
        self.backend.disconnect()
        self._is_connected = False
