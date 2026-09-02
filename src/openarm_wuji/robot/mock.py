from __future__ import annotations

import time
from typing import Any, Mapping, Sequence

import numpy as np

from .base import OpenArmWujiRobot
from ..safety.actions import ActionSafety


class MockOpenArmWuji(OpenArmWujiRobot):
    """Deterministic backend used before any hardware protocol is known."""

    def __init__(self, arm_dof: int = 7, hand_dof: int = 20, control_hz: float = 30.0):
        self.dof = arm_dof + hand_dof
        self.dt = 1.0 / control_hz
        self.state = np.zeros(self.dof, dtype=np.float64)
        self.connected = False
        self.safety = ActionSafety(-np.ones(self.dof), np.ones(self.dof), np.full(self.dof, 2.0))

    def connect(self) -> None:
        self.connected = True

    def get_observation(self) -> Mapping[str, Any]:
        if not self.connected:
            raise RuntimeError("robot is disconnected")
        return {"timestamp": time.monotonic(), "joint_position": self.state.copy()}

    def send_action(self, action: Sequence[float]) -> None:
        if not self.connected:
            raise RuntimeError("robot is disconnected")
        self.state = self.safety.validate(action, previous=self.state, dt=self.dt)

    def disconnect(self) -> None:
        self.connected = False

