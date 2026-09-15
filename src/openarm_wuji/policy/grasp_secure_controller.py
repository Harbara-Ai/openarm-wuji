"""Independent ACT stage for Wuji Grasp+Preload control."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .act_controller import ACTController


GRASP_SECURE_TASK = (
    "Close Wuji from a stable grasp-start pose, establish a multi-finger "
    "grasp and controller preload, then hold the load-bearing-ready state."
)


class GraspSecurePolicy:
    """Strict H_exec=1 Grasp+Preload ACT with the shared 27-D interface."""

    def __init__(self, checkpoint: str | Path, *, device: str = "cpu") -> None:
        self.controller = ACTController(
            checkpoint, device=device, task=GRASP_SECURE_TASK
        )

    def reset(self) -> None:
        self.controller.reset()

    def select_action(self, observation: dict) -> np.ndarray:
        # Discard every unused chunk tail and re-observe on each control frame.
        self.controller.reset()
        state = np.concatenate([
            observation["arm_joint_position"],
            observation["hand_joint_position"],
        ]).astype(np.float32)
        return self.controller.predict(
            state=state,
            front_rgb=observation["front_rgb"],
            wrist_rgb=observation["wrist_rgb"],
        )
