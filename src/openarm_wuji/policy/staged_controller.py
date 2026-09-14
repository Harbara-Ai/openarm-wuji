"""Small explicit stage-policy API for ACT-based task composition."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .act_controller import ACTController


REACH_TASK = "Move OpenArm and the open Wuji hand to the cube pregrasp pose and hold it stable."
APPROACH_TASK = "Move the open Wuji hand from stable pregrasp to the cube grasp-start pose and hold."
RECOVERY_TASK = (
    "Correct a terminal Approach error, reach the cube grasp-start pose, "
    "and stop while keeping the Wuji hand open."
)


@dataclass(frozen=True)
class StageStatus:
    phase: str
    frame: int
    success: bool
    timeout: bool
    failure_reason: str | None
    position_error_m: float
    orientation_error_deg: float
    consecutive_gate_frames: int
    cube_displacement_m: float


class CartesianACTStagePolicy:
    """ACT policy plus an explicit Cartesian success/timeout gate.

    The policy never owns phase transitions.  An outer FSM reads ``status``
    and explicitly chooses the next stage.
    """

    def __init__(self, checkpoint: str | Path, *, phase: str, task: str,
                 timeout_frames: int, position_tolerance_m: float = 0.012,
                 orientation_tolerance_deg: float = 2.0,
                 hold_frames: int = 5,
                 max_cube_displacement_m: float | None = None,
                 device: str = "cpu") -> None:
        self.phase = str(phase)
        self.timeout_frames = int(timeout_frames)
        self.position_tolerance_m = float(position_tolerance_m)
        self.orientation_tolerance_deg = float(orientation_tolerance_deg)
        self.hold_frames = int(hold_frames)
        self.max_cube_displacement_m = max_cube_displacement_m
        if self.timeout_frames < self.hold_frames or self.hold_frames < 1:
            raise ValueError("invalid timeout/hold configuration")
        self.controller = ACTController(checkpoint, device=device, task=task)
        self.reset()

    def reset(self) -> None:
        self.controller.reset()
        self._frame = 0
        self._consecutive = 0
        self._success = False
        self._failure_reason: str | None = None
        self._last_position_error = float("inf")
        self._last_orientation_error = float("inf")
        self._last_cube_displacement = 0.0

    def select_action(self, observation: dict) -> np.ndarray:
        """Select one action from the current observation (H_exec=1)."""
        # Discard the unused remainder of ACT's predicted chunk.  This makes
        # each stage a strict image+actual-qpos closed-loop controller.
        self.controller.reset()
        state = np.concatenate([
            observation["arm_joint_position"], observation["hand_joint_position"]
        ]).astype(np.float32)
        return self.controller.predict(
            state=state,
            front_rgb=observation["front_rgb"],
            wrist_rgb=observation["wrist_rgb"],
        )

    def observe(self, *, position_error_m: float,
                orientation_error_deg: float,
                cube_displacement_m: float = 0.0) -> StageStatus:
        """Update the explicit gate from post-action task telemetry."""
        self._frame += 1
        self._last_position_error = float(position_error_m)
        self._last_orientation_error = float(orientation_error_deg)
        self._last_cube_displacement = float(cube_displacement_m)
        unsafe = (
            self.max_cube_displacement_m is not None
            and cube_displacement_m > self.max_cube_displacement_m
        )
        inside = (
            position_error_m <= self.position_tolerance_m
            and orientation_error_deg <= self.orientation_tolerance_deg
            and not unsafe
        )
        self._consecutive = self._consecutive + 1 if inside else 0
        self._success = self._consecutive >= self.hold_frames
        if unsafe:
            self._failure_reason = "cube_displacement"
        elif self._frame >= self.timeout_frames and not self._success:
            self._failure_reason = "timeout"
        return self.status

    @property
    def status(self) -> StageStatus:
        return StageStatus(
            phase=self.phase,
            frame=self._frame,
            success=self._success,
            timeout=self._failure_reason == "timeout",
            failure_reason=self._failure_reason,
            position_error_m=self._last_position_error,
            orientation_error_deg=self._last_orientation_error,
            consecutive_gate_frames=self._consecutive,
            cube_displacement_m=self._last_cube_displacement,
        )

    def is_success(self) -> bool:
        return self._success

    def is_timeout(self) -> bool:
        return self._failure_reason == "timeout"

    def is_failed(self) -> bool:
        return self._failure_reason is not None


class ReachPolicy(CartesianACTStagePolicy):
    def __init__(self, checkpoint: str | Path, *, timeout_frames: int = 160,
                 device: str = "cpu") -> None:
        super().__init__(
            checkpoint, phase="reach", task=REACH_TASK,
            timeout_frames=timeout_frames,
            # Freeze the already evaluated Reach gate exactly as specified:
            # position <= 12 mm for five consecutive frames.
            orientation_tolerance_deg=float("inf"), device=device,
        )


class ApproachPolicy(CartesianACTStagePolicy):
    def __init__(self, checkpoint: str | Path, *, timeout_frames: int = 90,
                 max_cube_displacement_m: float = 0.025,
                 device: str = "cpu") -> None:
        super().__init__(
            checkpoint, phase="approach", task=APPROACH_TASK,
            timeout_frames=timeout_frames,
            max_cube_displacement_m=max_cube_displacement_m,
            device=device,
        )


class RecoveryPolicy(CartesianACTStagePolicy):
    """Independent ACT controller for a single terminal recovery attempt."""

    def __init__(self, checkpoint: str | Path, *, timeout_frames: int = 90,
                 max_cube_displacement_m: float = 0.025,
                 device: str = "cpu") -> None:
        super().__init__(
            checkpoint, phase="recovery", task=RECOVERY_TASK,
            timeout_frames=timeout_frames,
            max_cube_displacement_m=max_cube_displacement_m,
            device=device,
        )
