"""Deterministic hand-written router shared by mining and deployment."""
from __future__ import annotations

import numpy as np


TRIGGER_PRIORITY = (
    "cube_displacement",
    "moving_away",
    "terminal_plateau",
    "gate_stall",
    "near_timeout_terminal",
)


def detect_near_failure(*, errors: list[float], cube_displacement_m: float,
                        gate_frames_total: int, gate_consecutive: int,
                        frame: int, timeout_frames: int,
                        status_success: bool) -> dict[str, float]:
    """Return the unchanged mining detectors and their diagnostic scores."""
    result: dict[str, float] = {}
    current = errors[-1]
    if len(errors) >= 7 and current < 0.030:
        improvement = errors[-7] - current
        if improvement < 0.001:
            result["terminal_plateau"] = float(0.001 - improvement)
    if len(errors) >= 4 and min(errors[:-1], default=current) < 0.020:
        away = current - min(errors)
        recent = np.diff(errors[-4:])
        if away > 0.005 and np.count_nonzero(recent > 0.0) >= 2:
            result["moving_away"] = float(away)
    if cube_displacement_m > 0.003:
        result["cube_displacement"] = float(cube_displacement_m)
    if frame >= timeout_frames - 10 and current < 0.030:
        result["near_timeout_terminal"] = float(
            (frame - (timeout_frames - 10) + 1) / 10 + (0.030 - current)
        )
    if (
        not status_success
        and current < 0.020
        and gate_frames_total >= 4
        and gate_consecutive < 3
    ):
        result["gate_stall"] = float(
            gate_frames_total - gate_consecutive + (0.020 - current)
        )
    return result


def primary_trigger(triggers: dict[str, float]) -> str | None:
    """Choose a stable label if multiple unchanged detectors fire together."""
    return next((name for name in TRIGGER_PRIORITY if name in triggers), None)
