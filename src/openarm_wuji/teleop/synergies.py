from __future__ import annotations

import json
from pathlib import Path

import numpy as np


class HandSynergyMapper:
    """Config-driven open/close, pinch, and spread mapping for a 20-DoF hand."""

    def __init__(self, config: dict):
        self.joint_names = tuple(config["joint_names"])
        self.lower = np.asarray(config["position_min"], dtype=float)
        self.upper = np.asarray(config["position_max"], dtype=float)
        self.open_pose = np.asarray(config["open_pose"], dtype=float)
        self.close_pose = np.asarray(config["close_pose"], dtype=float)
        self.pinch_pose = np.asarray(config["pinch_pose"], dtype=float)
        self.spread_vector = np.asarray(config["spread_vector"], dtype=float)
        self.max_velocity = np.asarray(config["max_velocity_rad_s"], dtype=float)
        arrays = (self.lower, self.upper, self.open_pose, self.close_pose,
                  self.pinch_pose, self.spread_vector, self.max_velocity)
        if len(self.joint_names) != 20 or any(x.shape != (20,) for x in arrays):
            raise ValueError("Wuji synergy config must define exactly 20 joints")
        if any(not np.isfinite(x).all() for x in arrays):
            raise ValueError("Wuji synergy config contains NaN or infinity")

    @classmethod
    def from_json(cls, path: str | Path):
        return cls(json.loads(Path(path).read_text(encoding="utf-8")))

    def map(self, open_close: float, pinch: float, spread: float) -> np.ndarray:
        weights = np.asarray([open_close, pinch, spread], dtype=float)
        if not np.isfinite(weights).all():
            raise ValueError("synergy contains NaN or infinity")
        close_w, pinch_w = np.clip(weights[:2], 0.0, 1.0)
        spread_w = float(np.clip(weights[2], -1.0, 1.0))
        target = self.open_pose.copy()
        target += close_w * (self.close_pose - self.open_pose)
        target += pinch_w * (self.pinch_pose - self.open_pose)
        target += spread_w * self.spread_vector
        return np.clip(target, self.lower, self.upper)

    def next(self, command, *, previous, dt: float) -> np.ndarray:
        if dt <= 0:
            raise ValueError("dt must be positive")
        target = self.map(*command)
        previous = np.asarray(previous, dtype=float)
        if previous.shape != (20,) or not np.isfinite(previous).all():
            raise ValueError("previous hand action must be finite and 20-D")
        delta = self.max_velocity * dt
        return np.clip(target, previous - delta, previous + delta)


def synergy_to_joints(open_close: float, pinch: float, spread: float) -> np.ndarray:
    """Backward-compatible default mapping used by lightweight tests."""
    root = Path(__file__).resolve().parents[3]
    mapper = HandSynergyMapper.from_json(root / "configs/wuji_hand_left_synergies.json")
    return mapper.map(open_close, pinch, spread)
