from __future__ import annotations

import numpy as np


class UnsafeAction(ValueError):
    pass


class ActionSafety:
    def __init__(self, lower, upper, max_velocity):
        self.lower = np.asarray(lower, dtype=float)
        self.upper = np.asarray(upper, dtype=float)
        self.max_velocity = np.asarray(max_velocity, dtype=float)
        if not (self.lower.shape == self.upper.shape == self.max_velocity.shape):
            raise ValueError("safety arrays must have identical shapes")

    def validate(self, action, *, previous=None, dt=None):
        value = np.asarray(action, dtype=float)
        if value.shape != self.lower.shape:
            raise UnsafeAction(f"expected action shape {self.lower.shape}, got {value.shape}")
        if not np.isfinite(value).all():
            raise UnsafeAction("action contains NaN or infinity")
        value = np.clip(value, self.lower, self.upper)
        if previous is not None:
            if dt is None or dt <= 0:
                raise ValueError("positive dt required for velocity limiting")
            prev = np.asarray(previous, dtype=float)
            delta = self.max_velocity * dt
            value = np.clip(value, prev - delta, prev + delta)
        return value

