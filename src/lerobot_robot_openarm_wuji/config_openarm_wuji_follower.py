from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from lerobot.robots import RobotConfig


@RobotConfig.register_subclass("openarm_wuji_follower")
@dataclass
class OpenArmWujiFollowerConfig(RobotConfig):
    backend: Literal["mock", "mujoco", "real"] = "mujoco"
    model_path: Path | None = None
    synergy_config: Path | None = None
    arm_side: Literal["left", "right"] = "left"
    control_hz: float = 30.0
    image_height: int = 240
    image_width: int = 320

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.control_hz <= 0:
            raise ValueError("control_hz must be positive")
        if self.image_height <= 0 or self.image_width <= 0:
            raise ValueError("image dimensions must be positive")
        if self.backend in {"mock", "mujoco"} and self.synergy_config is None:
            raise ValueError(f"synergy_config is required for backend={self.backend!r}")
        if self.backend == "mujoco" and self.model_path is None:
            raise ValueError("model_path is required for backend='mujoco'")
