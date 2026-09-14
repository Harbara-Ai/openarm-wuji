"""Minimal LeRobot ACT inference adapter for synchronized MuJoCo control."""
from __future__ import annotations

from pathlib import Path

import numpy as np


DEFAULT_TASK = "Coordinate OpenArm and Wuji to grasp the cube, lift it, and hold it unsupported."


class ACTController:
    """Load a LeRobot ACT checkpoint and predict absolute 27-D targets."""

    STATE_DIM = 27
    ACTION_DIM = 27

    def __init__(self, checkpoint: str | Path, *, device: str = "cpu",
                 task: str = DEFAULT_TASK):
        try:
            import torch
            from lerobot.policies import make_pre_post_processors
            from lerobot.policies.act import ACTPolicy
        except ModuleNotFoundError as error:
            raise RuntimeError(
                "ACT inference requires LeRobot 0.6.2 and its training extras"
            ) from error

        self.torch = torch
        self.device = torch.device(device)
        self.checkpoint = Path(checkpoint)
        self.task = str(task)
        self.policy = ACTPolicy.from_pretrained(self.checkpoint)
        self.policy.to(self.device)
        self.policy.eval()
        self.preprocessor, self.postprocessor = make_pre_post_processors(
            self.policy.config,
            pretrained_path=str(self.checkpoint),
            preprocessor_overrides={
                "device_processor": {"device": str(self.device)}
            },
            postprocessor_overrides={
                "device_processor": {"device": "cpu"}
            },
        )
        if tuple(self.policy.config.input_features["observation.state"].shape) != (27,):
            raise ValueError("ACT checkpoint does not accept a 27-D observation.state")
        if tuple(self.policy.config.output_features["action"].shape) != (27,):
            raise ValueError("ACT checkpoint does not emit a 27-D action")

    def reset(self) -> None:
        """Clear ACT's action chunk between independent episodes."""
        self.policy.reset()

    @staticmethod
    def _validate_rgb(name: str, image: np.ndarray) -> np.ndarray:
        image = np.asarray(image)
        if image.shape != (240, 320, 3):
            raise ValueError(f"{name} must have shape (240, 320, 3), got {image.shape}")
        if image.dtype != np.uint8:
            raise ValueError(f"{name} must be uint8, got {image.dtype}")
        return image

    def predict(self, *, state: np.ndarray, front_rgb: np.ndarray,
                wrist_rgb: np.ndarray) -> np.ndarray:
        """Return the next unnormalized absolute controller target."""
        from lerobot.policies.utils import prepare_observation_for_inference

        state = np.asarray(state, dtype=np.float32)
        if state.shape != (self.STATE_DIM,) or not np.isfinite(state).all():
            raise ValueError("state must be finite and 27-D")
        observation = {
            "observation.state": state,
            "observation.images.front": self._validate_rgb("front_rgb", front_rgb),
            "observation.images.wrist": self._validate_rgb("wrist_rgb", wrist_rgb),
        }
        batch = prepare_observation_for_inference(
            observation,
            self.device,
            task=self.task,
            robot_type="openarm_wuji",
        )
        with self.torch.inference_mode():
            action = self.postprocessor(
                self.policy.select_action(self.preprocessor(batch))
            )
        action = action.squeeze(0).detach().cpu().numpy().astype(np.float64)
        if action.shape != (self.ACTION_DIM,) or not np.isfinite(action).all():
            raise RuntimeError(f"ACT returned an invalid action with shape {action.shape}")
        return action
