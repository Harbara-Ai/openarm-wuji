from __future__ import annotations

from .base import OpenArmWujiRobot


class RealOpenArmWuji(OpenArmWujiRobot):
    """Deliberately inert until the customized arm/hand protocol is supplied."""

    def connect(self) -> None:
        raise NotImplementedError("real hardware interface is not specified")

    def get_observation(self):
        raise NotImplementedError("real hardware interface is not specified")

    def send_action(self, action) -> None:
        raise NotImplementedError("real hardware interface is not specified")

    def disconnect(self) -> None:
        return None

