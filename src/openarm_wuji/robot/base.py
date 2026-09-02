from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Mapping, Sequence


class OpenArmWujiRobot(ABC):
    @abstractmethod
    def connect(self) -> None: ...

    @abstractmethod
    def get_observation(self) -> Mapping[str, Any]: ...

    @abstractmethod
    def send_action(self, action: Sequence[float]) -> None: ...

    @abstractmethod
    def disconnect(self) -> None: ...

