from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List


@dataclass
class SandboxConfig:
    network_enabled: bool = False
    allowlist_paths: List[str] | None = None
    time_limit_s: int = 30


class Sandbox(ABC):
    def __init__(self, config: SandboxConfig) -> None:
        self.config = config

    @abstractmethod
    def create_workspace(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def run_command(self, cmd: str, timeout_s: float) -> Dict[str, str | int]:
        raise NotImplementedError

    @abstractmethod
    def read_file(self, path: str) -> str:
        raise NotImplementedError

    @abstractmethod
    def write_file(self, path: str, content: str) -> None:
        raise NotImplementedError

    @abstractmethod
    def list_dir(self, path: str) -> List[str]:
        raise NotImplementedError

    @abstractmethod
    def teardown(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def describe(self) -> Dict[str, str]:
        raise NotImplementedError
