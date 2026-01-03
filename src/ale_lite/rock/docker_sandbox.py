from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Dict, List

from ale_lite.rock.sandbox import Sandbox, SandboxConfig


class DockerSandbox(Sandbox):
    def __init__(self, config: SandboxConfig, image: str = "python:3.11-slim") -> None:
        super().__init__(config)
        self.image = image
        self.workspace: Path | None = None

    def create_workspace(self) -> None:
        raise NotImplementedError("Docker sandbox requires external workspace creation")

    def run_command(self, cmd: str, timeout_s: float) -> Dict[str, str | int]:
        raise NotImplementedError("Docker sandbox execution is not implemented in ALE-lite")

    def read_file(self, path: str) -> str:
        raise NotImplementedError("Docker sandbox read not implemented")

    def write_file(self, path: str, content: str) -> None:
        raise NotImplementedError("Docker sandbox write not implemented")

    def list_dir(self, path: str) -> List[str]:
        raise NotImplementedError("Docker sandbox list not implemented")

    def teardown(self) -> None:
        return

    def describe(self) -> Dict[str, str]:
        return {"type": "docker", "image": self.image}


def docker_available() -> bool:
    return shutil.which("docker") is not None
