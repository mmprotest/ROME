from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List

from ale_lite.rock.filesystem import safe_path
from ale_lite.rock.limits import ResourceLimits, apply_limits
from ale_lite.rock.sandbox import Sandbox, SandboxConfig


class LocalSandbox(Sandbox):
    def __init__(self, config: SandboxConfig, limits: ResourceLimits | None = None) -> None:
        super().__init__(config)
        self._tmpdir: tempfile.TemporaryDirectory[str] | None = None
        self.workspace: Path | None = None
        self.limits = limits or ResourceLimits()

    def create_workspace(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory(prefix="ale-lite-")
        self.workspace = Path(self._tmpdir.name)

    def run_command(self, cmd: str, timeout_s: float) -> Dict[str, str | int]:
        if self.workspace is None:
            raise RuntimeError("Workspace not initialized")

        def preexec() -> None:
            apply_limits(self.limits)

        env = os.environ.copy()
        if not self.config.network_enabled:
            env["NO_PROXY"] = "*"
            env["HTTP_PROXY"] = ""
            env["HTTPS_PROXY"] = ""

        process = subprocess.run(
            cmd,
            cwd=self.workspace,
            env=env,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            preexec_fn=preexec,
        )
        return {
            "stdout": process.stdout,
            "stderr": process.stderr,
            "exit_code": process.returncode,
        }

    def read_file(self, path: str) -> str:
        if self.workspace is None:
            raise RuntimeError("Workspace not initialized")
        target = safe_path(self.workspace, path)
        return target.read_text(encoding="utf-8")

    def write_file(self, path: str, content: str) -> None:
        if self.workspace is None:
            raise RuntimeError("Workspace not initialized")
        target = safe_path(self.workspace, path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")

    def list_dir(self, path: str) -> List[str]:
        if self.workspace is None:
            raise RuntimeError("Workspace not initialized")
        target = safe_path(self.workspace, path)
        return [p.name for p in target.iterdir()]

    def teardown(self) -> None:
        if self._tmpdir is not None:
            self._tmpdir.cleanup()
        self._tmpdir = None
        self.workspace = None

    def describe(self) -> Dict[str, str]:
        return {
            "type": "local",
            "network_enabled": str(self.config.network_enabled),
        }


def docker_available() -> bool:
    return shutil.which("docker") is not None
