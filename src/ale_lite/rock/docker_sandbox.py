from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Callable, Dict, List

from ale_lite.rock.filesystem import safe_path
from ale_lite.rock.limits import DockerResourceLimits
from ale_lite.rock.sandbox import Sandbox, SandboxConfig

CommandRunner = Callable[[List[str], float], subprocess.CompletedProcess[str]]


def _default_runner(args: List[str], timeout_s: float) -> subprocess.CompletedProcess[str]:
    return subprocess.run(args, capture_output=True, text=True, timeout=timeout_s)


class DockerSandbox(Sandbox):
    def __init__(
        self,
        config: SandboxConfig,
        image: str = "python:3.11-slim",
        limits: DockerResourceLimits | None = None,
        runner: CommandRunner | None = None,
    ) -> None:
        super().__init__(config)
        self.image = image
        self._tmpdir: tempfile.TemporaryDirectory[str] | None = None
        self.workspace: Path | None = None
        self.limits = limits or DockerResourceLimits()
        self.runner = runner or _default_runner

    def create_workspace(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory(prefix="ale-lite-docker-")
        self.workspace = Path(self._tmpdir.name)

    def run_command(self, cmd: str, timeout_s: float) -> Dict[str, str | int]:
        if self.workspace is None:
            raise RuntimeError("Workspace not initialized")
        args = self._build_docker_args(cmd)
        process = self.runner(args, timeout_s)
        return {
            "stdout": process.stdout or "",
            "stderr": process.stderr or "",
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

    def describe(self) -> Dict[str, object]:
        return {
            "type": "docker",
            "image": self.image,
            "network_enabled": self.config.network_enabled,
        }

    def _build_docker_args(self, cmd: str) -> List[str]:
        if self.workspace is None:
            raise RuntimeError("Workspace not initialized")
        args = [
            "docker",
            "run",
            "--rm",
            "-v",
            f"{self.workspace}:/work",
            "-w",
            "/work",
        ]
        if not self.config.network_enabled:
            args.extend(["--network=none"])
        if self.limits.cpus is not None:
            args.extend(["--cpus", str(self.limits.cpus)])
        if self.limits.memory_mb is not None:
            args.extend(["--memory", f"{self.limits.memory_mb}m"])
        args.extend(self._allowlist_mounts())
        args.extend([self.image, "/bin/sh", "-lc", cmd])
        return args

    def _allowlist_mounts(self) -> List[str]:
        mounts: List[str] = []
        if not self.config.allowlist_paths:
            return mounts
        for index, raw in enumerate(self.config.allowlist_paths):
            path = Path(raw)
            if not path.is_absolute():
                raise ValueError(f"Allowlist path must be absolute: {raw}")
            if not path.exists():
                raise ValueError(f"Allowlist path does not exist: {raw}")
            target_name = path.name or f"path{index}"
            target = f"/mnt/allow/{index}_{target_name}"
            mounts.extend(["-v", f"{path.resolve()}:{target}:ro"])
        return mounts


def docker_available() -> bool:
    return shutil.which("docker") is not None
