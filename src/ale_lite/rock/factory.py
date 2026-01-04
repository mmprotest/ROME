from __future__ import annotations

from typing import Optional

from ale_lite.rock.docker_sandbox import DockerSandbox, docker_available
from ale_lite.rock.local_sandbox import LocalSandbox
from ale_lite.rock.sandbox import Sandbox, SandboxConfig


def make_sandbox(
    sandbox_config: SandboxConfig,
    *,
    prefer_docker: bool,
    image: Optional[str] = None,
) -> Sandbox:
    if docker_available() and (image or prefer_docker):
        return DockerSandbox(sandbox_config, image=image or "python:3.11-slim")
    return LocalSandbox(sandbox_config)
