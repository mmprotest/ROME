from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Optional

from ale_lite.rock.docker_sandbox import DockerSandbox, docker_available
from ale_lite.rock.local_sandbox import LocalSandbox
from ale_lite.rock.sandbox import Sandbox, SandboxConfig

DEFAULT_DOCKER_IMAGE = "python:3.11-slim"
_SUPPORTED_BACKENDS = {"auto", "docker", "local"}
_LOG = logging.getLogger(__name__)


@dataclass(frozen=True)
class ResolvedBackend:
    backend: str
    image: str | None = None


def resolve_backend(sandbox_config: SandboxConfig, task_image: Optional[str] = None) -> ResolvedBackend:
    backend = sandbox_config.backend or "auto"
    if sandbox_config.backend is None and sandbox_config.prefer_docker:
        _LOG.warning("sandbox.prefer_docker is deprecated; use sandbox.backend instead.")
    backend = backend.lower()
    if backend not in _SUPPORTED_BACKENDS:
        raise ValueError(f"Unsupported sandbox backend: {backend}")

    if backend == "docker":
        if not docker_available():
            raise RuntimeError("Docker backend requested but Docker is not available.")
        image = task_image or sandbox_config.image or DEFAULT_DOCKER_IMAGE
        return ResolvedBackend(backend="docker", image=image)

    if backend == "local":
        return ResolvedBackend(backend="local", image=None)

    if docker_available():
        image = task_image or sandbox_config.image or DEFAULT_DOCKER_IMAGE
        return ResolvedBackend(backend="docker", image=image)
    return ResolvedBackend(backend="local", image=None)


def make_sandbox(
    sandbox_config: SandboxConfig,
    *,
    task_image: Optional[str] = None,
) -> Sandbox:
    resolved = resolve_backend(sandbox_config, task_image=task_image)
    if resolved.backend == "docker":
        return DockerSandbox(sandbox_config, image=resolved.image or DEFAULT_DOCKER_IMAGE)
    return LocalSandbox(sandbox_config)
