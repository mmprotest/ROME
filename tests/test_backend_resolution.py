from __future__ import annotations

import pytest

from ale_lite.rock.factory import DEFAULT_DOCKER_IMAGE, resolve_backend
from ale_lite.rock.sandbox import SandboxConfig


def test_backend_auto_prefers_docker_when_available(monkeypatch) -> None:
    monkeypatch.setattr("ale_lite.rock.factory.docker_available", lambda: True)
    resolved = resolve_backend(SandboxConfig(backend="auto"))
    assert resolved.backend == "docker"
    assert resolved.image == DEFAULT_DOCKER_IMAGE


def test_backend_auto_falls_back_to_local_when_docker_missing(monkeypatch) -> None:
    monkeypatch.setattr("ale_lite.rock.factory.docker_available", lambda: False)
    resolved = resolve_backend(SandboxConfig(backend="auto"))
    assert resolved.backend == "local"
    assert resolved.image is None


def test_backend_docker_requires_availability(monkeypatch) -> None:
    monkeypatch.setattr("ale_lite.rock.factory.docker_available", lambda: False)
    with pytest.raises(RuntimeError):
        resolve_backend(SandboxConfig(backend="docker"))


def test_backend_local_always_local(monkeypatch) -> None:
    monkeypatch.setattr("ale_lite.rock.factory.docker_available", lambda: True)
    resolved = resolve_backend(SandboxConfig(backend="local"))
    assert resolved.backend == "local"


def test_image_precedence_task_over_config(monkeypatch) -> None:
    monkeypatch.setattr("ale_lite.rock.factory.docker_available", lambda: True)
    config = SandboxConfig(backend="auto", image="cfg-image")
    resolved = resolve_backend(config, task_image="task-image")
    assert resolved.image == "task-image"


def test_image_precedence_config_over_default(monkeypatch) -> None:
    monkeypatch.setattr("ale_lite.rock.factory.docker_available", lambda: True)
    config = SandboxConfig(backend="auto", image="cfg-image")
    resolved = resolve_backend(config)
    assert resolved.image == "cfg-image"
