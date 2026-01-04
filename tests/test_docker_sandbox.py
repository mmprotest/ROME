from __future__ import annotations

from pathlib import Path

from ale_lite.rock.docker_sandbox import DockerSandbox
from ale_lite.rock.limits import DockerResourceLimits
from ale_lite.rock.sandbox import SandboxConfig


def test_docker_sandbox_builds_command_args(tmp_path: Path) -> None:
    captured = {}

    def fake_runner(args: list[str], timeout_s: float):
        captured["args"] = args
        captured["timeout_s"] = timeout_s
        return type("Result", (), {"stdout": "", "stderr": "", "returncode": 0})()

    allow_path = tmp_path / "allow.txt"
    allow_path.write_text("ok", encoding="utf-8")
    sandbox = DockerSandbox(
        SandboxConfig(network_enabled=False, allowlist_paths=[str(allow_path)]),
        image="python:3.11-slim",
        limits=DockerResourceLimits(cpus=1.5, memory_mb=128),
        runner=fake_runner,
    )
    sandbox.create_workspace()
    sandbox.run_command("echo hi", timeout_s=5)
    args = captured["args"]
    assert args[:2] == ["docker", "run"]
    assert "--network=none" in args
    assert "--cpus" in args
    assert "--memory" in args
    assert f"{sandbox.workspace}:/work" in " ".join(args)
    assert "/mnt/allow/0_allow.txt" in " ".join(args)
