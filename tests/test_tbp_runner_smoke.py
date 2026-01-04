from __future__ import annotations

from pathlib import Path

from ale_lite.iflow.trajectory import TrajectoryWriter
from ale_lite.rock.docker_sandbox import DockerSandbox
from ale_lite.tbp.runner import run_task
from ale_lite.tbp.tasks import TaskSpec, SuccessCriteria


class FakeAgent:
    def __init__(self, sandbox, trajectory: TrajectoryWriter) -> None:
        self.sandbox = sandbox
        self.trajectory = trajectory

    def run(self, task) -> object:
        self.sandbox.write_file("note.txt", "fixed")
        return type("Result", (), {"success": True, "reason": "done"})


def test_tbp_runner_smoke(tmp_path: Path) -> None:
    task = TaskSpec(
        id="smoke",
        description="smoke",
        goal="fix note",
        setup_steps=["printf 'broken' > note.txt"],
        success_criteria=SuccessCriteria(type="file_contains", file="note.txt", contains="fixed"),
        constraints={"network": False},
        scoring={},
    )
    config = {"llm": {"base_url": "http://", "api_key": "x", "model": "x"}}

    def factory(sandbox, trajectory: TrajectoryWriter):
        return FakeAgent(sandbox, trajectory)

    result = run_task(task, config, tmp_path, agent_factory=factory)
    assert result.success is True
    assert result.trajectory_path.exists()


def test_tbp_runner_selects_docker_when_image_set(tmp_path: Path, monkeypatch) -> None:
    task = TaskSpec(
        id="docker",
        description="docker",
        goal="write note",
        setup_steps=[],
        success_criteria=SuccessCriteria(type="file_contains", file="note.txt", contains="fixed"),
        constraints={"network": False},
        scoring={},
        image="python:3.11-slim",
    )
    config = {"llm": {"base_url": "http://", "api_key": "x", "model": "x"}}
    monkeypatch.setattr("ale_lite.rock.factory.docker_available", lambda: True)

    def factory(sandbox, trajectory: TrajectoryWriter):
        assert isinstance(sandbox, DockerSandbox)
        return FakeAgent(sandbox, trajectory)

    result = run_task(task, config, tmp_path, agent_factory=factory)
    assert result.success is True
