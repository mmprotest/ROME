from __future__ import annotations

from pathlib import Path
from typing import Iterator

from ale_lite.api.openai_client import OpenAIConfig
from ale_lite.iflow.agent import Agent, AgentConfig
from ale_lite.iflow.trajectory import TrajectoryWriter, load_trajectory
from ale_lite.tbp.runner import run_task
from ale_lite.tbp.tasks import SuccessCriteria, TaskSpec


class FakeClient:
    def __init__(self) -> None:
        self.config = OpenAIConfig(base_url="http://", api_key="x", model="x")
        self.calls = 0

    def chat(self, messages, tools=None, tool_choice=None, max_retries=3):
        self.calls += 1
        return {"content": "still working", "tool_calls": []}


def _monotonic_sequence(values: list[float]):
    iterator: Iterator[float] = iter(values)
    last = values[-1]

    def _fake() -> float:
        nonlocal last
        try:
            last = next(iterator)
        except StopIteration:
            pass
        return last

    return _fake


def test_agent_enforces_deadline_and_records_timeout(tmp_path: Path, monkeypatch) -> None:
    task = TaskSpec(
        id="deadline",
        description="deadline",
        goal="timeout",
        setup_steps=[],
        success_criteria=SuccessCriteria(type="command_exit_code", command="true"),
        constraints={"time_limit_s": 1},
        scoring={},
    )

    def agent_factory(sandbox, trajectory: TrajectoryWriter):
        agent = Agent(
            FakeClient(),
            sandbox,
            trajectory,
            AgentConfig(max_steps=10, time_limit_s=1),
        )
        return agent

    fake_time = _monotonic_sequence([0.0, 0.1, 1.1, 1.2])
    monkeypatch.setattr("ale_lite.iflow.agent.time.monotonic", fake_time)

    result = run_task(task, {"sandbox": {"backend": "local"}}, tmp_path, agent_factory=agent_factory)
    assert result.success is False
    events = load_trajectory(result.trajectory_path)
    outcome_event = next(event for event in events if event["type"] == "outcome")
    payload = outcome_event["payload"]
    assert payload["outcome"] == "timeout"
    assert payload["reason"] == "timeout"
    assert payload["duration_s"] >= 1
