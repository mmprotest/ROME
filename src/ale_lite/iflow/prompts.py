from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TaskSpec:
    goal: str
    evaluation: str


def system_prompt() -> str:
    return (
        "You are an agent operating inside a sandbox. "
        "Use the provided tools to inspect and modify files and execute commands. "
        "Be concise and follow the task goal."
    )


def task_prompt(task: TaskSpec) -> str:
    return f"Task goal:\n{task.goal}\n\nEvaluation:\n{task.evaluation}"
