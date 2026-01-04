from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any

from ale_lite.api.openai_client import OpenAIChatClient, build_messages, tool_choice_auto, tool_schema
from ale_lite.iflow.context import WorkingMemory
from ale_lite.iflow.prompts import TaskSpec, system_prompt, task_prompt
from ale_lite.iflow.tools import dispatch_tool_args
from ale_lite.iflow.trajectory import TrajectoryWriter, config_event, message_event, tool_event
from ale_lite.rock.sandbox import Sandbox


@dataclass
class AgentConfig:
    max_steps: int = 40
    max_turns: int = 80
    tool_timeout_s: float = 60
    time_limit_s: float = 600
    context_max_tokens: int = 8000
    memory_items: int = 20


@dataclass
class AgentResult:
    success: bool
    reason: str
    outcome: str
    duration_s: float


class Agent:
    def __init__(
        self,
        client: OpenAIChatClient,
        sandbox: Sandbox,
        trajectory: TrajectoryWriter,
        config: AgentConfig,
    ) -> None:
        self.client = client
        self.sandbox = sandbox
        self.trajectory = trajectory
        self.config = config
        self.memory = WorkingMemory(max_items=config.memory_items)

    def run(self, task: TaskSpec) -> AgentResult:
        start_time = time.monotonic()
        deadline = start_time + self.config.time_limit_s
        self.trajectory.log(
            "config",
            config_event(
                model={
                    "base_url": self.client.config.base_url,
                    "model": self.client.config.model,
                    "temperature": self.client.config.temperature,
                },
                sandbox=self.sandbox.describe(),
                agent={
                    "max_steps": self.config.max_steps,
                    "max_turns": self.config.max_turns,
                    "tool_timeout_s": self.config.tool_timeout_s,
                    "time_limit_s": self.config.time_limit_s,
                    "context_max_tokens": self.config.context_max_tokens,
                    "memory_items": self.config.memory_items,
                },
            ),
        )
        for step in range(self.config.max_steps):
            if time.monotonic() > deadline:
                duration = time.monotonic() - start_time
                return AgentResult(
                    success=False,
                    reason="timeout",
                    outcome="timeout",
                    duration_s=duration,
                )
            messages = build_messages(
                system_prompt(),
                task_prompt(task),
                self.memory.to_messages(),
                max_tokens=self.config.context_max_tokens,
            )
            summary_message = next(
                (
                    message["content"]
                    for message in messages
                    if message.get("role") == "system"
                    and message.get("content", "").startswith("Working memory summary:")
                ),
                None,
            )
            if summary_message:
                self.trajectory.log("message", message_event("system", summary_message))
            response = self.client.chat(
                messages=messages,
                tools=tool_schema(),
                tool_choice=tool_choice_auto(),
            )
            assistant_text = response["content"]
            tool_calls = response["tool_calls"]
            self.memory.add({"role": "assistant", "content": assistant_text, "tool_calls": tool_calls})
            self.trajectory.log("message", message_event("assistant", assistant_text, tool_calls))

            if not tool_calls:
                if "SUCCESS" in assistant_text.upper():
                    duration = time.monotonic() - start_time
                    return AgentResult(
                        success=True,
                        reason="assistant reported success",
                        outcome="success",
                        duration_s=duration,
                    )
                continue

            for tool_call in tool_calls:
                if time.monotonic() > deadline:
                    duration = time.monotonic() - start_time
                    return AgentResult(
                        success=False,
                        reason="timeout",
                        outcome="timeout",
                        duration_s=duration,
                    )
                name = tool_call["function"]["name"]
                args = tool_call["function"]["arguments"]
                arguments = json.loads(args) if args else {}
                if name == "terminal.exec":
                    remaining = max(1.0, deadline - time.monotonic())
                    arguments["timeout_s"] = min(self.config.tool_timeout_s, remaining)
                result = dispatch_tool_args(self.sandbox, name, arguments)
                self.trajectory.log(
                    "tool",
                    tool_event(name, arguments, result.raw),
                )
                tool_message = {
                    "role": "tool",
                    "name": name,
                    "content": result.output,
                }
                self.memory.add(tool_message)
                self.trajectory.log("message", message_event("tool", result.output))

        duration = time.monotonic() - start_time
        return AgentResult(
            success=False,
            reason="max steps reached",
            outcome="max_steps",
            duration_s=duration,
        )
