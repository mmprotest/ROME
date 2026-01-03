from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Optional

from ale_lite.api.openai_client import OpenAIChatClient, build_messages, tool_choice_auto, tool_schema
from ale_lite.iflow.context import WorkingMemory
from ale_lite.iflow.prompts import TaskSpec, system_prompt, task_prompt
from ale_lite.iflow.tools import dispatch_tool
from ale_lite.iflow.trajectory import TrajectoryWriter, config_event, message_event, tool_event
from ale_lite.rock.sandbox import Sandbox


@dataclass
class AgentConfig:
    max_steps: int = 40
    max_turns: int = 80
    tool_timeout_s: float = 60
    context_max_tokens: int = 8000
    memory_items: int = 20


@dataclass
class AgentResult:
    success: bool
    reason: str


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
        self.trajectory.log(
            "config",
            config_event(
                model={
                    "base_url": self.client.config.base_url,
                    "model": self.client.config.model,
                    "temperature": self.client.config.temperature,
                },
                sandbox=self.sandbox.describe(),
            ),
        )
        for step in range(self.config.max_steps):
            summary = self.memory.summarize()
            messages = build_messages(system_prompt(), task_prompt(task, summary), self.memory.to_messages())
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
                    return AgentResult(success=True, reason="assistant reported success")
                continue

            for tool_call in tool_calls:
                name = tool_call["function"]["name"]
                args = tool_call["function"]["arguments"]
                result = dispatch_tool(self.sandbox, name, args)
                self.trajectory.log(
                    "tool",
                    tool_event(name, json.loads(args) if args else {}, result.raw),
                )
                tool_message = {
                    "role": "tool",
                    "name": name,
                    "content": result.output,
                }
                self.memory.add(tool_message)
                self.trajectory.log("message", message_event("tool", result.output))

        return AgentResult(success=False, reason="max steps reached")
