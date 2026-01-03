from __future__ import annotations

import importlib
import time
from dataclasses import dataclass
from typing import Any, Iterable, List, Optional, Tuple, Type


@dataclass(frozen=True)
class OpenAIConfig:
    base_url: str
    api_key: str
    model: str
    temperature: float = 0.2
    max_tokens: int = 1024
    timeout_s: float = 120.0


class OpenAIChatClient:
    def __init__(self, config: OpenAIConfig, client: Optional[Any] = None) -> None:
        self.config = config
        if client is None:
            self.client = self._build_client()
            self._retry_exceptions = self._load_retry_exceptions()
        else:
            self.client = client
            self._retry_exceptions = (Exception,)

    def _build_client(self) -> Any:
        openai_module = importlib.import_module("openai")
        return openai_module.OpenAI(base_url=self.config.base_url, api_key=self.config.api_key)

    def _load_retry_exceptions(self) -> Tuple[Type[BaseException], ...]:
        openai_module = importlib.import_module("openai")
        return (
            openai_module.APIError,
            openai_module.APITimeoutError,
            openai_module.APIConnectionError,
            openai_module.RateLimitError,
        )

    def chat(
        self,
        messages: List[dict[str, Any]],
        tools: Optional[List[dict[str, Any]]] = None,
        tool_choice: Optional[str | dict[str, Any]] = None,
        max_retries: int = 3,
    ) -> dict[str, Any]:
        backoff = 1.0
        for attempt in range(max_retries + 1):
            try:
                response = self.client.chat.completions.create(
                    model=self.config.model,
                    messages=messages,
                    temperature=self.config.temperature,
                    max_tokens=self.config.max_tokens,
                    timeout=self.config.timeout_s,
                    tools=tools,
                    tool_choice=tool_choice,
                )
                message = response.choices[0].message
                tool_calls = [
                    {
                        "id": call.id,
                        "type": call.type,
                        "function": {
                            "name": call.function.name,
                            "arguments": call.function.arguments,
                        },
                    }
                    for call in (message.tool_calls or [])
                ]
                return {
                    "content": message.content or "",
                    "tool_calls": tool_calls,
                    "raw": response,
                }
            except self._retry_exceptions as exc:
                if attempt >= max_retries:
                    raise
                time.sleep(backoff)
                backoff *= 2


def tool_schema() -> List[dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {
                "name": "terminal.exec",
                "description": "Run a shell command inside the sandbox.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "cmd": {"type": "string"},
                        "timeout_s": {"type": "number"},
                    },
                    "required": ["cmd"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "filesystem.read",
                "description": "Read a file inside the sandbox.",
                "parameters": {
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                    "required": ["path"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "filesystem.write",
                "description": "Write a file inside the sandbox.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                        "content": {"type": "string"},
                    },
                    "required": ["path", "content"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "filesystem.list",
                "description": "List a directory inside the sandbox.",
                "parameters": {
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                    "required": ["path"],
                },
            },
        },
    ]


def tool_choice_auto() -> str:
    return "auto"


def build_messages(system_prompt: str, user_prompt: str, memory: Iterable[dict[str, Any]]) -> List[dict[str, Any]]:
    messages: List[dict[str, Any]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    messages.extend(memory)
    return messages
