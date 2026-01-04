from __future__ import annotations

import importlib
import time
import json
import math
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


def estimate_tokens(text: str) -> int:
    """Approximate token usage as len(text) / 4 for deterministic budgeting."""
    if not text:
        return 0
    return max(1, math.floor(len(text) / 4))


def _message_token_cost(message: dict[str, Any]) -> int:
    content = message.get("content", "")
    tool_calls = message.get("tool_calls")
    if tool_calls:
        content += json.dumps(tool_calls, sort_keys=True)
    return estimate_tokens(content)


def _truncate_to_budget(text: str, max_tokens: int) -> str:
    if max_tokens <= 0:
        return ""
    max_chars = max_tokens * 4
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "...<truncated>"


def _summarize_messages(messages: Iterable[dict[str, Any]]) -> str:
    lines: List[str] = []
    for message in messages:
        role = message.get("role", "unknown")
        content = message.get("content", "")
        lines.append(f"{role}: {content[:200]}")
    return "\n".join(lines) if lines else "No prior steps."


def build_messages(
    system_prompt: str,
    user_prompt: str,
    memory: Iterable[dict[str, Any]],
    *,
    max_tokens: Optional[int] = None,
    memory_summary: Optional[str] = None,
) -> List[dict[str, Any]]:
    messages: List[dict[str, Any]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    memory_list = list(memory)
    if max_tokens is None:
        messages.extend(memory_list)
        return messages

    base_tokens = sum(_message_token_cost(message) for message in messages)
    if base_tokens >= max_tokens:
        return messages

    selected: List[dict[str, Any]] = []
    remaining = max_tokens - base_tokens
    for message in reversed(memory_list):
        cost = _message_token_cost(message)
        if cost > remaining:
            break
        selected.append(message)
        remaining -= cost
    selected.reverse()

    omitted_count = len(memory_list) - len(selected)
    if omitted_count > 0:
        summary_text = memory_summary or _summarize_messages(memory_list[:omitted_count])
        prefix = "Working memory summary:\n"
        summary_budget = remaining - estimate_tokens(prefix)
        summary_text = _truncate_to_budget(summary_text, summary_budget)
        if summary_text:
            summary_message = f"{prefix}{summary_text}"
            messages.append({"role": "system", "content": summary_message})
            remaining -= estimate_tokens(summary_message)

    messages.extend(selected)

    total_tokens = sum(_message_token_cost(message) for message in messages)
    while total_tokens > max_tokens and selected:
        removed = selected.pop(0)
        messages.remove(removed)
        total_tokens -= _message_token_cost(removed)
    if total_tokens > max_tokens:
        summary_messages = [m for m in messages if m.get("role") == "system" and "summary" in m.get("content", "")]
        for summary_message in summary_messages:
            messages.remove(summary_message)
            total_tokens -= _message_token_cost(summary_message)
            if total_tokens <= max_tokens:
                break
    return messages
