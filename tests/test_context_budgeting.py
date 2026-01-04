from __future__ import annotations

import json

from ale_lite.api.openai_client import build_messages, estimate_tokens


def _message_cost(message: dict[str, object]) -> int:
    content = message.get("content", "")
    tool_calls = message.get("tool_calls")
    text = str(content)
    if tool_calls:
        text += json.dumps(tool_calls, sort_keys=True)
    return estimate_tokens(text)


def test_context_budgeting_keeps_recent_turns() -> None:
    system = "system"
    user = "user prompt"
    memory = []
    for index in range(6):
        memory.append({"role": "assistant", "content": f"message-{index}-" + ("x" * 80)})

    messages = build_messages(system, user, memory, max_tokens=60)
    total = sum(_message_cost(message) for message in messages)
    assert total <= 60
    assert any(message.get("content") == "message-5-" + ("x" * 80) for message in messages)
    assert any(
        message.get("role") == "system"
        and str(message.get("content", "")).startswith("Working memory summary:")
        for message in messages
    )
