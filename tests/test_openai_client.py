from __future__ import annotations

from dataclasses import dataclass

from ale_lite.api.openai_client import OpenAIChatClient, OpenAIConfig


class FakeMessage:
    def __init__(self) -> None:
        self.content = "hello"
        self.tool_calls = []


class FakeChoice:
    def __init__(self) -> None:
        self.message = FakeMessage()


class FakeResponse:
    def __init__(self) -> None:
        self.choices = [FakeChoice()]


class FakeChatCompletions:
    def create(self, **kwargs: object) -> FakeResponse:
        return FakeResponse()


class FakeChat:
    def __init__(self) -> None:
        self.completions = FakeChatCompletions()


class FakeOpenAI:
    def __init__(self) -> None:
        self.chat = FakeChat()


def test_openai_client_basic() -> None:
    config = OpenAIConfig(base_url="http://localhost", api_key="test", model="test")
    client = OpenAIChatClient(config, client=FakeOpenAI())
    result = client.chat(messages=[{"role": "user", "content": "hi"}])
    assert result["content"] == "hello"
    assert result["tool_calls"] == []
