from __future__ import annotations

from ale_lite.roll.ipa import assign_rewards, chunk_trajectory


def test_chunking_and_ipa() -> None:
    events = [
        {"type": "config", "payload": {"model": "x"}},
        {
            "type": "message",
            "payload": {"role": "assistant", "content": "step one"},
        },
        {
            "type": "tool",
            "payload": {"name": "terminal.exec", "arguments": {}, "result": {"exit_code": 0}},
        },
        {
            "type": "message",
            "payload": {"role": "tool", "content": "ok"},
        },
        {
            "type": "message",
            "payload": {"role": "assistant", "content": "step two"},
        },
    ]
    chunks = chunk_trajectory(events)
    assert len(chunks) == 2
    scored = assign_rewards(chunks, final_reward=1.0)
    assert scored[-1]["reward"] >= scored[0]["reward"]
    assert "advantage" in scored[0]
