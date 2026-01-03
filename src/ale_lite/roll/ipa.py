from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List


MICRO_STEP_START = "<micro-step>"
MICRO_STEP_END = "</micro-step>"


@dataclass
class Chunk:
    state_summary: str
    assistant_text: str
    tool_calls: List[Dict[str, Any]]
    observations: List[Dict[str, Any]]
    outcome_features: Dict[str, Any]


def _split_micro_steps(text: str) -> List[str]:
    if MICRO_STEP_START not in text:
        return [text]
    segments: List[str] = []
    remaining = text
    while MICRO_STEP_START in remaining:
        before, rest = remaining.split(MICRO_STEP_START, 1)
        if before.strip():
            segments.append(before.strip())
        if MICRO_STEP_END in rest:
            mid, remaining = rest.split(MICRO_STEP_END, 1)
            if mid.strip():
                segments.append(mid.strip())
        else:
            segments.append(rest.strip())
            remaining = ""
    if remaining.strip():
        segments.append(remaining.strip())
    return segments


def chunk_trajectory(events: List[Dict[str, Any]]) -> List[Chunk]:
    chunks: List[Chunk] = []
    current_tools: List[Dict[str, Any]] = []
    current_obs: List[Dict[str, Any]] = []
    state_summary = ""

    for event in events:
        if event["type"] == "message":
            payload = event["payload"]
            if payload["role"] == "assistant":
                for segment in _split_micro_steps(payload["content"]):
                    chunks.append(
                        Chunk(
                            state_summary=state_summary,
                            assistant_text=segment,
                            tool_calls=list(current_tools),
                            observations=list(current_obs),
                            outcome_features={},
                        )
                    )
                    current_tools.clear()
                    current_obs.clear()
            elif payload["role"] == "tool":
                current_obs.append({"content": payload["content"]})
        elif event["type"] == "tool":
            current_tools.append(event["payload"])
        elif event["type"] == "config":
            state_summary = json.dumps(event["payload"], sort_keys=True)

    return chunks


def assign_rewards(chunks: List[Chunk], final_reward: float) -> List[Dict[str, Any]]:
    scored = []
    total = len(chunks) or 1
    for idx, chunk in enumerate(chunks):
        weight = (idx + 1) / total
        reward = final_reward * weight
        tool_error = any(
            call.get("result", {}).get("exit_code") not in (0, None)
            for call in chunk.tool_calls
        )
        if tool_error:
            reward = max(0.0, reward - 0.1)
        scored.append({"chunk": chunk, "reward": reward})
    mean_reward = sum(item["reward"] for item in scored) / total
    for item in scored:
        item["advantage"] = item["reward"] - mean_reward
    return scored


def write_ipa_scored(records: Iterable[Dict[str, Any]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True, default=str) + "\n")
            handle.flush()
