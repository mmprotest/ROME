from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List

from ale_lite.iflow.tools import dispatch_tool
from ale_lite.iflow.trajectory import load_trajectory
from ale_lite.rock.sandbox import Sandbox


@dataclass
class ReplaySummary:
    steps: int
    tool_calls: int
    outcome: Dict[str, Any] | None


@dataclass
class ReplayDiff:
    name: str
    differences: List[str]


def replay(path: Path) -> List[Dict[str, Any]]:
    return load_trajectory(path)


def summarize(events: Iterable[Dict[str, Any]]) -> ReplaySummary:
    steps = 0
    tool_calls = 0
    outcome: Dict[str, Any] | None = None
    for event in events:
        if event["type"] == "message" and event["payload"].get("role") == "assistant":
            steps += 1
        if event["type"] == "tool":
            tool_calls += 1
        if event["type"] == "outcome":
            outcome = event["payload"]
    return ReplaySummary(steps=steps, tool_calls=tool_calls, outcome=outcome)


def reexec_tools(events: Iterable[Dict[str, Any]], sandbox: Sandbox) -> List[ReplayDiff]:
    diffs: List[ReplayDiff] = []
    for event in events:
        if event["type"] != "tool":
            continue
        payload = event["payload"]
        name = payload.get("name", "")
        arguments = payload.get("arguments", {})
        expected = payload.get("result", {})
        try:
            result = dispatch_tool(sandbox, name, json.dumps(arguments))
            actual = result.raw
            differences = _compare_dicts(expected, actual)
        except Exception as exc:  # pragma: no cover - defensive safety
            differences = [f"reexec error: {exc}"]
        if differences:
            diffs.append(ReplayDiff(name=name, differences=differences))
    return diffs


def _compare_dicts(expected: Dict[str, Any], actual: Dict[str, Any]) -> List[str]:
    diffs: List[str] = []
    keys = set(expected.keys()) | set(actual.keys())
    for key in sorted(keys):
        if expected.get(key) != actual.get(key):
            diffs.append(f"{key}: expected={expected.get(key)!r} actual={actual.get(key)!r}")
    return diffs
