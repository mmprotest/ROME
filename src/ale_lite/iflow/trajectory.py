from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List

SCHEMA_VERSION = "1.0"


@dataclass
class TrajectoryWriter:
    path: Path
    events: List[Dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, event_type: str, payload: Dict[str, Any]) -> None:
        event = {
            "schema_version": SCHEMA_VERSION,
            "timestamp": time.time(),
            "type": event_type,
            "payload": payload,
        }
        self.events.append(event)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, sort_keys=True) + "\n")
            handle.flush()

    def to_list(self) -> List[Dict[str, Any]]:
        return list(self.events)


def load_trajectory(path: Path) -> List[Dict[str, Any]]:
    events: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            events.append(json.loads(line))
    return events


def message_event(role: str, content: str, tool_calls: List[dict[str, Any]] | None = None) -> Dict[str, Any]:
    return {
        "role": role,
        "content": content,
        "tool_calls": tool_calls or [],
    }


def tool_event(name: str, arguments: dict[str, Any], result: dict[str, Any]) -> Dict[str, Any]:
    return {
        "name": name,
        "arguments": arguments,
        "result": result,
    }


def config_event(model: Dict[str, Any], sandbox: Dict[str, Any]) -> Dict[str, Any]:
    return {"model": model, "sandbox": sandbox}


def outcome_event(success: bool, score: float, reason: str) -> Dict[str, Any]:
    return {"success": success, "score": score, "reason": reason}
