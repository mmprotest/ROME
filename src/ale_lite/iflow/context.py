from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, List


@dataclass
class WorkingMemory:
    max_items: int
    items: List[dict[str, Any]] = field(default_factory=list)

    def add(self, message: dict[str, Any]) -> None:
        self.items.append(message)
        if len(self.items) > self.max_items:
            self.items = self.items[-self.max_items :]

    def summarize(self, items: Iterable[dict[str, Any]] | None = None, max_chars: int | None = None) -> str:
        target_items = list(items) if items is not None else self.items
        if not target_items:
            return "No prior steps."
        summary_lines = []
        for item in target_items:
            role = item.get("role", "unknown")
            content = item.get("content", "")
            summary_lines.append(f"{role}: {content[:200]}")
        summary = "\n".join(summary_lines)
        if max_chars is not None and len(summary) > max_chars:
            return summary[:max_chars] + "...<truncated>"
        return summary

    def to_messages(self) -> List[dict[str, Any]]:
        return list(self.items)
