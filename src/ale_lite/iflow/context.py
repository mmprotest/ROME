from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List


@dataclass
class WorkingMemory:
    max_items: int
    items: List[dict[str, Any]] = field(default_factory=list)

    def add(self, message: dict[str, Any]) -> None:
        self.items.append(message)
        if len(self.items) > self.max_items:
            self.items = self.items[-self.max_items :]

    def summarize(self) -> str:
        if not self.items:
            return "No prior steps."
        summary_lines = []
        for item in self.items:
            role = item.get("role", "unknown")
            content = item.get("content", "")
            summary_lines.append(f"{role}: {content[:200]}")
        return "\n".join(summary_lines)

    def to_messages(self) -> List[dict[str, Any]]:
        return list(self.items)
