from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class SuccessCriteria:
    type: str
    command: Optional[str] = None
    file: Optional[str] = None
    contains: Optional[str] = None
    regex: Optional[str] = None


@dataclass
class TaskSpec:
    id: str
    description: str
    goal: str
    setup_steps: List[str] = field(default_factory=list)
    success_criteria: SuccessCriteria = field(default_factory=SuccessCriteria)
    constraints: Dict[str, Any] = field(default_factory=dict)
    scoring: Dict[str, Any] = field(default_factory=dict)
    image: Optional[str] = None

    @property
    def evaluation_text(self) -> str:
        return f"Success criteria: {self.success_criteria.type}"


def load_task(path: Path) -> TaskSpec:
    import yaml

    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    criteria = SuccessCriteria(**data.get("success_criteria", {}))
    return TaskSpec(
        id=data["id"],
        description=data["description"],
        goal=data["goal"],
        setup_steps=data.get("setup_steps", []),
        success_criteria=criteria,
        constraints=data.get("constraints", {}),
        scoring=data.get("scoring", {}),
        image=data.get("image"),
    )


def load_tasks_from_dir(path: Path) -> List[TaskSpec]:
    return [load_task(p) for p in sorted(path.glob("*.yaml"))]
