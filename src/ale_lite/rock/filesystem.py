from __future__ import annotations

from pathlib import Path


class PathViolationError(ValueError):
    pass


def safe_path(base: Path, target: str) -> Path:
    candidate = (base / target).resolve()
    if not str(candidate).startswith(str(base.resolve())):
        raise PathViolationError(f"Path traversal blocked: {target}")
    return candidate
