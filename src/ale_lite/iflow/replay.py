from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List

from ale_lite.iflow.trajectory import load_trajectory


def replay(path: Path) -> List[Dict[str, Any]]:
    return load_trajectory(path)
