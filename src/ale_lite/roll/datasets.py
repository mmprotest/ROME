from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List

from ale_lite.iflow.trajectory import load_trajectory


@dataclass
class RawTrajectory:
    task_id: str
    path: Path
    events: List[dict[str, object]]


def collect_runs(runs_dir: Path) -> List[RawTrajectory]:
    trajectories = []
    for path in sorted(runs_dir.glob("*_trajectory.jsonl")):
        task_id = path.name.replace("_trajectory.jsonl", "")
        events = load_trajectory(path)
        trajectories.append(RawTrajectory(task_id=task_id, path=path, events=events))
    return trajectories


def write_raw_dataset(trajectories: Iterable[RawTrajectory], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        for traj in trajectories:
            record = {
                "task_id": traj.task_id,
                "path": str(traj.path),
                "events": traj.events,
            }
            handle.write(json.dumps(record, sort_keys=True) + "\n")
            handle.flush()
