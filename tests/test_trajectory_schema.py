from __future__ import annotations

from pathlib import Path

from ale_lite.iflow.trajectory import SCHEMA_VERSION, TrajectoryWriter, load_trajectory


def test_trajectory_schema(tmp_path: Path) -> None:
    path = tmp_path / "traj.jsonl"
    writer = TrajectoryWriter(path)
    writer.log("message", {"role": "assistant", "content": "hi"})
    events = load_trajectory(path)
    assert events[0]["schema_version"] == SCHEMA_VERSION
    assert events[0]["type"] == "message"
