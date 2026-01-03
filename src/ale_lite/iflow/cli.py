from __future__ import annotations

from pathlib import Path

import typer

from ale_lite.tbp.runner import load_config, run_task
from ale_lite.tbp.tasks import load_task

app = typer.Typer(help="iFlow agent runtime")


@app.command()
def run(
    task: Path = typer.Option(..., "--task", exists=True, dir_okay=False),
    config: Path = typer.Option(..., "--config", exists=True, dir_okay=False),
    out: Path = typer.Option(Path("runs"), "--out"),
) -> None:
    cfg = load_config(config)
    task_spec = load_task(task)
    result = run_task(task_spec, cfg, out)
    typer.echo(f"{task_spec.id}: score={result.score} success={result.success}")
