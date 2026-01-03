from __future__ import annotations

from pathlib import Path

import typer

from ale_lite.tbp.runner import load_config, run_task
from ale_lite.tbp.tasks import load_tasks_from_dir

app = typer.Typer(help="TerminalBenchPro harness")


@app.command()
def run(
    tasks: Path = typer.Option(..., "--tasks", exists=True, dir_okay=True),
    config: Path = typer.Option(..., "--config", exists=True, dir_okay=False),
    out: Path = typer.Option(Path("runs"), "--out"),
) -> None:
    cfg = load_config(config)
    results = []
    for task in load_tasks_from_dir(tasks):
        result = run_task(task, cfg, out)
        results.append(result)
        typer.echo(f"{task.id}: {result.score}")
    typer.echo(f"Completed {len(results)} tasks")
