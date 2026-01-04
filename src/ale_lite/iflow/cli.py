from __future__ import annotations

from pathlib import Path

import typer

from ale_lite.iflow.replay import reexec_tools, replay, summarize
from ale_lite.rock.factory import make_sandbox
from ale_lite.rock.sandbox import SandboxConfig
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


@app.command("replay")
def replay_command(
    trajectory: Path = typer.Option(..., "--trajectory", exists=True, dir_okay=False),
    reexec_tools_flag: bool = typer.Option(False, "--reexec-tools"),
    prefer_docker: bool = typer.Option(False, "--prefer-docker"),
    image: str | None = typer.Option(None, "--image"),
) -> None:
    events = replay(trajectory)
    summary = summarize(events)
    typer.echo(f"steps: {summary.steps}")
    typer.echo(f"tool_calls: {summary.tool_calls}")
    typer.echo(f"outcome: {summary.outcome}")

    if not reexec_tools_flag:
        return

    sandbox_info = next(
        (event["payload"].get("sandbox", {}) for event in events if event["type"] == "config"),
        {},
    )
    network_enabled = bool(sandbox_info.get("network_enabled", False))
    prefer_docker = prefer_docker or sandbox_info.get("type") == "docker"
    image = image or sandbox_info.get("image")
    sandbox = make_sandbox(SandboxConfig(network_enabled=network_enabled), prefer_docker=prefer_docker, image=image)
    sandbox.create_workspace()
    diffs = reexec_tools(events, sandbox)
    sandbox.teardown()
    if not diffs:
        typer.echo("reexec: no differences found")
        return
    typer.echo("reexec differences:")
    for diff in diffs:
        typer.echo(f"- {diff.name}")
        for line in diff.differences:
            typer.echo(f"  {line}")
