from __future__ import annotations

from pathlib import Path

import typer

from ale_lite.rock.docker_sandbox import docker_available
from ale_lite.rock.factory import resolve_backend
from ale_lite.rock.sandbox import SandboxConfig

app = typer.Typer(help="ROCK sandbox manager")


@app.command()
def doctor(config: Path | None = typer.Option(None, "--config", exists=True, dir_okay=False)) -> None:
    docker_ok = docker_available()
    status = "yes" if docker_ok else "no"
    typer.echo(f"docker_available: {status}")
    typer.echo("local_available: yes")
    sandbox_config = SandboxConfig()
    if config is not None:
        import yaml

        data = yaml.safe_load(config.read_text(encoding="utf-8")) or {}
        sandbox_cfg = data.get("sandbox", {})
        sandbox_config.backend = sandbox_cfg.get("backend")
        sandbox_config.prefer_docker = bool(sandbox_cfg.get("prefer_docker", False))
        sandbox_config.image = sandbox_cfg.get("image")
    configured_backend = sandbox_config.backend or "auto"
    typer.echo(f"configured_backend: {configured_backend}")
    try:
        resolved = resolve_backend(sandbox_config)
        typer.echo(f"resolved_backend: {resolved.backend}")
        if resolved.backend == "docker":
            typer.echo(f"resolved_image: {resolved.image}")
    except RuntimeError as exc:
        typer.echo("resolved_backend: error")
        typer.echo(f"error: {exc}")
