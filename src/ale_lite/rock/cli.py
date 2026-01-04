from __future__ import annotations

from pathlib import Path

import typer

from ale_lite.rock.docker_sandbox import docker_available

app = typer.Typer(help="ROCK sandbox manager")


@app.command()
def doctor(config: Path | None = typer.Option(None, "--config", exists=True, dir_okay=False)) -> None:
    docker_ok = docker_available()
    status = "available" if docker_ok else "not available"
    typer.echo(f"docker: {status}")
    typer.echo("local: available")
    prefer_docker = False
    has_image = False
    if config is not None:
        import yaml

        data = yaml.safe_load(config.read_text(encoding="utf-8")) or {}
        sandbox_cfg = data.get("sandbox", {})
        prefer_docker = bool(sandbox_cfg.get("prefer_docker", False))
        has_image = sandbox_cfg.get("image") is not None
    if docker_ok and (prefer_docker or has_image):
        default_backend = "docker"
    else:
        default_backend = "docker" if docker_ok else "local"
    typer.echo(f"default_backend: {default_backend}")
