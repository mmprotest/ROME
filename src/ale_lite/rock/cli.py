from __future__ import annotations

import typer

from ale_lite.rock.local_sandbox import docker_available

app = typer.Typer(help="ROCK sandbox manager")


@app.command()
def doctor() -> None:
    docker_ok = docker_available()
    status = "available" if docker_ok else "not available"
    typer.echo(f"docker: {status}")
    typer.echo("local: available")
