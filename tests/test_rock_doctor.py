from __future__ import annotations

from typer.testing import CliRunner

from ale_lite.rock.cli import app


def test_rock_doctor_reports_docker_available(monkeypatch) -> None:
    runner = CliRunner()
    monkeypatch.setattr("ale_lite.rock.cli.docker_available", lambda: True)
    result = runner.invoke(app, [])
    assert result.exit_code == 0
    assert "docker: available" in result.output
    assert "default_backend: docker" in result.output


def test_rock_doctor_reports_docker_missing(monkeypatch) -> None:
    runner = CliRunner()
    monkeypatch.setattr("ale_lite.rock.cli.docker_available", lambda: False)
    result = runner.invoke(app, [])
    assert result.exit_code == 0
    assert "docker: not available" in result.output
    assert "default_backend: local" in result.output
