from __future__ import annotations

from typer.testing import CliRunner

from ale_lite.rock.cli import app


def test_rock_doctor_reports_docker_available(monkeypatch) -> None:
    runner = CliRunner()
    monkeypatch.setattr("ale_lite.rock.cli.docker_available", lambda: True)
    result = runner.invoke(app, [])
    assert result.exit_code == 0
    assert "docker_available: yes" in result.output
    assert "configured_backend: auto" in result.output
    assert "resolved_backend: docker" in result.output


def test_rock_doctor_reports_docker_missing(monkeypatch) -> None:
    runner = CliRunner()
    monkeypatch.setattr("ale_lite.rock.cli.docker_available", lambda: False)
    result = runner.invoke(app, [])
    assert result.exit_code == 0
    assert "docker_available: no" in result.output
    assert "configured_backend: auto" in result.output
    assert "resolved_backend: local" in result.output
