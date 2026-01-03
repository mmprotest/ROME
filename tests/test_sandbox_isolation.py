from __future__ import annotations

import pytest

from ale_lite.rock.filesystem import PathViolationError
from ale_lite.rock.local_sandbox import LocalSandbox
from ale_lite.rock.sandbox import SandboxConfig


def test_local_sandbox_path_safety(tmp_path) -> None:
    sandbox = LocalSandbox(SandboxConfig())
    sandbox.create_workspace()
    sandbox.write_file("safe.txt", "ok")
    assert sandbox.read_file("safe.txt") == "ok"
    with pytest.raises(PathViolationError):
        sandbox.read_file("../etc/passwd")
    sandbox.teardown()
