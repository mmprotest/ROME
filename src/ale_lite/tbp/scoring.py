from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict

from ale_lite.rock.sandbox import Sandbox
from ale_lite.tbp.tasks import SuccessCriteria


@dataclass
class ScoreResult:
    score: float
    success: bool
    details: Dict[str, str]


def evaluate(sandbox: Sandbox, criteria: SuccessCriteria) -> ScoreResult:
    if criteria.type == "command_exit_code" and criteria.command:
        result = sandbox.run_command(criteria.command, timeout_s=30)
        success = result["exit_code"] == 0
        return ScoreResult(score=1.0 if success else 0.0, success=success, details=result)
    if criteria.type == "file_contains" and criteria.file and criteria.contains is not None:
        content = sandbox.read_file(criteria.file)
        success = criteria.contains in content
        return ScoreResult(
            score=1.0 if success else 0.0,
            success=success,
            details={"file": criteria.file},
        )
    if criteria.type == "regex_in_stdout" and criteria.command and criteria.regex:
        result = sandbox.run_command(criteria.command, timeout_s=30)
        match = re.search(criteria.regex, result["stdout"]) is not None
        return ScoreResult(
            score=1.0 if match else 0.0,
            success=match,
            details=result,
        )
    if criteria.type == "unit_tests_pass" and criteria.command:
        result = sandbox.run_command(criteria.command, timeout_s=60)
        success = result["exit_code"] == 0
        return ScoreResult(score=1.0 if success else 0.0, success=success, details=result)
    return ScoreResult(score=0.0, success=False, details={"error": "unsupported criteria"})
