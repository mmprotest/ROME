from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable, Dict

from ale_lite.rock.sandbox import Sandbox


@dataclass
class ToolResult:
    name: str
    output: str
    success: bool
    truncated: bool
    raw: Dict[str, Any]


MAX_OUTPUT_CHARS = 2000


def _truncate(text: str) -> tuple[str, bool]:
    if len(text) <= MAX_OUTPUT_CHARS:
        return text, False
    return text[:MAX_OUTPUT_CHARS] + "...<truncated>", True


def terminal_exec(sandbox: Sandbox, args: dict[str, Any]) -> ToolResult:
    cmd = args.get("cmd", "")
    timeout_s = float(args.get("timeout_s", 60))
    result = sandbox.run_command(cmd, timeout_s=timeout_s)
    output, truncated = _truncate(result["stdout"] + result["stderr"])
    return ToolResult(
        name="terminal.exec",
        output=output,
        success=result["exit_code"] == 0,
        truncated=truncated,
        raw=result,
    )


def filesystem_read(sandbox: Sandbox, args: dict[str, Any]) -> ToolResult:
    path = args.get("path", "")
    content = sandbox.read_file(path)
    output, truncated = _truncate(content)
    return ToolResult(
        name="filesystem.read",
        output=output,
        success=True,
        truncated=truncated,
        raw={"path": path, "content": content},
    )


def filesystem_write(sandbox: Sandbox, args: dict[str, Any]) -> ToolResult:
    path = args.get("path", "")
    content = args.get("content", "")
    sandbox.write_file(path, content)
    return ToolResult(
        name="filesystem.write",
        output=f"Wrote {len(content)} bytes to {path}",
        success=True,
        truncated=False,
        raw={"path": path, "content": content},
    )


def filesystem_list(sandbox: Sandbox, args: dict[str, Any]) -> ToolResult:
    path = args.get("path", ".")
    entries = sandbox.list_dir(path)
    output = "\n".join(entries)
    output, truncated = _truncate(output)
    return ToolResult(
        name="filesystem.list",
        output=output,
        success=True,
        truncated=truncated,
        raw={"path": path, "entries": entries},
    )


ToolHandler = Callable[[Sandbox, dict[str, Any]], ToolResult]


TOOL_REGISTRY: dict[str, ToolHandler] = {
    "terminal.exec": terminal_exec,
    "filesystem.read": filesystem_read,
    "filesystem.write": filesystem_write,
    "filesystem.list": filesystem_list,
}


def dispatch_tool(sandbox: Sandbox, name: str, arguments_json: str) -> ToolResult:
    handler = TOOL_REGISTRY[name]
    args = json.loads(arguments_json) if arguments_json else {}
    return handler(sandbox, args)
