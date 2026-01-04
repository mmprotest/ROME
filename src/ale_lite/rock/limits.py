from __future__ import annotations

import resource
from dataclasses import dataclass


@dataclass(frozen=True)
class ResourceLimits:
    cpu_time_s: int = 10
    memory_mb: int = 512


@dataclass(frozen=True)
class DockerResourceLimits:
    cpus: float | None = None
    memory_mb: int | None = None


def apply_limits(limits: ResourceLimits) -> None:
    resource.setrlimit(resource.RLIMIT_CPU, (limits.cpu_time_s, limits.cpu_time_s))
    memory_bytes = limits.memory_mb * 1024 * 1024
    resource.setrlimit(resource.RLIMIT_AS, (memory_bytes, memory_bytes))
