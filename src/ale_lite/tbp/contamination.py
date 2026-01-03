from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ContaminationControl:
    allowlist: list[str]
    network_enabled: bool = False
