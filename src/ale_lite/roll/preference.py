from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List


def make_dpo_records(ipa_scored: Iterable[Dict[str, object]]) -> List[Dict[str, str]]:
    records: List[Dict[str, str]] = []
    for record in ipa_scored:
        chunks = record["chunks"]
        if not chunks:
            continue
        sorted_chunks = sorted(chunks, key=lambda c: c["reward"])
        rejected = sorted_chunks[0]
        chosen = sorted_chunks[-1]
        prompt = chosen["chunk"]["state_summary"]
        records.append(
            {
                "prompt": prompt,
                "chosen": chosen["chunk"]["assistant_text"],
                "rejected": rejected["chunk"]["assistant_text"],
            }
        )
    return records


def write_dpo(records: Iterable[Dict[str, str]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
            handle.flush()
