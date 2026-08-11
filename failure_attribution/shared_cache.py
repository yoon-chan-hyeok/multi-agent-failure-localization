from __future__ import annotations

from functools import lru_cache
import hashlib
import json
from pathlib import Path
from typing import Any

from .io import open_text
from .schema import Case


def case_fingerprint(case: Case) -> str:
    payload = {
        "problem": case.problem,
        "ground_truth": case.ground_truth,
        "final_answer": case.final_answer,
        "steps": [
            {
                "step": int(step.step),
                "agent": step.agent,
                "content": step.content,
            }
            for step in case.steps
        ],
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def constraint_fingerprint(constraints: list[dict[str, Any]]) -> str:
    encoded = json.dumps(constraints, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@lru_cache(maxsize=16)
def load_ccv_constraint_cache(path_value: str) -> dict[str, dict[str, Any]]:
    path = Path(path_value).resolve(strict=False)
    if not path.is_file():
        raise FileNotFoundError(f"CCV constraint cache not found: {path}")

    entries: dict[str, dict[str, Any]] = {}
    with open_text(path, "r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            fingerprint = str(item.get("case_fingerprint") or "").strip()
            if not fingerprint:
                raise ValueError(f"Missing case_fingerprint in CCV constraint cache line {line_number}: {path}")
            constraints = item.get("constraints")
            if not isinstance(constraints, list) or not constraints or not all(
                isinstance(constraint, dict) for constraint in constraints
            ):
                raise ValueError(f"Invalid constraints in CCV constraint cache line {line_number}: {path}")
            if fingerprint in entries:
                raise ValueError(f"Duplicate case_fingerprint in CCV constraint cache: {fingerprint}")
            entries[fingerprint] = item
    return entries
