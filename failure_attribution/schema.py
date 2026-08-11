from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class LogStep:
    step: int
    agent: str
    content: str


@dataclass
class Case:
    case_id: str
    problem: str
    steps: list[LogStep]
    ground_truth: str | None = None
    final_answer: str | None = None
    gold_agent: str | None = None
    gold_step: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Prediction:
    case_id: str
    method: str
    agent: str | None
    step: int | None
    confidence: float | None = None
    reason: str | None = None
    trace: dict[str, Any] = field(default_factory=dict)

