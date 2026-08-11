from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, TextIO

from .schema import Case, LogStep, Prediction


def extended_path(path: str | Path) -> Path:
    path = Path(path)
    if os.name != "nt":
        return path
    text = str(path)
    if text.startswith("\\\\?\\"):
        return path
    if not path.is_absolute():
        path = path.resolve(strict=False)
        text = str(path)
    if text.startswith("\\\\"):
        return Path("\\\\?\\UNC\\" + text.lstrip("\\"))
    return Path("\\\\?\\" + text)


def open_text(path: str | Path, mode: str, encoding: str = "utf-8", **kwargs: Any) -> TextIO:
    path = Path(path)
    if any(flag in mode for flag in ("w", "a", "x", "+")):
        extended_path(path.parent).mkdir(parents=True, exist_ok=True)
    return extended_path(path).open(mode, encoding=encoding, **kwargs)


def write_text_file(path: str | Path, text: str, encoding: str = "utf-8") -> None:
    with open_text(path, "w", encoding=encoding) as f:
        f.write(text)


def load_cases(path: str | Path, generated_step_base: int = 1) -> list[Case]:
    path = Path(path)
    cases: list[Case] = []
    with open_text(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            raw = json.loads(line)
            cases.append(parse_case(raw, line_no=line_no, generated_step_base=generated_step_base))
    return cases


def parse_case(raw: dict[str, Any], line_no: int, generated_step_base: int = 1) -> Case:
    case_id = str(raw.get("id") or raw.get("case_id") or raw.get("task_id") or raw.get("question_ID") or f"line-{line_no}")
    problem = str(raw.get("problem") or raw.get("query") or raw.get("question") or raw.get("task") or "")
    if not problem:
        raise ValueError(f"Missing problem/query field for case {case_id}")

    log_value = (
        raw.get("failure_log")
        or raw.get("log")
        or raw.get("logs")
        or raw.get("trajectory")
        or raw.get("conversation")
        or raw.get("history")
    )
    steps = parse_steps(log_value, generated_step_base=generated_step_base)
    if not steps:
        raise ValueError(f"Missing or empty failure log for case {case_id}")

    label = raw.get("label") if isinstance(raw.get("label"), dict) else {}
    gold_agent = first_present(
        label,
        raw,
        keys=["agent", "responsible_agent", "failure_responsible_agent", "gold_agent", "mistake_agent"],
    )
    gold_step_raw = first_present(
        label,
        raw,
        keys=["step", "error_step", "decisive_error_step", "gold_step", "s_star", "mistake_step"],
    )
    gold_step = int(gold_step_raw) if gold_step_raw is not None and str(gold_step_raw) != "" else None

    return Case(
        case_id=case_id,
        problem=problem,
        ground_truth=optional_str(raw.get("ground_truth") or raw.get("groundtruth") or raw.get("answer") or raw.get("gold_answer")),
        final_answer=optional_str(raw.get("final_answer") or raw.get("final_wrong_answer") or raw.get("prediction")),
        steps=steps,
        gold_agent=optional_str(gold_agent),
        gold_step=gold_step,
        metadata={k: v for k, v in raw.items() if k not in {"failure_log", "log", "logs", "trajectory", "conversation", "history"}},
    )


def parse_steps(log_value: Any, generated_step_base: int = 1) -> list[LogStep]:
    if isinstance(log_value, str):
        return [LogStep(step=generated_step_base, agent="Unknown", content=log_value)]
    if not isinstance(log_value, list):
        return []

    steps: list[LogStep] = []
    for idx, item in enumerate(log_value):
        default_step = generated_step_base + idx
        if isinstance(item, str):
            steps.append(LogStep(step=default_step, agent="Unknown", content=item))
            continue
        if not isinstance(item, dict):
            steps.append(LogStep(step=default_step, agent="Unknown", content=str(item)))
            continue
        step = int(item.get("step") or item.get("step_number") or item.get("turn") or default_step)
        agent = str(item.get("agent") or item.get("name") or item.get("role") or "Unknown")
        content = str(item.get("content") or item.get("message") or item.get("text") or item.get("action") or "")
        steps.append(LogStep(step=step, agent=agent, content=content))
    return steps


def write_predictions(path: str | Path, predictions: list[Prediction]) -> None:
    path = Path(path)
    with open_text(path, "w", encoding="utf-8") as f:
        for pred in predictions:
            f.write(json.dumps(pred_to_dict(pred), ensure_ascii=False) + "\n")


def write_json(path: str | Path, value: Any) -> None:
    path = Path(path)
    write_text_file(path, json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def pred_to_dict(pred: Prediction) -> dict[str, Any]:
    return {
        "case_id": pred.case_id,
        "method": pred.method,
        "agent": pred.agent,
        "step": pred.step,
        "confidence": pred.confidence,
        "reason": pred.reason,
        "trace": pred.trace,
    }


def first_present(*dicts: dict[str, Any], keys: list[str]) -> Any:
    for d in dicts:
        for key in keys:
            if key in d and d[key] is not None:
                return d[key]
    return None


def optional_str(value: Any) -> str | None:
    if value is None:
        return None
    value = str(value)
    return value if value else None
