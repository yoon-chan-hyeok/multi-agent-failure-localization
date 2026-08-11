from __future__ import annotations

import json
import re
from typing import Any


def extract_json(text: str) -> Any:
    text = text.strip()
    if not text:
        raise ValueError("empty response")

    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        text = fenced.group(1).strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    obj_start = text.find("{")
    arr_start = text.find("[")
    starts = [i for i in [obj_start, arr_start] if i >= 0]
    if not starts:
        raise ValueError("no JSON object or array found")
    start = min(starts)
    open_char = text[start]
    close_char = "}" if open_char == "{" else "]"
    end = text.rfind(close_char)
    if end < start:
        raise
    return json.loads(text[start : end + 1])


def as_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        out = float(value)
        if out < 0:
            return 0.0
        if out > 1:
            return 1.0
        return out
    except (TypeError, ValueError):
        return default


def as_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"true", "yes", "y", "1"}
    if value is None:
        return default
    return bool(value)
