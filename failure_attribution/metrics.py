from __future__ import annotations

from statistics import mean
from typing import Any

from .schema import Case, Prediction


def evaluate(cases: list[Case], predictions: list[Prediction]) -> dict[str, Any]:
    paired = pair_cases_and_predictions(cases, predictions)
    rows: list[dict[str, Any]] = []
    for case_index, case, pred in paired:
        pred_agent_attribution = prediction_agent_attribution(pred)
        row = {
            "case_index": case_index,
            "case_id": pred.case_id,
            "pred_agent": pred.agent,
            "pred_agent_attribution": pred_agent_attribution,
            "pred_step": pred.step,
            "gold_agent": case.gold_agent,
            "gold_step": case.gold_step,
            "agent_correct": agent_correct(pred.agent, case.gold_agent),
            "agent_in_attribution": agent_in_attribution(pred_agent_attribution, case.gold_agent),
            "step_correct": step_correct(pred.step, case.gold_step, tolerance=0),
            "step_pm1": step_correct(pred.step, case.gold_step, tolerance=1),
            "step_pm3": step_correct(pred.step, case.gold_step, tolerance=3),
            "step_pm5": step_correct(pred.step, case.gold_step, tolerance=5),
            "distance": abs(pred.step - case.gold_step) if pred.step is not None and case.gold_step is not None else None,
        }
        rows.append(row)

    return {
        "count": len(rows),
        "labeled_agent_count": sum(1 for r in rows if r["gold_agent"] is not None),
        "labeled_step_count": sum(1 for r in rows if r["gold_step"] is not None),
        "agent_accuracy": avg_bool([r["agent_correct"] for r in rows]),
        "agent_attribution_accuracy": avg_bool([r["agent_in_attribution"] for r in rows]),
        "step_accuracy": avg_bool([r["step_correct"] for r in rows]),
        "step_pm1_accuracy": avg_bool([r["step_pm1"] for r in rows]),
        "step_pm3_accuracy": avg_bool([r["step_pm3"] for r in rows]),
        "step_pm5_accuracy": avg_bool([r["step_pm5"] for r in rows]),
        "mean_abs_distance": avg_float([r["distance"] for r in rows]),
        "rows": rows,
    }


def pair_cases_and_predictions(cases: list[Case], predictions: list[Prediction]) -> list[tuple[int, Case, Prediction]]:
    case_ids = [case.case_id for case in cases]
    duplicate_case_ids = len(set(case_ids)) != len(case_ids)
    if duplicate_case_ids and len(cases) == len(predictions):
        return [(idx, case, pred) for idx, (case, pred) in enumerate(zip(cases, predictions))]

    case_by_id = {case.case_id: (idx, case) for idx, case in enumerate(cases)}
    paired: list[tuple[int, Case, Prediction]] = []
    for pred in predictions:
        idx, case = case_by_id[pred.case_id]
        paired.append((idx, case, pred))
    return paired


def usage_summary(predictions: list[Prediction]) -> dict[str, Any]:
    totals = {
        "llm_calls": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
    }
    for pred in predictions:
        usage = (pred.trace or {}).get("usage") or {}
        totals["llm_calls"] += safe_int(usage.get("calls", usage.get("llm_calls", 0)))
        totals["input_tokens"] += safe_int(usage.get("input_tokens", 0))
        totals["output_tokens"] += safe_int(usage.get("output_tokens", 0))
        totals["total_tokens"] += safe_int(
            usage.get("total_tokens", safe_int(usage.get("input_tokens", 0)) + safe_int(usage.get("output_tokens", 0)))
        )

    case_count = len(predictions)
    if case_count:
        totals["avg_llm_calls_per_case"] = totals["llm_calls"] / case_count
        totals["avg_input_tokens_per_case"] = totals["input_tokens"] / case_count
        totals["avg_output_tokens_per_case"] = totals["output_tokens"] / case_count
        totals["avg_total_tokens_per_case"] = totals["total_tokens"] / case_count
    else:
        totals["avg_llm_calls_per_case"] = None
        totals["avg_input_tokens_per_case"] = None
        totals["avg_output_tokens_per_case"] = None
        totals["avg_total_tokens_per_case"] = None
    return totals


def agent_correct(pred: str | None, gold: str | None) -> bool | None:
    if gold is None:
        return None
    if pred is None:
        return False
    return normalize_agent(pred) == normalize_agent(gold)


def agent_in_attribution(pred_agents: list[str], gold: str | None) -> bool | None:
    if gold is None:
        return None
    if not pred_agents:
        return False
    gold_norm = normalize_agent(gold)
    return any(normalize_agent(agent) == gold_norm for agent in pred_agents)


def prediction_agent_attribution(pred: Prediction) -> list[str]:
    trace = pred.trace or {}
    consensus = trace.get("consensus")
    if isinstance(consensus, dict):
        conclusion = consensus.get("consensus_conclusion")
        if isinstance(conclusion, dict):
            attribution = conclusion.get("attribution")
            if isinstance(attribution, list):
                return [str(agent).strip() for agent in attribution if str(agent).strip()]
            if isinstance(attribution, str) and attribution.strip():
                return [attribution.strip()]
    return [pred.agent] if pred.agent else []


def step_correct(pred: int | None, gold: int | None, tolerance: int) -> bool | None:
    if gold is None:
        return None
    if pred is None:
        return False
    return abs(pred - gold) <= tolerance


def avg_bool(values: list[bool | None]) -> float | None:
    usable = [v for v in values if v is not None]
    if not usable:
        return None
    return sum(1 for v in usable if v) / len(usable)


def avg_float(values: list[float | int | None]) -> float | None:
    usable = [float(v) for v in values if v is not None]
    if not usable:
        return None
    return mean(usable)


def normalize_agent(value: str) -> str:
    value = value.strip().lower()
    if "(" in value:
        value = value.split("(", 1)[0].strip()
    return value


def safe_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0
