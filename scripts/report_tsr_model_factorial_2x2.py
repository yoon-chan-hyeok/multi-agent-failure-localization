from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from failure_attribution.io import load_cases
from failure_attribution.metrics import evaluate, usage_summary
from failure_attribution.schema import Prediction


CELL_ORDER = ("L->L", "G->L", "L->G", "G->G")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def load_predictions(path: Path) -> list[Prediction]:
    return [
        Prediction(
            case_id=str(row["case_id"]),
            method=str(row.get("method") or "ccv_ablation_no_gt"),
            agent=row.get("agent"),
            step=row.get("step"),
            confidence=row.get("confidence"),
            reason=row.get("reason"),
            trace=row.get("trace") or {},
        )
        for row in load_jsonl(path)
    ]


def require_parallel(cases: list[Any], predictions: list[Prediction], label: str) -> None:
    if len(cases) != len(predictions):
        raise ValueError(
            f"{label}: case/prediction count mismatch {len(cases)} != {len(predictions)}"
        )
    mismatches = [
        (index, case.case_id, prediction.case_id)
        for index, (case, prediction) in enumerate(zip(cases, predictions, strict=True))
        if case.case_id != prediction.case_id
    ]
    if mismatches:
        raise ValueError(f"{label}: prediction order mismatch: {mismatches[:5]}")


def stage_usage(predictions: list[Prediction]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for stage in ("requirement_generation", "failure_localization"):
        rows = []
        for prediction in predictions:
            stages = prediction.trace.get("stage_usage")
            if isinstance(stages, dict) and isinstance(stages.get(stage), dict):
                rows.append(stages[stage])
        if len(rows) != len(predictions):
            result[stage] = None
            continue
        result[stage] = {
            "calls": sum(int(row.get("calls") or 0) for row in rows),
            "input_tokens": sum(int(row.get("input_tokens") or 0) for row in rows),
            "output_tokens": sum(int(row.get("output_tokens") or 0) for row in rows),
            "total_tokens": sum(int(row.get("total_tokens") or 0) for row in rows),
        }
    return result


def summarize(cases: list[Any], predictions: list[Prediction]) -> dict[str, Any]:
    result = evaluate(cases, predictions)
    return {
        "n": len(cases),
        "agent": result["agent_accuracy"],
        "step": result["step_accuracy"],
        "pm3": result["step_pm3_accuracy"],
        "pm5": result["step_pm5_accuracy"],
        "mad": result["mean_abs_distance"],
        "null_steps": sum(prediction.step is None for prediction in predictions),
        "usage": usage_summary(predictions),
        "stage_usage": stage_usage(predictions),
    }


def exact_binomial_p(left_only: int, right_only: int) -> float:
    discordant = left_only + right_only
    if discordant == 0:
        return 1.0
    lower = min(left_only, right_only)
    probability = 2 * sum(math.comb(discordant, k) for k in range(lower + 1)) / (2**discordant)
    return min(1.0, probability)


def paired_metric(
    cases: list[Any],
    left: list[Prediction],
    right: list[Prediction],
    metric: str,
) -> dict[str, Any]:
    left_rows = evaluate(cases, left)["rows"]
    right_rows = evaluate(cases, right)["rows"]
    field = "agent_correct" if metric == "agent" else "step_correct"
    values = [
        (int(bool(a.get(field))), int(bool(b.get(field))))
        for a, b in zip(left_rows, right_rows, strict=True)
    ]
    left_only = sum(a and not b for a, b in values)
    right_only = sum(b and not a for a, b in values)
    return {
        "left_only": left_only,
        "right_only": right_only,
        "right_minus_left": sum(b - a for a, b in values) / len(values),
        "mcnemar_exact_p": exact_binomial_p(left_only, right_only),
    }


def pct(value: Any) -> str:
    return "NA" if value is None else f"{float(value) * 100:.2f}%"


def number(value: Any) -> str:
    return "NA" if value is None else f"{float(value):.2f}"


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# TSR-Loc Requirement-Generator x Localizer 2x2",
        "",
        f"- Light model: `{report['light_model']}` ({report['light_provider']})",
        f"- Strong model: `{report['strong_model']}`",
        "- L/G denote the light/strong model respectively.",
        "",
        "| split | cell | n | Agent | Exact | +/-3 | +/-5 | MAD | null | calls | tokens | tok/case |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for split in ("All", "AG", "HC", "HC-long"):
        for cell in CELL_ORDER:
            item = report["summaries"][split][cell]
            usage = item["usage"]
            lines.append(
                f"| {split} | {cell} | {item['n']} | {pct(item['agent'])} | {pct(item['step'])} | "
                f"{pct(item['pm3'])} | {pct(item['pm5'])} | "
                f"{number(item['mad'])} | {item['null_steps']} | {usage['llm_calls']} | "
                f"{usage['total_tokens']} | {number(usage['avg_total_tokens_per_case'])} |"
            )

    lines.extend(
        [
            "",
            "## Factorial contrasts on all cases",
            "",
            "Positive delta means the right-hand condition is better.",
            "",
            "| contrast | metric | delta | left-only/right-only | McNemar p |",
            "| --- | --- | ---: | ---: | ---: |",
        ]
    )
    for contrast, metrics in report["paired_contrasts"].items():
        for metric, value in metrics.items():
            lines.append(
                f"| {contrast} | {metric} | {value['right_minus_left'] * 100:.2f} pp | "
                f"{value['left_only']}/{value['right_only']} | {value['mcnemar_exact_p']:.4f} |"
            )

    lines.extend(["", "## Recorded stage usage", ""])
    for cell in CELL_ORDER:
        stage = report["summaries"]["All"][cell]["stage_usage"]
        if not stage or all(value is None for value in stage.values()):
            lines.append(f"- **{cell}**: unavailable in the historical prediction file")
            continue
        req = stage.get("requirement_generation")
        loc = stage.get("failure_localization")
        lines.append(
            f"- **{cell}**: requirement={req['total_tokens'] if req else 'NA'} tokens; "
            f"localization={loc['total_tokens'] if loc else 'NA'} tokens"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize the TSR-Loc requirement-generator x localizer 2x2 experiment."
    )
    parser.add_argument("--data", required=True)
    parser.add_argument("--cell", action="append", required=True, help="LABEL=PREDICTIONS_JSONL")
    parser.add_argument("--light-model", default="meta-llama/llama-3.1-8b-instruct")
    parser.add_argument("--light-provider", default="OpenRouter/Groq")
    parser.add_argument("--strong-model", default="gpt-4o")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    cell_paths: dict[str, Path] = {}
    for value in args.cell:
        label, separator, raw_path = value.partition("=")
        if not separator or label not in CELL_ORDER:
            raise ValueError(f"Invalid --cell value: {value}; expected one of {CELL_ORDER}=PATH")
        cell_paths[label] = Path(raw_path)
    missing = set(CELL_ORDER) - set(cell_paths)
    if missing:
        raise ValueError(f"Missing factorial cells: {sorted(missing)}")

    data_path = Path(args.data)
    raw_rows = load_jsonl(data_path)
    cases = load_cases(data_path, generated_step_base=0)
    predictions = {label: load_predictions(cell_paths[label]) for label in CELL_ORDER}
    for label, values in predictions.items():
        require_parallel(cases, values, label)

    groups = {
        "All": list(range(len(cases))),
        "AG": [i for i, row in enumerate(raw_rows) if row.get("source_config") == "Algorithm-Generated"],
        "HC": [i for i, row in enumerate(raw_rows) if row.get("source_config") != "Algorithm-Generated"],
        "HC-long": [
            i
            for i, (row, case) in enumerate(zip(raw_rows, cases, strict=True))
            if row.get("source_config") != "Algorithm-Generated" and len(case.steps) > 50
        ],
    }
    summaries: dict[str, dict[str, Any]] = {}
    for group, indices in groups.items():
        group_cases = [cases[i] for i in indices]
        summaries[group] = {
            label: summarize(group_cases, [predictions[label][i] for i in indices])
            for label in CELL_ORDER
        }

    contrast_pairs = {
        "Generator effect with light localizer (L->L vs G->L)": ("L->L", "G->L"),
        "Generator effect with strong localizer (L->G vs G->G)": ("L->G", "G->G"),
        "Localizer effect with light requirements (L->L vs L->G)": ("L->L", "L->G"),
        "Localizer effect with strong requirements (G->L vs G->G)": ("G->L", "G->G"),
    }
    paired_contrasts = {
        name: {
            metric: paired_metric(cases, predictions[left], predictions[right], metric)
            for metric in ("agent", "step")
        }
        for name, (left, right) in contrast_pairs.items()
    }

    report = {
        "light_model": args.light_model,
        "light_provider": args.light_provider,
        "strong_model": args.strong_model,
        "cells": {label: str(path.resolve(strict=False)) for label, path in cell_paths.items()},
        "summaries": summaries,
        "paired_contrasts": paired_contrasts,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    markdown = render_markdown(report)
    out.with_suffix(".md").write_text(markdown, encoding="utf-8")
    print(markdown)


if __name__ == "__main__":
    main()
