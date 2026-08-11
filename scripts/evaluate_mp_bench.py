from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate one or more failure-attribution prediction files with MP-Bench multi-perspective labels."
    )
    parser.add_argument("--data", required=True, help="Converted MP-Bench JSONL.")
    parser.add_argument(
        "--predictions",
        nargs="+",
        required=True,
        help="Prediction JSONL files. Multiple files are treated as repeated independent runs.",
    )
    parser.add_argument("--out", required=True, help="Output JSON report.")
    parser.add_argument("--csv", default=None, help="Optional case-level CSV output.")
    parser.add_argument(
        "--method",
        default=None,
        help="Optional method filter when a prediction file contains multiple methods.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cases = load_jsonl(Path(args.data))
    predictions = load_predictions([Path(path) for path in args.predictions], method=args.method)
    report, rows = evaluate(cases, predictions)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.csv:
        write_csv(Path(args.csv), rows)

    print(json.dumps(report["prediction_subset"]["overall"], ensure_ascii=False, indent=2))
    print(f"Report: {out_path}")
    if args.csv:
        print(f"Case rows: {args.csv}")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def load_predictions(
    paths: list[Path],
    *,
    method: str | None,
) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for run_index, path in enumerate(paths, 1):
        for row_index, row in enumerate(load_jsonl(path)):
            if method and str(row.get("method")) != method:
                continue
            case_id = str(row.get("case_id") or "").strip()
            if not case_id:
                continue
            grouped[case_id].append(
                {
                    **row,
                    "_run_index": run_index,
                    "_row_index": row_index,
                    "_source_path": str(path),
                }
            )
    return grouped


def evaluate(
    cases: list[dict[str, Any]],
    predictions: dict[str, list[dict[str, Any]]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    case_rows: list[dict[str, Any]] = []
    for case in cases:
        case_id = str(case.get("id") or case.get("case_id"))
        case_predictions = predictions.get(case_id, [])
        case_rows.append(evaluate_case(case, case_predictions))

    summaries = {
        "overall": summarize(case_rows),
        "manual": summarize([row for row in case_rows if row["configuration"] == "manual"]),
        "automatic": summarize([row for row in case_rows if row["configuration"] == "automatic"]),
    }
    predicted_rows = [row for row in case_rows if row["valid_prediction_count"] > 0]
    prediction_subset = {
        "overall": summarize(predicted_rows),
        "manual": summarize(
            [row for row in predicted_rows if row["configuration"] == "manual"]
        ),
        "automatic": summarize(
            [row for row in predicted_rows if row["configuration"] == "automatic"]
        ),
    }
    report = {
        "benchmark": "MP-Bench",
        "evaluation_semantics": {
            "top1_any_expert_hit": "Top-ranked predicted step was selected by at least one expert.",
            "top1_majority_hit": "Top-ranked predicted step was selected by at least two of three experts.",
            "top1_unanimous_hit": "Top-ranked predicted step was selected by all three experts.",
            "expected_expert_agreement": "Human vote fraction for the top-ranked predicted step.",
            "pair_any_expert_hit": (
                "Top-ranked predicted step was selected by an expert and the predicted agent matches "
                "the trace role at that step."
            ),
            "ndcg": (
                "Predicted steps are ranked by frequency across supplied runs. Human relevance is the "
                "number of experts selecting each step. Linear gain uses relevance directly; "
                "exponential gain uses 2^relevance-1."
            ),
        },
        "prediction_files": sorted(
            {
                pred["_source_path"]
                for case_predictions in predictions.values()
                for pred in case_predictions
            }
        ),
        "prediction_subset": prediction_subset,
        **summaries,
        "rows": case_rows,
    }
    return report, case_rows


def evaluate_case(case: dict[str, Any], predictions: list[dict[str, Any]]) -> dict[str, Any]:
    mp = case.get("mp_bench") or {}
    human_votes = {
        int(item["step"]): int(item["votes"])
        for item in mp.get("consensus_ranking", [])
    }
    human_rank = [
        step for step, _ in sorted(human_votes.items(), key=lambda item: (-item[1], item[0]))
    ]
    step_roles = {
        int(step["step"]): str(step.get("agent") or step.get("name") or step.get("role") or "Unknown")
        for step in case.get("history", [])
    }

    valid_predictions: list[dict[str, Any]] = []
    invalid_prediction_count = 0
    for order, pred in enumerate(predictions):
        step = parse_int(pred.get("step"))
        if step is None or step not in step_roles:
            invalid_prediction_count += 1
            continue
        valid_predictions.append({**pred, "_parsed_step": step, "_order": order})

    predicted_counts = Counter(pred["_parsed_step"] for pred in valid_predictions)
    first_order = {
        step: min(pred["_order"] for pred in valid_predictions if pred["_parsed_step"] == step)
        for step in predicted_counts
    }
    predicted_rank = [
        step
        for step, _ in sorted(
            predicted_counts.items(),
            key=lambda item: (-item[1], first_order[item[0]], item[0]),
        )
    ]
    top_step = predicted_rank[0] if predicted_rank else None
    top_pred = next(
        (pred for pred in valid_predictions if pred["_parsed_step"] == top_step),
        None,
    )
    top_agent = optional_text(top_pred.get("agent")) if top_pred else None
    top_role = step_roles.get(top_step) if top_step is not None else None
    top_votes = human_votes.get(top_step, 0) if top_step is not None else 0
    max_human_votes = max(human_votes.values()) if human_votes else None
    has_human_labels = bool(human_votes)

    full_k = max(len(human_rank), len(predicted_rank), 1)
    return {
        "case_id": str(case.get("id") or case.get("case_id")),
        "configuration": str(case.get("source_config") or ""),
        "source_dataset": str(case.get("source_dataset") or ""),
        "step_count": len(step_roles),
        "human_annotated_step_count": len(human_rank),
        "human_ranked_steps": human_rank,
        "human_step_votes": human_votes,
        "prediction_count": len(predictions),
        "valid_prediction_count": len(valid_predictions),
        "invalid_prediction_count": invalid_prediction_count,
        "predicted_ranked_steps": predicted_rank,
        "predicted_step_counts": dict(predicted_counts),
        "top1_step": top_step,
        "top1_agent": top_agent,
        "top1_trace_role": top_role,
        "top1_human_votes": top_votes,
        "top1_any_expert_hit": bool(top_step is not None and top_votes >= 1) if has_human_labels else None,
        "top1_majority_hit": bool(top_step is not None and top_votes >= 2) if has_human_labels else None,
        "top1_unanimous_hit": bool(top_step is not None and top_votes == 3) if has_human_labels else None,
        "top_consensus_tier_hit": (
            bool(top_step is not None and top_votes == max_human_votes)
            if has_human_labels
            else None
        ),
        "expected_expert_agreement": (
            top_votes / 3.0 if has_human_labels and top_step is not None else (0.0 if has_human_labels else None)
        ),
        "agent_matches_predicted_step_role": bool(
            top_agent and top_role and normalize_agent(top_agent) == normalize_agent(top_role)
        ),
        "agent_any_expert_hit": (
            bool(
                top_agent
                and any(
                    normalize_agent(top_agent) == normalize_agent(step_roles[step])
                    for step in human_rank
                )
            )
            if has_human_labels
            else None
        ),
        "pair_any_expert_hit": (
            bool(
                top_step is not None
                and top_votes >= 1
                and top_agent
                and top_role
                and normalize_agent(top_agent) == normalize_agent(top_role)
            )
            if has_human_labels
            else None
        ),
        "top_consensus_pair_hit": (
            bool(
                top_step is not None
                and top_votes == max_human_votes
                and top_agent
                and top_role
                and normalize_agent(top_agent) == normalize_agent(top_role)
            )
            if has_human_labels
            else None
        ),
        "ndcg_at_5_linear": (
            ndcg(predicted_rank, human_votes, k=5, exponential=False) if has_human_labels else None
        ),
        "ndcg_at_5_exponential": (
            ndcg(predicted_rank, human_votes, k=5, exponential=True) if has_human_labels else None
        ),
        "ndcg_full_linear": (
            ndcg(predicted_rank, human_votes, k=full_k, exponential=False) if has_human_labels else None
        ),
        "ndcg_full_exponential": (
            ndcg(predicted_rank, human_votes, k=full_k, exponential=True) if has_human_labels else None
        ),
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    metric_keys = [
        "top1_any_expert_hit",
        "top1_majority_hit",
        "top1_unanimous_hit",
        "top_consensus_tier_hit",
        "expected_expert_agreement",
        "agent_matches_predicted_step_role",
        "agent_any_expert_hit",
        "pair_any_expert_hit",
        "top_consensus_pair_hit",
        "ndcg_at_5_linear",
        "ndcg_at_5_exponential",
        "ndcg_full_linear",
        "ndcg_full_exponential",
    ]
    return {
        "count": len(rows),
        "labeled_case_count": sum(bool(row["human_ranked_steps"]) for row in rows),
        "unlabeled_case_count": sum(not bool(row["human_ranked_steps"]) for row in rows),
        "prediction_coverage": (
            mean(1.0 if row["valid_prediction_count"] > 0 else 0.0 for row in rows)
            if rows
            else None
        ),
        "avg_prediction_runs_per_case": (
            mean(row["prediction_count"] for row in rows) if rows else None
        ),
        **{
            key: average_optional(row[key] for row in rows)
            for key in metric_keys
        },
    }


def ndcg(
    predicted_rank: list[int],
    human_votes: dict[int, int],
    *,
    k: int,
    exponential: bool,
) -> float:
    if not human_votes:
        return 0.0
    cutoff = max(1, k)
    gains = [
        gain(human_votes.get(step, 0), exponential=exponential)
        for step in predicted_rank[:cutoff]
    ]
    dcg = sum(value / math.log2(rank + 2) for rank, value in enumerate(gains))
    ideal_relevances = sorted(human_votes.values(), reverse=True)[:cutoff]
    idcg = sum(
        gain(relevance, exponential=exponential) / math.log2(rank + 2)
        for rank, relevance in enumerate(ideal_relevances)
    )
    return dcg / idcg if idcg > 0 else 0.0


def gain(relevance: int, *, exponential: bool) -> float:
    return float((2**relevance) - 1 if exponential else relevance)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "case_id",
        "configuration",
        "source_dataset",
        "step_count",
        "human_annotated_step_count",
        "prediction_count",
        "valid_prediction_count",
        "top1_step",
        "top1_agent",
        "top1_trace_role",
        "top1_human_votes",
        "top1_any_expert_hit",
        "top1_majority_hit",
        "top1_unanimous_hit",
        "top_consensus_tier_hit",
        "expected_expert_agreement",
        "agent_matches_predicted_step_role",
        "agent_any_expert_hit",
        "pair_any_expert_hit",
        "top_consensus_pair_hit",
        "ndcg_at_5_linear",
        "ndcg_at_5_exponential",
        "ndcg_full_linear",
        "ndcg_full_exponential",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def normalize_agent(value: str) -> str:
    return reformat_agent(value).lower()


def reformat_agent(value: str) -> str:
    value = str(value).strip()
    if "(" in value:
        value = value.split("(", 1)[0].strip()
    return "".join(char for char in value if char.isalnum())


def parse_int(value: Any) -> int | None:
    try:
        if value is None or isinstance(value, bool):
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def optional_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def average_optional(values: Any) -> float | None:
    usable = [float(value) for value in values if value is not None]
    return mean(usable) if usable else None


if __name__ == "__main__":
    main()
