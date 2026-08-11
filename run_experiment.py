from __future__ import annotations

import argparse
import json
import traceback
import time
from pathlib import Path

from failure_attribution.config import load_config
from failure_attribution.io import load_cases, open_text, pred_to_dict, write_json, write_predictions, write_text_file
from failure_attribution.llm import build_llm, get_usage_snapshot, usage_delta
from failure_attribution.metrics import evaluate, usage_summary
from failure_attribution.methods import run_method
from failure_attribution.schema import Prediction


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Who&When-style failure attribution experiments.")
    parser.add_argument("--config", required=True, help="Path to TOML config.")
    parser.add_argument("--data", required=True, help="Path to input JSONL data.")
    parser.add_argument(
        "--methods",
        default="all_at_once,step_by_step,binary_search,paper_hybrid,mvbs10,ccv10",
        help="Comma-separated methods.",
    )
    parser.add_argument("--out", default="outputs/run", help="Output directory.")
    parser.add_argument("--limit", type=int, default=None, help="Optional case limit for debugging.")
    parser.add_argument("--offset", type=int, default=0, help="Optional starting case offset for debugging.")
    parser.add_argument("--repeats", type=int, default=1, help="Number of independent runs per method.")
    parser.add_argument("--continue-on-error", action="store_true", help="Record a failed case and continue.")
    args = parser.parse_args()

    config = load_config(args.config)
    generated_step_base = int(config.get("data", {}).get("generated_step_base", 1))
    cases = load_cases(args.data, generated_step_base=generated_step_base)
    if args.offset < 0:
        raise ValueError("--offset must be non-negative")
    if args.offset:
        cases = cases[args.offset :]
    if args.limit is not None:
        cases = cases[: args.limit]

    llm = build_llm(config.get("llm", {}))
    out_dir = Path(args.out)
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]

    if args.repeats < 1:
        raise ValueError("--repeats must be at least 1")

    manifest = {
        "config": args.config,
        "data": args.data,
        "methods": methods,
        "case_count": len(cases),
        "offset": args.offset,
        "limit": args.limit,
        "repeats": args.repeats,
        "started_at_epoch": time.time(),
    }
    write_json(out_dir / "manifest.json", manifest)

    for method in methods:
        repeat_predictions = []
        for repeat_idx in range(1, args.repeats + 1):
            start = time.time()
            predictions = []
            pred_name = f"{method}_predictions.jsonl" if args.repeats == 1 else f"{method}_run{repeat_idx}_predictions.jsonl"
            pred_path = out_dir / pred_name
            write_text_file(pred_path, "", encoding="utf-8")
            for idx, case in enumerate(cases, 1):
                print(f"[{method} run {repeat_idx}/{args.repeats}] {idx}/{len(cases)} {case.case_id}", flush=True)
                usage_before = get_usage_snapshot(llm)
                try:
                    pred = run_method(method, case, llm, config)
                except Exception as exc:  # noqa: BLE001
                    clear_cuda_cache()
                    if not args.continue_on_error:
                        raise
                    tb = traceback.format_exc()
                    print(f"[{method}] ERROR on {case.case_id}: {type(exc).__name__}: {exc}", flush=True)
                    pred = Prediction(
                        case_id=case.case_id,
                        method=method,
                        agent=None,
                        step=None,
                        confidence=None,
                        reason=f"ERROR: {type(exc).__name__}: {exc}",
                        trace={"error_type": type(exc).__name__, "error": str(exc), "traceback": tb},
                    )
                usage_after = get_usage_snapshot(llm)
                pred.trace = pred.trace or {}
                pred.trace["usage"] = usage_delta(usage_before, usage_after)
                predictions.append(pred)
                append_prediction(pred_path, pred)
            elapsed = time.time() - start
            repeat_predictions.append(predictions)

            summary_name = f"{method}_summary.json" if args.repeats == 1 else f"{method}_run{repeat_idx}_summary.json"
            write_predictions(out_dir / pred_name, predictions)
            summary = evaluate(cases, predictions)
            summary["method"] = method
            summary["repeat"] = repeat_idx
            summary["elapsed_seconds"] = elapsed
            summary["usage"] = usage_summary(predictions)
            write_json(out_dir / summary_name, summary)
            print_summary(summary)

        if args.repeats > 1:
            stability = stability_summary(method, repeat_predictions)
            write_json(out_dir / f"{method}_stability.json", stability)
            print_stability(stability)

    print(f"Done. Outputs written to {out_dir}")


def print_summary(summary: dict) -> None:
    compact = {
        "method": summary["method"],
        "count": summary["count"],
        "agent_accuracy": summary["agent_accuracy"],
        "step_accuracy": summary["step_accuracy"],
        "step_pm3_accuracy": summary["step_pm3_accuracy"],
        "step_pm5_accuracy": summary["step_pm5_accuracy"],
        "mean_abs_distance": summary["mean_abs_distance"],
        "elapsed_seconds": round(summary["elapsed_seconds"], 2),
    }
    usage = summary.get("usage") or {}
    if usage:
        compact["llm_calls"] = usage.get("llm_calls")
        compact["total_tokens"] = usage.get("total_tokens")
    print(compact, flush=True)


def append_prediction(path: Path, pred: Prediction) -> None:
    with open_text(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(pred_to_dict(pred), ensure_ascii=False) + "\n")


def clear_cuda_cache() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception:
        pass


def stability_summary(method: str, repeat_predictions: list[list]) -> dict:
    by_case: dict[str, list[tuple[str | None, int | None]]] = {}
    for predictions in repeat_predictions:
        for pred in predictions:
            by_case.setdefault(pred.case_id, []).append((pred.agent, pred.step))

    rows = []
    for case_id, values in sorted(by_case.items()):
        unique_values = sorted({value for value in values}, key=lambda x: (str(x[0]), -1 if x[1] is None else x[1]))
        rows.append(
            {
                "case_id": case_id,
                "predictions": [{"agent": agent, "step": step} for agent, step in values],
                "unique_prediction_count": len(unique_values),
                "stable": len(unique_values) == 1,
            }
        )
    stable_count = sum(1 for row in rows if row["stable"])
    return {
        "method": method,
        "repeat_count": len(repeat_predictions),
        "case_count": len(rows),
        "stable_case_count": stable_count,
        "stable_rate": stable_count / len(rows) if rows else None,
        "rows": rows,
    }


def print_stability(stability: dict) -> None:
    compact = {
        "method": stability["method"],
        "repeat_count": stability["repeat_count"],
        "stable_rate": stability["stable_rate"],
        "stable_case_count": stability["stable_case_count"],
        "case_count": stability["case_count"],
    }
    print(compact, flush=True)


if __name__ == "__main__":
    main()
