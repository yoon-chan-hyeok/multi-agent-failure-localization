from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any


METRICS = [
    ("agent", "agent_correct"),
    ("exact", "step_correct"),
    ("pm3", "step_pm3"),
    ("pm5", "step_pm5"),
]


COMPARISONS = [
    {
        "name": "Who&When official",
        "original_paths": [
            {
                "path": "outputs/who_when_official_two/openai_g4o_all_who_when_official_two/who_when_official_all_at_once_combined_summary.json",
                "index_scope": "all",
            }
        ],
        "wrapper_paths": [
            {
                "path": "outputs/who_when_official_two/openai_g4o_all_who_when_official_two/who_when_official_beam_joint_combined_summary.json",
                "index_scope": "all",
            }
        ],
    },
    {
        "name": "A2P official",
        "original_paths": [
            {
                "path": "outputs/who_and_when_all_gpt-4o_a2p_official_r1/a2p_official_summary.json",
                "index_scope": "all",
            }
        ],
        "wrapper_paths": [
            {
                "path": "outputs/fixed_v4_beam_joint_rest/gpt_g4o_ag_a2pbeamj/a2p_official_beam_joint_combined_summary.json",
                "index_scope": "ag",
            },
            {
                "path": "outputs/fixed_v4_beam_joint_rest/gpt_g4o_hc_rest_a2pbeamj/a2p_official_beam_joint_combined_summary.json",
                "index_scope": "hc_nonlong",
            },
            {
                "path": "outputs/fixed_v4_new3/who_and_when_long_hand-crafted_gpt-4o_a2p_official_beam_joint_beam25pct_k3-8_chunk_s6_t3000_max28_maxtok4096_r1/a2p_official_beam_joint_combined_summary.json",
                "index_scope": "hc_long",
            },
        ],
    },
    {
        "name": "ECHO strict GPT-4o HC-long",
        "original_paths": [
            {
                "path": "outputs/gpt4o_echo_strict_hc_long_only/wwlong_hc_openai_gpt-4o_echo_strict_s51+_tok4096_r1/echo_appendix_strict_original_combined_summary.json",
                "index_scope": "hc_long",
            }
        ],
        "wrapper_paths": [
            {
                "path": "outputs/gpt4o_echo_strict_hc_long_only/wwlong_hc_openai_gpt-4o_echo_strict_s51+_tok4096_r1/echo_appendix_strict_panel_router_wrapper_combined_summary.json",
                "index_scope": "hc_long",
            }
        ],
    },
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Paired bootstrap confidence intervals for wrapper deltas.")
    parser.add_argument("--data", default="data/who_and_when_all.jsonl")
    parser.add_argument("--samples", type=int, default=50000)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--out", default="outputs/bootstrap_ci_report.md")
    parser.add_argument("--json-out", default="outputs/bootstrap_ci_report.json")
    args = parser.parse_args()

    data_rows = load_jsonl(Path(args.data))
    subset_indices = build_subset_indices(data_rows)
    meta_by_global_index = {idx: row for idx, row in enumerate(data_rows)}
    rng = random.Random(args.seed)

    reports = []
    for spec in COMPARISONS:
        reports.append(
            compare_pair(
                spec,
                subset_indices=subset_indices,
                meta_by_global_index=meta_by_global_index,
                samples=args.samples,
                rng=rng,
            )
        )

    result = {
        "data": args.data,
        "samples": args.samples,
        "seed": args.seed,
        "reports": reports,
    }
    json_out = Path(args.json_out)
    json_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render_markdown(result), encoding="utf-8")
    print(f"Wrote {out}")
    print(f"Wrote {json_out}")


def compare_pair(
    spec: dict[str, Any],
    subset_indices: dict[str, list[int]],
    meta_by_global_index: dict[int, dict[str, Any]],
    samples: int,
    rng: random.Random,
) -> dict[str, Any]:
    original = load_summary_rows(spec["original_paths"], subset_indices)
    wrapper = load_summary_rows(spec["wrapper_paths"], subset_indices)
    shared = sorted(set(original) & set(wrapper), key=int)
    scopes = {
        "all_shared": shared,
        "hc_long": [
            key
            for key in shared
            if is_hc_long(meta_by_global_index.get(int(key), {}))
        ],
    }
    report = {
        "name": spec["name"],
        "shared_count": len(shared),
        "scopes": {},
    }
    for scope_name, keys in scopes.items():
        if not keys:
            continue
        report["scopes"][scope_name] = {}
        for metric_name, row_key in METRICS:
            diffs = [
                int(bool(wrapper[key].get(row_key))) - int(bool(original[key].get(row_key)))
                for key in keys
            ]
            report["scopes"][scope_name][metric_name] = bootstrap_delta(diffs, samples=samples, rng=rng)
    return report


def bootstrap_delta(diffs: list[int], samples: int, rng: random.Random) -> dict[str, Any]:
    n = len(diffs)
    observed = sum(diffs) / n if n else 0.0
    values = []
    positive = nonnegative = 0
    for _ in range(samples):
        total = 0
        for _ in range(n):
            total += diffs[rng.randrange(n)]
        value = total / n
        values.append(value)
        positive += int(value > 0)
        nonnegative += int(value >= 0)
    values.sort()
    return {
        "n": n,
        "observed_delta": observed,
        "ci95_low": percentile(values, 0.025),
        "ci95_high": percentile(values, 0.975),
        "prob_delta_gt_0": positive / samples,
        "prob_delta_ge_0": nonnegative / samples,
        "samples": samples,
    }


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    pos = q * (len(values) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(values) - 1)
    frac = pos - lo
    return values[lo] * (1.0 - frac) + values[hi] * frac


def load_summary_rows(specs: list[dict[str, str]], subset_indices: dict[str, list[int]]) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for spec in specs:
        path = Path(spec["path"])
        index_scope = spec.get("index_scope") or "all"
        index_map = subset_indices[index_scope]
        if not path.exists():
            raise FileNotFoundError(path)
        summary = json.loads(path.read_text(encoding="utf-8"))
        for row in summary.get("rows") or []:
            subset_idx = int(row.get("case_index"))
            if subset_idx < 0 or subset_idx >= len(index_map):
                raise IndexError(f"case_index {subset_idx} out of range for {index_scope}: {path}")
            global_idx = index_map[subset_idx]
            key = str(global_idx)
            if key in rows:
                raise ValueError(f"Duplicate global case index in input summaries: {key}")
            enriched = dict(row)
            enriched["global_case_index"] = global_idx
            enriched["summary_case_index"] = subset_idx
            rows[key] = enriched
    return rows


def build_subset_indices(data_rows: list[dict[str, Any]]) -> dict[str, list[int]]:
    return {
        "all": list(range(len(data_rows))),
        "ag": [idx for idx, row in enumerate(data_rows) if row.get("source_config") == "Algorithm-Generated"],
        "hc_nonlong": [
            idx
            for idx, row in enumerate(data_rows)
            if row.get("source_config") == "Hand-Crafted" and len(row.get("history") or []) <= 50
        ],
        "hc_long": [
            idx
            for idx, row in enumerate(data_rows)
            if is_hc_long(row)
        ],
    }


def is_hc_long(row: dict[str, Any]) -> bool:
    return row.get("source_config") == "Hand-Crafted" and len(row.get("history") or []) > 50


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def render_markdown(result: dict[str, Any]) -> str:
    lines = [
        "# Bootstrap CI Report",
        "",
        f"- samples: {result['samples']}",
        f"- seed: {result['seed']}",
        "",
        "CI is for wrapper minus original accuracy, using paired case-level bootstrap resampling.",
        "",
    ]
    for report in result["reports"]:
        lines.extend([f"## {report['name']}", "", f"- shared cases: {report['shared_count']}", ""])
        for scope_name, metrics in report["scopes"].items():
            lines.extend(
                [
                    f"### {scope_name}",
                    "",
                    "| metric | n | observed delta pp | 95% CI pp | P(delta > 0) |",
                    "| --- | ---: | ---: | ---: | ---: |",
                ]
            )
            for metric_name, row in metrics.items():
                lines.append(
                    "| {metric} | {n} | {delta} | [{lo}, {hi}] | {prob} |".format(
                        metric=metric_name,
                        n=row["n"],
                        delta=fmt_pp(row["observed_delta"]),
                        lo=fmt_pp(row["ci95_low"]),
                        hi=fmt_pp(row["ci95_high"]),
                        prob=f"{row['prob_delta_gt_0']:.3f}",
                    )
                )
            lines.append("")
    return "\n".join(lines)


def fmt_pp(value: float) -> str:
    return f"{value * 100:+.2f}"


if __name__ == "__main__":
    main()
