from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


METRICS = [
    ("agent", "agent_correct"),
    ("exact", "step_correct"),
    ("pm3", "step_pm3"),
    ("pm5", "step_pm5"),
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute paired significance for original vs wrapper summaries.")
    parser.add_argument("--data", default="data/who_and_when_all.jsonl")
    parser.add_argument("--out", default="outputs/paired_significance_report.md")
    parser.add_argument("--json-out", default="outputs/paired_significance_report.json")
    args = parser.parse_args()

    data_rows = load_jsonl(Path(args.data))
    meta_by_global_index = {idx: row for idx, row in enumerate(data_rows)}
    subset_indices = {
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
    reports = []

    reports.append(
        compare_pair(
            name="Who&When official",
            original_paths=[
                {
                    "path": "outputs/who_when_official_two/openai_g4o_all_who_when_official_two/who_when_official_all_at_once_combined_summary.json",
                    "index_scope": "all",
                }
            ],
            wrapper_paths=[
                {
                    "path": "outputs/who_when_official_two/openai_g4o_all_who_when_official_two/who_when_official_beam_joint_combined_summary.json",
                    "index_scope": "all",
                }
            ],
            meta_by_global_index=meta_by_global_index,
            subset_indices=subset_indices,
        )
    )
    reports.append(
        compare_pair(
            name="A2P official",
            original_paths=[
                {
                    "path": "outputs/who_and_when_all_gpt-4o_a2p_official_r1/a2p_official_summary.json",
                    "index_scope": "all",
                }
            ],
            wrapper_paths=[
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
            meta_by_global_index=meta_by_global_index,
            subset_indices=subset_indices,
        )
    )
    reports.append(
        compare_pair(
            name="ECHO strict GPT-4o HC-long",
            original_paths=[
                {
                    "path": "outputs/gpt4o_echo_strict_hc_long_only/wwlong_hc_openai_gpt-4o_echo_strict_s51+_tok4096_r1/echo_appendix_strict_original_combined_summary.json",
                    "index_scope": "hc_long",
                }
            ],
            wrapper_paths=[
                {
                    "path": "outputs/gpt4o_echo_strict_hc_long_only/wwlong_hc_openai_gpt-4o_echo_strict_s51+_tok4096_r1/echo_appendix_strict_panel_router_wrapper_combined_summary.json",
                    "index_scope": "hc_long",
                }
            ],
            meta_by_global_index=meta_by_global_index,
            subset_indices=subset_indices,
        )
    )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render_markdown(reports), encoding="utf-8")
    json_out = Path(args.json_out)
    json_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.write_text(json.dumps(reports, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {out}")
    print(f"Wrote {json_out}")


def compare_pair(
    name: str,
    original_paths: list[dict[str, str]],
    wrapper_paths: list[dict[str, str]],
    meta_by_global_index: dict[int, dict[str, Any]],
    subset_indices: dict[str, list[int]],
) -> dict[str, Any]:
    original = load_summary_rows(original_paths, subset_indices)
    wrapper = load_summary_rows(wrapper_paths, subset_indices)
    shared_ids = sorted(set(original) & set(wrapper))
    missing = {
        "original_only": sorted(set(original) - set(wrapper)),
        "wrapper_only": sorted(set(wrapper) - set(original)),
    }
    scopes = {
        "all_shared": shared_ids,
        "hc_long": [
            str(gid)
            for cid in shared_ids
            for gid in [int(cid)]
            if is_hc_long(meta_by_global_index.get(gid, {}))
        ],
    }
    result = {
        "name": name,
        "original_paths": original_paths,
        "wrapper_paths": wrapper_paths,
        "shared_count": len(shared_ids),
        "missing": {key: len(value) for key, value in missing.items()},
        "scopes": {},
    }
    for scope_name, ids in scopes.items():
        if not ids:
            continue
        result["scopes"][scope_name] = {
            metric_name: paired_metric(original, wrapper, ids, row_key)
            for metric_name, row_key in METRICS
        }
    return result


def paired_metric(
    original: dict[str, dict[str, Any]],
    wrapper: dict[str, dict[str, Any]],
    ids: list[str],
    row_key: str,
) -> dict[str, Any]:
    both = orig_only = wrap_only = neither = 0
    for cid in ids:
        o = bool(original[cid].get(row_key))
        w = bool(wrapper[cid].get(row_key))
        if o and w:
            both += 1
        elif o and not w:
            orig_only += 1
        elif (not o) and w:
            wrap_only += 1
        else:
            neither += 1
    n = len(ids)
    orig_correct = both + orig_only
    wrap_correct = both + wrap_only
    discordant = orig_only + wrap_only
    return {
        "n": n,
        "original_correct": orig_correct,
        "wrapper_correct": wrap_correct,
        "original_accuracy": orig_correct / n if n else None,
        "wrapper_accuracy": wrap_correct / n if n else None,
        "delta": (wrap_correct - orig_correct) / n if n else None,
        "both_correct": both,
        "both_wrong": neither,
        "original_only": orig_only,
        "wrapper_only": wrap_only,
        "discordant": discordant,
        "mcnemar_exact_two_sided_p": exact_binomial_two_sided(orig_only, wrap_only),
        "one_sided_wrapper_better_p": exact_binomial_one_sided_wrapper_better(orig_only, wrap_only),
    }


def exact_binomial_two_sided(b: int, c: int) -> float | None:
    n = b + c
    if n == 0:
        return None
    lo = min(b, c)
    return min(1.0, 2.0 * sum(binom_pmf(k, n, 0.5) for k in range(lo + 1)))


def exact_binomial_one_sided_wrapper_better(b: int, c: int) -> float | None:
    n = b + c
    if n == 0:
        return None
    # b = original-only wins, c = wrapper-only wins.
    return sum(binom_pmf(k, n, 0.5) for k in range(c, n + 1))


def binom_pmf(k: int, n: int, p: float) -> float:
    return math.comb(n, k) * (p**k) * ((1 - p) ** (n - k))


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


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def case_id(row: dict[str, Any]) -> str:
    return str(row.get("question_ID") or row.get("case_id") or row.get("id"))


def is_hc_long(row: dict[str, Any]) -> bool:
    return row.get("source_config") == "Hand-Crafted" and len(row.get("history") or []) > 50


def render_markdown(reports: list[dict[str, Any]]) -> str:
    lines = [
        "# Paired Significance Report",
        "",
        "McNemar exact p-values are computed from discordant pairs only.",
        "`original_only` means original was correct and wrapper was wrong; `wrapper_only` means wrapper was correct and original was wrong.",
        "",
    ]
    for report in reports:
        lines.extend(
            [
                f"## {report['name']}",
                "",
                f"- shared cases: {report['shared_count']}",
                f"- missing: {report['missing']}",
                "",
            ]
        )
        for scope_name, metrics in report["scopes"].items():
            lines.extend(
                [
                    f"### {scope_name}",
                    "",
                    "| metric | n | original | wrapper | delta pp | original_only | wrapper_only | two-sided p | one-sided p(wrapper better) |",
                    "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
                ]
            )
            for metric_name, row in metrics.items():
                lines.append(
                    "| {metric} | {n} | {orig} | {wrap} | {delta} | {orig_only} | {wrap_only} | {p2} | {p1} |".format(
                        metric=metric_name,
                        n=row["n"],
                        orig=fmt_pct(row["original_accuracy"]),
                        wrap=fmt_pct(row["wrapper_accuracy"]),
                        delta=fmt_pp(row["delta"]),
                        orig_only=row["original_only"],
                        wrap_only=row["wrapper_only"],
                        p2=fmt_p(row["mcnemar_exact_two_sided_p"]),
                        p1=fmt_p(row["one_sided_wrapper_better_p"]),
                    )
                )
            lines.append("")
    return "\n".join(lines)


def fmt_pct(value: float | None) -> str:
    return "-" if value is None else f"{value * 100:.2f}"


def fmt_pp(value: float | None) -> str:
    return "-" if value is None else f"{value * 100:+.2f}"


def fmt_p(value: float | None) -> str:
    return "-" if value is None else f"{value:.4f}"


if __name__ == "__main__":
    main()
