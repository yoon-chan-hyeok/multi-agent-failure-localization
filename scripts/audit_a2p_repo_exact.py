from __future__ import annotations

import argparse
import ast
import hashlib
import json
import subprocess
import sys
import tomllib
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from failure_attribution.io import load_cases  # noqa: E402
from failure_attribution.prompts import a2p_repo_exact_prompt  # noqa: E402


A2P_COMMIT = "7953d780c85054721a7b4bf246bcf60a16bb28af"
A2P_SOURCE_PATH = "Automated_FA/Lib/utils.py"
A2P_SYSTEM_PROMPT = "You are a helpful assistant skilled in analyzing conversations."


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit the pinned A2P repository-exact prompt implementation.")
    parser.add_argument("--repo", default=str(ROOT / ".external" / "A2P"))
    parser.add_argument("--who-when", default=str(ROOT / "data" / "who_and_when_all.jsonl"))
    parser.add_argument(
        "--mast",
        default=str(ROOT / "data" / "mp_bench_who_when_style_manual_mast_positive.jsonl"),
    )
    parser.add_argument(
        "--who-config",
        default=str(ROOT / "outputs" / "who_and_when_a2p_repo_exact_gpt4o" / "run_config.toml"),
    )
    parser.add_argument(
        "--mast-config",
        default=str(ROOT / "outputs" / "mp_bench_a2p_repo_exact_mast_positive_manual" / "run_config.toml"),
    )
    parser.add_argument(
        "--json-out",
        default=str(ROOT / "docs" / "a2p_repo_exact_audit.json"),
    )
    parser.add_argument(
        "--md-out",
        default=str(ROOT / "docs" / "a2p_repo_exact_audit_ko.md"),
    )
    args = parser.parse_args()

    repo = Path(args.repo).resolve()
    source, head, blob = pinned_repository_source(repo)
    construct = extract_construct_function(source)

    who_result = audit_who_when(Path(args.who_when), construct)
    mast_result = audit_mast(Path(args.mast), construct)
    config_results = [
        audit_config("Who&When", Path(args.who_config)),
        audit_config("MAST 100", Path(args.mast_config)),
    ]

    report = {
        "status": "PASS"
        if (
            head == A2P_COMMIT
            and who_result["mismatch_count"] == 0
            and mast_result["mismatch_count"] == 0
            and all(item["system_prompt_match"] and item["temperature_omitted"] for item in config_results)
        )
        else "FAIL",
        "repository": {
            "path": str(repo),
            "origin": "https://github.com/ResearAI/A2P",
            "expected_commit": A2P_COMMIT,
            "actual_head": head,
            "source_path": A2P_SOURCE_PATH,
            "source_blob": blob,
            "working_tree_note": (
                "The checkout has local transport edits. This audit reads the pinned Git object with git show, "
                "not the working-tree file."
            ),
        },
        "who_and_when": who_result,
        "mast": mast_result,
        "configs": config_results,
        "intentional_differences": [
            {
                "item": "Model",
                "repository_reported_setting": "gpt-oss-120b for the README headline results",
                "controlled_experiment_setting": "gpt-4o",
                "implication": "This is not a numerical reproduction of the A2P paper/README result.",
            },
            {
                "item": "Maximum completion tokens",
                "repository_reported_setting": "CLI default 20000",
                "controlled_experiment_setting": "4096",
                "implication": "The prompt requests a response under 150 words, so this should not truncate a valid response.",
            },
            {
                "item": "API backend",
                "repository_reported_setting": "Azure OpenAI or OpenAI-compatible endpoint",
                "controlled_experiment_setting": "OpenAI-compatible endpoint",
                "implication": "Message roles and request fields are matched; transport implementation differs.",
            },
            {
                "item": "Parser and evaluator",
                "repository_reported_setting": "Repository regex parser and substring-based equality",
                "controlled_experiment_setting": "Unified parser and strict integer/normalized-agent equality",
                "implication": "Scores are stricter and must be labeled as a unified-evaluator rerun.",
            },
            {
                "item": "MAST serialization",
                "repository_reported_setting": "Not present in the A2P repository",
                "controlled_experiment_setting": "MP-Bench agent field mapped to the repository Agent_Name slot",
                "implication": "MAST is an external transfer evaluation, not an official A2P benchmark reproduction.",
            },
        ],
        "repository_bug_note": (
            "The synchronous CLI passes the string value of --is_handcrafted directly, so the literal string "
            "'False' is truthy in Python. The async path converts it to a boolean and implements the documented "
            "AG=name, HC=role behavior. The local adapter follows that intended async behavior."
        ),
    }

    json_out = Path(args.json_out)
    md_out = Path(args.md_out)
    json_out.parent.mkdir(parents=True, exist_ok=True)
    md_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    md_out.write_text(render_markdown(report), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if report["status"] != "PASS":
        raise SystemExit(1)


def pinned_repository_source(repo: Path) -> tuple[str, str, str]:
    if not (repo / ".git").exists():
        raise FileNotFoundError(f"A2P repository checkout not found: {repo}")
    safe = str(repo).replace("\\", "/")
    base = ["git", "-c", f"safe.directory={safe}", "-C", str(repo)]
    head = subprocess.check_output(base + ["rev-parse", "HEAD"], text=True, encoding="utf-8").strip()
    source = subprocess.check_output(
        base + ["show", f"{A2P_COMMIT}:{A2P_SOURCE_PATH}"],
        text=True,
        encoding="utf-8",
    )
    blob = subprocess.check_output(
        base + ["rev-parse", f"{A2P_COMMIT}:{A2P_SOURCE_PATH}"],
        text=True,
        encoding="utf-8",
    ).strip()
    return source, head, blob


def extract_construct_function(source: str) -> Callable[..., str]:
    module = ast.parse(source)
    function = next(
        node
        for node in module.body
        if isinstance(node, ast.FunctionDef) and node.name == "construct_a2p_prompt"
    )
    namespace: dict[str, Any] = {}
    exec(compile(ast.Module(body=[function], type_ignores=[]), "<a2p-repository>", "exec"), namespace)
    return namespace["construct_a2p_prompt"]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def audit_who_when(path: Path, construct: Callable[..., str]) -> dict[str, Any]:
    raw_rows = read_jsonl(path)
    cases = load_cases(path, generated_step_base=0)
    if len(raw_rows) != len(cases):
        raise ValueError("Who&When raw and parsed case counts differ.")

    mismatches = []
    split_counts = {"AG": 0, "HC": 0}
    prompt_hashes = []
    for index, (raw, case) in enumerate(zip(raw_rows, cases)):
        handcrafted = raw.get("source_config") == "Hand-Crafted"
        split = "HC" if handcrafted else "AG"
        split_counts[split] += 1
        index_agent = "role" if handcrafted else "name"
        ground_truth = raw.get("ground_truth")
        if ground_truth is None:
            ground_truth = raw.get("groundtruth", "")
        expected = construct(
            problem=raw.get("question", ""),
            ground_truth=ground_truth,
            chat_content="unused",
            chat_history=raw.get("history", []),
            index_agent=index_agent,
            a2p=True,
        )
        actual = a2p_repo_exact_prompt(case)
        prompt_hashes.append(hashlib.sha256(actual.encode("utf-8")).hexdigest())
        if expected != actual:
            mismatches.append(prompt_mismatch(index, case.case_id, split, expected, actual))

    return {
        "data": str(path),
        "case_count": len(cases),
        "split_counts": split_counts,
        "exact_prompt_match_count": len(cases) - len(mismatches),
        "mismatch_count": len(mismatches),
        "mismatch_examples": mismatches[:5],
        "unique_prompt_hashes": len(set(prompt_hashes)),
        "adapter": "HC groundtruth is normalized to the repository ground_truth field before prompt construction.",
    }


def audit_mast(path: Path, construct: Callable[..., str]) -> dict[str, Any]:
    raw_rows = read_jsonl(path)
    cases = load_cases(path, generated_step_base=0)
    if len(raw_rows) != len(cases):
        raise ValueError("MAST raw and parsed case counts differ.")

    mismatches = []
    step_coordinate_failures = []
    for index, (raw, case) in enumerate(zip(raw_rows, cases)):
        adapted_history = [
            {
                "name": item.get("agent", "Unknown Agent"),
                "content": item.get("content", ""),
            }
            for item in raw.get("history", [])
        ]
        expected = construct(
            problem=raw.get("question", ""),
            ground_truth=raw.get("ground_truth", ""),
            chat_content="unused",
            chat_history=adapted_history,
            index_agent="name",
            a2p=True,
        )
        actual = a2p_repo_exact_prompt(case)
        if expected != actual:
            mismatches.append(prompt_mismatch(index, case.case_id, "MAST", expected, actual))
        raw_steps = [item.get("step") for item in raw.get("history", [])]
        if raw_steps != list(range(len(raw_steps))):
            step_coordinate_failures.append(case.case_id)

    return {
        "data": str(path),
        "case_count": len(cases),
        "exact_prompt_match_count_after_declared_adapter": len(cases) - len(mismatches),
        "mismatch_count": len(mismatches),
        "mismatch_examples": mismatches[:5],
        "contiguous_zero_based_step_count": len(cases) - len(step_coordinate_failures),
        "step_coordinate_failures": step_coordinate_failures[:5],
        "adapter": "Each MP-Bench history item.agent is mapped to the repository Agent_Name/name slot.",
    }


def prompt_mismatch(index: int, case_id: str, split: str, expected: str, actual: str) -> dict[str, Any]:
    position = next(
        (pos for pos, (left, right) in enumerate(zip(expected, actual)) if left != right),
        min(len(expected), len(actual)),
    )
    return {
        "index": index,
        "case_id": case_id,
        "split": split,
        "position": position,
        "expected_excerpt": expected[position : position + 160],
        "actual_excerpt": actual[position : position + 160],
    }


def audit_config(label: str, path: Path) -> dict[str, Any]:
    if not path.exists():
        return {
            "label": label,
            "path": str(path),
            "exists": False,
            "system_prompt_match": False,
            "temperature_omitted": False,
        }
    config = tomllib.loads(path.read_text(encoding="utf-8"))
    llm = config.get("llm", {})
    return {
        "label": label,
        "path": str(path),
        "exists": True,
        "backend": llm.get("backend"),
        "model": llm.get("model"),
        "system_prompt": llm.get("system_prompt"),
        "system_prompt_match": llm.get("system_prompt") == A2P_SYSTEM_PROMPT,
        "temperature_value": llm.get("temperature"),
        "temperature_omitted": bool(llm.get("omit_temperature", False)),
        "max_tokens": llm.get("max_tokens"),
    }


def render_markdown(report: dict[str, Any]) -> str:
    who = report["who_and_when"]
    mast = report["mast"]
    lines = [
        "# A2P Repository-Exact 감사 보고서",
        "",
        f"**최종 판정: {report['status']}**",
        "",
        "## 고정 원문",
        "",
        f"- 저장소: `{report['repository']['origin']}`",
        f"- 커밋: `{report['repository']['actual_head']}`",
        f"- 원문 파일 blob: `{report['repository']['source_blob']}`",
        f"- 원문 함수: `{A2P_SOURCE_PATH}::construct_a2p_prompt(a2p=True)`",
        "",
        "로컬 작업 트리에는 API transport 수정이 있으므로, 감사 과정은 작업 파일이 아니라",
        "고정 커밋의 Git object를 `git show`로 직접 읽었다.",
        "",
        "## Prompt 전수 비교",
        "",
        "| 데이터 | 사례 수 | 원문과 최종 prompt 완전 일치 | 불일치 |",
        "|---|---:|---:|---:|",
        (
            f"| Who&When | {who['case_count']} | {who['exact_prompt_match_count']} "
            f"| {who['mismatch_count']} |"
        ),
        (
            f"| MAST positive-labeled | {mast['case_count']} "
            f"| {mast['exact_prompt_match_count_after_declared_adapter']} | {mast['mismatch_count']} |"
        ),
        "",
        f"- Who&When 구성: AG {who['split_counts']['AG']}개, HC {who['split_counts']['HC']}개",
        (
            f"- MAST 0-based 연속 step 검증: "
            f"{mast['contiguous_zero_based_step_count']}/{mast['case_count']}"
        ),
        "- Who&When HC의 `groundtruth`는 A2P 저장소 데이터의 `ground_truth` 필드로 정규화했다.",
        "- MAST의 `agent` 필드는 A2P prompt의 Agent_Name 슬롯으로 매핑했다.",
        "",
        "## API 설정",
        "",
        "| 데이터 | 모델 | system prompt 일치 | temperature 전송 | max output |",
        "|---|---|---:|---|---:|",
    ]
    for item in report["configs"]:
        lines.append(
            f"| {item['label']} | {item.get('model')} | "
            f"{'예' if item.get('system_prompt_match') else '아니오'} | "
            f"{'생략' if item.get('temperature_omitted') else item.get('temperature_value')} | "
            f"{item.get('max_tokens')} |"
        )
    lines.extend(
        [
            "",
            "## 원문과 의도적으로 다른 조건",
            "",
            "| 항목 | A2P 저장소/보고 조건 | 현재 통제 실험 | 해석 |",
            "|---|---|---|---|",
        ]
    )
    for item in report["intentional_differences"]:
        lines.append(
            f"| {item['item']} | {item['repository_reported_setting']} | "
            f"{item['controlled_experiment_setting']} | {item['implication']} |"
        )
    lines.extend(
        [
            "",
            "## 최종 해석",
            "",
            "현재 구현은 **A2P 공개 저장소 prompt와 single-pass 입력 직렬화를 정확히 복제한",
            "통합-evaluator 재실행**이다. GPT-OSS-120B와 저장소 evaluator를 그대로 사용한 수치 재현은",
            "아니므로 논문에서는 `official reproduction`이 아니라",
            "`repository-exact prompt reimplementation under a unified backend and evaluator`로 표기한다.",
            "",
            "저장소 동기 CLI에는 `--is_handcrafted False` 문자열을 그대로 전달하는 문제가 있다.",
            "현재 구현은 이 버그가 없는 비동기 경로의 문서화된 AG=`name`, HC=`role` 동작을 따른다.",
        ]
    )
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
