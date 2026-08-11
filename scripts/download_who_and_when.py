from __future__ import annotations

import argparse
import json
import sys
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


DATASET = "Kevin355/Who_and_When"
BASE_URL = "https://datasets-server.huggingface.co"
CONFIGS = {
    "Algorithm-Generated": "who_and_when_algorithm_generated.jsonl",
    "Hand-Crafted": "who_and_when_hand_crafted.jsonl",
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Download Who&When rows from the Hugging Face Dataset Viewer API.")
    parser.add_argument("--out-dir", default="data", help="Output directory.")
    parser.add_argument("--page-size", type=int, default=100, help="Rows per request. Dataset Viewer max is usually 100.")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict[str, Any]] = []
    for config, filename in CONFIGS.items():
        rows = download_split(config=config, split="train", page_size=args.page_size)
        for row in rows:
            row["source_config"] = config
            row["source_split"] = "train"
        write_jsonl(out_dir / filename, rows)
        all_rows.extend(rows)
        print(f"{config}: {len(rows)} rows -> {out_dir / filename}")

    write_jsonl(out_dir / "who_and_when_all.jsonl", all_rows)
    print(f"all: {len(all_rows)} rows -> {out_dir / 'who_and_when_all.jsonl'}")


def download_split(config: str, split: str, page_size: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    offset = 0
    total: int | None = None
    while total is None or offset < total:
        params = urllib.parse.urlencode(
            {
                "dataset": DATASET,
                "config": config,
                "split": split,
                "offset": offset,
                "length": page_size,
            }
        )
        url = f"{BASE_URL}/rows?{params}"
        payload = read_json(url)
        total = int(payload.get("num_rows_total") or len(payload.get("rows", [])))
        page_rows = [item["row"] for item in payload.get("rows", [])]
        rows.extend(page_rows)
        if not page_rows:
            break
        offset += len(page_rows)
    return rows


def read_json(url: str) -> dict[str, Any]:
    with urllib.request.urlopen(url, timeout=120) as response:
        return json.loads(response.read().decode("utf-8"))


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
