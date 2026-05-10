import argparse
import json
from pathlib import Path
from typing import Any


def load_json_array(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(f"Expected top-level JSON array in {path}")

    for index, item in enumerate(data):
        if not isinstance(item, dict):
            raise ValueError(f"Item {index} in {path} is not a JSON object")

    return data


def save_json(path: Path, data: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.write("\n")


def save_jsonl(path: Path, data: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge multiple JSON array dataset files into one JSON/JSONL file."
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        type=Path,
        help="Input JSON files to merge."
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        required=True,
        help="Merged JSON output path."
    )
    parser.add_argument(
        "--jsonl-output",
        type=Path,
        default=None,
        help="Optional merged JSONL output path."
    )
    args = parser.parse_args()

    merged: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    seen_syllogisms: set[str] = set()

    for input_path in args.inputs:
        data = load_json_array(input_path)

        for item in data:
            item_id = item.get("id")
            syllogism = item.get("syllogism")

            if isinstance(item_id, str):
                if item_id in seen_ids:
                    raise ValueError(f"Duplicate id found: {item_id}")
                seen_ids.add(item_id)

            if isinstance(syllogism, str):
                if syllogism in seen_syllogisms:
                    raise ValueError(f"Duplicate syllogism found: {syllogism}")
                seen_syllogisms.add(syllogism)

            merged.append(item)

    save_json(args.output, merged)

    if args.jsonl_output is not None:
        save_jsonl(args.jsonl_output, merged)

    print(f"Merged {len(merged)} examples.")
    print(f"Saved JSON to: {args.output}")

    if args.jsonl_output is not None:
        print(f"Saved JSONL to: {args.jsonl_output}")


if __name__ == "__main__":
    main()