import argparse
import json
from pathlib import Path
from typing import Any

"""
What it does:
- reads every .json file you pass
- expects each file to contain a top level JSON array
- checks every "id"
- reports duplicates across all files
- exits with code 1 if duplicates are found
"""

"""
Usage: 
python check_duplicates path/to/data_file.json
python check_duplicates path/to/
python check_duplicates path/to/ path/to2 path/to3/data_file.json
"""

def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def collect_json_files(paths: list[str]) -> list[Path]:
    files: list[Path] = []

    for raw_path in paths:
        path = Path(raw_path)

        if not path.exists():
            raise FileNotFoundError(f"Path not found: {path}")

        if path.is_file():
            if path.suffix.lower() == ".json":
                files.append(path)
        elif path.is_dir():
            files.extend(sorted(path.rglob("*.json")))

    if not files:
        raise ValueError("No JSON files found in the provided paths.")

    return files


def check_duplicate_ids(files: list[Path]) -> int:
    seen: dict[str, tuple[Path, int]] = {}
    duplicates_found = 0

    for file_path in files:
        data = load_json(file_path)

        if not isinstance(data, list):
            raise ValueError(
                f"Expected top-level JSON array in file: {file_path}"
            )

        for index, item in enumerate(data):
            if not isinstance(item, dict):
                raise ValueError(
                    f"Item at index {index} in {file_path} is not a JSON object."
                )

            if "id" not in item:
                raise ValueError(
                    f'Item at index {index} in {file_path} is missing the "id" field.'
                )

            item_id = item["id"]
            if not isinstance(item_id, str):
                raise ValueError(
                    f'"id" at index {index} in {file_path} must be a string.'
                )

            if item_id in seen:
                first_file, first_index = seen[item_id]
                print(
                    f"DUPLICATE ID FOUND: {item_id}\n"
                    f"  First occurrence : {first_file} (item index {first_index})\n"
                    f"  Duplicate        : {file_path} (item index {index})\n"
                )
                duplicates_found += 1
            else:
                seen[item_id] = (file_path, index)

    return duplicates_found


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Check JSON files for duplicate example IDs."
    )
    parser.add_argument(
        "paths",
        nargs="+",
        help="One or more JSON files or directories containing JSON files."
    )
    args = parser.parse_args()

    files = collect_json_files(args.paths)
    duplicates_found = check_duplicate_ids(files)

    if duplicates_found == 0:
        print(f"No duplicate IDs found across {len(files)} JSON file(s).")
    else:
        print(f"Found {duplicates_found} duplicate ID occurrence(s).")
        raise SystemExit(1)


if __name__ == "__main__":
    main()