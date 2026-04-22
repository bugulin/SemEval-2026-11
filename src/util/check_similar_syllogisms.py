import argparse
import json
from pathlib import Path
from typing import Any

"""
What it does:
- reads every .json file you pass
- expects each file to contain a top level JSON array
- checks every "syllogism"
- reports extremely similar syllogisms across all files
- uses Levenshtein distance with a default limit of 3
- exits with code 1 if similar syllogisms are found
"""

"""
Usage:
python check_similar_syllogisms.py path/to/data_file.json
python check_similar_syllogisms.py path/to/
python check_similar_syllogisms.py path/to/ path/to2 path/to3/data_file.json
python check_similar_syllogisms.py path/to/ --max-distance 3
"""

"""
NOTE: 
This is a pairwise comparison, so it is O(n^2) in the number of syllogisms. 
Please tell me if this becomes too slow and a faster alternative is required.
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


def normalize_syllogism(text: str) -> str:
    return " ".join(text.strip().lower().split())


def levenshtein_distance_with_limit(a: str, b: str, max_distance: int) -> int | None:
    if a == b:
        return 0

    if abs(len(a) - len(b)) > max_distance:
        return None

    if len(a) > len(b):
        a, b = b, a

    previous_row = list(range(len(b) + 1))

    for i, char_a in enumerate(a, start=1):
        current_row = [i]
        row_min = current_row[0]

        for j, char_b in enumerate(b, start=1):
            insert_cost = current_row[j - 1] + 1
            delete_cost = previous_row[j] + 1
            replace_cost = previous_row[j - 1] + (char_a != char_b)

            value = min(insert_cost, delete_cost, replace_cost)
            current_row.append(value)

            if value < row_min:
                row_min = value

        if row_min > max_distance:
            return None

        previous_row = current_row

    distance = previous_row[-1]
    return distance if distance <= max_distance else None


def collect_syllogisms(files: list[Path]) -> list[tuple[Path, int, str, str]]:
    syllogisms: list[tuple[Path, int, str, str]] = []

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

            if "syllogism" not in item:
                raise ValueError(
                    f'Item at index {index} in {file_path} is missing the "syllogism" field.'
                )

            syllogism = item["syllogism"]
            if not isinstance(syllogism, str):
                raise ValueError(
                    f'"syllogism" at index {index} in {file_path} must be a string.'
                )

            normalized = normalize_syllogism(syllogism)
            syllogisms.append((file_path, index, syllogism, normalized))

    return syllogisms


def check_similar_syllogisms(files: list[Path], max_distance: int) -> int:
    syllogisms = collect_syllogisms(files)
    similar_found = 0

    for i in range(len(syllogisms)):
        first_file, first_index, first_original, first_normalized = syllogisms[i]

        for j in range(i + 1, len(syllogisms)):
            second_file, second_index, second_original, second_normalized = syllogisms[j]

            distance = levenshtein_distance_with_limit(
                first_normalized,
                second_normalized,
                max_distance
            )

            if distance is not None:
                print(
                    f"SIMILAR SYLLOGISM FOUND (distance {distance}):\n"
                    f"  First occurrence : {first_file} (item index {first_index})\n"
                    f"  Similar          : {second_file} (item index {second_index})\n"
                    f"  First text       : {first_original}\n"
                    f"  Similar text     : {second_original}\n"
                )
                similar_found += 1

    return similar_found


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Check JSON files for extremely similar syllogisms."
    )
    parser.add_argument(
        "paths",
        nargs="+",
        help="One or more JSON files or directories containing JSON files."
    )
    parser.add_argument(
        "--max-distance",
        type=int,
        default=3,
        help="Maximum Levenshtein distance allowed to count as similar (default: 3)."
    )
    args = parser.parse_args()

    if args.max_distance < 0:
        raise ValueError("--max-distance must be a non-negative integer.")

    files = collect_json_files(args.paths)
    similar_found = check_similar_syllogisms(files, args.max_distance)

    if similar_found == 0:
        print(
            f'No extremely similar syllogisms found across {len(files)} JSON file(s) '
            f'with max distance {args.max_distance}.'
        )
    else:
        print(
            f"Found {similar_found} extremely similar syllogism pair(s) "
            f"with max distance {args.max_distance}."
        )
        raise SystemExit(1)


if __name__ == "__main__":
    main()