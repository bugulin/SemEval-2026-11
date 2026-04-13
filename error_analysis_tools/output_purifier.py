#!/usr/bin/env python3
"""
Create a purified JSON dataset from the enriched syllogism output file.

Keeps only entries with:
    form_flag == "valid_form"
or
    form_flag == "valid"

And preserves only:
    id, syllogism, validity, plausibility

Usage:
    python purify_valid_form_dataset.py \
        --input output_train_data_formal_subtask1.json \
        --output train_data_formal_subtask1.json
"""

from __future__ import annotations

import argparse
import json
from typing import Any


VALID_FORM_VALUES = {"valid_form", "valid"}


def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: str, obj: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def extract_plausibility(record: dict) -> Any:
    """
    Support both:
      - 'plausibility'
      - 'plausible'
    Output key will always be 'plausibility'.
    """
    if "plausibility" in record:
        return record["plausibility"]
    if "plausible" in record:
        return record["plausible"]
    return None


def purify_records(data: list[dict]) -> list[dict]:
    purified = []

    for record in data:
        form_flag = record.get("form_flag")

        if form_flag not in VALID_FORM_VALUES:
            continue

        purified_record = {
            "id": record.get("id"),
            "syllogism": record.get("syllogism"),
            "validity": record.get("validity"),
            "plausibility": extract_plausibility(record),
        }

        purified.append(purified_record)

    return purified


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to enriched output JSON file")
    parser.add_argument("--output", required=True, help="Path to purified output JSON file")
    args = parser.parse_args()

    data = load_json(args.input)

    if not isinstance(data, list):
        raise ValueError("Expected top-level JSON array")

    purified = purify_records(data)
    save_json(args.output, purified)

    print(f"Read {len(data)} entries from: {args.input}")
    print(f"Kept {len(purified)} entries with valid form")
    print(f"Wrote purified dataset to: {args.output}")


if __name__ == "__main__":
    main()