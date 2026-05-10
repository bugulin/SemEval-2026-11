#!/usr/bin/env python3
import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path


def load_examples(path: Path):
    text = path.read_text(encoding="utf-8").strip()

    if not text:
        return []

    if path.suffix == ".jsonl":
        return [json.loads(line) for line in text.splitlines() if line.strip()]

    data = json.loads(text)
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for key in ["data", "examples", "train", "validation", "items"]:
            if key in data and isinstance(data[key], list):
                return data[key]
        return [data]

    raise ValueError(f"Unsupported data format in {path}")


def normalize_bool(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        v = value.strip().lower()
        if v in {"true", "1", "yes", "valid"}:
            return True
        if v in {"false", "0", "no", "invalid"}:
            return False
    return value


def get_relevant_premises(example):
    value = example.get("relevant_premises", example.get("relevant", None))

    if value is None:
        return []

    if isinstance(value, list):
        return [int(x) for x in value]

    if isinstance(value, str):
        return [int(x) for x in re.findall(r"\d+", value)]

    return []


def get_num_premises(example):
    if isinstance(example.get("premises"), list):
        return len(example["premises"])

    text = example.get("syllogism", "")
    if not isinstance(text, str):
        return None

    premise_matches = re.findall(
        r"(?im)^\s*(?:premise\s*)?\d+\s*[\.:)\-]",
        text,
    )
    if premise_matches:
        return len(premise_matches)

    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("files", nargs="+", help="JSON or JSONL dataset files")
    args = parser.parse_args()

    all_examples = []
    for file in args.files:
        all_examples.extend(load_examples(Path(file)))

    print(f"Total examples: {len(all_examples)}")
    print()

    validity_counts = Counter()
    plausibility_counts = Counter()
    validity_plausibility = Counter()
    relevant_pair_counts = Counter()
    relevant_index_counts = Counter()
    premise_count_distribution = Counter()

    missing_relevant = 0
    max_relevant_index = -1
    examples_with_late_relevant = 0
    examples_with_only_early_relevant = 0
    examples_with_non_adjacent_relevant = 0

    for ex in all_examples:
        validity = normalize_bool(ex.get("validity"))
        plausibility = normalize_bool(ex.get("plausibility", "MISSING"))

        validity_counts[validity] += 1
        plausibility_counts[plausibility] += 1
        validity_plausibility[(validity, plausibility)] += 1

        n_premises = get_num_premises(ex)
        premise_count_distribution[n_premises] += 1

        rel = get_relevant_premises(ex)

        if not rel:
            missing_relevant += 1
            continue

        rel_sorted = tuple(sorted(rel))
        relevant_pair_counts[rel_sorted] += 1

        for idx in rel_sorted:
            relevant_index_counts[idx] += 1
            max_relevant_index = max(max_relevant_index, idx)

        if any(idx >= 3 for idx in rel_sorted):
            examples_with_late_relevant += 1
        else:
            examples_with_only_early_relevant += 1

        if len(rel_sorted) >= 2:
            gaps = [b - a for a, b in zip(rel_sorted, rel_sorted[1:])]
            if any(gap > 1 for gap in gaps):
                examples_with_non_adjacent_relevant += 1

    print("Validity label balance:")
    for key, count in validity_counts.most_common():
        print(f"  {key}: {count}")
    print()

    print("Plausibility label balance:")
    for key, count in plausibility_counts.most_common():
        print(f"  {key}: {count}")
    print()

    print("Validity x plausibility:")
    for key, count in validity_plausibility.most_common():
        print(f"  validity={key[0]}, plausibility={key[1]}: {count}")
    print()

    print("Number of premises distribution:")
    for key, count in sorted(premise_count_distribution.items(), key=lambda x: str(x[0])):
        print(f"  {key}: {count}")
    print()

    print("Relevant premise pair distribution:")
    for pair, count in relevant_pair_counts.most_common(20):
        print(f"  {pair}: {count}")
    print()

    print("Relevant premise index distribution:")
    for idx, count in sorted(relevant_index_counts.items()):
        print(f"  index {idx}: {count}")
    print()

    print("Warnings / diagnostics:")
    print(f"  Missing relevant_premises: {missing_relevant}")
    print(f"  Max relevant premise index: {max_relevant_index}")
    print(f"  Examples with relevant premise at index >= 3: {examples_with_late_relevant}")
    print(f"  Examples where all relevant premises are among indices 0,1,2: {examples_with_only_early_relevant}")
    print(f"  Examples with non-adjacent relevant premises: {examples_with_non_adjacent_relevant}")

    if examples_with_late_relevant == 0:
        print("  WARNING: No relevant premise appears after index 2. Strong positional artifact.")

    if relevant_pair_counts:
        most_common_pair, count = relevant_pair_counts.most_common(1)[0]
        ratio = count / sum(relevant_pair_counts.values())
        if ratio > 0.4:
            print(f"  WARNING: Pair {most_common_pair} appears in {ratio:.1%} of examples.")

    if len(validity_counts) == 2:
        values = list(validity_counts.values())
        minority_ratio = min(values) / sum(values)
        if minority_ratio < 0.35:
            print(f"  WARNING: Validity label imbalance. Minority class is only {minority_ratio:.1%}.")


if __name__ == "__main__":
    main()