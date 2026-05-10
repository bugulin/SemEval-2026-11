#!/usr/bin/env python3
import json
import re
import sys
from collections import Counter
from pathlib import Path


def split_syllogism(s):
    if "Therefore," not in s:
        return [], None

    premise_part, conclusion = s.split("Therefore,", 1)

    premises = [p.strip() for p in re.split(r"\.\s+", premise_part.strip()) if p.strip()]

    premises = [p[:-1] if p.endswith(".") else p for p in premises]
    conclusion = conclusion.strip()

    return premises, conclusion


def load(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def main():
    if len(sys.argv) != 2:
        print("Usage: python scripts/check_stress_set.py data/2/stress_valid_data.json")
        sys.exit(1)

    path = Path(sys.argv[1])
    data = load(path)

    errors = []
    premise_counts = Counter()
    validity_counts = Counter()
    plausibility_counts = Counter()
    vp_counts = Counter()
    relevant_pair_counts = Counter()
    relevant_index_counts = Counter()

    seen_syllogisms = set()

    valid_count = 0
    valid_with_late_relevant = 0
    valid_with_non_adjacent = 0

    for i, ex in enumerate(data):
        prefix = f"Example {i}"

        required = {"id", "syllogism", "validity", "plausibility", "relevant_premises"}
        missing = required - set(ex)
        if missing:
            errors.append(f"{prefix}: missing fields {missing}")
            continue

        if not isinstance(ex["id"], str):
            errors.append(f"{prefix}: id must be string")

        if not isinstance(ex["syllogism"], str):
            errors.append(f"{prefix}: syllogism must be string")
            continue

        if ex["syllogism"] in seen_syllogisms:
            errors.append(f"{prefix}: duplicate syllogism")
        seen_syllogisms.add(ex["syllogism"])

        if not isinstance(ex["validity"], bool):
            errors.append(f"{prefix}: validity must be bool")

        if not isinstance(ex["plausibility"], bool):
            errors.append(f"{prefix}: plausibility must be bool")

        if not isinstance(ex["relevant_premises"], list):
            errors.append(f"{prefix}: relevant_premises must be list")
            continue

        premises, conclusion = split_syllogism(ex["syllogism"])
        n = len(premises)
        premise_counts[n] += 1

        if n not in {4, 5, 6}:
            errors.append(f"{prefix}: has {n} premises, expected 4, 5, or 6")

        rel = ex["relevant_premises"]

        if ex["validity"] is True:
            valid_count += 1

            if len(rel) != 2:
                errors.append(f"{prefix}: valid example must have exactly 2 relevant premises, got {rel}")

            if len(rel) == 2:
                relevant_pair_counts[tuple(rel)] += 1

                if any(r >= 3 for r in rel):
                    valid_with_late_relevant += 1

                if abs(rel[0] - rel[1]) > 1:
                    valid_with_non_adjacent += 1

        else:
            if rel != []:
                errors.append(f"{prefix}: invalid example must have empty relevant_premises, got {rel}")

        for r in rel:
            if not isinstance(r, int):
                errors.append(f"{prefix}: relevant premise index {r} is not int")
                continue

            relevant_index_counts[r] += 1

            if r < 0 or r >= n:
                errors.append(f"{prefix}: relevant premise index {r} out of range for {n} premises")

        validity_counts[ex["validity"]] += 1
        plausibility_counts[ex["plausibility"]] += 1
        vp_counts[(ex["validity"], ex["plausibility"])] += 1

    print("Total examples:", len(data))
    print()

    print("Premise count distribution:")
    for k, v in sorted(premise_counts.items()):
        print(f"  {k}: {v}")
    print()

    print("Validity distribution:")
    for k, v in validity_counts.items():
        print(f"  {k}: {v}")
    print()

    print("Plausibility distribution:")
    for k, v in plausibility_counts.items():
        print(f"  {k}: {v}")
    print()

    print("Validity x plausibility:")
    for k, v in sorted(vp_counts.items()):
        print(f"  validity={k[0]}, plausibility={k[1]}: {v}")
    print()

    print("Relevant pair distribution:")
    for k, v in relevant_pair_counts.most_common():
        print(f"  {k}: {v}")
    print()

    print("Relevant index distribution:")
    for k, v in sorted(relevant_index_counts.items()):
        print(f"  index {k}: {v}")
    print()

    if valid_count:
        print("Stress properties:")
        print(f"  Valid examples: {valid_count}")
        print(f"  Valid examples with relevant premise index >= 3: {valid_with_late_relevant} / {valid_count}")
        print(f"  Valid examples with non-adjacent relevant premises: {valid_with_non_adjacent} / {valid_count}")
        print()

    if errors:
        print("ERRORS:")
        for e in errors[:100]:
            print("  -", e)
        if len(errors) > 100:
            print(f"  ... and {len(errors) - 100} more")
        sys.exit(1)

    print("No structural errors found.")


if __name__ == "__main__":
    main()