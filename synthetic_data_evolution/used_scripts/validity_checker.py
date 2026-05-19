import json
import re
from itertools import permutations
from pathlib import Path


INPUT_FILE = "raw_synthetic_2_formalized.json"
OUTPUT_FILE = "raw_synthetic_2_validity_check.json"


# Base valid deductions from Aristotle's table, before permutations
base_valid_deductions = [
    # First figure
    ("Aba", "Acb", "Aca"),  # Barbara
    ("Aba", "Acb", "Ica"),  # Barbara + existential import
    ("Aba", "Acb", "Iac"),  # Barbara + existential import
    ("Eba", "Acb", "Eca"),  # Celarent
    ("Eba", "Acb", "Eac"),  # Celarent
    ("Eab", "Acb", "Eca"),  # Celarent
    ("Eab", "Acb", "Eac"),  # Celarent
    ("Aba", "Icb", "Ica"),  # Darii
    ("Aba", "Icb", "Iac"),  # Darii
    ("Aba", "Ibc", "Ica"),  # Darii
    ("Aba", "Ibc", "Iac"),  # Darii
    ("Eba", "Icb", "Oca"),  # Ferio
    ("Eba", "Ibc", "Oca"),  # Ferio
    ("Eab", "Icb", "Oca"),  # Ferio
    ("Eab", "Ibc", "Oca"),  # Ferio
    ("Eba", "Acb", "Oca"),
    ("Eab", "Acb", "Oca"),
    ("Eba", "Acb", "Oac"),
    ("Eab", "Acb", "Oac"),

    # Second figure
    ("Eba", "Aca", "Ecb"),  # Cesare
    ("Eba", "Aca", "Ebc"),  # Cesare
    ("Eab", "Aca", "Ecb"),  # Cesare
    ("Eab", "Aca", "Ebc"),  # Cesare
    ("Aba", "Eca", "Ecb"),  # Camestres
    ("Aba", "Eca", "Ebc"),  # Camestres
    ("Aba", "Eac", "Ecb"),  # Camestres
    ("Aba", "Eac", "Ebc"),  # Camestres
    ("Eba", "Ica", "Ocb"),  # Festino
    ("Eba", "Ica", "Ocb"),  # Festino
    ("Eab", "Ica", "Ocb"),  # Festino
    ("Eab", "Ica", "Ocb"),  # Festino
    ("Aba", "Oca", "Ocb"),  # Baroco
    ("Eba", "Aca", "Ocb"),
    ("Eab", "Aca", "Oca"),
    ("Aab", "Eac", "Ocb"),
    ("Aab", "Eca", "Ocb"),

    # Third figure
    ("Aca", "Acb", "Iba"),  # Darapti
    ("Aca", "Acb", "Iab"),  # Darapti
    ("Eca", "Acb", "Oba"),  # Felapton
    ("Eac", "Acb", "Oba"),  # Felapton
    ("Ica", "Acb", "Iba"),  # Disamis
    ("Ica", "Acb", "Iab"),  # Disamis
    ("Iac", "Acb", "Iba"),  # Disamis
    ("Iac", "Acb", "Iab"),  # Disamis
    ("Aca", "Icb", "Iba"),  # Datisi
    ("Aca", "Icb", "Iba"),  # Datisi
    ("Aca", "Ibc", "Iba"),  # Datisi
    ("Aca", "Ibc", "Iba"),  # Datisi
    ("Oca", "Acb", "Oba"),  # Bocardo
    ("Eca", "Icb", "Oba"),  # Ferison
    ("Eca", "Ibc", "Oba"),  # Ferison
    ("Eac", "Icb", "Oba"),  # Ferison
    ("Eac", "Ibc", "Oba"),  # Ferison
]


def permute_formula(formula, mapping):
    """
    Example:
    formula = "Aab"
    mapping = {"a": "b", "b": "c", "c": "a"}
    returns "Abc"

    The mood letter A/E/I/O is not changed.
    """
    mood = formula[0]
    terms = formula[1:]
    return mood + "".join(mapping[t] for t in terms)


def build_valid_deduction_set():
    """
    Creates the full set of valid deductions:
    14 base rows * 6 permutations * 2 premise orders = 168 strings.
    """
    valid_deductions = []

    for premise_1, premise_2, conclusion in base_valid_deductions:
        for perm in permutations("abc"):
            mapping = dict(zip("abc", perm))

            p1 = permute_formula(premise_1, mapping)
            p2 = permute_formula(premise_2, mapping)
            c = permute_formula(conclusion, mapping)

            # Original premise order
            valid_deductions.append(
                f"premise_1: {p1}, premise_2: {p2}, conclusion: {c}"
            )

            # Switched premise order
            valid_deductions.append(
                f"premise_1: {p2}, premise_2: {p1}, conclusion: {c}"
            )
    print(valid_deductions)
    return valid_deductions


def normalize_formal_language(formal_language):
    """
    Extracts the three formal statements from strings like:

    "premise_1: Aab, premise_2: Abc, conclusion: Aac"

    and returns them in standardized formatting.
    """
    pattern = re.compile(
        r"premise_1:\s*([AEIO][abc]{2}),\s*"
        r"premise_2:\s*([AEIO][abc]{2}),\s*"
        r"conclusion:\s*([AEIO][abc]{2})"
    )

    match = pattern.fullmatch(formal_language.strip())

    if not match:
        return None

    p1, p2, c = match.groups()

    return f"premise_1: {p1}, premise_2: {p2}, conclusion: {c}"


def evaluate_file(input_file):
    valid_deductions = build_valid_deduction_set()
    print(len(valid_deductions))

    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    results = []

    total = len(data)
    correct = 0
    invalid_format_count = 0

    true_positive = 0
    true_negative = 0
    false_positive = 0
    false_negative = 0

    for item in data:
        formal_language = item.get("formal_language", "")
        normalized = normalize_formal_language(formal_language)

        if normalized is None:
            predicted_validity = False
            invalid_format = True
            invalid_format_count += 1
        else:
            predicted_validity = normalized in valid_deductions
            invalid_format = False

        actual_validity = bool(item.get("validity"))
        is_correct = predicted_validity == actual_validity

        if is_correct:
            correct += 1

        if actual_validity and predicted_validity:
            true_positive += 1
        elif not actual_validity and not predicted_validity:
            true_negative += 1
        elif not actual_validity and predicted_validity:
            false_positive += 1
        elif actual_validity and not predicted_validity:
            false_negative += 1

        results.append({
            "id": item.get("id"),
            "syllogism": item.get("syllogism"),
            "formal_language": formal_language,
            "normalized_formal_language": normalized,
            "actual_validity": actual_validity,
            "predicted_validity": predicted_validity,
            "correct": is_correct,
            "invalid_format": invalid_format,
        })

    statistics = {
        "total_syllogisms": total,
        "correctly_evaluated": correct,
        "incorrectly_evaluated": total - correct,
        "accuracy": correct / total if total else 0,
        "true_positive": true_positive,
        "true_negative": true_negative,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "invalid_formal_language_formats": invalid_format_count,
        "number_of_valid_deduction_patterns": len(valid_deductions),
    }

    return statistics, results


def main():
    statistics, results = evaluate_file(INPUT_FILE)

    output = {
        "statistics": statistics,
        "results": results,
    }

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print("Statistics")
    print("-" * 40)
    print(f"Total syllogisms: {statistics['total_syllogisms']}")
    print(f"Correctly evaluated: {statistics['correctly_evaluated']}")
    print(f"Incorrectly evaluated: {statistics['incorrectly_evaluated']}")
    print(f"Accuracy: {statistics['accuracy']:.2%}")
    print()
    print(f"True positives: {statistics['true_positive']}")
    print(f"True negatives: {statistics['true_negative']}")
    print(f"False positives: {statistics['false_positive']}")
    print(f"False negatives: {statistics['false_negative']}")
    print(f"Invalid formal_language formats: {statistics['invalid_formal_language_formats']}")
    print()
    print(f"Saved detailed results to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()