import json
import re
from pathlib import Path


INPUT_FILE = "raw_synthetic_3_output.json"
OUTPUT_FILE = "raw_synthetic_3_formalized.json"


def is_valid_syllogism(item):
    """
    Handles both possible formats:
    - "validity": true
    - "validity_flag": "valid"
    """
    if item.get("form_flag")=="valid_form":
        return True

    return False


def corrected_mood(clause):
    """
    If the stored mood is A, but the raw sentence contains
    'no' or 'not', translate it as E instead.

    I also included 'nothing' and 'never' because your file has
    cases like 'Nothing that is ...' which are clearly negative.
    Remove those two words if you only want exactly 'no' and 'not'.
    """
    mood = clause.get("form")
    raw = clause.get("raw", "").lower()
    raw = " "+raw

    negative_pattern = r"\b( no | not | nothing | never )\b"

    if mood == "A" and (" no " in raw or " not " in raw or " nothing " in raw or " never " in raw):
        return "E"
    
    if mood == "E" and (" some " in raw or " exist " in raw):
        return "O"

    return mood


def get_ordered_clauses(item):
    """
    Returns clauses in the required order:
    premise_1, premise_2, conclusion.
    """
    role_map = {
        "premise1": None,
        "premise2": None,
        "conclusion": None,
    }

    for clause in item.get("parsed_clauses", []):
        role = clause.get("role")
        if role in role_map:
            role_map[role] = clause

    return [
        ("premise_1", role_map["premise1"]),
        ("premise_2", role_map["premise2"]),
        ("conclusion", role_map["conclusion"]),
    ]


def formalize_syllogism(item):
    """
    Assigns letters a, b, c to terms in the order they first appear
    within the valid syllogism.

    Example:
    fish -> a
    shark -> b
    white -> c

    Then:
    A fish shark -> Aab
    A shark white -> Abc
    A fish white -> Aac
    """
    term_to_letter = {}
    available_letters = iter("abcdefghijklmnopqrstuvwxyz")

    parts = {}

    for output_role, clause in get_ordered_clauses(item):
        if clause is None:
            parts[output_role] = None
            continue

        mood = corrected_mood(clause)
        subject = clause.get("subject")
        predicate = clause.get("predicate")

        if mood is None or subject is None or predicate is None:
            parts[output_role] = None
            continue

        for term in [subject, predicate]:
            if term not in term_to_letter:
                term_to_letter[term] = next(available_letters)

        subject_letter = term_to_letter[subject]
        predicate_letter = term_to_letter[predicate]

        parts[output_role] = f"{mood}{subject_letter}{predicate_letter}"

    return {
        "id": item.get("id"),
        "syllogism": item.get("syllogism"),
        "formal_language": (
            f"premise_1: {parts['premise_1']}, "
            f"premise_2: {parts['premise_2']}, "
            f"conclusion: {parts['conclusion']}"
        ),
        "term_mapping": term_to_letter,
        "validity": item.get("validity")
    }


def main():
    input_path = Path(INPUT_FILE)

    with input_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    formalized = []

    for item in data:
        if is_valid_syllogism(item):
            formalized.append(formalize_syllogism(item))

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(formalized, f, indent=2, ensure_ascii=False)

    print(f"Converted {len(formalized)} valid syllogisms.")
    print(f"Saved result to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()