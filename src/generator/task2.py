from __future__ import annotations
import argparse
import hashlib
import itertools
import json
import random
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Literal

"""
Deterministic Subtask 2 syllogism generator.

Goal:
- generate Subtask 2 examples without local/api LLM calls.
- keep validity labels correct by construction
- add irrelevant premises and compute zero-based relevant_premises mechanically
- output JSON and optionally JSONL

This intentionally does not ask a language model whether a syllogism is valid.
The generator chooses a formal schema first, then fills it with natural terms.

Formal convention used internally:
- Forms: A/E/I/O
  A(S, P): All S are P.
  E(S, P): No S are P.
  I(S, P): Some S are P.
  O(S, P): Some S are not P.
- S = minor term, P = major term, M = middle term.
- Figures:
  1: major M-P, minor S-M, conclusion S-P
  2: major P-M, minor S-M, conclusion S-P
  3: major M-P, minor M-S, conclusion S-P
  4: major P-M, minor M-S, conclusion S-P
"""

"""
ARGUMENTS:

-n / --count              number of examples
-p / --premises           one or more premise counts
--seed                    reproducible generation
--semantics               modern or aristotelian
-o / --output             main JSON output
--jsonl-output            optional JSONL output
--audit-output            audit report path
--metadata-output         metadata rich output path
--include-metadata        include metadata directly in main JSON
--no-id                   omit id fields
"""

Form = Literal["A", "E", "I", "O"]
Semantics = Literal["modern", "aristotelian"]
Role = Literal["S", "P", "M"]

FORMS: tuple[Form, ...] = ("A", "E", "I", "O")
FIGURES: dict[int, tuple[tuple[Role, Role], tuple[Role, Role]]] = {
    1: (("M", "P"), ("S", "M")),
    2: (("P", "M"), ("S", "M")),
    3: (("M", "P"), ("M", "S")),
    4: (("P", "M"), ("M", "S")),
}

# Traditional names for readability/debugging. The code does not trust this map for validity.
SCHEMA_NAMES: dict[tuple[int, str], str] = {
    (1, "AAA"): "barbara",
    (1, "EAE"): "celarent",
    (1, "AII"): "darii",
    (1, "EIO"): "ferio",
    (1, "AAI"): "barbari_ei",
    (1, "EAO"): "celaront_ei",
    (2, "EAE"): "cesare",
    (2, "AEE"): "camestres",
    (2, "EIO"): "festino",
    (2, "AOO"): "baroco",
    (2, "EAO"): "cesaro_ei",
    (2, "AEO"): "camestrop_ei",
    (3, "AAI"): "darapti_ei",
    (3, "IAI"): "disamis",
    (3, "AII"): "datisi",
    (3, "EAO"): "felapton_ei",
    (3, "OAO"): "bocardo",
    (3, "EIO"): "ferison",
    (4, "AAI"): "bramantip_ei",
    (4, "AEE"): "camenes",
    (4, "IAI"): "dimaris",
    (4, "EAO"): "fesapo_ei",
    (4, "EIO"): "fresison",
}

TOPICS = [
    "Baking Bread", "Spicy Food", "Morning Coffee", "Vegetarianism", "Grilling Steak",
    "Restaurant Service", "Fresh Herbs", "Fast Food", "Salad Making", "Chocolate Desserts",
    "Laundry Day", "Ironing Clothes", "Vacuuming Carpets", "Interior Design", "Fixing Leaky Faucets",
    "Painting Walls", "Houseplants", "Mowing the Lawn", "Organizing Closets", "Taking Out Trash",
    "Commuting to Work", "Driving in Traffic", "Cycling to School", "Public Transportation", "Car Maintenance",
    "Gas Prices", "Finding Parking", "Airline Travel", "Train Journeys", "Walking the Dog",
    "Sending Emails", "Office Meetings", "Working from Home", "Job Interviews", "Coffee Breaks",
    "Office Supplies", "Project Deadlines", "Professional Networking", "Career Changes", "Retirement Planning",
    "First Dates", "Wedding Planning", "Birthday Parties", "Holiday Gift Giving", "Family Dinners",
    "Neighbor Disputes", "Long-distance Friendships", "Small Talk", "Text Messaging", "Social Media Etiquette",
    "Watching Movies", "Playing Video Games", "Reading Novels", "Attending Concerts", "Photography",
    "Hiking Trails", "Camping Trips", "Museum Visits", "Board Game Nights", "Learning a Musical Instrument",
    "Morning Exercise", "Skincare Routines", "Getting a Haircut", "Quality of Sleep", "Taking Vitamins",
    "Dental Hygiene", "Fashion Trends", "Applying Sunscreen", "Mental Health Breaks", "Yoga Practice",
    "Rainy Weather", "Summer Heatwaves", "Winter Snowfall", "Recycling Habits", "Composting Waste",
    "Electricity Usage", "Water Conservation", "Seasonal Changes", "Spring Allergies", "Using Umbrellas",
    "Cat Behavior", "Pet Adoption", "Feeding Birds", "Visiting the Vet", "Dog Training",
    "Aquarium Maintenance", "Local Wildlife", "Farm Animals", "Beekeeping", "Horseback Riding",
    "Grocery Shopping", "Budgeting Finances", "Paying Taxes", "Setting Alarm Clocks", "Online Shopping",
    "Battery Life", "Lost Keys", "Mirror Reflections", "Renewing Passports", "Library Books",
]

# Base triples are (P, M, S). They are used as natural language term material.
DOMAIN_TRIPLES: dict[str, list[tuple[str, str, str]]] = {
    "food": [
        ("prepared foods", "bakery items", "sourdough loaves"),
        ("drink orders", "coffee orders", "double espresso orders"),
        ("vegetarian dishes", "salad bowls", "lunch specials"),
        ("savory dishes", "spicy curries", "pepper stews"),
        ("desserts", "chocolate cakes", "birthday menu items"),
        ("meat dishes", "vegetarian meals", "lentil salads"),
        ("raw foods", "baked goods", "rye loaves"),
        ("beverages", "solid foods", "sandwich orders"),
    ],
    "home": [
        ("household chores", "laundry tasks", "towel loads"),
        ("cleaning jobs", "vacuuming tasks", "hallway vacuuming tasks"),
        ("home projects", "painting jobs", "ceiling painting jobs"),
        ("plumbing repairs", "leaky faucet repairs", "kitchen sink repairs"),
        ("home maintenance tasks", "closet organizing tasks", "shelf labeling tasks"),
        ("outdoor chores", "ironing tasks", "shirt pressing tasks"),
        ("garden chores", "vacuuming tasks", "apartment chore lists"),
        ("decor decisions", "plumbing repairs", "maintenance reports"),
    ],
    "transport": [
        ("transport options", "public transport routes", "night tram lines"),
        ("commuting routines", "driving routines", "morning highway commutes"),
        ("vehicle care tasks", "car maintenance jobs", "oil change jobs"),
        ("travel plans", "airline itineraries", "multi city itineraries"),
        ("urban trips", "train journeys", "regional rail journeys"),
        ("air travel", "train journeys", "regional rail journeys"),
        ("bike rides", "airline itineraries", "multi city itineraries"),
        ("vehicle care tasks", "public transport routes", "city travel plans"),
    ],
    "office": [
        ("office tasks", "email tasks", "client follow up emails"),
        ("work activities", "office meetings", "budget review meetings"),
        ("work arrangements", "remote work schedules", "hybrid Friday schedules"),
        ("career steps", "job interviews", "final round interviews"),
        ("planning tasks", "deadline reviews", "release readiness reviews"),
        ("coffee breaks", "email tasks", "client follow up emails"),
        ("vacation days", "office meetings", "weekly calendars"),
        ("personal hobbies", "deadline reviews", "team schedules"),
    ],
    "social": [
        ("social events", "birthday parties", "surprise parties"),
        ("family occasions", "family dinners", "holiday meals"),
        ("relationship messages", "text messages", "good morning texts"),
        ("formal ceremonies", "weddings", "evening receptions"),
        ("online interactions", "social media posts", "birthday posts"),
        ("private conversations", "public posts", "birthday posts"),
        ("weekday errands", "wedding events", "evening receptions"),
        ("legal contracts", "friendly texts", "small talk messages"),
    ],
    "media": [
        ("entertainment activities", "movie nights", "Friday screenings"),
        ("interactive games", "video games", "cooperative games"),
        ("written works", "novels", "mystery novels"),
        ("public performances", "concerts", "orchestra concerts"),
        ("creative hobbies", "photography sessions", "portrait shoots"),
        ("silent activities", "concerts", "orchestra concerts"),
        ("outdoor hikes", "board games", "strategy games"),
        ("fictional works", "documentary films", "nature documentaries"),
    ],
    "health": [
        ("wellness routines", "morning exercise sessions", "short workouts"),
        ("skin care products", "cleansers", "gentle face washes"),
        ("health appointments", "dental checkups", "annual cleanings"),
        ("protective habits", "sunscreen applications", "beach day routines"),
        ("relaxation practices", "yoga sessions", "evening stretches"),
        ("medical treatments", "haircuts", "barber appointments"),
        ("sleep periods", "exercise sessions", "morning workouts"),
        ("food groups", "vitamin tablets", "daily supplements"),
    ],
    "environment": [
        ("environmental habits", "recycling routines", "paper sorting routines"),
        ("waste reduction practices", "composting habits", "kitchen scrap collections"),
        ("resource saving actions", "water conservation habits", "short shower routines"),
        ("weather events", "rainy days", "stormy afternoons"),
        ("seasonal conditions", "winter weather patterns", "snowfall warnings"),
        ("summer heatwaves", "winter weather patterns", "snowfall warnings"),
        ("dry conditions", "rainy days", "stormy afternoons"),
        ("trash disposal", "composting habits", "kitchen scrap collections"),
    ],
    "animals": [
        ("pets", "adopted animals", "shelter cats"),
        ("training activities", "dog training sessions", "recall practice sessions"),
        ("animal care visits", "veterinary appointments", "vaccination appointments"),
        ("farm animals", "dairy cows", "barn animals"),
        ("outdoor animals", "local wildlife", "garden birds"),
        ("wild animals", "house pets", "indoor cats"),
        ("bird feeders", "aquarium equipment", "filter pumps"),
        ("farm animals", "aquarium fish", "neon tetras"),
    ],
    "errands": [
        ("errands", "grocery trips", "weekly shopping trips"),
        ("financial tasks", "tax payments", "income tax filings"),
        ("online purchases", "online shopping orders", "bookstore orders"),
        ("official documents", "passport renewals", "embassy appointments"),
        ("borrowed items", "library books", "reserved novels"),
        ("free activities", "online purchases", "bookstore orders"),
        ("shopping trips", "tax filings", "income tax filings"),
        ("personal possessions", "library books", "reserved novels"),
    ],
}

TOPIC_TO_DOMAIN = {
    **dict.fromkeys(TOPICS[0:10], "food"),
    **dict.fromkeys(TOPICS[10:20], "home"),
    **dict.fromkeys(TOPICS[20:30], "transport"),
    **dict.fromkeys(TOPICS[30:40], "office"),
    **dict.fromkeys(TOPICS[40:50], "social"),
    **dict.fromkeys(TOPICS[50:60], "media"),
    **dict.fromkeys(TOPICS[60:70], "health"),
    **dict.fromkeys(TOPICS[70:80], "environment"),
    **dict.fromkeys(TOPICS[80:90], "animals"),
    **dict.fromkeys(TOPICS[90:100], "errands"),
}

ABSURD_TRIPLES = [
    ("silver shadows", "curious teapots", "rainy orchestras"),
    ("edible constellations", "singing spoons", "midnight kitchens"),
    ("plastic forests", "whispering boots", "museum pockets"),
    ("floating teacups", "invisible mittens", "backward comets"),
    ("tidy hurricanes", "sleeping lanterns", "attic festivals"),
    ("paper moons", "singing ladders", "transparent violins"),
    ("wooden oceans", "flying teacups", "festival baskets"),
    ("quiet thunderstorms", "singing clocks", "library windows"),
    ("friendly meteors", "glass bicycles", "garden pockets"),
    ("liquid mountains", "paper shoes", "kitchen moons"),
    ("velvet equations", "mirror planets", "copper melodies"),
    ("ceramic rainbows", "sleeping compasses", "paper submarines"),
    ("electric candles", "frozen radios", "violet staircases"),
]

PLAUSIBLE_MODIFIERS = [
    "local", "weekly", "monthly", "weekday", "weekend", "morning", "evening", "family", "shared", "scheduled",
    "ordinary", "routine", "urban", "neighborhood", "student", "office", "home", "school", "community", "seasonal",
    "planned", "regular", "daily", "public", "private", "digital", "printed", "online", "in-person", "annual",
    "short", "long", "early", "late", "informal", "formal", "practical", "common", "optional", "recurring",
    "central", "small", "large", "indoor", "outdoor", "regional", "personal", "professional", "budget", "calendar",
    "summer", "winter", "spring", "autumn", "quiet", "busy", "simple", "detailed", "confirmed", "draft",
    "main", "secondary", "temporary", "permanent", "updated", "archived", "reserved", "approved", "basic", "advanced",
]

ABSURD_MODIFIERS = [
    "crystal", "nocturnal", "backward", "invisible", "transparent", "clockwork", "floating", "singing", "sleeping", "upside-down",
    "silver", "purple", "paper", "wooden", "liquid", "neon", "hollow", "velvet", "electric", "mirror",
    "lunar", "copper", "feathered", "recursive", "silent", "glowing", "ceramic", "marble", "cloudy", "magnetic",
    "midnight", "attic", "festival", "garden", "library", "accordion", "rubber", "painted", "borrowed", "triangular",
]

CATEGORICAL_RE = re.compile(
    r"^(All|No|Some) .+ are( not)? .+\.$"
)


@dataclass(frozen=True)
class Schema:
    figure: int
    mood: str
    valid: bool
    name: str
    fallacy: str | None = None


@dataclass(frozen=True)
class Core:
    premises: list[str]
    conclusion: str
    schema: Schema
    terms: dict[Role, str]
    signature: str


def normalize_text(text: str) -> str:
    return " ".join(text.strip().split())


def hash_text(text: str) -> str:
    return hashlib.sha256(normalize_text(text).encode("utf-8")).hexdigest()


def clean_topic(topic: str) -> str:
    return topic.lower().replace("-", " ")


def sentence(form: Form, subject: str, predicate: str, *, conclusion: bool = False) -> str:
    prefix = "Therefore, " if conclusion else ""
    if form == "A":
        body = f"all {subject} are {predicate}"
    elif form == "E":
        body = f"no {subject} are {predicate}"
    elif form == "I":
        body = f"some {subject} are {predicate}"
    elif form == "O":
        body = f"some {subject} are not {predicate}"
    else:
        raise ValueError(f"Unknown categorical form: {form}")
    if not conclusion:
        body = body[0].upper() + body[1:]
    return f"{prefix}{body}."


def schema_statements(mood: str, figure: int) -> tuple[tuple[Form, Role, Role], tuple[Form, Role, Role], tuple[Form, Role, Role]]:
    if figure not in FIGURES:
        raise ValueError(f"Invalid figure: {figure}")
    if len(mood) != 3 or any(ch not in FORMS for ch in mood):
        raise ValueError(f"Invalid mood: {mood}")
    major_roles, minor_roles = FIGURES[figure]
    return (
        (mood[0], major_roles[0], major_roles[1]),
        (mood[1], minor_roles[0], minor_roles[1]),
        (mood[2], "S", "P"),
    )


def occupied(model: int, role: Role, cell: int) -> bool:
    # Cell bit layout: S=4, P=2, M=1. The model bitset says which Venn cells are non empty.
    mask = {"S": 4, "P": 2, "M": 1}[role]
    return bool(cell & mask)


def nonempty(model: int, role: Role) -> bool:
    return any((model & (1 << cell)) and occupied(model, role, cell) for cell in range(8))


def truth(form: Form, subject: Role, predicate: Role, model: int, semantics: Semantics) -> bool:
    subject_cells = [cell for cell in range(8) if model & (1 << cell) and occupied(model, subject, cell)]
    both_cells = [cell for cell in subject_cells if occupied(model, predicate, cell)]
    outside_cells = [cell for cell in subject_cells if not occupied(model, predicate, cell)]

    if form == "A":
        if semantics == "aristotelian" and not subject_cells:
            return False
        return not outside_cells
    if form == "E":
        if semantics == "aristotelian" and not subject_cells:
            return False
        return not both_cells
    if form == "I":
        return bool(both_cells)
    if form == "O":
        return bool(outside_cells)
    raise ValueError(f"Unknown categorical form: {form}")


def is_schema_valid(mood: str, figure: int, semantics: Semantics) -> bool:
    major, minor, conclusion_stmt = schema_statements(mood, figure)
    for model in range(1 << 8):
        major_true = truth(*major, model=model, semantics=semantics)
        minor_true = truth(*minor, model=model, semantics=semantics)
        conclusion_true = truth(*conclusion_stmt, model=model, semantics=semantics)
        if major_true and minor_true and not conclusion_true:
            return False
    return True


def distributed_terms(form: Form, subject: Role, predicate: Role) -> set[Role]:
    if form == "A":
        return {subject}
    if form == "E":
        return {subject, predicate}
    if form == "I":
        return set()
    if form == "O":
        return {predicate}
    raise ValueError(form)


def classify_invalid_schema(mood: str, figure: int, semantics: Semantics) -> str:
    major, minor, conclusion_stmt = schema_statements(mood, figure)
    forms = [major[0], minor[0], conclusion_stmt[0]]
    negative_premises = sum(form in {"E", "O"} for form in forms[:2])
    particular_premises = sum(form in {"I", "O"} for form in forms[:2])
    conclusion_negative = forms[2] in {"E", "O"}
    conclusion_particular = forms[2] in {"I", "O"}

    major_dist = distributed_terms(*major)
    minor_dist = distributed_terms(*minor)
    conclusion_dist = distributed_terms(*conclusion_stmt)
    premise_dist = major_dist | minor_dist

    if negative_premises == 2:
        return "exclusive_premises"
    if negative_premises == 1 and not conclusion_negative:
        return "affirmative_conclusion_from_negative_premise"
    if negative_premises == 0 and conclusion_negative:
        return "negative_conclusion_from_affirmative_premises"
    if particular_premises == 2:
        return "two_particular_premises"
    if "M" not in premise_dist:
        return "undistributed_middle"
    if "P" in conclusion_dist and "P" not in major_dist:
        return "illicit_major"
    if "S" in conclusion_dist and "S" not in minor_dist:
        return "illicit_minor"
    if semantics == "modern" and conclusion_particular and particular_premises == 0:
        return "existential_fallacy"
    return "other_invalid_mood_figure"


def build_schema_catalog(semantics: Semantics) -> tuple[list[Schema], list[Schema]]:
    valid: list[Schema] = []
    invalid: list[Schema] = []
    for figure in sorted(FIGURES):
        for mood_tuple in itertools.product(FORMS, repeat=3):
            mood = "".join(mood_tuple)
            schema_valid = is_schema_valid(mood, figure, semantics)
            name = SCHEMA_NAMES.get((figure, mood), f"{mood}_{figure}")
            if schema_valid:
                valid.append(Schema(figure=figure, mood=mood, valid=True, name=name))
            else:
                invalid.append(
                    Schema(
                        figure=figure,
                        mood=mood,
                        valid=False,
                        name=name,
                        fallacy=classify_invalid_schema(mood, figure, semantics),
                    )
                )
    valid.sort(key=lambda s: (s.figure, s.mood, s.name))
    invalid.sort(key=lambda s: (s.fallacy or "", s.figure, s.mood, s.name))
    return valid, invalid


def choose_base_triple(topic: str, plausibility: bool, rng: random.Random) -> tuple[str, str, str]:
    if not plausibility:
        return rng.choice(ABSURD_TRIPLES)
    domain = TOPIC_TO_DOMAIN.get(topic, "errands")
    return rng.choice(DOMAIN_TRIPLES[domain])


def vary_terms(base: tuple[str, str, str], plausibility: bool, rng: random.Random) -> dict[Role, str]:
    modifiers = PLAUSIBLE_MODIFIERS if plausibility else ABSURD_MODIFIERS
    # Use two modifiers often enough to make exact premise repetition rare, but keep phrases readable.
    m1 = rng.choice(modifiers)
    m2 = rng.choice(modifiers)
    while m2 == m1:
        m2 = rng.choice(modifiers)
    style = rng.randrange(3)
    if style == 0:
        prefix = m1
    elif style == 1:
        prefix = f"{m1} {m2}"
    else:
        prefix = f"{m2} {m1}"

    p, m, s = base
    return {
        "P": f"{prefix} {p}",
        "M": f"{prefix} {m}",
        "S": f"{prefix} {s}",
    }


def build_core(schema: Schema, plausibility: bool, topic: str, rng: random.Random) -> Core:
    terms = vary_terms(choose_base_triple(topic, plausibility, rng), plausibility, rng)
    major, minor, conclusion_stmt = schema_statements(schema.mood, schema.figure)

    premises = [
        sentence(major[0], terms[major[1]], terms[major[2]]),
        sentence(minor[0], terms[minor[1]], terms[minor[2]]),
    ]
    conclusion_text = sentence(conclusion_stmt[0], terms[conclusion_stmt[1]], terms[conclusion_stmt[2]], conclusion=True)
    signature = normalize_text(" ".join(premises + [conclusion_text])).lower()
    return Core(premises=premises, conclusion=conclusion_text, schema=schema, terms=terms, signature=signature)


def all_pairs(num_premises: int) -> list[tuple[int, int]]:
    return [(i, j) for i in range(num_premises) for j in range(i + 1, num_premises)]


def choose_relevant_positions(num_premises: int, serial: int) -> tuple[int, int]:
    pairs = all_pairs(num_premises)
    # Rotate through all pairs. Sorting emphasizes later positions early, but all pairs are used.
    pairs.sort(key=lambda pair: (-pair[1], -pair[0], pair[0]))
    return pairs[serial % len(pairs)]


def build_categorical_distractors(
    needed: int,
    plausibility: bool,
    rng: random.Random,
    used_sentences: set[str],
    topic: str,
) -> list[str]:
    distractors: list[str] = []
    attempts = 0
    while len(distractors) < needed:
        attempts += 1
        if attempts > 1000:
            raise RuntimeError("Could not build enough unique categorical distractors.")

        # Use a domain not necessarily equal to the core topic domain to reduce accidental term overlap.
        if plausibility:
            domain = rng.choice(list(DOMAIN_TRIPLES))
            base = rng.choice(DOMAIN_TRIPLES[domain])
        else:
            base = rng.choice(ABSURD_TRIPLES)
        terms = vary_terms(base, plausibility, rng)
        roles = rng.sample(["S", "P", "M"], 2)
        form = rng.choice(FORMS)
        candidate = sentence(form, terms[roles[0]], terms[roles[1]])
        key = normalize_text(candidate).lower()
        if key in used_sentences:
            continue
        if " of them " in key:
            continue
        if not CATEGORICAL_RE.match(candidate):
            continue
        used_sentences.add(key)
        distractors.append(candidate)
    return distractors


def make_example(
    example_index: int,
    num_premises: int,
    schema: Schema,
    validity: bool,
    plausibility: bool,
    rng: random.Random,
    include_id: bool,
    include_metadata: bool,
    position_serial: int,
    semantics: Semantics,
) -> dict[str, Any]:
    if num_premises < 2:
        raise ValueError("Subtask 2 examples need at least two premises.")

    topic = TOPICS[example_index % len(TOPICS)]
    core = build_core(schema, plausibility, topic, rng)

    positions = choose_relevant_positions(num_premises, position_serial)
    premises: list[str | None] = [None] * num_premises
    premises[positions[0]] = core.premises[0]
    premises[positions[1]] = core.premises[1]

    used_sentences = {normalize_text(p).lower() for p in core.premises}
    distractors = build_categorical_distractors(num_premises - 2, plausibility, rng, used_sentences, topic)
    distractor_iter = iter(distractors)
    for i, premise in enumerate(premises):
        if premise is None:
            premises[i] = next(distractor_iter)

    final_premises = [str(p) for p in premises]
    syllogism = normalize_text(" ".join(final_premises + [core.conclusion]))
    relevant_premises = list(positions) if validity else []

    metadata = {
        "semantics": semantics,
        "schema_name": schema.name,
        "figure": schema.figure,
        "mood": schema.mood,
        "fallacy": schema.fallacy,
        "candidate_premises": list(positions),
        "core_signature": core.signature,
        "core_premises": core.premises,
        "core_conclusion": core.conclusion,
        "topic": topic,
        "premise_count": num_premises,
    }

    example: dict[str, Any] = {
        "syllogism": syllogism,
        "validity": validity,
        "plausibility": plausibility,
        "relevant_premises": relevant_premises,
    }
    if include_id:
        example = {"id": hash_text(syllogism), **example}
    # Always keep private metadata while generating. Strip later unless requested.
    example["_metadata"] = metadata
    if include_metadata:
        example["metadata"] = metadata
    return example


def validation_errors(examples: list[dict[str, Any]], semantics: Semantics) -> list[str]:
    errors: list[str] = []
    seen_ids: set[str] = set()
    seen_syllogisms: set[str] = set()
    seen_cores: set[str] = set()

    for i, example in enumerate(examples):
        prefix = f"example {i}"
        syllogism = example.get("syllogism")
        if not isinstance(syllogism, str):
            errors.append(f"{prefix}: syllogism is not a string")
            continue

        normalized_syllogism = normalize_text(syllogism).lower()
        if normalized_syllogism in seen_syllogisms:
            errors.append(f"{prefix}: duplicate full syllogism")
        seen_syllogisms.add(normalized_syllogism)

        if "id" in example:
            if not isinstance(example["id"], str):
                errors.append(f"{prefix}: id is not a string")
            elif example["id"] in seen_ids:
                errors.append(f"{prefix}: duplicate id {example['id']}")
            seen_ids.add(str(example.get("id")))

        metadata = example.get("_metadata")
        if not isinstance(metadata, dict):
            errors.append(f"{prefix}: missing internal metadata")
            continue

        core_signature = metadata.get("core_signature")
        if not isinstance(core_signature, str):
            errors.append(f"{prefix}: missing core signature")
        elif core_signature in seen_cores:
            errors.append(f"{prefix}: duplicate core syllogism")
        else:
            seen_cores.add(core_signature)

        premise_count = metadata.get("premise_count")
        if not isinstance(premise_count, int):
            errors.append(f"{prefix}: invalid premise_count metadata")
            continue

        parts = [part.strip() for part in syllogism.split(".") if part.strip()]
        if len(parts) != premise_count + 1:
            errors.append(f"{prefix}: expected {premise_count + 1} sentences, found {len(parts)}")
            continue

        for premise in parts[:premise_count]:
            if not CATEGORICAL_RE.match(premise + "."):
                errors.append(f"{prefix}: non-categorical premise: {premise}.")

        conclusion = parts[-1]
        if not conclusion.startswith("Therefore,"):
            errors.append(f"{prefix}: conclusion lacks Therefore marker")
        if " of them " in normalized_syllogism:
            errors.append(f"{prefix}: ambiguous 'of them' phrase")

        validity = example.get("validity")
        relevant = example.get("relevant_premises")
        if not isinstance(validity, bool):
            errors.append(f"{prefix}: validity is not bool")
        if not isinstance(example.get("plausibility"), bool):
            errors.append(f"{prefix}: plausibility is not bool")
        if not isinstance(relevant, list):
            errors.append(f"{prefix}: relevant_premises is not list")
        else:
            if validity and len(relevant) != 2:
                errors.append(f"{prefix}: valid example must have two relevant premises")
            if not validity and relevant:
                errors.append(f"{prefix}: invalid example must have empty relevant_premises")
            for value in relevant:
                if not isinstance(value, int) or not (0 <= value < premise_count):
                    errors.append(f"{prefix}: bad relevant premise index {value}")

        mood = metadata.get("mood")
        figure = metadata.get("figure")
        if isinstance(mood, str) and isinstance(figure, int):
            schema_valid = is_schema_valid(mood, figure, semantics)
            if schema_valid != validity:
                errors.append(f"{prefix}: schema validity mismatch for {mood}-{figure}")

    return errors


def strip_internal_metadata(examples: list[dict[str, Any]], include_metadata: bool) -> list[dict[str, Any]]:
    stripped: list[dict[str, Any]] = []
    for example in examples:
        item = dict(example)
        internal_metadata = item.pop("_metadata", None)
        if include_metadata:
            if "metadata" not in item and internal_metadata is not None:
                item["metadata"] = internal_metadata
        else:
            item.pop("metadata", None)
        stripped.append(item)
    return stripped


def audit_examples(examples: list[dict[str, Any]], semantics: Semantics) -> dict[str, Any]:
    premise_counter: Counter[int] = Counter()
    validity_counter: Counter[str] = Counter()
    plausibility_counter: Counter[str] = Counter()
    combo_counter: Counter[str] = Counter()
    schema_counter: Counter[str] = Counter()
    mood_counter: Counter[str] = Counter()
    figure_counter: Counter[str] = Counter()
    fallacy_counter: Counter[str] = Counter()
    relevant_pair_counter: Counter[str] = Counter()
    candidate_pair_counter: Counter[str] = Counter()
    premise_text_counter: Counter[str] = Counter()
    core_counter: Counter[str] = Counter()
    noncategorical_premises = 0
    ambiguous_them = 0

    for example in examples:
        meta = example["_metadata"]
        premise_count = meta["premise_count"]
        premise_counter[premise_count] += 1
        validity_counter[str(example["validity"])] += 1
        plausibility_counter[str(example["plausibility"])] += 1
        combo_counter[f"validity={example['validity']},plausibility={example['plausibility']}"] += 1
        schema_counter[meta["schema_name"]] += 1
        mood_counter[f"{meta['mood']}-{meta['figure']}"] += 1
        figure_counter[str(meta["figure"])] += 1
        if meta.get("fallacy"):
            fallacy_counter[meta["fallacy"]] += 1
        if example["validity"]:
            relevant_pair_counter[str(tuple(example["relevant_premises"]))] += 1
        candidate_pair_counter[str(tuple(meta["candidate_premises"]))] += 1
        core_counter[meta["core_signature"]] += 1

        parts = [part.strip() + "." for part in example["syllogism"].split(".") if part.strip()]
        for premise in parts[:premise_count]:
            key = normalize_text(premise).lower()
            premise_text_counter[key] += 1
            if not CATEGORICAL_RE.match(premise):
                noncategorical_premises += 1
        if " of them " in normalize_text(example["syllogism"]).lower():
            ambiguous_them += 1

    duplicate_cores = {core: count for core, count in core_counter.items() if count > 1}
    duplicate_full = len(examples) - len({normalize_text(e["syllogism"]).lower() for e in examples})
    validation = validation_errors(examples, semantics)

    return {
        "semantics": semantics,
        "num_examples": len(examples),
        "premise_count_distribution": dict(sorted(premise_counter.items())),
        "validity_counts": dict(validity_counter),
        "plausibility_counts": dict(plausibility_counter),
        "validity_plausibility_counts": dict(combo_counter),
        "figure_counts": dict(sorted(figure_counter.items())),
        "mood_figure_counts": dict(mood_counter),
        "schema_counts": dict(schema_counter),
        "fallacy_counts": dict(fallacy_counter),
        "relevant_pair_counts_valid_only": dict(relevant_pair_counter),
        "candidate_pair_counts_all_examples": dict(candidate_pair_counter),
        "duplicate_core_count": sum(count - 1 for count in duplicate_cores.values()),
        "duplicate_full_syllogism_count": duplicate_full,
        "ambiguous_them_count": ambiguous_them,
        "noncategorical_premise_count": noncategorical_premises,
        "max_exact_premise_repetition": max(premise_text_counter.values(), default=0),
        "most_repeated_premises": premise_text_counter.most_common(20),
        "validation_error_count": len(validation),
        "validation_errors_preview": validation[:30],
    }


def split_counts(total: int, premise_counts: list[int]) -> dict[int, int]:
    base = total // len(premise_counts)
    remainder = total % len(premise_counts)
    return {p: base + (1 if i < remainder else 0) for i, p in enumerate(premise_counts)}


def generate_subtask2_examples(
    count: int,
    premise_counts: list[int],
    seed: int = 42,
    semantics: Semantics = "modern",
    include_id: bool = True,
    include_metadata: bool = False,
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    valid_schemas, invalid_schemas = build_schema_catalog(semantics)
    if not valid_schemas:
        raise RuntimeError("No valid schemas found. This should be impossible.")
    if not invalid_schemas:
        raise RuntimeError("No invalid schemas found. This should be impossible.")

    counts_by_premise = split_counts(count, premise_counts)
    examples: list[dict[str, Any]] = []
    seen_cores: set[str] = set()
    seen_syllogisms: set[str] = set()
    schema_serial: Counter[tuple[bool, bool]] = Counter()
    position_serial: Counter[int] = Counter()

    # Generate in target strata: each premise count bucket cycles through the four validity/plausibility combinations.
    for premise_count in premise_counts:
        target = counts_by_premise[premise_count]
        local_index = 0
        attempts = 0
        while local_index < target:
            attempts += 1
            if attempts > target * 500:
                raise RuntimeError(
                    f"Too many duplicate generations for {premise_count} premises; increase lexical/schema variation."
                )

            cycle = local_index % 4
            validity = cycle in {0, 2}
            plausibility = cycle in {0, 1}

            catalog = valid_schemas if validity else invalid_schemas
            serial_key = (validity, plausibility)
            schema = catalog[schema_serial[serial_key] % len(catalog)]
            schema_serial[serial_key] += 1

            example = make_example(
                example_index=len(examples) + attempts,
                num_premises=premise_count,
                schema=schema,
                validity=validity,
                plausibility=plausibility,
                rng=rng,
                include_id=include_id,
                include_metadata=include_metadata,
                position_serial=position_serial[premise_count],
                semantics=semantics,
            )
            position_serial[premise_count] += 1

            core_sig = example["_metadata"]["core_signature"]
            syllo_sig = normalize_text(example["syllogism"]).lower()
            if core_sig in seen_cores or syllo_sig in seen_syllogisms:
                continue
            seen_cores.add(core_sig)
            seen_syllogisms.add(syllo_sig)
            examples.append(example)
            local_index += 1

    errors = validation_errors(examples, semantics)
    if errors:
        raise ValueError("Generated data failed validation:\n" + "\n".join(errors[:30]))
    return examples


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.write("\n")


def save_jsonl(path: Path, data: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def print_summary(audit: dict[str, Any]) -> None:
    print(f"Generated examples: {audit['num_examples']}")
    print(f"Semantics: {audit['semantics']}")
    print(f"Premise counts: {audit['premise_count_distribution']}")
    print(f"Validity counts: {audit['validity_counts']}")
    print(f"Plausibility counts: {audit['plausibility_counts']}")
    print(f"Figure counts: {audit['figure_counts']}")
    print(f"Fallacy counts: {audit['fallacy_counts']}")
    print(f"Duplicate core count: {audit['duplicate_core_count']}")
    print(f"Duplicate full syllogism count: {audit['duplicate_full_syllogism_count']}")
    print(f"Ambiguous 'of them' count: {audit['ambiguous_them_count']}")
    print(f"Non-categorical premise count: {audit['noncategorical_premise_count']}")
    print(f"Max exact premise repetition: {audit['max_exact_premise_repetition']}")
    print(f"Validation error count: {audit['validation_error_count']}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate high diversity Subtask 2 syllogism data.")
    parser.add_argument("-n", "--count", type=int, default=1200, help="Number of examples to generate.")
    parser.add_argument(
        "-p", "--premises", type=int, nargs="+", default=[5, 6, 7],
        help="Premise counts to generate. Example: -p 5 6 7. Counts are split as evenly as possible.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducible generation.")
    parser.add_argument(
        "--semantics", choices=["modern", "aristotelian"], default="modern",
        help="modern = no existential import; aristotelian = universal statements imply subject existence.",
    )
    parser.add_argument("-o", "--output", type=Path, default=Path("data/2/data.json"))
    parser.add_argument("--jsonl-output", type=Path, default=None)
    parser.add_argument("--audit-output", type=Path, default=Path("data/2/data_audit.json"))
    parser.add_argument("--metadata-output", type=Path, default=None, help="Optional full output retaining metadata.")
    parser.add_argument("--include-metadata", action="store_true", help="Include metadata inside the main JSON examples.")
    parser.add_argument("--no-id", action="store_true", help="Do not include id fields.")
    args = parser.parse_args()

    examples = generate_subtask2_examples(
        count=args.count,
        premise_counts=args.premises,
        seed=args.seed,
        semantics=args.semantics,
        include_id=not args.no_id,
        include_metadata=args.include_metadata,
    )
    audit = audit_examples(examples, args.semantics)

    output_examples = strip_internal_metadata(examples, include_metadata=args.include_metadata)
    save_json(args.output, output_examples)
    if args.jsonl_output:
        save_jsonl(args.jsonl_output, output_examples)
    if args.audit_output:
        save_json(args.audit_output, audit)
    if args.metadata_output:
        metadata_examples = strip_internal_metadata(examples, include_metadata=True)
        save_json(args.metadata_output, metadata_examples)

    print(f"Saved JSON to: {args.output}")
    if args.audit_output:
        print(f"Saved audit to: {args.audit_output}")
    if args.metadata_output:
        print(f"Saved metadata JSON to: {args.metadata_output}")
    print_summary(audit)


if __name__ == "__main__":
    main()
