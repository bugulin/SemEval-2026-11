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

Goals:
- generate Subtask 2 examples without local/api LLM calls.
- keep synthetic validity labels correct by formal model checking.
- add irrelevant premises and compute zero based relevant_premises mechanically.
- support both fully synthetic examples and UFAL derived examples with injected distractors.
- reduce shortcut prone surface patterns by using varied categorical and conclusion templates.
- output JSON, optionally JSONL, plus audit and metadata rich files.

This intentionally does not ask a language model whether a syllogism is valid.

For synthetic examples, the generator chooses a formal A/E/I/O schema first,
model checks its validity under the selected semantics, then realizes it as
natural language premises and a conclusion.

For UFAL integrated examples, the generator preserves the original UFAL
two premise syllogism and label, optionally swaps the two core premises,
injects unrelated synthetic distractors, and computes relevant_premises after
placement. UFAL examples are not formally re-parsed into A/E/I/O. Their labels
come from the cleaned UFAL dataset.

Formal convention used internally for synthetic examples:
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

Figure 4 is defined internally for the later modern 4 figure convention.

Semantics:
- modern: evaluate validity over all Boolean models of S/P/M.
- aristotelian: evaluate validity only over models where S, P, and M are all
  nonempty. This gives the traditional existential-import behavior and makes
  E/I conversion symmetric while preserving A/O directionality.

Invalid schemata are not manually listed. The synthetic schema catalog is built
by enumerating all A/E/I/O mood triples for the selected figures and
model checking each one. classify_invalid_schema() only assigns a fallacy label
to schemas already determined invalid.
"""

"""
ARGUMENTS:

-n / --count              number of examples to generate
-p / --premises           one or more target premise counts, e.g. -p 2 3 4 5 6 7
--seed                    random seed for reproducible generation
--semantics               validity semantics: modern or aristotelian
--figures                 syllogistic figures to use, e.g. --figures 1 2 3
--style                   surface style: canonical, varied, or mixed
--source-mode             data source: synthetic, ufal, or mixed
--ufal-input              path to cleaned UFAL train set for ufal/mixed mode
--ufal-ratio              fraction of examples generated from UFAL in mixed mode
-o / --output             main JSON output path
--jsonl-output            optional JSONL output path
--audit-output            audit report output path
--metadata-output         metadata-rich JSON output path
--include-metadata        include metadata directly in the main JSON output
--no-id                   omit id fields from generated examples
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

# Used only for word overlap checks when injecting UFAL distractors.
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "can", "from", "in", "is", "it", "its", "no", "not", "of",
    "on", "or", "some", "that", "the", "there", "this", "to", "with", "all", "any", "every", "single", "thing",
    "things", "item", "items", "member", "members", "among", "included", "classified", "considered", "known",
    "therefore", "consequently", "thus", "follows", "conclusion", "conclude", "must", "case", "true", "fact",
}

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

def content_words(text: str) -> set[str]:
    words = re.findall(r"[A-Za-z][A-Za-z'-]*", text.lower())
    return {w for w in words if len(w) > 3 and w not in STOPWORDS}

def split_sentences(text: str) -> list[str]:
    """Split the task's short syllogism strings into sentence-like units, preserving punctuation."""
    normalized = normalize_text(text)
    pieces = re.findall(r"[^.!?]+[.!?]", normalized)
    if pieces:
        return [piece.strip() for piece in pieces]
    # Fallback for malformed inputs without final punctuation.
    rough = [part.strip() for part in re.split(r"(?<=[.!?])\s+|\.\s*", normalized) if part.strip()]
    return [part if part.endswith((".", "!", "?")) else part + "." for part in rough]

def capitalize_first(text: str) -> str:
    return text[:1].upper() + text[1:] if text else text

def categorical_body(form: Form, subject: str, predicate: str, *, rng: random.Random | None = None, style: str = "canonical") -> str:
    """Return a lower-case categorical statement body without final punctuation.

    style="canonical" keeps the original all/no/some template.
    style="varied" samples paraphrases to reduce surface-form overfitting.
    style="mixed" uses both.
    """
    if style == "mixed":
        style = "canonical" if (rng is None or rng.random() < 0.35) else "varied"
    if style == "canonical" or rng is None:
        if form == "A":
            return f"all {subject} are {predicate}"
        if form == "E":
            return f"no {subject} are {predicate}"
        if form == "I":
            return f"some {subject} are {predicate}"
        if form == "O":
            return f"some {subject} are not {predicate}"
        raise ValueError(f"Unknown categorical form: {form}")

    templates: dict[Form, list[str]] = {
        "A": [
            "all {S} are {P}",
            "every member of {S} is included among {P}",
            "anything that is one of {S} is also one of {P}",
            "each item classified as {S} is also classified as {P}",
            "the group of {S} is contained in the group of {P}",
        ],
        "E": [
            "no {S} are {P}",
            "there are no {S} that are {P}",
            "not a single member of {S} is among {P}",
            "nothing classified as {S} is also classified as {P}",
            "the groups {S} and {P} do not overlap",
        ],
        "I": [
            "some {S} are {P}",
            "at least some {S} are {P}",
            "a portion of {S} are also {P}",
            "there are {S} that are {P}",
            "some members of {S} are included among {P}",
        ],
        "O": [
            "some {S} are not {P}",
            "at least some {S} are not {P}",
            "a portion of {S} are outside the group of {P}",
            "not all {S} are {P}",
            "some members of {S} fail to be {P}",
        ],
    }
    return rng.choice(templates[form]).format(S=subject, P=predicate)

def sentence(
    form: Form,
    subject: str,
    predicate: str,
    *,
    conclusion: bool = False,
    rng: random.Random | None = None,
    style: str = "canonical",
) -> str:
    body = categorical_body(form, subject, predicate, rng=rng, style=style)
    if not conclusion:
        return f"{capitalize_first(body)}."
    if style == "mixed":
        style_for_conclusion = "canonical" if (rng is None or rng.random() < 0.35) else "varied"
    else:
        style_for_conclusion = style
    if style_for_conclusion == "canonical" or rng is None:
        return f"Therefore, {body}."
    conclusion_templates = [
        "Therefore, {body}.",
        "It follows that {body}.",
        "Consequently, {body}.",
        "Thus, {body}.",
        "The conclusion is that {body}.",
        "From this, we can conclude that {body}.",
        "The result is that {body}.",
    ]
    return rng.choice(conclusion_templates).format(body=body)

def allowed_figures(figures: list[int] | None) -> list[int]:
    chosen = sorted(set(figures if figures is not None else FIGURES.keys()))
    bad = [figure for figure in chosen if figure not in FIGURES]
    if bad:
        raise ValueError(f"Unsupported figure(s): {bad}. Supported figures are {sorted(FIGURES)}")
    return chosen


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
    # Cell bit layout: S=4, P=2, M=1. The model bitset says which Venn cells are non-empty.
    mask = {"S": 4, "P": 2, "M": 1}[role]
    return bool(cell & mask)


def nonempty(model: int, role: Role) -> bool:
    return any((model & (1 << cell)) and occupied(model, role, cell) for cell in range(8))


def model_allowed(model: int, semantics: Semantics) -> bool:
    # Aristotelian mode assumes all three syllogistic terms are non-empty.
    if semantics == "aristotelian":
        return all(nonempty(model, role) for role in ("S", "P", "M"))
    return True

def truth(form: Form, subject: Role, predicate: Role, model: int, semantics: Semantics) -> bool:
    subject_cells = [cell for cell in range(8) if model & (1 << cell) and occupied(model, subject, cell)]
    both_cells = [cell for cell in subject_cells if occupied(model, predicate, cell)]
    outside_cells = [cell for cell in subject_cells if not occupied(model, predicate, cell)]

    if form == "A":
        return not outside_cells
    if form == "E":
        return not both_cells
    if form == "I":
        return bool(both_cells)
    if form == "O":
        return bool(outside_cells)
    raise ValueError(f"Unknown categorical form: {form}")


def is_schema_valid(mood: str, figure: int, semantics: Semantics) -> bool:
    major, minor, conclusion_stmt = schema_statements(mood, figure)
    for model in range(1 << 8):
        if not model_allowed(model, semantics):
            continue
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


def build_schema_catalog(semantics: Semantics, figures: list[int] | None = None) -> tuple[list[Schema], list[Schema]]:
    valid: list[Schema] = []
    invalid: list[Schema] = []
    for figure in allowed_figures(figures):
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


def build_core(schema: Schema, plausibility: bool, topic: str, rng: random.Random, style: str) -> Core:
    terms = vary_terms(choose_base_triple(topic, plausibility, rng), plausibility, rng)
    major, minor, conclusion_stmt = schema_statements(schema.mood, schema.figure)

    premises = [
        sentence(major[0], terms[major[1]], terms[major[2]], rng=rng, style=style),
        sentence(minor[0], terms[minor[1]], terms[minor[2]], rng=rng, style=style),
    ]
    conclusion_text = sentence(
        conclusion_stmt[0],
        terms[conclusion_stmt[1]],
        terms[conclusion_stmt[2]],
        conclusion=True,
        rng=rng,
        style=style,
    )
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
    *,
    style: str = "mixed",
    avoid_words: set[str] | None = None,
) -> list[str]:
    distractors: list[str] = []
    attempts = 0
    avoid_words = avoid_words or set()
    while len(distractors) < needed:
        attempts += 1
        if attempts > 3000:
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
        candidate = sentence(form, terms[roles[0]], terms[roles[1]], rng=rng, style=style)
        key = normalize_text(candidate).lower()
        if key in used_sentences:
            continue
        if " of them " in key:
            continue
        # For UFAL integration, reject distractors with meaningful lexical overlap with the original core.
        if avoid_words and (content_words(candidate) & avoid_words):
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
    *,
    style: str = "mixed",
    swap_core_order: bool = True,
) -> dict[str, Any]:
    if num_premises < 2:
        raise ValueError("Subtask 2 examples need at least two premises.")

    topic = TOPICS[example_index % len(TOPICS)]
    core = build_core(schema, plausibility, topic, rng, style)

    positions = choose_relevant_positions(num_premises, position_serial)
    core_premises = list(core.premises)
    core_order = "major_minor"
    if swap_core_order and rng.random() < 0.5:
        core_premises.reverse()
        core_order = "minor_major"

    premises: list[str | None] = [None] * num_premises
    premises[positions[0]] = core_premises[0]
    premises[positions[1]] = core_premises[1]

    used_sentences = {normalize_text(p).lower() for p in core.premises}
    distractors = build_categorical_distractors(
        num_premises - 2,
        plausibility,
        rng,
        used_sentences,
        topic,
        style=style,
    )
    distractor_iter = iter(distractors)
    for i, premise in enumerate(premises):
        if premise is None:
            premises[i] = next(distractor_iter)

    final_premises = [str(p) for p in premises]
    syllogism = normalize_text(" ".join(final_premises + [core.conclusion]))
    relevant_premises = list(positions) if validity else []

    metadata = {
        "source": "synthetic",
        "semantics": semantics,
        "style": style,
        "schema_name": schema.name,
        "figure": schema.figure,
        "mood": schema.mood,
        "fallacy": schema.fallacy,
        "candidate_premises": list(positions),
        "core_signature": core.signature,
        "core_premises": core.premises,
        "core_premise_order_in_output": core_order,
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


def build_ufal_integrated_example(
    base: dict[str, Any],
    base_index: int,
    num_premises: int,
    rng: random.Random,
    include_id: bool,
    include_metadata: bool,
    position_serial: int,
    *,
    style: str = "mixed",
) -> dict[str, Any] | None:
    syllogism = base.get("syllogism")
    if not isinstance(syllogism, str):
        return None
    sentences = split_sentences(syllogism)
    if len(sentences) != 3:
        return None
    if num_premises < 2:
        raise ValueError("UFAL integration needs at least two premises.")

    core_premises = sentences[:2]
    conclusion = sentences[2]
    core_order = "original"
    if rng.random() < 0.5:
        core_premises = list(reversed(core_premises))
        core_order = "swapped"

    validity = bool(base.get("validity"))
    plausibility = bool(base.get("plausibility"))
    positions = choose_relevant_positions(num_premises, position_serial)
    premises: list[str | None] = [None] * num_premises
    premises[positions[0]] = core_premises[0]
    premises[positions[1]] = core_premises[1]

    used_sentences = {normalize_text(p).lower() for p in core_premises}
    avoid = content_words(" ".join(sentences))
    distractors = build_categorical_distractors(
        num_premises - 2,
        plausibility,
        rng,
        used_sentences,
        topic=TOPICS[base_index % len(TOPICS)],
        style=style,
        avoid_words=avoid,
    )
    distractor_iter = iter(distractors)
    for i, premise in enumerate(premises):
        if premise is None:
            premises[i] = next(distractor_iter)

    final_premises = [str(p) for p in premises]
    integrated_syllogism = normalize_text(" ".join(final_premises + [conclusion]))
    relevant_premises = list(positions) if validity else []
    base_id = str(base.get("id", hash_text(syllogism)))

    metadata = {
        "source": "ufal_integrated",
        "style": style,
        "base_id": base_id,
        "base_index": base_index,
        "base_syllogism": syllogism,
        "base_premises": sentences[:2],
        "base_conclusion": conclusion,
        "core_signature": normalize_text(syllogism).lower(),
        "core_premises": sentences[:2],
        "core_premise_order_in_output": core_order,
        "candidate_premises": list(positions),
        "num_injected_distractors": num_premises - 2,
        "premise_count": num_premises,
    }

    example: dict[str, Any] = {
        "syllogism": integrated_syllogism,
        "validity": validity,
        "plausibility": plausibility,
        "relevant_premises": relevant_premises,
    }
    if include_id:
        example = {"id": hash_text(integrated_syllogism), **example}
    example["_metadata"] = metadata
    if include_metadata:
        example["metadata"] = metadata
    return example


def load_ufal_examples(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"UFAL input must be a JSON array: {path}")
    usable: list[dict[str, Any]] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        if isinstance(item.get("syllogism"), str) and isinstance(item.get("validity"), bool):
            usable.append(item)
    if not usable:
        raise ValueError(f"No usable UFAL examples found in {path}")
    return usable


def validation_errors(examples: list[dict[str, Any]], semantics: Semantics, figures: list[int] | None = None) -> list[str]:
    errors: list[str] = []
    seen_ids: set[str] = set()
    seen_syllogisms: set[str] = set()
    seen_cores: set[str] = set()
    figure_set = set(allowed_figures(figures)) if figures is not None else None

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
        elif (metadata.get("source") == "synthetic") and core_signature in seen_cores:
            errors.append(f"{prefix}: duplicate synthetic core syllogism")
        else:
            seen_cores.add(core_signature)

        premise_count = metadata.get("premise_count")
        if not isinstance(premise_count, int):
            errors.append(f"{prefix}: invalid premise_count metadata")
            continue

        parts = split_sentences(syllogism)
        if len(parts) != premise_count + 1:
            errors.append(f"{prefix}: expected {premise_count + 1} sentences, found {len(parts)}")
            continue

        # "of them" is tracked in the audit. It is not a hard error just in case the UFAL set might contain it.

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
            if figure_set is not None and figure not in figure_set:
                errors.append(f"{prefix}: figure {figure} is outside allowed figures {sorted(figure_set)}")
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


def audit_examples(examples: list[dict[str, Any]], semantics: Semantics, figures: list[int] | None = None) -> dict[str, Any]:
    premise_counter: Counter[int] = Counter()
    source_counter: Counter[str] = Counter()
    style_counter: Counter[str] = Counter()
    validity_counter: Counter[str] = Counter()
    plausibility_counter: Counter[str] = Counter()
    combo_counter: Counter[str] = Counter()
    schema_counter: Counter[str] = Counter()
    mood_counter: Counter[str] = Counter()
    figure_counter: Counter[str] = Counter()
    fallacy_counter: Counter[str] = Counter()
    relevant_pair_counter: Counter[str] = Counter()
    candidate_pair_counter: Counter[str] = Counter()
    core_order_counter: Counter[str] = Counter()
    injected_counter: Counter[int] = Counter()
    premise_text_counter: Counter[str] = Counter()
    core_counter: Counter[str] = Counter()
    canonical_premises = 0
    noncanonical_premises = 0
    therefore_conclusions = 0
    non_therefore_conclusions = 0
    ambiguous_them = 0

    for example in examples:
        meta = example["_metadata"]
        source = str(meta.get("source", "unknown"))
        premise_count = int(meta["premise_count"])
        premise_counter[premise_count] += 1
        source_counter[source] += 1
        style_counter[str(meta.get("style", "unknown"))] += 1
        validity_counter[str(example["validity"])] += 1
        plausibility_counter[str(example["plausibility"])] += 1
        combo_counter[f"validity={example['validity']},plausibility={example['plausibility']}"] += 1
        if meta.get("schema_name"):
            schema_counter[str(meta["schema_name"])] += 1
        if meta.get("mood") and meta.get("figure"):
            mood_counter[f"{meta['mood']}-{meta['figure']}"] += 1
        if meta.get("figure"):
            figure_counter[str(meta["figure"])] += 1
        if meta.get("fallacy"):
            fallacy_counter[str(meta["fallacy"])] += 1
        if example["validity"]:
            relevant_pair_counter[str(tuple(example["relevant_premises"]))] += 1
        candidate_pair_counter[str(tuple(meta["candidate_premises"]))] += 1
        core_order_counter[str(meta.get("core_premise_order_in_output", "unknown"))] += 1
        injected_counter[int(meta.get("num_injected_distractors", max(0, premise_count - 2)))] += 1
        core_counter[str(meta["core_signature"])] += 1

        parts = split_sentences(example["syllogism"])
        for premise in parts[:premise_count]:
            key = normalize_text(premise).lower()
            premise_text_counter[key] += 1
            if CATEGORICAL_RE.match(premise):
                canonical_premises += 1
            else:
                noncanonical_premises += 1
        if parts and parts[-1].startswith("Therefore,"):
            therefore_conclusions += 1
        elif parts:
            non_therefore_conclusions += 1
        if " of them " in normalize_text(example["syllogism"]).lower():
            ambiguous_them += 1

    duplicate_cores = {core: count for core, count in core_counter.items() if count > 1}
    duplicate_full = len(examples) - len({normalize_text(e["syllogism"]).lower() for e in examples})
    validation = validation_errors(examples, semantics, figures)

    return {
        "semantics": semantics,
        "figures": allowed_figures(figures),
        "num_examples": len(examples),
        "source_counts": dict(source_counter),
        "style_counts": dict(style_counter),
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
        "core_premise_order_counts": dict(core_order_counter),
        "injected_distractor_counts": dict(sorted(injected_counter.items())),
        "duplicate_core_count": sum(count - 1 for count in duplicate_cores.values()),
        "duplicate_full_syllogism_count": duplicate_full,
        "ambiguous_them_count": ambiguous_them,
        "canonical_premise_count": canonical_premises,
        "noncanonical_premise_count": noncanonical_premises,
        "therefore_conclusion_count": therefore_conclusions,
        "non_therefore_conclusion_count": non_therefore_conclusions,
        "max_exact_premise_repetition": max(premise_text_counter.values(), default=0),
        "most_repeated_premises": premise_text_counter.most_common(20),
        "validation_error_count": len(validation),
        "validation_errors_preview": validation[:30],
    }


def split_counts(total: int, premise_counts: list[int]) -> dict[int, int]:
    if total < 0:
        raise ValueError("total must be non-negative")
    if not premise_counts:
        raise ValueError("premise_counts must not be empty")
    base = total // len(premise_counts)
    remainder = total % len(premise_counts)
    return {p: base + (1 if i < remainder else 0) for i, p in enumerate(premise_counts)}


def generate_synthetic_examples(
    count: int,
    premise_counts: list[int],
    rng: random.Random,
    semantics: Semantics,
    figures: list[int],
    include_id: bool,
    include_metadata: bool,
    *,
    style: str,
    existing_syllogisms: set[str] | None = None,
) -> list[dict[str, Any]]:
    valid_schemas, invalid_schemas = build_schema_catalog(semantics, figures)
    if not valid_schemas:
        raise RuntimeError("No valid schemas found. This should be impossible.")
    if not invalid_schemas:
        raise RuntimeError("No invalid schemas found. This should be impossible.")

    counts_by_premise = split_counts(count, premise_counts)
    examples: list[dict[str, Any]] = []
    seen_cores: set[str] = set()
    seen_syllogisms: set[str] = existing_syllogisms if existing_syllogisms is not None else set()
    schema_serial: Counter[tuple[bool, bool]] = Counter()
    position_serial: Counter[int] = Counter()

    # Generate in target strata: each premise count bucket cycles through the four validity/plausibility combinations.
    for premise_count in premise_counts:
        target = counts_by_premise[premise_count]
        local_index = 0
        attempts = 0
        while local_index < target:
            attempts += 1
            if attempts > max(1000, target * 700):
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
                style=style,
                swap_core_order=True,
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
    return examples


def generate_ufal_integrated_examples(
    count: int,
    premise_counts: list[int],
    rng: random.Random,
    ufal_examples: list[dict[str, Any]],
    include_id: bool,
    include_metadata: bool,
    *,
    style: str,
    existing_syllogisms: set[str] | None = None,
) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    seen_syllogisms: set[str] = existing_syllogisms if existing_syllogisms is not None else set()
    position_serial: Counter[int] = Counter()
    usable_indices = list(range(len(ufal_examples)))
    rng.shuffle(usable_indices)
    idx_cursor = 0
    attempts = 0
    while len(examples) < count:
        attempts += 1
        if attempts > max(2000, count * 1000):
            raise RuntimeError("Too many failed UFAL integration attempts.")
        base_index = usable_indices[idx_cursor % len(usable_indices)]
        idx_cursor += 1
        base = ufal_examples[base_index]
        premise_count = premise_counts[len(examples) % len(premise_counts)]
        if premise_count < 2:
            continue
        example = build_ufal_integrated_example(
            base,
            base_index,
            premise_count,
            rng,
            include_id,
            include_metadata,
            position_serial[premise_count],
            style=style,
        )
        position_serial[premise_count] += 1
        if example is None:
            continue
        syllo_sig = normalize_text(example["syllogism"]).lower()
        if syllo_sig in seen_syllogisms:
            continue
        seen_syllogisms.add(syllo_sig)
        examples.append(example)
    return examples


def generate_subtask2_examples(
    count: int,
    premise_counts: list[int],
    seed: int = 42,
    semantics: Semantics = "aristotelian",
    include_id: bool = True,
    include_metadata: bool = False,
    *,
    figures: list[int] | None = None,
    style: str = "mixed",
    source_mode: str = "synthetic",
    ufal_input: Path | None = None,
    ufal_ratio: float = 0.5,
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    figures = allowed_figures(figures)
    if any(p < 2 for p in premise_counts):
        raise ValueError("All premise counts must be at least 2.")
    if style not in {"canonical", "varied", "mixed"}:
        raise ValueError("style must be one of: canonical, varied, mixed")
    if source_mode not in {"synthetic", "ufal", "mixed"}:
        raise ValueError("source_mode must be one of: synthetic, ufal, mixed")
    if not (0.0 <= ufal_ratio <= 1.0):
        raise ValueError("ufal_ratio must be between 0 and 1")

    examples: list[dict[str, Any]] = []
    seen_syllogisms: set[str] = set()

    if source_mode == "synthetic":
        examples.extend(
            generate_synthetic_examples(
                count,
                premise_counts,
                rng,
                semantics,
                figures,
                include_id,
                include_metadata,
                style=style,
                existing_syllogisms=seen_syllogisms,
            )
        )
    else:
        if ufal_input is None:
            raise ValueError("--ufal-input is required when --source-mode is ufal or mixed")
        ufal_examples = load_ufal_examples(ufal_input)
        if source_mode == "ufal":
            ufal_count = count
            synthetic_count = 0
        else:
            ufal_count = round(count * ufal_ratio)
            synthetic_count = count - ufal_count
        if ufal_count:
            examples.extend(
                generate_ufal_integrated_examples(
                    ufal_count,
                    premise_counts,
                    rng,
                    ufal_examples,
                    include_id,
                    include_metadata,
                    style=style,
                    existing_syllogisms=seen_syllogisms,
                )
            )
        if synthetic_count:
            examples.extend(
                generate_synthetic_examples(
                    synthetic_count,
                    premise_counts,
                    rng,
                    semantics,
                    figures,
                    include_id,
                    include_metadata,
                    style=style,
                    existing_syllogisms=seen_syllogisms,
                )
            )

    rng.shuffle(examples)
    errors = validation_errors(examples, semantics, figures)
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
    print(f"Figures: {audit['figures']}")
    print(f"Sources: {audit['source_counts']}")
    print(f"Styles: {audit['style_counts']}")
    print(f"Premise counts: {audit['premise_count_distribution']}")
    print(f"Validity counts: {audit['validity_counts']}")
    print(f"Plausibility counts: {audit['plausibility_counts']}")
    print(f"Figure counts: {audit['figure_counts']}")
    print(f"Fallacy counts: {audit['fallacy_counts']}")
    print(f"Core premise order counts: {audit['core_premise_order_counts']}")
    print(f"Injected distractor counts: {audit['injected_distractor_counts']}")
    print(f"Duplicate core count: {audit['duplicate_core_count']}")
    print(f"Duplicate full syllogism count: {audit['duplicate_full_syllogism_count']}")
    print(f"Ambiguous 'of them' count: {audit['ambiguous_them_count']}")
    print(f"Canonical premise count: {audit['canonical_premise_count']}")
    print(f"Non-canonical premise count: {audit['noncanonical_premise_count']}")
    print(f"Therefore conclusion count: {audit['therefore_conclusion_count']}")
    print(f"Non-Therefore conclusion count: {audit['non_therefore_conclusion_count']}")
    print(f"Max exact premise repetition: {audit['max_exact_premise_repetition']}")
    print(f"Validation error count: {audit['validation_error_count']}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate high diversity Subtask 2 syllogism data.")
    parser.add_argument("-n", "--count", type=int, default=1200, help="Number of examples to generate.")
    parser.add_argument(
        "-p", "--premises", type=int, nargs="+", default=[5, 6, 7],
        help="Premise counts to generate. Example: -p 2 3 4 5 6 7. Counts are split as evenly as possible.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducible generation.")
    parser.add_argument(
        "--semantics", choices=["modern", "aristotelian"], default="aristotelian",
        help=(
            "modern = no existential import; aristotelian = evaluate only models where S/P/M are non-empty "
            "so traditional E/I conversion is respected."
        ),
    )
    parser.add_argument(
        "--figures", type=int, nargs="+", default=[1, 2, 3],
        help="Syllogistic figures to use. Use '--figures 1 2 3 4' for the later modern fourth figure.",
    )
    parser.add_argument(
        "--style", choices=["canonical", "varied", "mixed"], default="mixed",
        help="Surface form style. 'mixed' combines canonical all/no/some with paraphrases.",
    )
    parser.add_argument(
        "--source-mode", choices=["synthetic", "ufal", "mixed"], default="synthetic",
        help="Generate synthetic data, UFAL-integrated data, or a mixture of both.",
    )
    parser.add_argument("--ufal-input", type=Path, default=None, help="Path to cleared_ufal_train_set.json.")
    parser.add_argument(
        "--ufal-ratio", type=float, default=0.5,
        help="When --source-mode mixed, fraction of output generated from UFAL cores.",
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
        figures=args.figures,
        style=args.style,
        source_mode=args.source_mode,
        ufal_input=args.ufal_input,
        ufal_ratio=args.ufal_ratio,
    )
    audit = audit_examples(examples, args.semantics, args.figures)

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