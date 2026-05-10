import argparse
import hashlib
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

"""
Deterministic Subtask 2 syllogism generator.

Goal:
- generate Subtask 2 examples without local/api LLM calls.
- keep validity labels correct by construction
- add irrelevant premises and compute zero-based relevant_premises mechanically
- output JSON and optionally JSONL

This intentionally does not ask a language model whether a syllogism is valid.
The generator chooses a formal schema first, then fills it with natural terms.
"""

"""
ARGUMENTS:

-n / --count       number of examples
-p / --premises    number of premises before the conclusion
--seed             reproducibility seed
-o / --output      output JSON path
--jsonl-output     optional JSONL output path
--no-id            generate examples without id fields
"""

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

# Each triple is (a, b, c), matching the project notation:
# Aab = All b are a.
DOMAIN_POOLS = {
    "food": {
        "chains": [
            ("prepared foods", "bakery items", "sourdough loaves"),
            ("drink orders", "coffee orders", "double espresso orders"),
            ("vegetarian dishes", "salad bowls", "lunch specials"),
            ("savory dishes", "spicy curries", "pepper stews"),
            ("desserts", "chocolate cakes", "birthday menu items"),
        ],
        "negative": [
            ("meat dishes", "vegetarian meals", "lentil salads"),
            ("raw foods", "baked goods", "rye loaves"),
            ("desserts", "savory dishes", "grilled steak dinners"),
            ("beverages", "solid foods", "sandwich orders"),
        ],
        "some": [
            ("prepared foods", "bakery items", "weekend orders"),
            ("drink orders", "coffee orders", "morning purchases"),
            ("vegetarian dishes", "salad bowls", "family dinner options"),
            ("savory dishes", "spicy curries", "restaurant specials"),
        ],
        "invalid_conclusions": [
            "some lunch specials are served with extra herbs",
            "all dinner orders are popular with tourists",
            "some restaurant specials are chosen for weekend meals",
            "all coffee orders are made before noon",
        ],
    },
    "home": {
        "chains": [
            ("household chores", "laundry tasks", "towel loads"),
            ("cleaning jobs", "vacuuming tasks", "hallway vacuuming tasks"),
            ("home projects", "painting jobs", "ceiling painting jobs"),
            ("plumbing repairs", "leaky faucet repairs", "kitchen sink repairs"),
            ("home maintenance tasks", "closet organizing tasks", "shelf labeling tasks"),
        ],
        "negative": [
            ("outdoor chores", "ironing tasks", "shirt pressing tasks"),
            ("garden chores", "vacuuming tasks", "apartment chore lists"),
            ("decor decisions", "plumbing repairs", "maintenance reports"),
            ("wet repairs", "painting jobs", "ceiling painting jobs"),
        ],
        "some": [
            ("household chores", "laundry tasks", "weekend task lists"),
            ("home care tasks", "ironing tasks", "travel prep lists"),
            ("home projects", "painting jobs", "renovation plans"),
            ("cleaning jobs", "vacuuming tasks", "apartment chore lists"),
        ],
        "invalid_conclusions": [
            "some task lists require new labels",
            "all repair jobs are finished before noon",
            "some apartment chores are noisy",
            "all maintenance tasks use fresh paint",
        ],
    },
    "transport": {
        "chains": [
            ("transport options", "public transport routes", "night tram lines"),
            ("commuting routines", "driving routines", "morning highway commutes"),
            ("vehicle care tasks", "car maintenance jobs", "oil change jobs"),
            ("travel plans", "airline itineraries", "multi city itineraries"),
            ("urban trips", "train journeys", "regional rail journeys"),
        ],
        "negative": [
            ("air travel", "train journeys", "regional rail journeys"),
            ("walking trips", "driving routines", "weekday travel plans"),
            ("bike rides", "airline itineraries", "multi city itineraries"),
            ("vehicle care tasks", "public transport routes", "city travel plans"),
        ],
        "some": [
            ("transport options", "public transport routes", "weekday commute plans"),
            ("commuting routines", "cycling trips", "school travel plans"),
            ("daily outings", "dog walking trips", "evening park walks"),
            ("travel plans", "airline itineraries", "summer vacation plans"),
        ],
        "invalid_conclusions": [
            "some routes are delayed in rain",
            "all commute plans use reserved seats",
            "some vehicle tasks happen before sunrise",
            "all travel plans need printed tickets",
        ],
    },
    "office": {
        "chains": [
            ("office tasks", "email tasks", "client follow up emails"),
            ("work activities", "office meetings", "budget review meetings"),
            ("work arrangements", "remote work schedules", "hybrid Friday schedules"),
            ("career steps", "job interviews", "final round interviews"),
            ("planning tasks", "deadline reviews", "release readiness reviews"),
        ],
        "negative": [
            ("coffee breaks", "email tasks", "client follow up emails"),
            ("vacation days", "office meetings", "weekly calendars"),
            ("retirement plans", "job interviews", "career plans"),
            ("personal hobbies", "deadline reviews", "team schedules"),
        ],
        "some": [
            ("office tasks", "email tasks", "morning admin blocks"),
            ("professional activities", "networking events", "conference plans"),
            ("work activities", "office meetings", "weekly calendars"),
            ("career steps", "job interviews", "candidate schedules"),
        ],
        "invalid_conclusions": [
            "some meetings last more than an hour",
            "all reviews happen on Mondays",
            "some interviews are discussed in Slack",
            "all email tasks require approval emails",
        ],
    },
    "social": {
        "chains": [
            ("social events", "birthday parties", "surprise parties"),
            ("family occasions", "family dinners", "holiday meals"),
            ("relationship messages", "text messages", "good morning texts"),
            ("formal ceremonies", "weddings", "evening receptions"),
            ("online interactions", "social media posts", "birthday posts"),
        ],
        "negative": [
            ("private conversations", "public posts", "birthday posts"),
            ("weekday errands", "wedding events", "evening receptions"),
            ("legal contracts", "friendly texts", "small talk messages"),
            ("family obligations", "first dates", "coffee dates"),
        ],
        "some": [
            ("social events", "birthday parties", "weekend plans"),
            ("friendly messages", "text messages", "evening reminders"),
            ("family occasions", "family dinners", "holiday plans"),
            ("online interactions", "social media posts", "public updates"),
        ],
        "invalid_conclusions": [
            "some plans require reservations",
            "all messages are sent after midnight",
            "some conversations happen during lunch",
            "all invitations are accepted quickly",
        ],
    },
    "media": {
        "chains": [
            ("entertainment activities", "movie nights", "Friday screenings"),
            ("interactive games", "video games", "cooperative games"),
            ("written works", "novels", "mystery novels"),
            ("public performances", "concerts", "orchestra concerts"),
            ("creative hobbies", "photography sessions", "portrait shoots"),
        ],
        "negative": [
            ("silent activities", "concerts", "orchestra concerts"),
            ("outdoor hikes", "board games", "strategy games"),
            ("fictional works", "documentary films", "nature documentaries"),
            ("digital media", "printed novels", "library books"),
        ],
        "some": [
            ("entertainment activities", "movie nights", "weekend plans"),
            ("creative hobbies", "photography sessions", "museum visits"),
            ("musical activities", "instrument practice sessions", "evening lessons"),
            ("tabletop games", "board games", "party activities"),
        ],
        "invalid_conclusions": [
            "some activities last two hours",
            "all performances need tickets",
            "some books are recommended by friends",
            "all games are played on weekends",
        ],
    },
    "health": {
        "chains": [
            ("wellness routines", "morning exercise sessions", "short workouts"),
            ("skin care products", "cleansers", "gentle face washes"),
            ("health appointments", "dental checkups", "annual cleanings"),
            ("protective habits", "sunscreen applications", "beach day routines"),
            ("relaxation practices", "yoga sessions", "evening stretches"),
        ],
        "negative": [
            ("medical treatments", "haircuts", "barber appointments"),
            ("sleep periods", "exercise sessions", "morning workouts"),
            ("food groups", "vitamin tablets", "daily supplements"),
            ("outdoor activities", "mental health breaks", "quiet pauses"),
        ],
        "some": [
            ("wellness routines", "morning exercise sessions", "weekday routines"),
            ("protective habits", "sunscreen applications", "summer routines"),
            ("relaxation practices", "yoga sessions", "evening routines"),
            ("health appointments", "dental checkups", "calendar entries"),
        ],
        "invalid_conclusions": [
            "some routines improve sleep quality",
            "all appointments are scheduled early",
            "some breaks happen after lunch",
            "all products are used daily",
        ],
    },
    "environment": {
        "chains": [
            ("environmental habits", "recycling routines", "paper sorting routines"),
            ("waste reduction practices", "composting habits", "kitchen scrap collections"),
            ("resource saving actions", "water conservation habits", "short shower routines"),
            ("weather events", "rainy days", "stormy afternoons"),
            ("seasonal conditions", "winter weather patterns", "snowfall warnings"),
        ],
        "negative": [
            ("summer heatwaves", "winter weather patterns", "snowfall warnings"),
            ("dry conditions", "rainy days", "stormy afternoons"),
            ("electricity usage", "water conservation habits", "short shower routines"),
            ("trash disposal", "composting habits", "kitchen scrap collections"),
        ],
        "some": [
            ("environmental habits", "recycling routines", "apartment routines"),
            ("weather events", "rainy days", "weekend forecasts"),
            ("seasonal conditions", "winter weather patterns", "travel delays"),
            ("resource saving actions", "water conservation habits", "household plans"),
        ],
        "invalid_conclusions": [
            "some habits save money",
            "all forecasts change overnight",
            "some routines are written on calendars",
            "all warnings arrive by phone",
        ],
    },
    "animals": {
        "chains": [
            ("pets", "adopted animals", "shelter cats"),
            ("training activities", "dog training sessions", "recall practice sessions"),
            ("animal care visits", "veterinary appointments", "vaccination appointments"),
            ("farm animals", "dairy cows", "barn animals"),
            ("outdoor animals", "local wildlife", "garden birds"),
        ],
        "negative": [
            ("wild animals", "house pets", "indoor cats"),
            ("bird feeders", "aquarium equipment", "filter pumps"),
            ("farm animals", "aquarium fish", "neon tetras"),
            ("horse rides", "vet appointments", "checkup visits"),
        ],
        "some": [
            ("pets", "adopted animals", "family animals"),
            ("training activities", "dog training sessions", "weekend plans"),
            ("animal care visits", "veterinary appointments", "calendar entries"),
            ("outdoor animals", "local wildlife", "garden sightings"),
        ],
        "invalid_conclusions": [
            "some animals need morning feeding",
            "all visits require a carrier",
            "some sessions happen in parks",
            "all sightings are photographed",
        ],
    },
    "errands": {
        "chains": [
            ("errands", "grocery trips", "weekly shopping trips"),
            ("financial tasks", "tax payments", "income tax filings"),
            ("online purchases", "online shopping orders", "bookstore orders"),
            ("official documents", "passport renewals", "embassy appointments"),
            ("borrowed items", "library books", "reserved novels"),
        ],
        "negative": [
            ("free activities", "online purchases", "bookstore orders"),
            ("shopping trips", "tax filings", "income tax filings"),
            ("digital files", "passport documents", "paper forms"),
            ("personal possessions", "library books", "reserved novels"),
        ],
        "some": [
            ("errands", "grocery trips", "Saturday plans"),
            ("financial tasks", "budgeting sessions", "monthly plans"),
            ("official documents", "passport renewals", "travel preparations"),
            ("borrowed items", "library books", "reading lists"),
        ],
        "invalid_conclusions": [
            "some errands take less than an hour",
            "all forms are submitted online",
            "some shopping trips happen after work",
            "all reminders ring in the morning",
        ],
    },
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

ABSURD_CHAINS = [
    ("silver shadows", "curious teapots", "rainy orchestras"),
    ("edible constellations", "singing spoons", "midnight kitchens"),
    ("plastic forests", "whispering boots", "museum pockets"),
    ("floating teacups", "invisible mittens", "backward comets"),
    ("tidy hurricanes", "sleeping lanterns", "attic festivals"),
    ("paper moons", "singing ladders", "transparent violins"),
]

ABSURD_NEGATIVE = [
    ("wooden oceans", "flying teacups", "festival baskets"),
    ("quiet thunderstorms", "singing clocks", "library windows"),
    ("friendly meteors", "glass bicycles", "garden pockets"),
    ("liquid mountains", "paper shoes", "kitchen moons"),
]

ABSURD_DISTRACTORS = [
    "Some purple staircases hum softly near {topic}.",
    "Many upside-down umbrellas whisper about {topic} at dawn.",
    "A few transparent pockets collect moonlight during {topic}.",
    "Some silver echoes drift past {topic} every Tuesday.",
    "Several clockwork peaches discuss {topic} in empty libraries.",
]

PLAUSIBLE_DISTRACTORS = [
    "Some people mention {topic} while planning their week.",
    "A few calendars include reminders related to {topic}.",
    "Many people compare notes about {topic} during casual conversations.",
    "Some online articles give practical advice about {topic}.",
    "Several families make small plans involving {topic}.",
    "Some people buy supplies before working on {topic}.",
]

VALID_SCHEMAS = ["barbara", "celarent", "darii", "ferio"]
INVALID_SCHEMAS = ["undistributed_middle", "invalid_particular_chain", "too_strong", "negative_too_strong"]


def clean_topic(topic: str) -> str:
    return topic.lower().replace("-", " ")


def normalize_text(text: str) -> str:
    return " ".join(text.strip().split())


def hash_syllogism(text: str) -> str:
    return hashlib.sha256(normalize_text(text).encode("utf-8")).hexdigest()


def all_sentence(b: str, a: str) -> str:
    return f"All {b} are {a}."


def no_sentence(b: str, a: str) -> str:
    return f"No {b} are {a}."


def some_sentence(c: str, b: str) -> str:
    return f"Some {c} are {b}."


def some_not_sentence(c: str, a: str) -> str:
    return f"Some {c} are not {a}."


def all_conclusion(c: str, a: str) -> str:
    return f"Therefore, all {c} are {a}."


def no_conclusion(c: str, a: str) -> str:
    return f"Therefore, no {c} are {a}."


def some_conclusion(c: str, a: str) -> str:
    return f"Therefore, some {c} are {a}."


def some_not_conclusion(c: str, a: str) -> str:
    return f"Therefore, some {c} are not {a}."


def choose_pool(topic: str, plausible: bool, rng: random.Random) -> dict[str, Any]:
    if not plausible:
        return {
            "chains": ABSURD_CHAINS,
            "negative": ABSURD_NEGATIVE,
            "some": ABSURD_CHAINS,
            "invalid_conclusions": [
                "some of them are louder than weather",
                "all of them dissolve into music",
                "some of them are made of moonlight",
                "all of them dream about umbrellas",
            ],
        }

    domain = TOPIC_TO_DOMAIN.get(topic, "errands")
    return DOMAIN_POOLS[domain]


def build_core(validity: bool, plausibility: bool, topic: str, rng: random.Random) -> tuple[list[str], str, str]:
    pool = choose_pool(topic, plausibility, rng)

    if validity:
        schema = rng.choice(VALID_SCHEMAS)
        if schema == "barbara":
            a, b, c = rng.choice(pool["chains"])
            return [all_sentence(b, a), all_sentence(c, b)], all_conclusion(c, a), schema
        if schema == "celarent":
            a, b, c = rng.choice(pool["negative"])
            return [no_sentence(b, a), all_sentence(c, b)], no_conclusion(c, a), schema
        if schema == "darii":
            a, b, c = rng.choice(pool["some"])
            return [all_sentence(b, a), some_sentence(c, b)], some_conclusion(c, a), schema
        if schema == "ferio":
            a, b, c = rng.choice(pool["negative"])
            return [no_sentence(b, a), some_sentence(c, b)], some_not_conclusion(c, a), schema

    schema = rng.choice(INVALID_SCHEMAS)
    if schema == "undistributed_middle":
        a, b, c = rng.choice(pool["chains"])
        conclusion = rng.choice(pool["invalid_conclusions"])
        return [all_sentence(b, a), all_sentence(c, a)], f"Therefore, {conclusion}.", schema
    if schema == "invalid_particular_chain":
        a, b, c = rng.choice(pool["some"])
        return [some_sentence(b, a), some_sentence(c, b)], some_conclusion(c, a), schema
    if schema == "too_strong":
        a, b, c = rng.choice(pool["some"])
        return [all_sentence(b, a), some_sentence(c, b)], all_conclusion(c, a), schema
    if schema == "negative_too_strong":
        a, b, c = rng.choice(pool["negative"])
        return [no_sentence(b, a), some_sentence(c, b)], f"Therefore, all {c} are not {a}.", schema

    raise AssertionError(f"Unknown schema: {schema}")


def build_distractors(topic: str, plausibility: bool, needed: int, rng: random.Random) -> list[str]:
    templates = PLAUSIBLE_DISTRACTORS if plausibility else ABSURD_DISTRACTORS
    candidates = [template.format(topic=clean_topic(topic)) for template in templates]
    rng.shuffle(candidates)

    distractors: list[str] = []
    while len(distractors) < needed:
        candidate = candidates[len(distractors) % len(candidates)]
        if candidate not in distractors:
            distractors.append(candidate)
        else:
            distractors.append(f"Some people keep separate notes about {clean_topic(topic)}.")
    return distractors


def choose_relevant_positions(num_premises: int, rng: random.Random) -> tuple[int, int]:
    if num_premises < 3:
        raise ValueError("Subtask 2 examples should have at least 3 premises.")

    preferred = [
        (2, num_premises - 1),
        (1, num_premises - 1),
        (3, num_premises - 1) if num_premises > 4 else (0, num_premises - 1),
        (2, num_premises - 2) if num_premises > 4 else (1, num_premises - 1),
        (0, num_premises - 1),
    ]
    valid_pairs = [pair for pair in preferred if 0 <= pair[0] < pair[1] < num_premises]
    all_pairs = [(i, j) for i in range(num_premises) for j in range(i + 1, num_premises)]

    if rng.random() < 0.75:
        return rng.choice(valid_pairs)
    return rng.choice(all_pairs)


def generate_example(index: int, num_premises: int, rng: random.Random, include_id: bool) -> dict[str, Any]:
    topic = TOPICS[index % len(TOPICS)]

    # Balanced cycle: valid/plausible, invalid/plausible, valid/implausible, invalid/implausible.
    cycle = index % 4
    validity = cycle in {0, 2}
    plausibility = cycle in {0, 1}

    core_premises, conclusion, schema = build_core(validity, plausibility, topic, rng)
    distractors = build_distractors(topic, plausibility, num_premises - 2, rng)

    premises: list[str | None] = [None] * num_premises
    relevant_premises: list[int] = []

    if validity:
        first_pos, second_pos = choose_relevant_positions(num_premises, rng)
        premises[first_pos] = core_premises[0]
        premises[second_pos] = core_premises[1]
        relevant_premises = [first_pos, second_pos]
    else:
        # Invalid examples officially use [] even though they still contain core looking premises.
        first_pos, second_pos = choose_relevant_positions(num_premises, rng)
        premises[first_pos] = core_premises[0]
        premises[second_pos] = core_premises[1]

    distractor_iter = iter(distractors)
    for i, value in enumerate(premises):
        if value is None:
            premises[i] = next(distractor_iter)

    syllogism = normalize_text(" ".join([str(p) for p in premises] + [conclusion]))

    example: dict[str, Any] = {
        "syllogism": syllogism,
        "validity": validity,
        "plausibility": plausibility,
        "relevant_premises": relevant_premises,
    }

    if include_id:
        example = {"id": hash_syllogism(syllogism), **example}

    return example


def validate_examples(examples: list[dict[str, Any]], num_premises: int) -> list[str]:
    errors: list[str] = []
    seen_ids: set[str] = set()
    seen_syllogisms: set[str] = set()

    for i, example in enumerate(examples):
        prefix = f"example {i}"
        required = {"syllogism", "validity", "plausibility", "relevant_premises"}
        missing = required - set(example)
        if missing:
            errors.append(f"{prefix}: missing fields {sorted(missing)}")
            continue

        if "id" in example:
            if not isinstance(example["id"], str):
                errors.append(f"{prefix}: id is not a string")
            elif example["id"] in seen_ids:
                errors.append(f"{prefix}: duplicate id {example['id']}")
            seen_ids.add(example.get("id", ""))

        syllogism = example["syllogism"]
        if not isinstance(syllogism, str):
            errors.append(f"{prefix}: syllogism is not a string")
            continue
        if syllogism in seen_syllogisms:
            errors.append(f"{prefix}: duplicate syllogism")
        seen_syllogisms.add(syllogism)

        if not isinstance(example["validity"], bool):
            errors.append(f"{prefix}: validity is not bool")
        if not isinstance(example["plausibility"], bool):
            errors.append(f"{prefix}: plausibility is not bool")

        relevant = example["relevant_premises"]
        if not isinstance(relevant, list):
            errors.append(f"{prefix}: relevant_premises is not a list")
            continue
        if example["validity"] and len(relevant) != 2:
            errors.append(f"{prefix}: valid example should have exactly 2 relevant premises")
        if not example["validity"] and relevant:
            errors.append(f"{prefix}: invalid example should have empty relevant_premises")
        for value in relevant:
            if not isinstance(value, int):
                errors.append(f"{prefix}: relevant premise index is not int")
            elif value < 0 or value >= num_premises:
                errors.append(f"{prefix}: relevant premise index {value} is out of range")

        sentence_count = syllogism.count(".")
        if sentence_count != num_premises + 1:
            errors.append(f"{prefix}: expected {num_premises + 1} sentences, found {sentence_count}")
        if "Therefore," not in syllogism:
            errors.append(f"{prefix}: missing 'Therefore,' conclusion marker")

    return errors


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


def summarize(examples: list[dict[str, Any]], num_premises: int) -> None:
    validity_counts = Counter(example["validity"] for example in examples)
    plausibility_counts = Counter(example["plausibility"] for example in examples)
    relevant_pair_counts = Counter(tuple(example["relevant_premises"]) for example in examples if example["validity"])

    print(f"Generated examples: {len(examples)}")
    print(f"Premises per example: {num_premises}")
    print(f"Validity counts: {dict(validity_counts)}")
    print(f"Plausibility counts: {dict(plausibility_counts)}")
    print("Most common relevant premise pairs:")
    for pair, count in relevant_pair_counts.most_common(10):
        print(f"  {pair}: {count}")


def generate_subtask2_examples(
    count: int,
    num_premises: int = 6,
    seed: int = 42,
    include_id: bool = True,
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    examples: list[dict[str, Any]] = []
    seen_syllogisms: set[str] = set()

    attempt = 0
    while len(examples) < count:
        example = generate_example(attempt, num_premises, rng, include_id)
        attempt += 1
        if example["syllogism"] in seen_syllogisms:
            continue
        seen_syllogisms.add(example["syllogism"])
        examples.append(example)

        if attempt > count * 20:
            raise RuntimeError("Too many duplicate generations; increase topic/schema variation.")

    errors = validate_examples(examples, num_premises)
    if errors:
        preview = "\n".join(errors[:20])
        raise ValueError(f"Generated data failed validation:\n{preview}")

    return examples


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate deterministic high-quality Subtask 2 syllogism data."
    )
    parser.add_argument("-n", "--count", type=int, default=1000, help="Number of examples to generate.")
    parser.add_argument(
        "-p",
        "--premises",
        type=int,
        default=6,
        help="Number of premises before the conclusion. Default is 6 to resemble official Subtask 2 data.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducible generation.")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("data/2/generated_subtask2.json"),
        help="Output JSON path.",
    )
    parser.add_argument(
        "--jsonl-output",
        type=Path,
        default=None,
        help="Optional JSONL output path.",
    )
    parser.add_argument("--no-id", action="store_true", help="Do not include id fields.")
    args = parser.parse_args()

    examples = generate_subtask2_examples(
        count=args.count,
        num_premises=args.premises,
        seed=args.seed,
        include_id=not args.no_id,
    )

    save_json(args.output, examples)
    print(f"Saved JSON to: {args.output}")

    if args.jsonl_output is not None:
        save_jsonl(args.jsonl_output, examples)
        print(f"Saved JSONL to: {args.jsonl_output}")

    summarize(examples, args.premises)


if __name__ == "__main__":
    main()
