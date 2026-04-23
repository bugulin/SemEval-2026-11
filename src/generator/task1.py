import asyncio
import os
import uuid

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# Note: Update this path if running outside of Kaggle
MODEL_PATH = "/kaggle/input/models/janflajk/qwen/pytorch/default/1/models/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28"

TOPICS = [
    "Baking Bread",
    "Spicy Food",
    "Morning Coffee",
    "Vegetarianism",
    "Grilling Steak",
    "Restaurant Service",
    "Fresh Herbs",
    "Fast Food",
    "Salad Making",
    "Chocolate Desserts",
    "Laundry Day",
    "Ironing Clothes",
    "Vacuuming Carpets",
    "Interior Design",
    "Fixing Leaky Faucets",
    "Painting Walls",
    "Houseplants",
    "Mowing the Lawn",
    "Organizing Closets",
    "Taking Out Trash",
    "Commuting to Work",
    "Driving in Traffic",
    "Cycling to School",
    "Public Transportation",
    "Car Maintenance",
    "Gas Prices",
    "Finding Parking",
    "Airline Travel",
    "Train Journeys",
    "Walking the Dog",
    "Sending Emails",
    "Office Meetings",
    "Working from Home",
    "Job Interviews",
    "Coffee Breaks",
    "Office Supplies",
    "Project Deadlines",
    "Professional Networking",
    "Career Changes",
    "Retirement Planning",
    "First Dates",
    "Wedding Planning",
    "Birthday Parties",
    "Holiday Gift Giving",
    "Family Dinners",
    "Neighbor Disputes",
    "Long-distance Friendships",
    "Small Talk",
    "Text Messaging",
    "Social Media Etiquette",
    "Watching Movies",
    "Playing Video Games",
    "Reading Novels",
    "Attending Concerts",
    "Photography",
    "Hiking Trails",
    "Camping Trips",
    "Museum Visits",
    "Board Game Nights",
    "Learning a Musical Instrument",
    "Morning Exercise",
    "Skincare Routines",
    "Getting a Haircut",
    "Quality of Sleep",
    "Taking Vitamins",
    "Dental Hygiene",
    "Fashion Trends",
    "Applying Sunscreen",
    "Mental Health Breaks",
    "Yoga Practice",
    "Rainy Weather",
    "Summer Heatwaves",
    "Winter Snowfall",
    "Recycling Habits",
    "Composting Waste",
    "Electricity Usage",
    "Water Conservation",
    "Seasonal Changes",
    "Spring Allergies",
    "Using Umbrellas",
    "Cat Behavior",
    "Pet Adoption",
    "Feeding Birds",
    "Visiting the Vet",
    "Dog Training",
    "Aquarium Maintenance",
    "Local Wildlife",
    "Farm Animals",
    "Beekeeping",
    "Horseback Riding",
    "Grocery Shopping",
    "Budgeting Finances",
    "Paying Taxes",
    "Setting Alarm Clocks",
    "Online Shopping",
    "Battery Life",
    "Lost Keys",
    "Mirror Reflections",
    "Renewing Passports",
    "Library Books",
]

FORMATS = [
    "Aab,Abc ⊢ Aac meaning: All b are a, All c are b therefore All c are a. E.g. All humans are mortal , All Greeks are humans . Therefore, all Greeks are mortal .",
    "Eab,Abc ⊢ Eac meaning: No b are a, All c are b therefore No c are a. E.g. No reptiles have fur , All snakes are reptiles . Therefore, no snakes have fur .",
    "Aab,Ibc ⊢ Iac meaning: All b are a, Some c are b therefore Some c are a. E.g. All birds have feathers , Some pets are birds . Therefore, some pets have feathers .",
    "Eab,Ibc ⊢ Oac meaning: No b are a, Some c are b therefore Some c are not a. E.g. No vegetarians eat meat , Some athletes are vegetarians. Therefore, some athletes do not eat meat.",
]

PREMISES_PROMPT = """
# ROLE:
You are an expert in the field of formal logic and Syllogism creation in natural language.

# WORKFLOW:
Your goal is to generate ONLY the first two premises of a syllogism (without a conclusion) in a predefined format {formats} based on the topic of {topic}.

# Step 1:
Generate exactly 2 premises of a syllogism based on the properties described above. The output should look something like this:
````There are people who are students. A few children are students.```

# NOTES:
- Do NOT include any special symbols like '⊢'.
- Do NOT generate a conclusion — stop after the second premise.
- Make sure there are always exactly 2 sentences.
- Please return only the two premises in the specified format.
- ALWAYS ANSWER IN ENGLISH
"""

SYLLOGISM_PROMPT = """
# ROLE:
You are an expert in the field of formal logic and Syllogism filling in natural language.

# WORKFLOW:
Your goal is to finish an incomplete syllogism (2 premises + 1 conclusion) in a predefined format: {formats} based on the topic of the provided premises.
USE THE PROVIDED PREMISES AS INSPIRATION.
The syllogism must match the following properties:
- Logical validity: {valid} (if True, the conclusion must follow necessarily from the premises; if False, it must not)
- Plausibility: {plausible} (if True, the premises and conclusion should reflect real-world believable facts; if False, at least one should be counterintuitive or false)

# Step 1:
Analyse the two premises as well as the format and whether you are asked to make the syllogism (in)valid or (im)plausible.
# Step 2:
Generate a conclusion to the 2 premises: {premises} that satisfies the properties above. Pay attention to the logical validity and plausibility. The output should look something like this:
```There are people who are students. A few children are students. It follows, then, that some children are people.```

# NOTES:
- If you are asked to create invalid or implausible syllogism DO NOT be afraid to use conclusion outside of the topic
- Do NOT include any special symbols like '⊢'.
- Make sure there are always exactly 3 sentences.
- Check if the final conclusion always fullfills the plausibility and validity.
- Please return only the syllogism in the specified format.
- ALWAYS ANSWER IN ENGLISH
"""


def remove_non_utf8(text):
    if isinstance(text, str):
        return text.encode("utf-8", "ignore").decode("utf-8")
    return text


async def create_premises(
    topics, formats, prompt, tokenizer, model, enable_thinking=True, max_topics=100
):
    ans = []
    max_topics = min(len(topics), max_topics)

    for topic in tqdm(topics[:max_topics], desc="Generating Premises"):
        for f in formats:
            try:
                messages = [
                    {"role": "user", "content": prompt.format(formats=f, topic=topic)}
                ]
                text = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=enable_thinking,
                )

                model_inputs = tokenizer([text], return_tensors="pt").to("cuda")
                generated_ids = model.generate(
                    **model_inputs,
                    max_new_tokens=64,
                    do_sample=True,
                    temperature=0.8,
                    top_k=10,
                    top_p=0.3,
                    pad_token_id=tokenizer.eos_token_id,
                )

                output_ids = generated_ids[0][len(model_inputs.input_ids[0]) :].tolist()
                response = tokenizer.decode(output_ids, skip_special_tokens=True).strip(
                    "\n"
                )

                ans.append(
                    {
                        "premises": response.replace("\n", " "),
                        "format": f,
                        "topic": topic,
                    }
                )
            except Exception as e:
                print(f"Error in create_premises: {e}")
                ans.append("network_or_connectivity_issue")

    return ans


async def finish_syllogism(premises, prompt, tokenizer, model, enable_thinking=True):
    ans = []

    for premise in tqdm(premises, desc="Finishing Syllogisms"):
        for v in [True, False]:
            for p in [True, False]:
                try:
                    messages = [
                        {
                            "role": "user",
                            "content": prompt.format(
                                formats=premise["format"],
                                topic=premise["topic"],
                                valid=str(v),
                                plausible=str(p),
                                premises=premise["premises"],
                            ),
                        }
                    ]

                    text = tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True,
                        enable_thinking=enable_thinking,
                    )

                    model_inputs = tokenizer([text], return_tensors="pt").to("cuda")
                    generated_ids = model.generate(
                        **model_inputs,
                        max_new_tokens=64,
                        do_sample=True,
                        temperature=2.0,
                        top_k=75,
                        top_p=1.7,
                        pad_token_id=tokenizer.eos_token_id,
                    )

                    output_ids = generated_ids[0][
                        len(model_inputs.input_ids[0]) :
                    ].tolist()
                    response = tokenizer.decode(
                        output_ids, skip_special_tokens=True
                    ).strip("\n")

                    ans.append(
                        {
                            "id": uuid.uuid4(),
                            "syllogism": response.replace("\n", " ")
                            .encode("utf-8", "ignore")
                            .decode("utf-8"),
                            "validity": v,
                            "plausibility": p,
                        }
                    )
                except Exception as e:
                    print(f"Error in finish_syllogism: {e}")
                    ans.append("network_or_connectivity_issue")

    return ans


async def main():
    # Set display options
    pd.set_option("display.max_colwidth", None)

    # Handle Token (Kaggle secrets alternative)
    # If not on Kaggle, set the environment variable HF_TOKEN manually in your terminal
    if "HF_TOKEN" not in os.environ:
        try:
            from kaggle_secrets import UserSecretsClient

            user_secrets = UserSecretsClient()
            os.environ["HF_TOKEN"] = user_secrets.get_secret("HF_TOKEN")
        except ImportError:
            print("Warning: HF_TOKEN not found in environment and not on Kaggle.")

    print("Loading Model and Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH, use_fast=False, trust_remote_code=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, device_map="cuda", torch_dtype=torch.float16, trust_remote_code=True
    )

    raw_premises = await create_premises(
        TOPICS, FORMATS, PREMISES_PROMPT, tokenizer, model, max_topics=100
    )

    syllogism_list = await finish_syllogism(
        raw_premises, SYLLOGISM_PROMPT, tokenizer, model
    )

    df = pd.DataFrame(syllogism_list)
    df = df.map(remove_non_utf8)

    df_cleaned = df.drop_duplicates(subset=["syllogism"], keep=False)

    print("Saving results to dataset_final.csv and data.json...")
    df.to_csv("dataset_final.csv", index=False)

    df_final = pd.read_csv("dataset_final.csv")
    df_final.to_json("data.json", orient="records", force_ascii=False)

    print("Process Complete.")


def generate_syllogisms(n: int):
    asyncio.run(main())
