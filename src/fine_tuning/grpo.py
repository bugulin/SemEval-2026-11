import re

from datasets import load_dataset
from peft import LoraConfig
from trl import GRPOTrainer, GRPOConfig

SYSTEM_PROMPT = (
    "You are a logic expert. Given a syllogism, determine whether the conclusion "
    "follows logically from the premises. Think step by step, then state your final "
    "answer in the last sentence as 'The syllogism is valid' or 'The syllogism is invalid'."
)


def make_prompt(example):
    return {
        "prompt": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": example["syllogism"]},
        ]
    }


dataset = load_dataset("json", data_files="data/1/train_data.json", split="train")
dataset = dataset.map(make_prompt)


def validity_reward(completions, validity, **kwargs):
    """Reward based on whether the last sentence correctly says 'valid' or 'invalid'."""
    rewards = []
    for completion, is_valid in zip(completions, validity):
        text = completion[0]["content"].strip().lower()
        last_sentence = text.split(".")[-1] if "." in text else text
        # Check for "invalid" first since it contains "valid" as a substring
        predictions = {
            True: re.search(r"\bvalid\b", last_sentence),
            False: re.search(r"\binvalid\b", last_sentence)
        }
        if predictions[True] == predictions[False]:
            rewards.append(0.0)
        elif predictions[is_valid]:
            rewards.append(1.0)
        else:
            rewards.append(-1.0)
    return rewards


peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules="all-linear",
    bias="none",
    task_type="CAUSAL_LM",
)

training_config = GRPOConfig(
    output_dir="grpo_output",
    logging_steps=10,
)

trainer = GRPOTrainer(
    model="Qwen/Qwen2-0.5B-Instruct",
    reward_funcs=validity_reward,
    args=training_config,
    peft_config=peft_config,
    train_dataset=dataset,
)
trainer.train()
