import json
import logging
import re
from pathlib import Path
from typing import Sequence

import torch
from datasets import Split, load_dataset
from jinja2 import Environment, FileSystemLoader
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import GRPOConfig, GRPOTrainer

logger = logging.getLogger(__name__)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)


def prepare_dataset(input_path: str, output_path: str | Path, prompt_path: str = "solver_thinking.j2") -> str:
    """Prepare training data.

    Llama 3.1 is an instruction-tuned model, so we need to change the JSON format to fit
    the instruct structure.
    """
    with open(input_path) as f:
        data = json.load(f)
    assert isinstance(data, list), "Training data should be stored as a list."

    system_prompt = Environment(
        # /path/to/src/training/grpo_lora.py -> /path/to/src/prompts
        loader=FileSystemLoader(Path(__file__).parent.parent / "prompts")
    ).get_template(prompt_path).render()

    with open(output_path, "w") as f:
        for entry in data:
            sample = {
                "prompt": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": entry["syllogism"]}
                ],
                "validity": entry["validity"],
            }
            json.dump(sample, f)
            f.write("\n")

    logger.info("Conversion complete! %d entries saved to %s.", len(data), output_path)
    return output_path


def validity_reward(completions, validity, **kwargs) -> list[float]:
    """
    Reward based on whether the last sentence correctly says 'valid' or 'invalid'.

    If the last sentence contains both the words 'valid' and 'invalid', or
    neither of them, or negation (n't, not), the reward is 0.
    If the last sentence contains exactly one of the words 'valid' or 'invalid',
    the reward is 1.0 for the correct answer, -1.0 for the incorrect one.

    :param completions: List of model completions.
    :param validity: List of ground truth validity labels, extracted from the
        `validity` field of the dataset.
    """
    rewards = []
    for completion, is_valid in zip(completions, validity):
        text = completion[0]["content"].strip().lower()
        last_sentence = text.split(".")[-1] if "." in text else text
        predictions = {
            True: re.search(r"\bvalid\b", last_sentence),
            False: re.search(r"\binvalid\b", last_sentence),
            None: re.search(r"(\bnot|n't)\b", last_sentence),
        }
        if predictions[None] or predictions[True] == predictions[False]:
            rewards.append(0.0)
        elif predictions[is_valid]:
            rewards.append(1.0)
        else:
            rewards.append(-1.0)
    return rewards


def fine_tune(
        model_name: str,
        datasets: Sequence[str],
        output_dir: str,
        output_repo: str | None = None,
) -> None:
    """Fine-tune an LLM using GRPO and LoRA."""
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    dataset = load_dataset("json", data_files=datasets, split=Split.TRAIN)

    trainer = GRPOTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules="all-linear",
            bias="none",
            task_type="CAUSAL_LM",
        ),
        reward_funcs=validity_reward,
        args=GRPOConfig(
            output_dir=output_dir,
            max_completion_length=1024,
            per_device_train_batch_size=5,
            gradient_accumulation_steps=4,
            learning_rate=2e-4,
            num_train_epochs=2,
            logging_steps=1,
            fp16=not torch.cuda.is_bf16_supported(),
            bf16=torch.cuda.is_bf16_supported(),
        ),
    )
    trainer.train()

    if output_repo is not None:
        trainer.push_to_hub(output_repo)
