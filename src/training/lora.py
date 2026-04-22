import json
import logging
from typing import TYPE_CHECKING

import torch
from datasets import Split, load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer

if TYPE_CHECKING:
    from collections.abc import Sequence


logger = logging.getLogger(__name__)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)


def prepare_dataset(input_path: str, output_path: str) -> str:
    """Prepare training data.

    Llama 3.1 is an instruction-tuned model, so we need to change the JSON format to fit
    the instruct structure.
    """
    with open(input_path) as f:
        data = json.load(f)
    assert isinstance(data, list), "Training data should be stored as a list."

    with open(output_path, "w") as f:
        for entry in data:
            sample = {
                "instruction": entry["syllogism"],
                "response": ("FALSE", "TRUE")[entry["validity"]],
            }
            json.dump(sample, f)
            f.write("\n")

    logger.info("Conversion complete! %d entries saved to %s.", len(data), output_path)
    return output_path


def formatting_prompts_func(example):
    # The Trainer now passes a single example (dict) at a time this is the place where
    # one can play with the training prompt format
    return (
        f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
        f"You are a logic expert specializing in syllogisms.<|eot_id|>"
        f"<|start_header_id|>user<|end_header_id|>\n\n"
        f"For the following syllogism decide whether it is valid or not. You should respond with sigle word, without change in capitalization or puctuation, either true, if it is valid, or false otherwise. Here start your syllogism: {example['instruction']}<|eot_id|>"
        f"<|start_header_id|>assistant<|end_header_id|>\n\n"
        f"{example['response']}<|eot_id|>"
    )


def fine_tune(model_name: str, datasets: Sequence[str], output_dir: str) -> None:
    """Fine-tune an LLM using LoRA."""
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
    )

    tokenizer = AutoTokenizer.from_pretrained(model)
    tokenizer.pad_token = tokenizer.eos_token

    dataset = load_dataset("json", data_files=datasets, split=Split.TRAIN)

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules="all-linear",
            bias="none",
            task_type="CAUSAL_LM",
        ),
        formatting_func=formatting_prompts_func,
        args=SFTConfig(
            output_dir=output_dir,
            max_length=512,
            dataset_text_field="text",
            packing=False,
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
    # trainer.push_to_hub()
