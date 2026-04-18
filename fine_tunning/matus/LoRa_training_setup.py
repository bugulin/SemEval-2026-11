import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from huggingface_hub import login
import json
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig


def HGlogin(HGtoken):
  login(token=HGtoken)
  return()

def Lamma_setup():
  # Load Llama 3.1 8B
  model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"
  
  bnb_config = BitsAndBytesConfig(
      load_in_4bit=True,
      bnb_4bit_quant_type="nf4",
      bnb_4bit_compute_dtype=torch.bfloat16,
      bnb_4bit_use_double_quant=True,
  )
  
  model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_config, device_map="auto")
  tokenizer = AutoTokenizer.from_pretrained(model_id)
  tokenizer.pad_token = tokenizer.eos_token
  return(model, tokenizer)


def training_data_setup(input_path="/content/train_data.json"):
   #Lamma 3.1 is instruct tuned model, so we need to change the json format to fit the instruct structure
    output_path = "my_syllogisms.jsonl"
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # If your JSON is a single list of dictionaries
    if not isinstance(data, list):
        data = [data]

    with open(output_path, 'w', encoding='utf-8') as f:
        for entry in data:
            # Create the new dictionary structure
            new_entry = {
                "instruction": entry["syllogism"],
                "response": str(entry["validity"]).lower()  # Converts False to "false"
            }
            # Write as JSONL (JSON Lines) which is best for training
            f.write(json.dumps(new_entry) + '\n')

    print(f"Conversion complete! {len(data)} rows saved to {output_path}")


def formatting_prompts_func(example):
    # The Trainer now passes a single example (dict) at a time this is the place where one can play with the training prompt format
    text = (
        f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
        f"You are a logic expert specializing in syllogisms.<|eot_id|>"
        f"<|start_header_id|>user<|end_header_id|>\n\n"
        f"For the following syllogism decide whether it is valid or not. You should respond with sigle word, without change in capitalization or puctuation, either true, if it is valid, or false otherwise. Here start your syllogism: {example['instruction']}<|eot_id|>"
        f"<|start_header_id|>assistant<|end_header_id|>\n\n"
        f"{example['response']}<|eot_id|>"
    )
    return text # Return a single string

def LoRa_configuration():
  # Load your local file from the sidebar
  dataset = load_dataset("json", data_files="/content/my_syllogisms.jsonl", split="train")
  
  # 1. LoRA Configuration
  peft_config = LoraConfig(
      r=16,
      lora_alpha=32,
      target_modules="all-linear",
      bias="none",
      task_type="CAUSAL_LM"
  )
  
  training_args = SFTConfig(
      output_dir="./syllogism_model_checkpoints",
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
  )
  
  # 3. Initialize the Trainer
  trainer = SFTTrainer(
      model=model,
      train_dataset=dataset,
      peft_config=peft_config,
      formatting_func=formatting_prompts_func,
      args=training_args,
  )
  return trainer

# Verify we are on the GPU
print(f"Working on GPU: {torch.cuda.get_device_name(0)}")
HGtoken = input('Your HG writing token:')

HGlogin(HGtoken)
model,tokenizer = Lamma_setup()

path_to_training_data = input('add path to the training data or add the default one: /content/my_syllogisms.jsonl')

test_dataset=load_dataset("json", data_files=path_to_training_data, split="train")

training_data_setup()

trainer = LoRa_configuration()

trainer.train()
