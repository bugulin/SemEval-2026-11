#this code block downloads Lamma 8B (if access is granted) and put it into LoRa adapter.

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from huggingface_hub import login


HGtoken=input('INSERT YOURS HUGGING FACE WRITING TOKEN') #this is needed for access to the Lamma 8B model one needs to ask permision from Meta, bud it is simple
login(token=HGtoken)
print("Login Successful!")

model_id = "meta-llama/Llama-3.1-8B-Instruct"
adapter_id = "MatusZelko/llama-3.1-syllogism-lora"

# Load base model in 4-bit
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_config, device_map="auto")
print('model is running')

# Load and attach the adapter
model = PeftModel.from_pretrained(model, adapter_id)
tokenizer = AutoTokenizer.from_pretrained(adapter_id)
print('LoRa is running')
