import mlflow
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM


def llama_3_2_1b_instruct_pipeline():
    pipe = pipeline("text-generation", model="meta-llama/Llama-3.2-1B-Instruct")
    messages = [
        {"role": "user", "content": "34.8*.181+144.2/8974="},
    ]
    return pipe(messages)


def llama_3_2_1b_instruct_direct():
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")
    model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")
    messages = [
        {"role": "user", "content": "Who are you?"},
    ]
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)

    outputs = model.generate(**inputs, max_new_tokens=200)
    return tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:])


def llama_3_2_1b_instruct_cuda():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")
    device = "cuda"

    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")
    model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Llama-3.2-1B-Instruct",
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    ).to(device)

    messages = [
        {"role": "user", "content": "Who are you?"},
    ]

    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(device)

    with torch.inference_mode():
        outputs = model.generate(**inputs, max_new_tokens=200)

    return tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[-1]:],
        skip_special_tokens=True,
    )


def llama_3_2_1b_instruct_mlflow():
    print("Creating pipeline...")
    pipe = pipeline(
        task="text-classification",
        model="meta-llama/Llama-3.2-1B-Instruct"
    )

    print("Setting up MLflow...")
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment("mlflow-demo")

    print("Entering MLflow run...")
    with mlflow.start_run():
        print("Logging pipeline...")
        mlflow.transformers.log_model(
            transformers_model=pipe,
            name="model",
            input_example=["MLflow works nicely with Transformers."]
        )


if __name__ == '__main__':
    print(llama_3_2_1b_instruct_mlflow())
