import json
import random
from pathlib import Path


def split_train_validation(
    input_file: str,
    train_output_file: str = "train_split.json",
    validation_output_file: str = "validation_split.json",
    validation_ratio: float = 0.10,
    seed: int = 42
):
    """
    Split a JSON training dataset into train and validation JSON files.

    Assumes the input JSON file contains a list of samples.
    """

    input_path = Path(input_file)

    with input_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("Expected the input JSON file to contain a list of samples.")

    random.seed(seed)
    random.shuffle(data)

    validation_size = int(len(data) * validation_ratio)

    validation_data = data[:validation_size]
    train_data = data[validation_size:]

    with open(train_output_file, "w", encoding="utf-8") as f:
        json.dump(train_data, f, indent=4, ensure_ascii=False)

    with open(validation_output_file, "w", encoding="utf-8") as f:
        json.dump(validation_data, f, indent=4, ensure_ascii=False)

    print(f"Total samples: {len(data)}")
    print(f"Training samples: {len(train_data)}")
    print(f"Validation samples: {len(validation_data)}")
    print(f"Saved train set to: {train_output_file}")
    print(f"Saved validation set to: {validation_output_file}")


if __name__ == "__main__":
    split_train_validation(
        input_file="merged_train_hand.json",
        train_output_file="train.json",
        validation_output_file="validation.json",
        validation_ratio=0.10,
        seed=42
    )